import { EventEmitter } from 'node:events';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AgentLoop } from '../agent/index.js';
import { TempestCommand } from '../index.js';
import { MAX_TASK_RETRIES } from '../mission/index.js';
import { OperatorAgent } from '../operators/index.js';
import { KillChainPhase, type Task } from '../types/index.js';

const TIMEOUT_MS = 10_000;
const MAX_MS = 60_000;
const BLOCKED_TICKS = 30;

/** A controllable stand-in for AgentLoop: each run() hands back a promise the test settles. */
function makeLoop() {
  const emitter = new EventEmitter();
  const runs: Array<{ resolve: (r: unknown) => void; reject: (e: Error) => void }> = [];
  return {
    runs,
    emit: (event: string, data: unknown) => emitter.emit(event, data),
    on: (event: string, fn: (...args: unknown[]) => void) => emitter.on(event, fn),
    off: (event: string, fn: (...args: unknown[]) => void) => emitter.off(event, fn),
    setLLM: vi.fn(),
    cancel: vi.fn(),
    run: vi.fn(() => new Promise((resolve, reject) => { runs.push({ resolve, reject }); })),
  };
}

function setup() {
  const command = new TempestCommand({ name: 'Stall Op', llm: { provider: 'mock', model: 'mock-model' } });
  command.targetEnv.addTarget({ name: 'localhost', address: '127.0.0.1', type: 'web_application', zone: 'external' });
  command.start();
  // Drive ticks by hand and skip target seeding, so each test owns its exact task list.
  clearInterval((command as any).tickInterval);
  (command as any).taskSeeded = true;
  const mission = command.mission.getActiveMission()!;
  const queue = command.mission.getTaskQueue();
  const addTask = (overrides: Partial<Task> = {}): Task => {
    const task: Task = {
      id: `task-${queue.size + 1}`,
      missionId: mission.id,
      name: 'Probe',
      description: 'probe 127.0.0.1',
      phase: KillChainPhase.RECON,
      operatorType: 'recon',
      status: 'pending',
      priority: 5,
      dependencies: [],
      createdAt: Date.now(),
      ...overrides,
    };
    queue.add(task);
    return task;
  };
  const tick = () => (command as any).tick() as Promise<void>;
  return { command, mission, queue, addTask, tick };
}

function setupWithLoop() {
  const ctx = setup();
  const operator = ctx.command.spawnOperator('Recon-1', 'recon');
  const loop = makeLoop();
  operator.attachArsenal(ctx.command.arsenal, loop as any);
  return { ...ctx, operator, loop };
}

const advance = (ms: number) => vi.setSystemTime(Date.now() + ms);
const flush = async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); };

describe('mission stall recovery', () => {
  const savedEnv = { timeout: process.env.T3MP3ST_TASK_TIMEOUT_MS, max: process.env.T3MP3ST_TASK_MAX_MS };

  beforeEach(() => {
    process.env.T3MP3ST_TASK_TIMEOUT_MS = String(TIMEOUT_MS);
    process.env.T3MP3ST_TASK_MAX_MS = String(MAX_MS);
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    if (savedEnv.timeout === undefined) delete process.env.T3MP3ST_TASK_TIMEOUT_MS;
    else process.env.T3MP3ST_TASK_TIMEOUT_MS = savedEnv.timeout;
    if (savedEnv.max === undefined) delete process.env.T3MP3ST_TASK_MAX_MS;
    else process.env.T3MP3ST_TASK_MAX_MS = savedEnv.max;
  });

  describe('dispatch backstop', () => {
    it('does not reap a slow task that is still making progress', async () => {
      const { command, addTask, tick, loop } = setupWithLoop();
      const task = addTask();
      await tick();
      for (let i = 0; i < 5; i++) {
        advance(TIMEOUT_MS - 2_000);
        loop.emit('agent:iteration', { iteration: i });
        await tick();
      }
      expect(task.status).toBe('assigned');
      expect(loop.cancel).not.toHaveBeenCalled();
      command.stop();
    });

    it('reaps a silent task, cancels its run, and retries a timeout once before stalling', async () => {
      const { command, addTask, tick, loop } = setupWithLoop();
      const task = addTask();
      await tick();

      advance(TIMEOUT_MS + 1);
      await tick();
      expect(loop.cancel).toHaveBeenCalledTimes(1);
      expect(task.retryCount).toBe(1);
      expect(task.status).toBe('assigned');
      expect(loop.run).toHaveBeenCalledTimes(2);

      advance(TIMEOUT_MS + 1);
      await tick();
      const status = command.getStatus();
      expect(task.status).toBe('failed');
      expect(status.paused).toBe(true);
      expect(status.stallReason).toContain('stalled in reconnaissance');
      command.stop();
    });

    it('enforces the hard ceiling even when a run keeps reporting progress', async () => {
      const { command, addTask, tick, loop } = setupWithLoop();
      const task = addTask();
      await tick();
      for (let i = 0; i < 8; i++) {
        advance(TIMEOUT_MS - 2_000);
        loop.emit('agent:iteration', { iteration: i });
        await tick();
      }
      expect(loop.cancel).toHaveBeenCalledTimes(1);
      expect(task.retryCount).toBe(1);
      command.stop();
    });

    it('does not count time spent paused against an in-flight task', async () => {
      const { command, addTask, tick, loop } = setupWithLoop();
      const task = addTask();
      await tick();
      command.pause();
      advance(MAX_MS * 10);
      command.resume();
      await tick();
      expect(task.status).toBe('assigned');
      expect(loop.cancel).not.toHaveBeenCalled();
      command.stop();
    });
  });

  describe('abandoned runs', () => {
    it('ignores a late result from a reaped run instead of clobbering the retry', async () => {
      const { command, addTask, tick, loop, operator } = setupWithLoop();
      const task = addTask();
      await tick();
      advance(TIMEOUT_MS + 1);
      await tick();
      expect(loop.run).toHaveBeenCalledTimes(2);

      loop.runs[0].resolve({ success: true, summary: 'late', findings: [] });
      await flush();

      expect(operator.status).toBe('executing');
      expect(operator.state.currentTask).toBe(task.id);
      expect(task.status).toBe('assigned');
      command.stop();
    });

    it('does not decompose or re-dispatch when a reaped run later throws', async () => {
      const { command, addTask, tick, loop, operator } = setupWithLoop();
      addTask();
      const decomposed = vi.fn();
      operator.on('task:decomposed', decomposed);
      const prompt = vi.spyOn((operator as any).llm, 'prompt');
      await tick();
      advance(TIMEOUT_MS + 1);
      await tick();

      loop.runs[0].reject(new Error('late failure'));
      await flush();

      expect(prompt).not.toHaveBeenCalled();
      expect(decomposed).not.toHaveBeenCalled();
      expect(operator.status).toBe('executing');
      command.stop();
    });
  });

  describe('AgentLoop.cancel', () => {
    function runLoop(chatWithTools: (...args: any[]) => Promise<unknown>) {
      const { command, addTask } = setup();
      const llm = { getProvider: () => 'mock', chatWithTools: vi.fn(chatWithTools), chat: vi.fn() };
      const loop = new AgentLoop(llm as any, command.arsenal, { maxIterations: 5 });
      const pending = loop.run(addTask(), 'system prompt');
      return { command, llm, loop, pending };
    }

    it('aborts the in-flight LLM call through its signal and stops iterating', async () => {
      const seen: AbortSignal[] = [];
      const { command, llm, loop, pending } = runLoop((_m, _t, options) => new Promise((_resolve, reject) => {
        seen.push(options.signal);
        options.signal.addEventListener('abort', () => reject(new Error('aborted')));
      }));
      await flush();
      loop.cancel('backstop');
      const result = await pending;
      expect(seen[0].aborted).toBe(true);
      expect(result.success).toBe(false);
      expect(llm.chatWithTools).toHaveBeenCalledTimes(1);
      expect(llm.chat).not.toHaveBeenCalled();
      command.stop();
    });

    it('settles even when the provider ignores the abort signal', async () => {
      const { command, llm, loop, pending } = runLoop(() => new Promise(() => undefined));
      await flush();
      loop.cancel('backstop');
      const result = await pending;
      expect(result.success).toBe(false);
      expect(llm.chat).not.toHaveBeenCalled();
      command.stop();
    });
  });

  describe('resume', () => {
    it('re-queues failed current-phase tasks instead of re-stalling', async () => {
      const { command, queue, addTask, tick } = setup();
      const task = addTask();
      queue.fail(task.id, 'provider unavailable');
      await tick();
      expect(command.getStatus().stallReason).toContain('provider unavailable');
      // Non-timeout failures are not auto-retried: they wait for the operator.
      expect(task.retryCount ?? 0).toBe(0);

      command.resume();

      expect(command.getStatus().paused).toBe(false);
      expect(command.getStatus().stallReason).toBeNull();
      expect(task.status).toBe('pending');
      expect(task.retryCount).toBe(1);
      command.stop();
    });

    it('says when retries are exhausted rather than offering another resume', async () => {
      const { command, queue, addTask, tick } = setup();
      const task = addTask();
      for (let i = 0; i < MAX_TASK_RETRIES; i++) {
        queue.fail(task.id, 'boom');
        await tick();
        expect(command.getStatus().stallReason).toContain('resume to retry');
        command.resume();
      }
      queue.fail(task.id, 'boom');
      await tick();
      expect(command.getStatus().stallReason).toContain('retries exhausted');
      command.resume();
      expect(task.status).toBe('failed');
      command.stop();
    });

    it('leaves failed tasks alone when the mission was paused by hand', async () => {
      const { command, queue, addTask } = setup();
      const task = addTask();
      queue.fail(task.id, 'boom');
      command.pause();
      command.resume();
      expect(task.status).toBe('failed');
      command.stop();
    });
  });

  describe('undispatchable work', () => {
    beforeEach(() => {
      vi.spyOn(OperatorAgent.prototype, 'assignTask').mockReturnValue(new Promise(() => undefined));
    });

    it('spawns the missing operator on local-agent backends', async () => {
      const { command, addTask, tick } = setup();
      vi.spyOn((command as any).llm, 'getProvider').mockReturnValue('local-agent');
      const task = addTask({ operatorType: 'scanner', phase: KillChainPhase.RECON });
      await tick();
      expect(command.cell.getAllOperators().map(op => op.archetype)).toContain('scanner');
      expect(task.status).toBe('assigned');
      command.stop();
    });

    it('spawns at most one operator per archetype on local-agent backends', async () => {
      const { command, addTask, tick } = setup();
      vi.spyOn((command as any).llm, 'getProvider').mockReturnValue('local-agent');
      addTask({ operatorType: 'scanner' });
      addTask({ operatorType: 'scanner' });
      await tick();
      expect(command.cell.getAllOperators().filter(op => op.archetype === 'scanner')).toHaveLength(1);
      command.stop();
    });

    it('does not let burned operators use up the per-archetype cap', async () => {
      const { command, addTask, tick } = setup();
      for (const callsign of ['Scanner-1', 'Scanner-2', 'Scanner-3']) {
        command.spawnOperator(callsign, 'scanner').increaseDetectionRisk(1);
      }
      const task = addTask({ operatorType: 'scanner' });
      await tick();
      expect(task.status).toBe('assigned');
      command.stop();
    });

    it('stalls with a reason when pending work can never be dispatched', async () => {
      const { command, addTask, tick } = setup();
      command.spawnOperator('Recon-1', 'recon');
      addTask({ dependencies: ['task-that-never-finishes'] });
      for (let i = 0; i < BLOCKED_TICKS - 1; i++) await tick();
      expect(command.getStatus().paused).toBe(false);
      await tick();
      const status = command.getStatus();
      expect(status.paused).toBe(true);
      expect(status.stallReason).toContain('cannot be dispatched');
      expect(status.stallReason).toContain('unmet dependencies');
      command.stop();
    });

    it('does not call work blocked while an operator is busy', async () => {
      const { command, addTask, tick } = setup();
      const operator = command.spawnOperator('Recon-1', 'recon');
      (operator as any)._state.status = 'executing';
      addTask({ dependencies: ['task-that-never-finishes'] });
      for (let i = 0; i < BLOCKED_TICKS * 2; i++) await tick();
      expect(command.getStatus().paused).toBe(false);
      command.stop();
    });
  });

  it('reports why OPSEC paused the mission', async () => {
    const { command, addTask, tick } = setup();
    addTask();
    vi.spyOn(command.opsec, 'isAbortRecommended').mockReturnValue(true);
    await tick();
    const status = command.getStatus();
    expect(status.paused).toBe(true);
    expect(status.stallReason).toContain('OPSEC');
    command.stop();
  });
});
