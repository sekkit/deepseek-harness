/**
 * Hot-reload behavior of the adaptive multi-key scheduler: a settings edit
 * that reshapes a live pool's limits (rpm ceiling, concurrency, gap) must
 * update the gate in place and keep every learned per-key meter, instead of
 * rebuilding the gate and making the pool relearn quota the hard way.
 *
 * @module dsh-llm-pi-ai/scheduler-hot-reload.spec
 */

import { describe, expect, it } from 'vitest'
import { PiAiAdapter } from '../src/adapter.ts'

type SchedulerOutcome = 'ok' | 'rpm' | 'tpm' | 'quota' | 'auth' | 'server'

interface SchedulerProbe {
  acquireSlot(
    provider: string,
    refs: readonly string[],
    signal: AbortSignal | undefined,
    initialRpm: number | undefined,
    maxConcurrency: number,
    poolRpm: number,
    minGapMs: number,
  ): Promise<string | undefined>
  reportOutcome(provider: string, ref: string | undefined, outcome: SchedulerOutcome, tokens?: number): void
  slots: Map<string, {
    poolId: string
    rpmCeiling: number
    maxConcurrency: number
    poolRpm: number
    minGapMs: number
    keys: Map<string, { rpmCapacity: number; cooldownUntil: number }>
  }>
}

function probe(): SchedulerProbe {
  const adapter = new PiAiAdapter({
    profiles: () => new Map(),
    resolveApiKey: async () => 'secret',
    auth: {
      credentials: {
        resolve: async () => undefined,
      } as unknown as import('../src/adapter.ts').PiAiAuthInjection['credentials'],
      authContext: {} as import('../src/adapter.ts').PiAiAuthInjection['authContext'],
    },
  })
  return adapter as unknown as SchedulerProbe
}

describe('scheduler hot reload', () => {
  it('reshapes a live gate in place and keeps learned meters when limits change', async () => {
    const s = probe()
    const refs = ['A', 'B', 'C']

    // First configuration: rpm 2/key, pool concurrency 1, gap 500ms.
    const first = await s.acquireSlot('p', refs, undefined, 2, 1, 0, 500)
    expect(first).toBeDefined()
    const gateBefore = s.slots.get('p')
    expect(gateBefore).toBeDefined()

    // A real failure teaches the pool: rpm capacity shrinks, key cools down.
    s.reportOutcome('p', first, 'rpm')
    const meterBefore = gateBefore!.keys.get(first!)!
    expect(meterBefore.rpmCapacity).toBeLessThan(2)
    const cooldown = meterBefore.cooldownUntil
    expect(cooldown).toBeGreaterThan(0)

    // Hot reconfig of the SAME pool: rpm 7, concurrency 4, gap 300ms.
    const second = await s.acquireSlot('p', refs, undefined, 7, 4, 0, 300)
    expect(second).toBeDefined()
    // The cooled-down first key is skipped in favour of a healthy one.
    expect(second).not.toBe(first)

    // The gate is reshaped, not rebuilt.
    expect(s.slots.size).toBe(1)
    const gateAfter = s.slots.get('p')
    expect(gateAfter).toBe(gateBefore)
    expect(gateAfter!.maxConcurrency).toBe(4)
    expect(gateAfter!.minGapMs).toBe(300)
    expect(gateAfter!.rpmCeiling).toBe(7)

    // Learned state survives: same meter object, same cooldown, same shrunken cap.
    const meterAfter = gateAfter!.keys.get(first!)!
    expect(meterAfter).toBe(meterBefore)
    expect(meterAfter.cooldownUntil).toBe(cooldown)
    expect(meterAfter.rpmCapacity).toBe(meterBefore.rpmCapacity)
  })

  it('rebuilds the gate when the key pool itself changes', async () => {
    const s = probe()
    const first = await s.acquireSlot('p', ['A', 'B'], undefined, 2, 1, 0, 500)
    expect(first).toBeDefined()
    const gateBefore = s.slots.get('p')

    // A different pool identity (different refs) is a new gate: learned state
    // cannot meaningfully transfer across accounts.
    const second = await s.acquireSlot('p', ['A', 'B', 'C'], undefined, 2, 1, 0, 500)
    expect(second).toBeDefined()
    expect(s.slots.size).toBe(1)
    expect(s.slots.get('p')).not.toBe(gateBefore)
  })
})
