/**
 * Event-only filesystem observation policy; it registers no service. A bounded owner/target map
 * records every authoritative presence/absence observation, single-slot intent listeners derive
 * guards from that state, and the provider performs the atomic freshness/no-clobber check. Without
 * this plugin, tools retain the bare provider's unconditional mutation behavior. See the package
 * README for composition rules.
 * @module @deepseek-ai/dsh-fs-observation-policy
 */

import type { Context } from '@deepseek-ai/cordis'
import { FsError } from '@deepseek-ai/dsh-fs'
import type { FsObservation, FsTarget, FsVersion, FsWriteIntent } from '@deepseek-ai/dsh-fs'
import type { FsObservationActor } from './types.ts'

export type { FsObservationActor } from './types.ts'

/**
 * Per-context observed-file state and the three `fs/*` decisions over it. One
 * instance is created per `apply()` so disposal can drop all state for HMR.
 *
 * State is keyed by a STABLE owner key rather than the session object itself:
 * a session exposing a string `id` keys directly on it, so its prior
 * observations survive the session-object replacement a resume performs
 * (within this process). Sessions without an `id` fall back to a stable
 * per-object uid, preserving the old strict per-object isolation. Strong keys
 * are no longer garbage-collected with their sessions, so both levels are
 * bounded explicitly: least-recently-set owners and oldest targets are evicted
 * at the caps below.
 */
class ObservedStateGate {
  /** Bound on tracked owners; defensive against unbounded growth across many sessions. */
  private static readonly MAX_OWNERS = 512
  /** Bound on tracked targets per owner; generous against any real read volume. */
  private static readonly MAX_TARGETS_PER_OWNER = 4096

  /**
   * Observed-file state, keyed first by the stable owner key, then by
   * {@link FsTarget.targetKey}. An entry's presence is the prior-observation record; its
   * discriminant keeps confirmed absence distinct from an unseen target.
   */
  private observed = new Map<string, Map<string, FsObservation>>()

  /**
   * Stable fallback keys for sessions that expose no string `id`: one uid per
   * anonymous session object, so such owners keep exactly their old per-object
   * isolation (two distinct objects never share observed state).
   */
  private anonymousIds = new WeakMap<object, string>()
  private anonymousCounter = 0

  /**
   * Derive the stable observed-state owner key from the opaque event actor —
   * normally the active agent session. A session carrying a non-empty string
   * `id` keys on it directly (resume-stable); otherwise the session object gets
   * a stable per-object uid. `undefined` when no owner can be derived (e.g. a
   * direct tool call with no agent); such calls read freely but cannot satisfy
   * the write/edit prior-observation policy.
   */
  private owner(actor: object | undefined): string | undefined {
    // tsgolint treats object as assignable to weak FsObservationActor, while tsc still requires the structural cast for property access.
    // See the analyzer-divergence consequence in .agents/notes/archived/process/2026-07-29-oxlint-linter.md.
    // oxlint-disable-next-line typescript/no-unnecessary-type-assertion -- The analyzers disagree on this weak type.
    const session = (actor as FsObservationActor | undefined)?.agent?.session
    if (!session) return undefined
    const id = (session as { id?: unknown }).id
    if (typeof id === 'string' && id.length > 0) return id
    let uid = this.anonymousIds.get(session)
    if (uid === undefined) {
      uid = `anon:${++this.anonymousCounter}`
      this.anonymousIds.set(session, uid)
    }
    return uid
  }

  private get(ownerKey: string, targetKey: string): FsObservation | undefined {
    return this.observed.get(ownerKey)?.get(targetKey)
  }

  private set(ownerKey: string, targetKey: string, observation: FsObservation): void {
    let byTarget = this.observed.get(ownerKey)
    if (byTarget === undefined) {
      if (this.observed.size >= ObservedStateGate.MAX_OWNERS) {
        // Evict the least-recently-set owner (Map insertion order).
        const oldest = this.observed.keys().next()
        if (!oldest.done) this.observed.delete(oldest.value)
      }
      byTarget = new Map()
    } else {
      // Refresh recency so an actively mutating owner is never the eviction victim.
      this.observed.delete(ownerKey)
    }
    this.observed.set(ownerKey, byTarget)
    if (!byTarget.has(targetKey) && byTarget.size >= ObservedStateGate.MAX_TARGETS_PER_OWNER) {
      const oldest = byTarget.keys().next()
      if (!oldest.done) byTarget.delete(oldest.value)
    }
    byTarget.set(targetKey, observation)
  }

  /** Drop all recorded state (HMR safety / disposal). */
  clear(): void {
    this.observed = new Map()
    this.anonymousIds = new WeakMap()
  }

  /**
   * Decide the write intent: unseen or confirmed absent ⇒ `createIfAbsent`;
   * confirmed present ⇒ `replaceIfVersion` at the observed version.
   */
  writeIntent(target: FsTarget, actor: object | undefined): FsWriteIntent {
    const ownerKey = this.owner(actor)
    const prior = ownerKey ? this.get(ownerKey, target.targetKey) : undefined
    return prior?.kind === 'present'
      ? { kind: 'replaceIfVersion', version: prior.version }
      : { kind: 'createIfAbsent' }
  }

  /**
   * Decide the edit version guard: unseen rejects with `FS_NOT_OBSERVED`,
   * confirmed absence rejects with `FS_NOT_FOUND`, and presence supplies the
   * observed version as the CAS basis.
   */
  editIntent(target: FsTarget, actor: object | undefined): { version: FsVersion } {
    const ownerKey = this.owner(actor)
    const prior = ownerKey ? this.get(ownerKey, target.targetKey) : undefined
    if (!ownerKey || prior === undefined) {
      throw new FsError(`edit requires reading "${target.displayPath}" first`, 'FS_NOT_OBSERVED')
    }
    if (prior.kind === 'absent') {
      throw new FsError(`cannot edit "${target.displayPath}": not found`, 'FS_NOT_FOUND')
    }
    return { version: prior.version }
  }

  /** Record an authoritative present or absent observation for this owner and target. */
  observe(target: FsTarget, observation: FsObservation, actor: object | undefined): void {
    const ownerKey = this.owner(actor)
    if (ownerKey) this.set(ownerKey, target.targetKey, observation)
  }
}

/** Cordis plugin name used by loader diagnostics. */
export const name = 'fs-observation-policy'

/**
 * Register the three `fs/*` listeners. No `inject` — this plugin reads no
 * services; it operates only on its own `WeakMap`. The waterfalls are unbound
 * (the tool dispatches them with no `this`), so the listeners take the raw
 * `(target, actor, next)` arguments.
 */
export function apply(ctx: Context): void {
  const gate = new ObservedStateGate()

  ctx.effect(() => () => {
    // Drop all recorded state on disposal so a reloaded plugin starts clean
    // (HMR safety). The WeakMap itself would be GC'd, but replacing it makes the
    // release observable and immediate for tests.
    gate.clear()
  }, 'fs-observation-policy observed-state teardown')

  // fs/write-intent: occupy the single decision slot — do NOT call next().
  // Deferred through Promise.resolve().then so the declared Promise return type
  // holds (a throw rejects, never escapes synchronously through the waterfall).
  ctx.on('fs/write-intent', (target, actor) => Promise.resolve().then(() => gate.writeIntent(target, actor)))

  // fs/edit-intent: occupy the single decision slot — do not call next().
  ctx.on('fs/edit-intent', (target, actor) => Promise.resolve().then(() => gate.editIntent(target, actor)))

  // fs/observed must remain synchronous and non-throwing: emit does not await
  // promises, and successful mutations have already committed. WeakMap.set
  // satisfies that contract for both presence and absence.
  ctx.on('fs/observed', (target, observation, actor) => {
    gate.observe(target, observation, actor)
  })
}
