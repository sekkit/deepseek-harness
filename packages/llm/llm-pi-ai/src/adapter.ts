/**
 * Generic pi-ai-backed implementation of the Harness LLM seam.
 *
 * Each resolution produces one **immutable** snapshot — the profiles plus a
 * `Models` collection holding the `Provider` each route built — and an
 * operation captures a whole snapshot before its first `await`. A
 * configuration change builds a *new* collection rather than mutating the one
 * in use, because `Models.streamSimple()` is lazy: it resolves the provider
 * when the stream is first consumed, which is after the credential await, so a
 * mutated collection would let a request that started under one configuration
 * finish under another — or fail with a provider that no longer exists. This is
 * what makes the seam's per-step call freeze (`llm.prepareCall()`) hold all the
 * way down: switching models mid-reply takes effect on the next step, never
 * inside the one in flight.
 *
 * Credentials stay outside that collection. The harness resolves a route's key
 * through its own seam and passes it as the request's `apiKey` option, which
 * pi-ai treats as the highest-priority auth override — so `Models` never holds
 * a credential store and the harness keeps its fail-loud reference semantics.
 *
 * @module dsh-llm-pi-ai/adapter
 */

import { createModels, getSupportedThinkingLevels } from '@earendil-works/pi-ai'
import type {
  Api,
  Model,
  Models,
  ModelThinkingLevel,
  MutableModels,
  SimpleStreamOptions,
  ThinkingLevel,
} from '@earendil-works/pi-ai'
import {
  attributionHeaders,
  contentHasImage,
  LlmAdapter,
  LlmError,
  QUOTA_EXCEEDED_CODE,
  ReasoningEffortId,
} from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ReasoningEffortId as ReasoningEffortIdType,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import type { ResolvedPiAiProviderProfile } from './config.ts'
import { toPiContext } from './context.ts'
import { toStreamChunks } from './stream.ts'

/** One resolution's frozen view: the profiles and the collection built from them. */
interface PiAiSnapshot {
  /** The resolved profiles this collection was built from, used as its identity. */
  profiles: ReadonlyMap<string, ResolvedPiAiProviderProfile>
  /** Providers for exactly those profiles; never mutated once published. */
  models: Models
}

/** Constructor options for {@link PiAiAdapter}: the two resolution hooks the plugin owns. */
export interface PiAiAdapterOptions {
  /** Current validated profiles by provider route; called once per operation. */
  profiles: () => ReadonlyMap<string, ResolvedPiAiProviderProfile>
  /**
   * Resolve the credential for one already-resolved profile; called once per
   * stream call and frozen for that call. `undefined` defers to the route's own
   * pi-ai auth, which for an installed catalog route is its provider-native
   * ambient discovery; the plugin allows that only for a profile naming no
   * credential at all, because a named reference that misses throws `LlmError`
   * `MISSING_CREDENTIAL` rather than falling back.
   *
   * When `ref` is supplied (the key-pool scheduler's chosen credential), the
   * resolution is scoped to that one reference; when omitted, the first
   * available credential in the pool is used (model discovery path).
   */
  resolveApiKey: (provider: string, profile: ResolvedPiAiProviderProfile, ref?: string) => Promise<string | undefined>
  /** Resolve the optional durable attachment service at request time. */
  resolveAttachments?: () => AttachmentStore | undefined
  /**
   * Observe one assistant history message degrading to provider-neutral
   * conversion because its stored replay state is unusable by this build.
   */
  onReplayDegrade?: (detail: { provider: string; model: string; reason: string }) => void
}

/** Copy profile stream knobs into pi-ai's common option vocabulary. */
function profileOptions(
  profile: ResolvedPiAiProviderProfile,
  reasoning: ModelThinkingLevel | undefined,
  apiKey: string | undefined,
): SimpleStreamOptions {
  const enabledReasoning: ThinkingLevel | undefined = reasoning === 'off' ? undefined : reasoning
  return {
    ...apiKey === undefined ? {} : { apiKey },
    ...enabledReasoning === undefined ? {} : { reasoning: enabledReasoning },
    ...profile.thinkingBudgets === undefined ? {} : { thinkingBudgets: profile.thinkingBudgets },
    ...profile.cacheRetention === undefined ? {} : { cacheRetention: profile.cacheRetention },
    ...profile.transport === undefined ? {} : { transport: profile.transport },
    ...profile.timeoutMs === undefined ? {} : { timeoutMs: profile.timeoutMs },
    ...profile.websocketConnectTimeoutMs === undefined ? {} : { websocketConnectTimeoutMs: profile.websocketConnectTimeoutMs },
    // The agent recovery layer owns visible attempts; one adapter call is one SDK attempt.
    maxRetries: 0,
  }
}

/**
 * The profile default this exact model can actually take, for DESCRIBING it.
 * A configured level the model does not support yields none rather than
 * throwing: `resolveModel` builds the model catalog, and a catalog that fails
 * takes its whole provider out of every picker — so one mis-set profile field
 * would hide every model on the route, including the ones that support the
 * level. The request path still refuses, which is where a bad configuration
 * belongs: describing what a model can do must not fail because a deployment
 * asked it for something it cannot.
 * @param model - the resolved model descriptor.
 * @param effort - the profile's configured level, if any.
 * @returns the level when this model supports it, otherwise undefined.
 */
function describableReasoningLevel(
  model: Model<Api>,
  effort: ReasoningEffortIdType | ModelThinkingLevel | undefined,
): ModelThinkingLevel | undefined {
  if (effort === undefined) return undefined
  return getSupportedThinkingLevels(model).some(level => level === effort)
    ? effort as ModelThinkingLevel
    : undefined
}

/** Validate an explicit Harness/profile effort without invoking pi-ai's clamp. */
function resolveReasoningLevel(
  model: Model<Api>,
  effort: ReasoningEffortIdType | ModelThinkingLevel | undefined,
): ModelThinkingLevel | undefined {
  if (effort === undefined) return undefined
  const supported = getSupportedThinkingLevels(model)
  if (supported.some(level => level === effort)) return effort as ModelThinkingLevel
  throw new LlmError(
    `pi-ai provider "${model.provider}" model "${model.id}" does not support reasoning effort "${effort}"`,
    'UNSUPPORTED_REASONING_EFFORT',
  )
}

/**
 * Selectable reasoning efforts for one model, or nothing at all.
 *
 * A model that carries no reasoning metadata — every hand-declared one, and
 * every catalog model pi-ai marks as non-reasoning — is reported by pi-ai as
 * supporting the single level `off`. Passing that through would offer a control
 * that cannot do what it says: `off` is translated to *omitting* the reasoning
 * option, which for such a model is byte-for-byte the same request as naming no
 * effort — so a provider whose own default is to think would keep thinking with
 * `off` selected. Omitting `reasoning` entirely is the seam's way of saying the
 * capability is unavailable, which leaves the surface offering only the
 * provider's default.
 * @param model - the resolved model descriptor.
 * @param defaultLevel - the profile's configured effort, already validated.
 * @returns the `reasoning` field, or an empty object when none can be offered.
 */
function reasoningInfo(
  model: Model<Api>,
  defaultLevel: ModelThinkingLevel | undefined,
): Pick<LlmResolvedModelInfo, 'reasoning'> | Record<string, never> {
  if (!model.reasoning) return {}
  const levels = getSupportedThinkingLevels(model)
  return {
    reasoning: {
      efforts: levels.map(level => ({
        id: ReasoningEffortId(level),
        name: `${level.charAt(0).toUpperCase()}${level.slice(1)}`,
      })),
      ...defaultLevel === undefined ? {} : { defaultEffort: ReasoningEffortId(defaultLevel) },
    },
  }
}

/** Merge deployment headers while removing case-insensitive attribution collisions. */
function requestHeaders(headers: Readonly<Record<string, string>> | undefined): Record<string, string> {
  const attribution = attributionHeaders()
  const reserved = new Set(Object.keys(attribution).map(name => name.toLowerCase()))
  return {
    ...Object.fromEntries(Object.entries(headers ?? {}).filter(([name]) => !reserved.has(name.toLowerCase()))),
    ...attribution,
  }
}

// ── Adaptive multi-key scheduler ──────────────────────────────────────────
// The scheduler learns each key's real rate limit (AIMD) from live outcomes,
// so the constants below are only the starting point and the hard floors.

/** Initial per-key request budget (requests per minute). */
const INIT_RPM_CAPACITY = 5
/** Hard floor the AIMD factor never undershoots. */
const RPM_MIN_CAPACITY = 1
/** Hard ceiling the add-increase step never overshoots. */
const RPM_MAX_CAPACITY = 12
/** Additive-increase step on a clean completion. */
const RPM_AIMD_STEP = 0.25
/** Multiplicative-decrease factor on a rate-limit failure. */
const RPM_AIMD_FACTOR = 0.6
/** Initial per-key token budget (tokens per minute). */
const INIT_TPM_LIMIT = 300_000
/** Hard floor the TPM AIMD factor never undershoots. */
const TPM_MIN_LIMIT = 8_000
/** Multiplicative-decrease factor on a token-rate failure. */
const TPM_AIMD_FACTOR = 0.7
/** Minimum token headroom a key must have to be eligible. */
const MIN_TPM_RESERVE = 1_000
/** Minimum gap (ms) between consecutive requests on the same key. */
const MIN_KEY_GAP_MS = 1_200
/** Cooldown (ms) after a hard quota-exhausted failure. */
const QUOTA_COOLDOWN_MS = 300_000
/** Cooldown (ms) after an authentication failure. */
const AUTH_COOLDOWN_MS = 600_000
/** Cooldown (ms) after a per-key rate-limit hit. */
const RPM_COOLDOWN_MS = 30_000
/** Cooldown (ms) after a token-rate hit. */
const TPM_COOLDOWN_MS = 60_000
/** Brief cooldown (ms) after a transient server failure. */
const SERVER_COOLDOWN_MS = 5_000

/** Per-key meter state. */
interface KeyMeter {
  rpmTokens: number
  rpmLast: number
  rpmCapacity: number
  tpmTokens: number
  tpmLast: number
  tpmCapacity: number
  cooldownUntil: number
  cooldownKind: string | null
  lastStartAt: number
  inFlight: number
}

/** Per-route scheduler gate. */
interface KeyGate {
  poolId: string
  keys: Map<string, KeyMeter>
  cursor: number
  initialRpm: number
}

/** Granularity of a scheduler outcome report. */
type SchedulerOutcome = 'ok' | 'rpm' | 'tpm' | 'quota' | 'auth' | 'server' | 'other'

/**
 * Abortable sleep used while the scheduler waits for the next free slot.
 * @param ms - sleep duration.
 * @param signal - caller abort; reject with ABORTED.
 */
function sleepAbortable(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    let settled = false
    const onAbort = (): void => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      reject(new LlmError('pi-ai request aborted by caller while waiting for its rate-limit slot', 'ABORTED'))
    }
    const timer = setTimeout(() => {
      if (signal !== undefined) signal.removeEventListener('abort', onAbort)
      if (settled) return
      settled = true
      resolve()
    }, Math.max(1, ms))
    if (signal !== undefined) {
      if (signal.aborted) onAbort()
      else signal.addEventListener('abort', onAbort, { once: true })
    }
  })
}

/**
 * Classify a terminal provider failure into the scheduler's outcome granularity
 * so each failure class gets its own cooldown and AIMD reaction.
 * @param code - the harness failure code (e.g. QUOTA, AUTH, RATE_LIMIT, SERVER).
 * @param message - the provider error text.
 * @returns one of "rpm" | "tpm" | "quota" | "auth" | "server" | "other".
 */
function failureOutcome(code: string, message = ''): SchedulerOutcome {
  if (code === 'AUTH') return 'auth'
  if (code === QUOTA_EXCEEDED_CODE) return 'quota'
  if (code === 'RATE_LIMIT') return /\btpm\b|token/i.test(message) ? 'tpm' : 'rpm'
  if (code === 'SERVER' || code === 'TIMEOUT' || code === 'TRANSPORT') return 'server'
  return 'other'
}

/**
 * pi-ai-backed multi-provider adapter. Each operation reads the current
 * profiles, so a configuration change reaches the next request without a
 * restart; model descriptors come from the collection those profiles built.
 */
export class PiAiAdapter extends LlmAdapter {
  private snapshot: PiAiSnapshot | undefined
  /** Per-route adaptive scheduler state (per-key meters, cooldowns, in-flight). */
  private readonly slots = new Map<string, KeyGate>()

  constructor(private readonly config: PiAiAdapterOptions) {
    super()
  }

  /**
   * The snapshot for the current profiles. Resolution memoizes its result, so
   * an unchanged configuration is recognized by identity; a changed one gets a
   * brand-new collection, leaving any snapshot an operation already captured
   * untouched for as long as that operation holds it.
   */
  private current(): PiAiSnapshot {
    const profiles = this.config.profiles()
    if (this.snapshot?.profiles === profiles) return this.snapshot
    const models: MutableModels = createModels()
    for (const profile of profiles.values()) models.setProvider(profile.piProvider)
    this.snapshot = { profiles, models }
    return this.snapshot
  }

  /** The profile for one route within one snapshot, or the not-owned failure. */
  private profileOf(snapshot: PiAiSnapshot, provider: string): ResolvedPiAiProviderProfile {
    const profile = snapshot.profiles.get(provider)
    if (profile === undefined) {
      throw new LlmError(`pi-ai adapter does not own provider "${provider}"`, 'NO_ADAPTER')
    }
    return profile
  }

  /** The configured descriptor for one exact route/model pair within one snapshot. */
  private modelOf(snapshot: PiAiSnapshot, provider: string, model: string): Model<Api> {
    this.profileOf(snapshot, provider)
    const resolved = snapshot.models.getModel(provider, model)
    if (resolved === undefined) {
      throw new LlmError(`pi-ai provider "${provider}" has no configured model "${model}"`, 'UNKNOWN_MODEL')
    }
    return resolved
  }

  override providerInfo(provider: string): LlmProviderInfo {
    // The configured name, not the route key: `displayName` exists so a
    // deployment can label a route, and a label only the configuration surface
    // reads would leave every selector showing the raw key.
    return { id: provider, name: this.current().profiles.get(provider)?.displayName ?? provider }
  }

  override providerRetryPolicy(provider: string): ResolvedRetryPolicy | undefined {
    return this.current().profiles.get(provider)?.retryPolicy
  }

  /**
   * Adaptive multi-key scheduler.  Each key of a {@link ResolvedPiAiProviderProfile.keyRefs}
   * pool is an independent worker with its own learned limits (AIMD), so the
   * pool converges to each account's real sustainable rate without knowing it
   * ahead of time. Routes with a single key bypass the scheduler entirely.
   *
   * Per-key state:
   * - rpm token bucket with an adaptive capacity (starts conservative,
   *   grows on success, shrinks on `rpm exhausted`).
   * - tpm token budget mirroring the account's tokens-per-minute limit with a
   *   learned ceiling (lowered on `tpm exhausted`, sized by observed usage).
   * - cooldown per failure class, so a hard quota/auth failure is shepherded
   *   out of rotation and a transient rate limit only pauses briefly.
   * - in-flight counter capped at 1 per key: concurrent requests are fanned
   *   out across keys, which keeps any single account from seeing a burst.
   * - min-gap between request starts on the same key to avoid burst blocks.
   *
   * Selection favours the key with the most remaining rpm budget and tpm
   * headroom, then the least-recently-used (round-robin) — spreading load to
   * the healthiest accounts and draining each account's capacity evenly.
   *
   * @param provider - provider route key.
   * @param refs - the credential names of the key pool.
   * @param signal - caller abort; waiting for a slot is aborted with it.
   * @param initialRpm - per-key initial rpm capacity (defaults to {@link INIT_RPM_CAPACITY}).
   * @returns the credential name chosen for this request, or undefined when
   *   every key is cooled down (the caller surfaces the last failure).
   */
  private async acquireSlot(
    provider: string,
    refs: readonly string[],
    signal?: AbortSignal,
    initialRpm?: number,
  ): Promise<string | undefined> {
    if (refs.length === 0) return undefined
    // A single-key route has nothing to rotate or throttle: skipping the
    // scheduler keeps exactly the pre-pool behavior (no rate limiting, no
    // cooldown, no gap) for the common case the pool feature does not need.
    if (refs.length === 1) return refs[0]
    const now = Date.now()
    const gate = this.ensureGate(provider, String(refs), initialRpm)
    gate.cursor = gate.cursor % refs.length
    // refresh every key's meters for this instant's decision
    for (const ref of refs) this.refreshKey(gate, ref, now)
    let chosen: string | undefined
    let best = -1
    let lookup = -1
    for (let i = 0; i < refs.length; i++) {
      const idx = (gate.cursor + i) % refs.length
      const ref = refs[idx]
      if (ref === undefined) continue
      const k = gate.keys.get(ref)
      if (k === undefined) continue
      if (k.cooldownUntil > now) continue
      if (k.inFlight >= 1) continue
      if (now - k.lastStartAt < MIN_KEY_GAP_MS) continue
      if (k.rpmTokens < 1) continue
      if (k.tpmTokens < MIN_TPM_RESERVE) continue
      const score = k.rpmTokens * 1e6 + k.tpmTokens
      if (score > best) { best = score; chosen = ref; lookup = idx }
    }
    if (chosen !== undefined) {
      const k = gate.keys.get(chosen)!
      k.rpmTokens -= 1
      k.inFlight += 1
      k.lastStartAt = now
      gate.cursor = (lookup + 1) % refs.length
      return chosen
    }
    // no key eligible right now: wait for the soonest to become eligible
    const waitFor: number[] = []
    for (const ref of refs) {
      const k = gate.keys.get(ref)
      if (k === undefined) continue
      if (k.cooldownUntil > now) continue
      if (k.inFlight >= 1) continue
      let wait = 0
      const gapWait = MIN_KEY_GAP_MS - (now - k.lastStartAt)
      if (gapWait > wait) wait = gapWait
      if (k.rpmTokens < 1) {
        const rpmWait = Math.ceil((1 - k.rpmTokens) * 6e4 / Math.max(1, k.rpmCapacity))
        if (rpmWait > wait) wait = rpmWait
      }
      if (k.tpmTokens < MIN_TPM_RESERVE) {
        const tpmWait = Math.ceil((MIN_TPM_RESERVE - k.tpmTokens) * 6e4 / Math.max(1, k.tpmCapacity))
        if (tpmWait > wait) wait = tpmWait
      }
      if (wait > 0) waitFor.push(wait)
    }
    const cooldownEarliest = Math.min(...refs.map(ref => {
      const k = gate.keys.get(ref)
      return k !== undefined && k.cooldownUntil > now ? k.cooldownUntil : Number.MAX_SAFE_INTEGER
    }))
    let waitMs = waitFor.length > 0 ? Math.min(...waitFor) : Math.max(1, cooldownEarliest - now)
    if (cooldownEarliest !== Number.MAX_SAFE_INTEGER) {
      waitMs = Math.min(waitMs, Math.max(1, cooldownEarliest - now))
    }
    waitMs = Math.min(waitMs, 60_000)
    await sleepAbortable(waitMs, signal)
    // retry until a key is free or all keys are hard-cooled
    if (refs.every(ref => { const k = gate.keys.get(ref); return k !== undefined && k.cooldownUntil > Date.now() })) {
      return undefined
    }
    return this.acquireSlot(provider, refs, signal, initialRpm)
  }

  /**
   * Get the scheduler gate for a route, rebuilding it when the key pool id
   * (the ordered ref list) changes.
   * @param provider - provider route key.
   * @param poolId - identity of the ordered key pool.
   * @param initialRpm - per-key initial rpm capacity for a new gate.
   */
  private ensureGate(provider: string, poolId: string, initialRpm?: number): KeyGate {
    let gate = this.slots.get(provider)
    if (gate === undefined || gate.poolId !== poolId) {
      gate = { poolId, keys: new Map(), cursor: 0, initialRpm: initialRpm ?? INIT_RPM_CAPACITY }
      this.slots.set(provider, gate)
    }
    return gate
  }

  /**
   * Refill one key's rpm token bucket and tpm budget for the current instant.
   * @param gate - the route's scheduler gate.
   * @param ref - the credential name.
   * @param now - current time.
   */
  private refreshKey(gate: KeyGate, ref: string, now: number): void {
    let k = gate.keys.get(ref)
    if (k === undefined) {
      k = {
        rpmTokens: gate.initialRpm,
        rpmLast: now,
        rpmCapacity: gate.initialRpm,
        tpmTokens: INIT_TPM_LIMIT,
        tpmLast: now,
        tpmCapacity: INIT_TPM_LIMIT,
        cooldownUntil: 0,
        cooldownKind: null,
        lastStartAt: 0,
        inFlight: 0,
      }
      gate.keys.set(ref, k)
    }
    const rpmElapsed = Math.max(0, now - k.rpmLast)
    if (rpmElapsed > 0) {
      k.rpmTokens = Math.min(k.rpmCapacity, k.rpmTokens + rpmElapsed * k.rpmCapacity / 6e4)
      k.rpmLast = now
    }
    const tpmElapsed = Math.max(0, now - k.tpmLast)
    if (tpmElapsed > 0) {
      k.tpmTokens = Math.min(k.tpmCapacity, k.tpmTokens + tpmElapsed * k.tpmCapacity / 6e4)
      k.tpmLast = now
    }
  }

  /**
   * Release a request slot (decrement in-flight) and feed back the outcome.
   * AIMD: successes slowly raise the rpm ceiling; failures shrink the
   * relevant budget and cool the key for a class-specific duration, so the
   * pool learns each account's real limits and never thrashes a dead key.
   * @param provider - provider route key.
   * @param ref - the credential name that served the request.
   * @param outcome - "ok" | "rpm" | "tpm" | "quota" | "auth" | "server".
   * @param tokens - input+output tokens consumed on success (for tpm debiting).
   */
  private reportOutcome(provider: string, ref: string | undefined, outcome: SchedulerOutcome, tokens = 0): void {
    const gate = this.slots.get(provider)
    const k = gate?.keys.get(ref!)
    if (k === undefined) return
    k.inFlight = Math.max(0, k.inFlight - 1)
    if (outcome === 'ok') {
      if (tokens > 0) k.tpmTokens = Math.max(0, k.tpmTokens - tokens)
      if (k.rpmCapacity < RPM_MAX_CAPACITY) {
        k.rpmCapacity = Math.min(RPM_MAX_CAPACITY, k.rpmCapacity + RPM_AIMD_STEP)
      }
      return
    }
    if (outcome === 'rpm') {
      k.rpmCapacity = Math.max(RPM_MIN_CAPACITY, k.rpmCapacity * RPM_AIMD_FACTOR)
      k.rpmTokens = Math.min(k.rpmTokens, k.rpmCapacity)
      k.cooldownUntil = Date.now() + RPM_COOLDOWN_MS
      k.cooldownKind = 'rpm'
      return
    }
    if (outcome === 'tpm') {
      const spent = k.tpmCapacity - k.tpmTokens
      k.tpmCapacity = Math.max(TPM_MIN_LIMIT, Math.min(INIT_TPM_LIMIT, spent > 0 ? spent : k.tpmCapacity * TPM_AIMD_FACTOR))
      k.tpmTokens = 0
      k.cooldownUntil = Date.now() + TPM_COOLDOWN_MS
      k.cooldownKind = 'tpm'
      return
    }
    if (outcome === 'quota' || outcome === 'auth') {
      k.cooldownUntil = Date.now() + (outcome === 'auth' ? AUTH_COOLDOWN_MS : QUOTA_COOLDOWN_MS)
      k.cooldownKind = outcome
      return
    }
    // server / transport: transient, brief cooldown only
    if (outcome === 'server') {
      k.cooldownUntil = Date.now() + SERVER_COOLDOWN_MS
      k.cooldownKind = 'server'
    }
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve().then(() => {
      const snapshot = this.current()
      this.profileOf(snapshot, provider)
      return snapshot.models.getModels(provider).map(model => ({
        provider,
        id: model.id,
        name: model.name,
        inputModalities: [...model.input],
      }))
    })
  }

  override resolveModel(
    provider: string,
    model: string,
    _signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    return Promise.resolve().then(() => {
      const snapshot = this.current()
      const profile = this.profileOf(snapshot, provider)
      const resolvedModel = this.modelOf(snapshot, provider, model)
      const defaultLevel = describableReasoningLevel(resolvedModel, profile.reasoning)
      // Only a cap the deployment configured is a request default; the
      // catalog's `maxTokens` sizes the model and stops there.
      const configuredMaxTokens = profile.configuredMaxTokens.get(model)
      return {
        provider,
        id: model,
        name: resolvedModel.name,
        inputModalities: [...resolvedModel.input],
        context: { contextWindow: resolvedModel.contextWindow },
        ...configuredMaxTokens === undefined ? {} : { defaultMaxTokens: configuredMaxTokens },
        ...reasoningInfo(resolvedModel, defaultLevel),
      }
    })
  }

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    if (options.stop !== undefined) {
      throw new LlmError('llm-pi-ai does not support GenerateOptions.stop', 'UNSUPPORTED_OPTION')
    }
    // One capture per stream call, taken before any await: the profile, the
    // model descriptor, and the collection all come from the same immutable
    // snapshot, and the credential freezes with them. A configuration change
    // mid-request builds a separate snapshot, so this request finishes under
    // the one it started with and the next call picks up the new one.
    const snapshot = this.current()
    const profile = this.profileOf(snapshot, options.provider)
    const model = this.modelOf(snapshot, options.provider, options.model)
    const reasoning = resolveReasoningLevel(
      model,
      options.reasoningEffort ?? profile.reasoning,
    )
    const refs = profile.keyRefs
    // With a key pool each attempt uses a different credential; without one
    // the single attempt shares the unauthenticated route behaviour.
    const attempts = refs.length === 0 ? 1 : refs.length
    let lastCredentialError: LlmError | undefined

    for (let attempt = 0; attempt < attempts; attempt++) {
      // Acquire a slot: picks the healthiest key, or waits briefly for one
      // to become eligible. A single-key route bypasses the scheduler, so its
      // behavior is unchanged from a route with no pool.
      const chosenRef = await this.acquireSlot(options.provider, refs, options.signal, profile.requestsPerMinute)
      if (chosenRef === undefined) {
        if (refs.length === 0) {
          // unauthenticated route: fall through with no key
        } else {
          if (lastCredentialError !== undefined) throw lastCredentialError
          throw new LlmError(
            `pi-ai provider "${options.provider}" has no credential with remaining quota`,
            QUOTA_EXCEEDED_CODE,
          )
        }
      }

      let apiKey: string | undefined
      try {
        apiKey = await this.config.resolveApiKey(options.provider, profile, chosenRef)
      } catch (error: unknown) {
        if (chosenRef !== undefined && refs.length > 1) {
          if (error instanceof LlmError && error.code === 'MISSING_CREDENTIAL') {
            lastCredentialError = error
            this.reportOutcome(options.provider, chosenRef, 'auth')
            continue
          }
          // Any other resolution failure releases the acquired slot so the
          // remaining keys of the pool stay usable on retry.
          this.reportOutcome(options.provider, chosenRef, 'other')
        }
        throw error
      }

      // Per-attempt token accounting for the scheduler's TPM feedback
      let attemptTokens = 0
      let reported = false
      const report = (outcome: SchedulerOutcome, tokens = attemptTokens): void => {
        if (reported) return
        reported = true
        this.reportOutcome(options.provider, chosenRef, outcome, tokens)
      }

      const consumer = new AbortController()
      const upstream = options.signal === undefined
        ? consumer.signal
        : AbortSignal.any([options.signal, consumer.signal])
      const streamIdleTimeoutMs = profile.streamIdleTimeoutMs
      using watchdog = idleWatchdog(upstream, streamIdleTimeoutMs, 'LLM_STREAM_IDLE_TIMEOUT')

      try {
        const containsImage = options.messages.some(message => contentHasImage(message.content))
        if (containsImage && !model.input.includes('image')) {
          throw new LlmError(`pi-ai model "${model.id}" does not support image input`, 'UNSUPPORTED_CONTENT')
        }
        const attachments = containsImage ? this.config.resolveAttachments?.() : undefined
        if (containsImage && attachments === undefined) {
          throw new LlmError('pi-ai image input requires the durable attachment service', 'UNSUPPORTED_CONTENT')
        }
        const onReplayDegrade = (reason: string): void => {
          this.config.onReplayDegrade?.({ provider: options.provider, model: options.model, reason })
        }
        const context = attachments === undefined
          ? toPiContext(options, undefined, onReplayDegrade)
          : await toPiContext(options, attachments, onReplayDegrade, profile.maxRequestImageBytes)
        const events = snapshot.models.streamSimple(model, context, {
          ...profileOptions(profile, reasoning, apiKey),
          ...options.temperature === undefined ? {} : { temperature: options.temperature },
          ...options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens },
          ...options.sessionId === undefined ? {} : { sessionId: String(options.sessionId) },
          signal: watchdog.signal,
          // Profile headers are deployment-owned; attribution names are
          // Harness-owned and therefore win collisions.
          headers: requestHeaders(profile.headers),
        })
        const iterator = toStreamChunks(events, model.contextWindow)[Symbol.asyncIterator]()
        let exhausted = false
        let yieldedAny = false
        // Failure codes that trigger a rotation to the next key when no
        // content has been yielded yet. Once any content reached the caller,
        // the response is already partial and switching mid-stream would
        // produce a corrupt conversation.
        const rotateKinds = new Set<string>([QUOTA_EXCEEDED_CODE, 'AUTH', 'RATE_LIMIT', 'SERVER'])
        try {
          while (true) {
            const result = await watchdog.next(iterator)
            const timeout = timeoutOf(watchdog.signal, 'LLM_STREAM_IDLE_TIMEOUT')
            if (timeout !== undefined) throw timeout
            if (result.done) {
              exhausted = true
              report('ok')
              return
            }
            const chunk = result.value
            // Track token consumption for the scheduler's TPM feedback
            if (chunk.type === 'usage' && chunk.usage !== undefined) {
              attemptTokens = (chunk.usage.inputTokens ?? 0) + (chunk.usage.outputTokens ?? 0)
            }
            if (chunk.type === 'finish' && chunk.reason !== undefined) {
              const reason = chunk.reason
              // Terminal success: report the scheduler outcome and yield.
              if (reason.kind === 'stop' || reason.kind === 'tool-calls' || reason.kind === 'max-tokens') {
                report('ok')
                yield chunk
                continue
              }
              // Error finish: rotate when the failure is rotatable and no
              // content has been yielded (partial response would corrupt the
              // conversation if we switched mid-stream).
              if (reason.kind === 'error' && reason.failure !== undefined) {
                const failureCode = reason.failure.code
                const outcome = failureOutcome(failureCode, reason.failure.message)
                const rotatable = rotateKinds.has(failureCode)
                  && chosenRef !== undefined
                  && refs.length > 1
                  && !yieldedAny
                if (rotatable) {
                  lastCredentialError = new LlmError(reason.failure.message, failureCode)
                  report(outcome)
                  break
                }
                // Non-rotatable provider error (bad request, etc.): surface
                report(outcome === 'server' ? 'server' : 'other')
                yield chunk
                continue
              }
              // Aborted: log and surface
              if (reason.kind === 'aborted') {
                report('other')
                yield chunk
                continue
              }
            }
            if (chunk.type !== 'finish' && chunk.type !== 'usage') yieldedAny = true
            yield chunk
          }
        } finally {
          if (!exhausted) {
            consumer.abort('pi-ai stream consumer stopped')
            try {
              await iterator.return(undefined)
            } catch (_abortedSdkTeardown) {
              // The stable signal already owns SDK termination; return-time abort cannot add an outcome.
            }
          }
        }
      } catch (error: unknown) {
        if (timeoutOf(watchdog.signal, 'LLM_STREAM_IDLE_TIMEOUT') !== undefined) {
          throw new LlmError(`pi-ai stream idle timeout after ${streamIdleTimeoutMs}ms`, 'TIMEOUT', { cause: error })
        }
        if (options.signal?.aborted) {
          throw new LlmError('pi-ai request aborted by caller', 'ABORTED', { cause: error })
        }
        throw error
      } finally {
        report('server')
        consumer.abort('pi-ai stream consumer stopped')
      }
    }
    // Every attempt exhausted without a terminal event
    if (lastCredentialError !== undefined) throw lastCredentialError
    throw new LlmError(
      `pi-ai provider "${options.provider}" request ended without a terminal event`,
      'STREAM_CLOSED',
    )
  }
}
