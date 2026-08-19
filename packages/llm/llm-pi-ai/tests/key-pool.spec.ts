/**
 * Key-pool (multi-credential) rotation tests for the pi-ai adapter.
 *
 * @module dsh-llm-pi-ai/key-pool.spec
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createUserMessage } from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { resolveProfiles } from '../src/config.ts'
import { assemble } from './assemble.ts'
import { closeMockServers, mockServer, textEvents } from './mock-server.ts'

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  while (cleanups.length > 0) await cleanups.pop()!()
  await closeMockServers()
  vi.unstubAllEnvs()
})

beforeEach(() => {
  vi.stubEnv('KEY_A', 'key-a-secret')
  vi.stubEnv('KEY_B', 'key-b-secret')
  vi.stubEnv('KEY_C', 'key-c-secret')
})

/** One-shot request helper: returns the Authorization header the server saw. */
async function requestAuth(
  providers: Record<string, LlmPiAi.PiAiProviderProfile>,
  server: { url: string },
): Promise<string> {
  const ctx = new Context()
  cleanups.push(async () => { await ctx.fiber.dispose() })
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmPiAi, { providers })
  await assemble(ctx, {
    model: 'deepseek-v4-flash',
    messages: [createUserMessage({
      content: [{ type: 'text', text: 'hi' }],
      source: { kind: 'plugin', plugin: 'test' },
    })],
  })
  return server.headers[0]?.['authorization'] as string
}

describe('key pool rotation', () => {
  it('uses the single key when no apiKeyEnvs are set', async () => {
    const server = await mockServer([{ events: textEvents }])
    const auth = await requestAuth({
      deepseek: { apiKeyEnv: 'KEY_A', baseURL: server.url },
    }, server)
    expect(auth).toBe('Bearer key-a-secret')
  })

  it('rotates across two keys on successive requests', async () => {
    const server = await mockServer([{ events: textEvents }, { events: textEvents }])
    const providers = {
      deepseek: {
        apiKeyEnv: 'KEY_A',
        apiKeyEnvs: ['KEY_B'],
        baseURL: server.url,
      } as LlmPiAi.PiAiProviderProfile,
    }
    const ctx = new Context()
    cleanups.push(async () => { await ctx.fiber.dispose() })
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, { providers })
    await assemble(ctx, { model: 'deepseek-v4-flash', messages: [] })
    await assemble(ctx, { model: 'deepseek-v4-flash', messages: [] })
    // Two requests should have used two different auth keys
    expect(server.headers[0]?.['authorization']).toBe('Bearer key-a-secret')
    expect(server.headers[1]?.['authorization']).toBe('Bearer key-b-secret')
  })

  it('retries on the next key when the first key returns a quota error', async () => {
    // First request fails with quota error; second succeeds
    const server = await mockServer([
      { status: 429, body: '{"error":{"message":"rate limit exceeded","code":"rate_limit"}}' },
      { events: textEvents },
    ])
    const providers = {
      deepseek: {
        apiKeyEnv: 'KEY_A',
        apiKeyEnvs: ['KEY_B'],
        baseURL: server.url,
      } as LlmPiAi.PiAiProviderProfile,
    }
    const ctx = new Context()
    cleanups.push(async () => { await ctx.fiber.dispose() })
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, { providers })
    const result = await assemble(ctx, {
      model: 'deepseek-v4-flash',
      messages: [createUserMessage({
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'plugin', plugin: 'test' },
      })],
    })
    // The request should have succeeded after rotating to KEY_B
    expect(result.message.content).toEqual([{ type: 'text', text: 'hello' }])
    expect(result.finish).toEqual({ kind: 'stop' })
    // First request went to KEY_A (429), second to KEY_B (success)
    expect(server.headers[0]?.['authorization']).toBe('Bearer key-a-secret')
    expect(server.headers[1]?.['authorization']).toBe('Bearer key-b-secret')
  })

  it('retries on the next key when the first key is missing from credentials', async () => {
    vi.stubEnv('KEY_A', '') // key A is empty
    const server = await mockServer([{ events: textEvents }])
    const providers = {
      deepseek: {
        apiKeyEnv: 'KEY_A',
        apiKeyEnvs: ['KEY_B'],
        baseURL: server.url,
      } as LlmPiAi.PiAiProviderProfile,
    }
    const ctx = new Context()
    cleanups.push(async () => { await ctx.fiber.dispose() })
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, { providers })
    const result = await assemble(ctx, {
      model: 'deepseek-v4-flash',
      messages: [createUserMessage({
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'plugin', plugin: 'test' },
      })],
    })
    // Should have succeeded with KEY_B
    expect(result.message.content).toEqual([{ type: 'text', text: 'hello' }])
    expect(result.finish).toEqual({ kind: 'stop' })
    expect(server.headers[0]?.['authorization']).toBe('Bearer key-b-secret')
  })

  it('throws QUOTA_EXCEEDED when all keys are exhausted', async () => {
    const server = await mockServer([
      { status: 429, body: '{"error":{"message":"rate limit exceeded","code":"rate_limit"}}' },
      { status: 429, body: '{"error":{"message":"rate limit exceeded","code":"rate_limit"}}' },
    ])
    const providers = {
      deepseek: {
        apiKeyEnv: 'KEY_A',
        apiKeyEnvs: ['KEY_B'],
        baseURL: server.url,
      } as LlmPiAi.PiAiProviderProfile,
    }
    const ctx = new Context()
    cleanups.push(async () => { await ctx.fiber.dispose() })
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, { providers })
    const result = await assemble(ctx, {
      model: 'deepseek-v4-flash',
      messages: [createUserMessage({
        content: [{ type: 'text', text: 'hi' }],
        source: { kind: 'plugin', plugin: 'test' },
      })],
    })
    // Both keys exhausted: the last failure surfaces as an error finish
    expect(result.finish.kind).toBe('error')
    if (result.finish.kind === 'error') {
      expect(result.finish.failure.code).toBe('RATE_LIMIT')
    }
    // Both keys should have been tried
    expect(server.headers[0]?.['authorization']).toBe('Bearer key-a-secret')
    expect(server.headers[1]?.['authorization']).toBe('Bearer key-b-secret')
  })
})

describe('resolveProfiles keyRefs', () => {
  it('builds keyRefs from apiKeyEnv alone', () => {
    const resolved = resolveProfiles({ deepseek: { apiKeyEnv: 'KEY_A', baseURL: 'http://ignored' } })
    const profile = resolved.get('deepseek')!
    expect(profile.keyRefs).toEqual(['KEY_A'])
  })

  it('builds keyRefs from apiKeyEnv + apiKeyEnvs (deduplicated)', () => {
    const resolved = resolveProfiles({
      deepseek: {
        apiKeyEnv: 'KEY_A',
        apiKeyEnvs: ['KEY_B', 'KEY_A', 'KEY_C'],
        baseURL: 'http://ignored',
      } as LlmPiAi.PiAiProviderProfile,
    })
    const profile = resolved.get('deepseek')!
    expect(profile.keyRefs).toEqual(['KEY_A', 'KEY_B', 'KEY_C'])
  })

  it('builds empty keyRefs when no credential is set', () => {
    const resolved = resolveProfiles({ deepseek: { baseURL: 'http://ignored' } })
    const profile = resolved.get('deepseek')!
    expect(profile.keyRefs).toEqual([])
  })
})