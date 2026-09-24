# Agent Note: Plugin-wide defaultModelThinking

Status: implemented

English | [中文](2026-09-24-plugin-wide-default-model-thinking.zh.md)

## Problem

Route-level `defaultModelThinking` gives a hand-declared gateway's undeclared models selectable thinking levels: a model whose entry sets no `reasoningEfforts` and whose route key names no installed catalog takes the route's field instead of silently resolving as non-reasoning. A deployment whose gateways all speak the same thinking dialect must restate the identical dict on every route, and a route that omits it — typically one added later — leaves every undescribed model without a thinking selector, with nothing at load time pointing at the omission.

## Decision

The plugin configuration carries the same field beside `providers`, and every route inherits it: a route's own `defaultModelThinking` replaces the shared one for that route alone. The per-model precedence is unchanged — an entry's `reasoningEfforts`, then an installed catalog model that already reasons, then the route field, then the plugin-wide field, then non-reasoning — so the shared field only ever fills routes and models nothing else described.

- The field is volatile, following `welcomeNoticeVersion`: `Volatile.get()`'s contract returns `undefined` for an absent value, and the adapter's memoized resolution joins the thinking snapshot to the raw providers snapshot as its cache key, so editing only the shared field re-resolves every route under configuration hot reload.
- `resolveProfiles` takes the shared value as a third parameter; `assertServiceable` forwards it, and `false` is rejected at the plugin level exactly as at the route level — omitting the field is the spelling of "no fallback".
- `Options` now erases `undefined` from volatile members (`Exclude<T, undefined>`): an absent optional value is an absent property, which keeps the resolver's plain-data shape assignable to the schema and plugin input shapes under `exactOptionalPropertyTypes` while the `Volatile` interface keeps its honest `undefined`-for-absent return.

## Alternatives considered

**Share the dict through YAML anchors or `!!js` composition.** Every route still carries its own field, a settings surface cannot see through the composition, and one wrong indentation silently leaves a route without the fallback — the exact failure the plugin-wide field removes.

**Default unknown models to reasoning at the catalog layer.** No listing endpoint reports a model's thinking dialect; `thinkingLevelMap` needs every level's wire spelling, which only a configuration author who read the gateway's documentation can supply.

**Cover `defaultContextWindow` and `defaultMaxTokens` too.** Both already have route-level defaults with fixed numeric fallbacks, so an undescribed model never lacks a capacity; only thinking had no answer below the route.

## Consequences

A gateway model the installed catalog has not caught up with offers the deployment's thinking levels by default, and a route that speaks a different dialect overrides the shared field locally. Models with their own `reasoningEfforts` and catalog models that already reason keep their declared levels under either field. The shared field changes nothing about capacity parameters, and a deployment that sets neither field keeps today's behavior: undeclared models resolve as non-reasoning.
