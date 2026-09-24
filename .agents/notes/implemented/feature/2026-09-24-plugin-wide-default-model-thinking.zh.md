# Agent Note：插件级 defaultModelThinking

状态：implemented

[English](2026-09-24-plugin-wide-default-model-thinking.md) | 中文

## 问题

route 级 `defaultModelThinking` 为手写网关上未声明的模型提供可选 thinking levels：模型条目没写 `reasoningEfforts`、route 名又不在安装目录里的，取 route 上的字段，而不是静默解析为非推理模型。但一个部署里多个网关说同一种 thinking 方言时，同一份 dict 要在每个 route 上重抄一遍；漏写的 route —— 通常是后来补的 —— 上所有未描述模型都没有 thinking 选择器，加载时也没有任何东西指出这个遗漏。

## 决策

插件配置在 `providers` 旁携带同名字段，所有 route 继承它：route 自己的 `defaultModelThinking` 只替换该 route 的共享值。每个模型的优先级不变 —— 条目的 `reasoningEfforts`，再到已声明推理能力的目录模型，再到 route 字段，再到插件级字段，最后非推理 —— 共享字段只填充没有任何描述的 route 和模型。

- 字段是 volatile 的，沿用 `welcomeNoticeVersion` 的先例：`Volatile.get()` 的契约对缺失值返回 `undefined`；adapter 的 memoized 解析把 thinking 快照和原始 providers 快照一起作为缓存键，配置热重载下只改共享字段也会对所有 route 重新解析。
- `resolveProfiles` 以第三参数接收共享值；`assertServiceable` 转发它；`false` 在插件层和 route 层一样被拒绝 —— 省略字段就是"无 fallback"的写法。
- `Options` 现在从 volatile 成员里擦除 `undefined`（`Exclude<T, undefined>`）：缺失的可选值即缺失的属性。这让 resolver 的纯数据形状在 `exactOptionalPropertyTypes` 下可直接赋给 schema 与 plugin 的输入形状，而 `Volatile` 接口仍保留"缺失返回 undefined"的诚实返回类型。

## 考虑过的替代方案

**用 YAML anchor 或 `!!js` 组合共享同一份 dict。** 每个 route 仍然各带一个字段，设置面板看不穿组合，缩进错一处就会让某个 route 静默失去 fallback —— 正是插件级字段要消除的失败。

**在 catalog 层把未知模型默认为推理。** 没有任何 listing 端点会报告模型的 thinking 方言；`thinkingLevelMap` 需要每个 level 的 wire 拼写，只有读过网关文档的配置作者能提供。

**把 `defaultContextWindow` 和 `defaultMaxTokens` 一并纳入。** 两者已有 route 级默认值和固定数值兜底，未描述模型从不缺容量；只有 thinking 在 route 之下没有答案。

## 后果

安装目录尚未收录的网关模型默认提供该部署的 thinking levels，说不同方言的 route 局部覆盖共享字段。自带 `reasoningEfforts` 的模型和目录里已声明推理的模型在两种字段下都保留已声明的 levels。共享字段不改变容量参数，两个字段都不设的部署保持现状：未声明模型解析为非推理。
