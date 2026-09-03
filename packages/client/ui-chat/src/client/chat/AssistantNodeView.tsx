import { memo, useCallback, useMemo } from 'react'
import type { ChatNodeViewProps, TurnTailOwnerProps } from '../contract/slots.ts'
import { AssistantMarkdown } from './AssistantMarkdown.tsx'
import { MessageIconActions } from './MessageIconActions.tsx'
import { assistantText } from './turn-assistant.ts'
import css from './AssistantNodeView.module.css'

/** Streaming, settled, and interrupted Assistant states share one keyed renderer instance. */
export const AssistantNodeView = memo(function AssistantNodeView({
  node, useTurnData, turnProcess, openFile, renderMessageImages, fileMentions, t,
}: ChatNodeViewProps<'assistant-step'>) {
  const data = node.data
  const turn = node.location.kind === 'turn' || node.location.kind === 'step'
    ? node.location.turn
    : undefined
  const tail = useTurnData('turn-tail')
  const owner = useMemo<TurnTailOwnerProps | undefined>(() => {
    if (turn?.status !== 'closed' || data.finalNode === undefined) return undefined
    if (tail?.closing?.finalNode.seq !== data.finalNode.seq) return undefined
    return { turn, seq: data.finalNode.seq, openFile }
  }, [data.finalNode, openFile, tail, turn])
  const mentions = useMemo(
    () => owner === undefined ? undefined : fileMentions(owner),
    [fileMentions, owner],
  )
// Per-output timing chrome: every settled assistant message gets its own
  // start timestamp + run duration, revealed on hover (data-time-hover-root)
  // so it stays out of the flow unless wanted. The turn's closing output is
  // already labeled by the turn-tail footer (which also carries TTFT /
  // throughput and the branch action), so it is not duplicated here.
  const timing = data.status === 'settled' && data.finalNode !== undefined
    ? data.finalNode.timing
    : undefined
  const isClosing = owner !== undefined
  const hasText = assistantText(data.blocks).trim() !== ''
  const startTime = timing?.stepStartTime ?? null
  const showTiming = timing !== undefined
    && startTime !== null
    && !isClosing
    && hasText
    && timing.completedTime >= startTime
  // Narrowed after showTiming: both are guaranteed defined when the timing row renders.
  const timingStart = startTime ?? 0
  const timingRun = timing === undefined ? 0 : timing.completedTime - timingStart
  const reasoningHidden = turnProcess !== undefined
    && turnProcess.foldable
    && turnProcess.spec.answerStep === data.step
    && turnProcess.spec.inlineReasoning
    && !turnProcess.open
  const revealProcess = useCallback(() => { turnProcess?.setOpen(true) }, [turnProcess])
  return (
    <div className={css.root} data-time-hover-root={showTiming ? '' : undefined}>
      <AssistantMarkdown
        blocks={data.blocks}
        streaming={data.status === 'running'}
        interrupted={data.status === 'interrupted'}
        renderMessageImages={renderMessageImages}
        reasoningHidden={reasoningHidden}
        revealProcess={revealProcess}
        mentions={mentions}
        t={t}
      />
      {showTiming && (
        <MessageIconActions
          text={assistantText(data.blocks)}
          time={timingStart}
          runMs={timingRun}
          clock="end"
          className={css.actions}
          t={t}
        />
      )}
    </div>
  )
})
