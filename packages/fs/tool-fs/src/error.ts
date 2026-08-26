/**
 * Model-facing remediation for guarded-mutation and literal-match failures. The
 * provider's `FS_STALE_VERSION` and `FS_NOT_OBSERVED` messages state the condition but
 * not the only correct recovery (re-read / read the file), so this package
 * appends the remedy at the model boundary; provider messages stay
 * machine-oriented and unchanged.
 * @module @deepseek-ai/dsh-tool-fs/src/error
 */

import { FsError } from '@deepseek-ai/dsh-fs'
import type { FsErrorCode } from '@deepseek-ai/dsh-fs'

/** The remedy appended to each remediable failure code's message. */
const REMEDIES: Partial<Record<FsErrorCode, string>> = {
  FS_STALE_VERSION: 're-read the file, then retry',
  FS_NOT_OBSERVED: 'read the file, then retry',
  FS_EDIT_NOT_FOUND: 're-read the file, then retry with an exact old_string copied from it',
}

/**
 * Append the correct recovery instruction to a remediable failure's message.
 * `FS_STALE_VERSION` (the file changed since this session's last observation,
 * including a missing target) recovers only by re-reading; `FS_NOT_OBSERVED`
 * (no prior read by this session) by reading; `FS_EDIT_NOT_FOUND` (zero
 * literal matches) by re-reading and copying `old_string` exactly. The
 * `FsError` code is preserved so retry/permission/UI layers keep routing on
 * it, and the original error chains as `cause`. Anything else passes through
 * untouched.
 * @param error - the caught value from a write/edit execution.
 * @returns a remediated `FsError` for the remediable codes, else the original value.
 */
export function remediateFsError(error: unknown): unknown {
  if (!(error instanceof FsError)) return error
  const remedy = REMEDIES[error.code]
  if (!remedy) return error
  return new FsError(`${error.message} — ${remedy}`, error.code, { cause: error })
}
