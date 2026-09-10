/**
 * Model-facing diagnostics for guarded-mutation failures. Providers and
 * policies retain operation-specific causes, while this package owns the
 * stable message shown to the model. The provider's `FS_STALE_VERSION`,
 * `FS_NOT_OBSERVED`, and `FS_EDIT_NOT_FOUND` messages state the condition but
 * not the only correct recovery (re-read / read the file / copy old_string),
 * so this package appends that remedy at the model boundary.
 * @module @deepseek-ai/dsh-tool-fs/src/error
 */

import { FsError } from '@deepseek-ai/dsh-fs'

/**
 * Render the stable model-facing diagnostic for a guarded-mutation failure.
 * `FS_STALE_VERSION` keeps the provider's reason and appends its re-read
 * remedy. `FS_NOT_OBSERVED` replaces operation-specific policy/provider text
 * with one path-aware reason and read remedy. `FS_EDIT_NOT_FOUND` (zero
 * literal matches) appends the exact-copy re-read remedy. The original error
 * remains the cause, and every diagnostic preserves its code for machine
 * routing. Anything else passes through untouched.
 * @param error - the caught value from a write/edit execution.
 * @param displayPath - the resolved target path shown to the model.
 * @returns a remediated `FsError` for the remediable codes, else the original value.
 */
export function remediateFsError(error: unknown, displayPath: string): unknown {
  if (!(error instanceof FsError)) return error
  if (error.code === 'FS_NOT_OBSERVED') {
    return new FsError(
      `cannot modify "${displayPath}": file has not been read — read the file, then retry`,
      error.code,
      { cause: error },
    )
  }
  if (error.code === 'FS_STALE_VERSION') {
    return new FsError(`${error.message} — re-read the file, then retry`, error.code, { cause: error })
  }
  if (error.code === 'FS_EDIT_NOT_FOUND') {
    return new FsError(
      `${error.message} — re-read the file, then retry with an exact old_string copied from it`,
      error.code,
      { cause: error },
    )
  }
  return error
}
