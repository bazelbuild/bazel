// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.cmdline;

import com.google.common.base.Preconditions;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkThread;

/**
 * Bazel-specific contextual information associated with a Starlark evaluation thread.
 *
 * <p>This is stored in the {@link StarlarkThread} object as a thread-local. A distinct
 * implementation of this interface should be defined and used for each different scenario of
 * Starlark evaluation; in any case, it is still keyed in the thread-locals under {@code
 * StarlarkThreadContext.class}. Users of this interface should prefer to use a {@code fromOrFail}
 * static method to retrieve an instance from a {@link StarlarkThread} instead of calling {@link
 * StarlarkThread#getThreadLocal} directly, and prefer to use {@link #storeInThread} instead of
 * calling {@link StarlarkThread#setThreadLocal} directly.
 *
 * <p>This object tends to be mutable and should not be accessed simultaneously or reused for more
 * than one Starlark thread.
 */
public interface StarlarkThreadContext {
  // TODO: decide the extent to which we should enforce that such a context object is available
  //  anywhere we execute Starlark code in Bazel. As of right now (Oct 2026), the only logic here is
  //  `getMainRepoMapping`, and even that one is not strictly necessary (can return null and things
  //  will still work).

  /**
   * Saves this {@link StarlarkThreadContext} in the specified Starlark thread. Call only once,
   * before evaluation begins.
   *
   * <p>Users of this interface should prefer to use this method instead of calling {@link
   * StarlarkThread#setThreadLocal} directly.
   */
  default void storeInThread(StarlarkThread thread) {
    Preconditions.checkState(thread.getThreadLocal(StarlarkThreadContext.class) == null);
    thread.setThreadLocal(StarlarkThreadContext.class, this);
  }

  /**
   * Returns the repository mapping of the main repository, or null if it isn't available in this
   * context.
   *
   * <p>This is only used by {@link Label#debugPrint} to render labels with apparent repository
   * names in {@code print()} and {@code fail()} output. Contexts that don't override this method
   * print labels with canonical repository names instead. Overrides are expected to look up the
   * mapping lazily so that a Skyframe dependency on it is only incurred by Starlark code that
   * actually prints a label.
   */
  @Nullable
  default RepositoryMapping getMainRepoMapping() throws InterruptedException {
    return null;
  }
}
