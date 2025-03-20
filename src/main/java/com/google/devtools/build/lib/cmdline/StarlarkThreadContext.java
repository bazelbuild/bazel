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
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkThread;

/**
 * Bazel-specific contextual information associated with a Starlark evaluation thread.
 *
 * <p>This is stored in the {@link StarlarkThread} object as a thread-local. A distinct subclass of
 * this class should be defined and used for each different scenario of Starlark evaluation; in any
 * case, it is still keyed in the thread-locals under {@code StarlarkThreadContext.class}. Users of
 * this class should prefer to use a {@code fromOrFail} static method to retrieve an instance from a
 * {@link StarlarkThread} instead of calling {@link StarlarkThread#getThreadLocal} directly, and
 * prefer to use {@link #storeInThread} instead of calling {@link StarlarkThread#setThreadLocal}
 * directly.
 *
 * <p>This object tends to be mutable and should not be accessed simultaneously or reused for more
 * than one Starlark thread.
 */
public abstract class StarlarkThreadContext {
  // TODO: decide the extent to which we should enforce that such a context object is available
  //  anywhere we execute Starlark code in Bazel. As of right now (Jun 2024), the only field here is
  //  `mainRepoMappingSupplier`, and even that one is not strictly necessary (can be null and things
  //  will still work).

  /**
   * Saves this {@link StarlarkThreadContext} in the specified Starlark thread. Call only once,
   * before evaluation begins.
   *
   * <p>Users of this class should prefer to use this method instead of calling {@link
   * StarlarkThread#setThreadLocal} directly.
   */
  public void storeInThread(StarlarkThread thread) {
    Preconditions.checkState(thread.getThreadLocal(StarlarkThreadContext.class) == null);
    thread.setThreadLocal(StarlarkThreadContext.class, this);
  }

  @Nullable private final InterruptibleSupplier<RepositoryMapping> mainRepoMappingSupplier;

  /**
   * @param mainRepoMappingSupplier a supplier for the repo mapping of the main repo. This is used
   *     for debug-printing {@link Label} objects. Can be null if the main repo mapping isn't
   *     readily available, which just causes the debug-printing to produce canonical label
   *     literals.
   */
  protected StarlarkThreadContext(
      @Nullable InterruptibleSupplier<RepositoryMapping> mainRepoMappingSupplier) {
    this.mainRepoMappingSupplier = mainRepoMappingSupplier;
  }

  /**
   * The repository mapping applicable to the main repository. This is purely meant to support
   * {@link Label#debugPrint}.
   */
  @Nullable
  public RepositoryMapping getMainRepoMapping() throws InterruptedException {
    return mainRepoMappingSupplier == null ? null : mainRepoMappingSupplier.get();
  }
}
