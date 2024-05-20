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
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/*
 * TODO(b/236456122): We should break this class up into separate classes for each kind of Starlark
 * evaluation environment, as opposed to storing possibly inapplicable nullable fields on this
 * class. Then we can use static methods with signatures like fromOrFail(StarlarkThread, String) in
 * place of the Phase enum and its associated check* methods.
 *
 * Some kinds of evaluation environments include:
 *   - .bzl loading
 *   - BUILD evaluation (should absorb PackageFactory.PackageContext -- no need to store multiple
 *     thread-locals in StarlarkThread)
 *   - WORKSPACE evaluation (shares logic with BUILD)
 *   - implicit outputs
 *   - computed defaults
 *   - transition implementation
 *   - Args.map_each
 *   - probably others
 *
 * BazelStarlarkContext itself could probably be deleted. If we do keep BazelStarlarkContext around
 * as a common base class of the other context classes, it should be renamed BazelThreadContext for
 * symmetry with BazelModuleContext.
 *
 * Even if we can otherwise get rid of BazelStarlarkContext, it may still be handy to retain this
 * class as a static namespace for helper methods for storing and retrieving contexts on
 * StarlarkThreads. In particular, it'd avoid the very likely bug of a storeInThread
 * implementation in one of the context classes forgetting to setUncheckedExceptionContext(this). It
 * also would give us a place to keep javadoc about the proper way to use these context objects in
 * Bazel.
 */
/**
 * Bazel-specific contextual information associated with a Starlark evaluation thread.
 *
 * <p>This is stored in the StarlarkThread object as a thread-local. A subclass of this class may be
 * used for certain kinds of Starlark evaluations; in that case it is still keyed in the
 * thread-locals under {@code BazelStarlarkContext.class}.
 *
 * <p>This object is mutable and should not be accessed simultaneously or reused for more than one
 * Starlark thread.
 */
public class BazelStarlarkContext implements StarlarkThread.UncheckedExceptionContext {

  /** The phase to which this Starlark thread belongs. */
  // TODO(b/236456122): Eliminate.
  public enum Phase {
    WORKSPACE,
    LOADING,
    ANALYSIS
  }

  /**
   * Retrieves this context from a Starlark thread, or throws {@link EvalException} if unavailable.
   */
  public static BazelStarlarkContext fromOrFail(StarlarkThread thread) throws EvalException {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (ctx == null) {
      throw Starlark.errorf(
          "this function cannot be called from %s", thread.getContextDescription());
    }
    return ctx;
  }

  /**
   * Saves this {@code BazelStarlarkContext} in the specified Starlark thread. Call only once,
   * before evaluation begins.
   */
  public void storeInThread(StarlarkThread thread) {
    Preconditions.checkState(thread.getThreadLocal(BazelStarlarkContext.class) == null);
    thread.setThreadLocal(BazelStarlarkContext.class, this);
    // TODO(b/236456122): We can probably replace the concept of setUncheckedExceptionContext with a
    // string parameter to StarlarkThread construction. In that case, storeInThread() becomes more
    // superfluous, and there's one less reason to keep the inheritance hierarchy of
    // BazelStarlarkContext and its children.
    thread.setUncheckedExceptionContext(this);
  }

  // TODO(b/236456122): Eliminate Phase, migrate analysisRuleLabel to a separate context class.
  private final Phase phase;
  @Nullable private final InterruptibleSupplier<RepositoryMapping> mainRepoMappingSupplier;

  /**
   * @param phase the phase to which this Starlark thread belongs
   */
  public BazelStarlarkContext(
      Phase phase, @Nullable InterruptibleSupplier<RepositoryMapping> mainRepoMappingSupplier) {
    this.phase = Preconditions.checkNotNull(phase);
    this.mainRepoMappingSupplier = mainRepoMappingSupplier;
  }

  /** Returns the phase associated with this context. */
  public Phase getPhase() {
    return phase;
  }

  /**
   * The repository mapping applicable to the main repository. This is purely meant to support
   * {@link Label#debugPrint}.
   */
  @Nullable
  public RepositoryMapping getMainRepoMapping() throws InterruptedException {
    return mainRepoMappingSupplier == null ? null : mainRepoMappingSupplier.get();
  }

  @Override
  public String getContextForUncheckedException() {
    return phase.toString();
  }

  /**
   * Checks that the Starlark thread is in the loading or the workspace phase.
   *
   * @param function name of a function that requires this check
   */
  // TODO(b/236456122): The Phase enum is incomplete. Ex: `Args.map_each` evaluation happens at
  // execution time. So this is a misnomer and possibly wrong in those contexts.
  public static void checkLoadingOrWorkspacePhase(StarlarkThread thread, String function)
      throws EvalException {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (ctx == null) {
      throw Starlark.errorf(
          "'%s' cannot be called from %s", function, thread.getContextDescription());
    }
    if (ctx.phase == Phase.ANALYSIS) {
      throw Starlark.errorf("'%s' cannot be called during the analysis phase", function);
    }
  }

  /**
   * Checks that the current StarlarkThread is in the loading phase.
   *
   * @param function name of a function that requires this check
   */
  public static void checkLoadingPhase(StarlarkThread thread, String function)
      throws EvalException {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (ctx == null) {
      throw Starlark.errorf(
          "'%s' cannot be called from %s", function, thread.getContextDescription());
    }
    if (ctx.phase != Phase.LOADING) {
      throw Starlark.errorf(
          "'%s' can only be called from a BUILD file, or a macro invoked from a BUILD file",
          function);
    }
  }

  /**
   * Checks that the current StarlarkThread is in the workspace phase.
   *
   * @param function name of a function that requires this check
   */
  public static void checkWorkspacePhase(StarlarkThread thread, String function)
      throws EvalException {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (ctx == null) {
      throw Starlark.errorf(
          "'%s' cannot be called from %s", function, thread.getContextDescription());
    }
    if (ctx.phase != Phase.WORKSPACE) {
      throw Starlark.errorf("'%s' can only be called during workspace loading", function);
    }
  }
}
