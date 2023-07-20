// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/**
 * Bazel application data for the Starlark thread that evaluates the top-level code in a .bzl module
 * (i.e. when evaluating that module's global symbols).
 */
public final class BzlInitThreadContext extends BazelStarlarkContext {

  private final Label bzlFile;

  /* Digest of the .bzl file being initialized along with all its transitive loads. */
  private final byte[] transitiveDigest;

  // For storing the result of calling `visibility()`.
  @Nullable private BzlVisibility bzlVisibility;

  // TODO(b/236456122): Are all these arguments needed for .bzl initialization?
  public BzlInitThreadContext(
      Label bzlFile,
      byte[] transitiveDigest,
      @Nullable RepositoryName toolsRepository,
      @Nullable ImmutableMap<String, Class<?>> fragmentNameToClass,
      SymbolGenerator<?> symbolGenerator,
      @Nullable Label networkAllowlistForTests) {
    super(
        BazelStarlarkContext.Phase.LOADING,
        toolsRepository,
        fragmentNameToClass,
        symbolGenerator,
        /* analysisRuleLabel= */ null,
        networkAllowlistForTests);
    this.bzlFile = bzlFile;
    this.transitiveDigest = transitiveDigest;
  }

  /**
   * Retrieves this context from a Starlark thread, or throws {@link IllegalStateException} if
   * unavailable.
   */
  public static BzlInitThreadContext from(StarlarkThread thread) {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    Preconditions.checkState(
        ctx instanceof BzlInitThreadContext,
        "Expected to be in a .bzl initialization (top-level evaluation) Starlark thread");
    return (BzlInitThreadContext) ctx;
  }

  /**
   * Retrieves this context from a Starlark thread. If not present, throws {@code EvalException}
   * with an error message indicating the failure was in a function named {@code function}.
   */
  public static BzlInitThreadContext fromOrFailFunction(StarlarkThread thread, String function)
      throws EvalException {
    @Nullable BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (!(ctx instanceof BzlInitThreadContext)) {
      throw Starlark.errorf(
          "'%s' can only be called during .bzl initialization (top-level evaluation)", function);
    }
    return (BzlInitThreadContext) ctx;
  }

  /**
   * Returns the label of the .bzl module being initialized.
   *
   * <p>Note that this is not necessarily the same as the module of the innermost stack frame (i.e.,
   * {@code BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread)).label()}),
   * since the module may call helper functions loaded from elsewhere.
   */
  public Label getBzlFile() {
    return bzlFile;
  }

  /** Returns the transitive digest of the .bzl module being initialized. */
  public byte[] getTransitiveDigest() {
    return transitiveDigest;
  }

  /**
   * Returns the saved BzlVisibility that was declared for the currently initializing .bzl module.
   */
  @Nullable
  public BzlVisibility getBzlVisibility() {
    return bzlVisibility;
  }

  /** Sets the BzlVisibility for the currently initializing .bzl module. */
  public void setBzlVisibility(BzlVisibility bzlVisibility) {
    this.bzlVisibility = bzlVisibility;
  }
}
