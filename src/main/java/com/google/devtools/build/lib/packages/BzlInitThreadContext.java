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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/**
 * Bazel application data for the Starlark thread that evaluates the top-level code in a .bzl (or
 * .scl) module (i.e. when evaluating that module's global symbols).
 */
public final class BzlInitThreadContext extends BazelStarlarkContext
    implements RuleDefinitionEnvironment {

  private final Label bzlFile;

  /* Digest of the .bzl file being initialized along with all its transitive loads. */
  private final byte[] transitiveDigest;

  // For storing the result of calling `visibility()`.
  @Nullable private BzlVisibility bzlVisibility;

  private final RepositoryName toolsRepository;

  // TODO(b/192694287): Remove once we migrate all tests from the allowlist
  private final Optional<Label> networkAllowlistForTests;

  // Used for `configuration_field`.
  private final ImmutableMap<String, Class<?>> fragmentNameToClass;

  /**
   * Constructs a new context for initializing a .bzl file.
   *
   * @param bzlFile the name of the .bzl being initialized
   * @param transitiveDigest the hash of that file and its transitive load()s
   * @param toolsRepository the name of the tools repository, such as "@bazel_tools"
   * @param networkAllowlistForTests an allowlist for rule classes created by this thread
   * @param fragmentNameToClass a map from configuration fragment name to configuration fragment
   *     class, such as "apple" to AppleConfiguration.class
   * @param symbolGenerator symbol generator for this context
   */
  public BzlInitThreadContext(
      Label bzlFile,
      byte[] transitiveDigest,
      RepositoryName toolsRepository,
      Optional<Label> networkAllowlistForTests,
      ImmutableMap<String, Class<?>> fragmentNameToClass,
      SymbolGenerator<?> symbolGenerator) {
    super(BazelStarlarkContext.Phase.LOADING, symbolGenerator, /* analysisRuleLabel= */ null);
    this.bzlFile = bzlFile;
    this.transitiveDigest = transitiveDigest;
    this.toolsRepository = toolsRepository;
    this.networkAllowlistForTests = networkAllowlistForTests;
    this.fragmentNameToClass = fragmentNameToClass;
  }

  /**
   * Retrieves this context from a Starlark thread. If not present, throws {@code EvalException}
   * with an error message indicating that {@code what} can't be used in this Starlark environment.
   */
  @CanIgnoreReturnValue
  public static BzlInitThreadContext fromOrFail(StarlarkThread thread, String what)
      throws EvalException {
    @Nullable BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (!(ctx instanceof BzlInitThreadContext)) {
      throw Starlark.errorf(
          "%s can only be used during .bzl initialization (top-level evaluation)", what);
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

  /** Returns the name of the tools repository, such as "@bazel_tools". */
  @Override
  public RepositoryName getToolsRepository() {
    return toolsRepository;
  }

  /** Returns a label for network allowlist for tests if one should be added. */
  // TODO(b/192694287): Remove once we migrate all tests from the allowlist.
  @Override
  public Optional<Label> getNetworkAllowlistForTests() {
    return networkAllowlistForTests;
  }

  /** Returns a map from configuration fragment name to configuration fragment class. */
  public ImmutableMap<String, Class<?>> getFragmentNameToClass() {
    return fragmentNameToClass;
  }
}
