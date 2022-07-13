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

package com.google.devtools.build.lib.packages;

import static com.google.common.base.MoreObjects.firstNonNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/**
 * Bazel-specific contextual information associated with a Starlark evaluation thread.
 *
 * <p>This is stored in the StarlarkThread object as a thread-local. A subclass of this class may be
 * used for certain kinds of Starlark evaluations; in that case it is still keyed in the
 * thread-locals under {@code BazelStarlarkContext.class}.
 *
 * <p>This object is mutable and should not be reused for more than one Starlark thread.
 */
// TODO(b/236456122): rename BazelThreadContext, for symmetry with BazelModuleContext.
// TODO(b/236456122): We should break this class up into subclasses for each kind of evaluation, as
// opposed to storing specialized fields on this class and setting them to null for inapplicable
// environments. Subclasses should define {@code from(StarlarkThread)} and {@code
// fromOrFailFunction(StarlarkThread, String)} methods to be used in place of the {@code
// check*Phase} methods in this class. Kinds of evaluation include:
//   - .bzl loading
//   - BUILD evaluation (should absorb PackageFactory.PackageContext -- no need to store multiple
//     thread-locals in StarlarkThread)
//   - WORKSPACE evaluation (shares logic with BUILD)
//   - rule and aspect analysis implementation (can/should we store the RuleContext here?)
//   - implicit outputs
//   - computed defaults
//   - transition implementation
//   - Args.map_each
//   - probably others
// TODO(b/236456122): The inheritance of RuleDefinitionEnvironment should be replaced by
// composition, in an appropriate subclass of this class. Things like the tools repository, network
// allowlist, etc. can be accessed from the RDE. (If any info needs to be duplicated between RDE and
// here, we should assert consistency with a precondition check.)
public class BazelStarlarkContext
    implements RuleDefinitionEnvironment, StarlarkThread.UncheckedExceptionContext {

  /** The phase to which this Starlark thread belongs. */
  // TODO(b/236456122): Eliminate.
  public enum Phase {
    WORKSPACE,
    LOADING,
    ANALYSIS
  }

  /**
   * Retrieves this context from a Starlark thread, or throws {@link IllegalStateException} if
   * unavailable.
   */
  public static BazelStarlarkContext from(StarlarkThread thread) {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    // ISE rather than NPE for symmetry with subclasses.
    Preconditions.checkState(
        ctx != null, "Expected BazelStarlarkContext to be available in this Starlark thread");
    return ctx;
  }

  /**
   * Saves this {@code BazelStarlarkContext} in the specified Starlark thread. Call only once,
   * before evaluation begins.
   */
  public void storeInThread(StarlarkThread thread) {
    Preconditions.checkState(thread.getThreadLocal(BazelStarlarkContext.class) == null);
    thread.setThreadLocal(BazelStarlarkContext.class, this);
    thread.setUncheckedExceptionContext(this);
  }

  // A generic counter for uniquely identifying symbols created in this Starlark evaluation.
  private final SymbolGenerator<?> symbolGenerator;

  // TODO(b/236456122): Migrate the below fields to subclasses.
  private final Phase phase;
  // Only necessary for loading phase threads.
  @Nullable private final RepositoryName toolsRepository;
  // Only necessary for loading phase threads to construct configuration_field.
  @Nullable private final ImmutableMap<String, Class<?>> fragmentNameToClass;
  @Nullable private final Label analysisRuleLabel;
  // TODO(b/192694287): Remove once we migrate all tests from the allowlist
  @Nullable private final Label networkAllowlistForTests;

  /**
   * @param phase the phase to which this Starlark thread belongs
   * @param toolsRepository the name of the tools repository, such as "@bazel_tools" for loading
   *     phase threads, null for other threads.
   * @param fragmentNameToClass a map from configuration fragment name to configuration fragment
   *     class, such as "apple" to AppleConfiguration.class for loading phase threads, null for
   *     other threads.
   * @param symbolGenerator a {@link SymbolGenerator} to be used when creating objects to be
   *     compared using reference equality.
   * @param analysisRuleLabel is the label of the rule for an analysis phase (rule or aspect
   */
  public BazelStarlarkContext(
      Phase phase,
      @Nullable RepositoryName toolsRepository,
      @Nullable ImmutableMap<String, Class<?>> fragmentNameToClass,
      SymbolGenerator<?> symbolGenerator,
      @Nullable Label analysisRuleLabel,
      @Nullable Label networkAllowlistForTests) {
    this.phase = Preconditions.checkNotNull(phase);
    this.toolsRepository = toolsRepository;
    this.fragmentNameToClass = fragmentNameToClass;
    this.symbolGenerator = Preconditions.checkNotNull(symbolGenerator);
    this.analysisRuleLabel = analysisRuleLabel;
    this.networkAllowlistForTests = networkAllowlistForTests;
  }

  /** Returns the phase associated with this context. */
  public Phase getPhase() {
    return phase;
  }

  /** Returns the name of the tools repository, such as "@bazel_tools". */
  @Nullable
  @Override
  public RepositoryName getToolsRepository() {
    return toolsRepository;
  }

  /** Returns a map from configuration fragment name to configuration fragment class. */
  @Nullable
  public ImmutableMap<String, Class<?>> getFragmentNameToClass() {
    return fragmentNameToClass;
  }

  public SymbolGenerator<?> getSymbolGenerator() {
    return symbolGenerator;
  }

  /**
   * Returns the label of the rule, if this is an analysis-phase (rule or aspect 'implementation')
   * thread, or null otherwise.
   */
  @Nullable
  public Label getAnalysisRuleLabel() {
    return analysisRuleLabel;
  }

  @Override
  public String getContextForUncheckedException() {
    return firstNonNull(analysisRuleLabel, phase).toString();
  }

  @Override
  public Optional<Label> getNetworkAllowlistForTests() {
    return Optional.ofNullable(networkAllowlistForTests);
  }

  /**
   * Checks that the Starlark thread is in the loading or the workspace phase.
   *
   * @param function name of a function that requires this check
   */
  // TODO(b/236456122): The Phase enum is incomplete. Ex: `Args.map_each` evaluation happens at
  // execution time. So this is a misnomer and possibly wrong in those contexts.
  public void checkLoadingOrWorkspacePhase(String function) throws EvalException {
    if (phase == Phase.ANALYSIS) {
      throw Starlark.errorf("'%s' cannot be called during the analysis phase", function);
    }
  }

  /**
   * Checks that the current StarlarkThread is in the loading phase.
   *
   * @param function name of a function that requires this check
   */
  public void checkLoadingPhase(String function) throws EvalException {
    if (phase != Phase.LOADING) {
      throw Starlark.errorf("'%s' can only be called during the loading phase", function);
    }
  }

  /**
   * Checks that the current StarlarkThread is in the workspace phase.
   *
   * @param function name of a function that requires this check
   */
  public void checkWorkspacePhase(String function) throws EvalException {
    if (phase != Phase.WORKSPACE) {
      throw Starlark.errorf("'%s' can only be called during workspace loading", function);
    }
  }
}
