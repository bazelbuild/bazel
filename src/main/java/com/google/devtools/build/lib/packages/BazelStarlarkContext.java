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
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.HashMap;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Contextual information associated with each Starlark thread created by Bazel. */
// TODO(adonovan): rename BazelThreadContext, for symmetry with BazelModuleContext.
// TODO(brandjon): Use composition rather than inheritance for RuleDefinitionEnvironment; clients
// should retrieve the RDE (if it exists) from this class in order to access e.g. the network
// allowlist. The toolsRepository info will be duplicated between this class and RDE but we can
// enforce consistency with a precondition check.
public final class BazelStarlarkContext
    implements RuleDefinitionEnvironment,
        Label.HasRepoMapping,
        StarlarkThread.UncheckedExceptionContext {

  /** The phase to which this Starlark thread belongs. */
  public enum Phase {
    WORKSPACE,
    LOADING,
    ANALYSIS
  }

  /** Return the Bazel information associated with the specified Starlark thread. */
  public static BazelStarlarkContext from(StarlarkThread thread) {
    return thread.getThreadLocal(BazelStarlarkContext.class);
  }

  /** Save this BazelStarlarkContext in the specified Starlark thread. */
  public void storeInThread(StarlarkThread thread) {
    thread.setThreadLocal(BazelStarlarkContext.class, this);
    thread.setThreadLocal(Label.HasRepoMapping.class, this);
    thread.setUncheckedExceptionContext(this);
  }

  private final Phase phase;
  // Only necessary for loading phase threads.
  @Nullable private final RepositoryName toolsRepository;
  // Only necessary for loading phase threads to construct configuration_field.
  @Nullable private final ImmutableMap<String, Class<?>> fragmentNameToClass;
  private final HashMap<String, Label> convertedLabelsInPackage;
  private final SymbolGenerator<?> symbolGenerator;
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
   * @param convertedLabelsInPackage a mutable map from String to Label, used during package loading
   *     of a single package.
   * @param symbolGenerator a {@link SymbolGenerator} to be used when creating objects to be
   *     compared using reference equality.
   * @param analysisRuleLabel is the label of the rule for an analysis phase (rule or aspect
   *     'implementation') thread, or null for all other threads.
   */
  // TODO(adonovan): clearly demarcate which fields are defined in which kinds of threads (loading,
  // analysis, workspace, implicit outputs, computed defaults, etc), perhaps by splitting these into
  // separate structs, exactly one of which is populated (plus the common fields). And eliminate
  // StarlarkUtils.Phase.
  // TODO(adonovan): move PackageFactory.PackageContext in here, for loading-phase threads.
  // TODO(adonovan): is there any reason not to put the entire RuleContext in this thread, for
  // analysis threads?
  public BazelStarlarkContext(
      Phase phase,
      @Nullable RepositoryName toolsRepository,
      @Nullable ImmutableMap<String, Class<?>> fragmentNameToClass,
      HashMap<String, Label> convertedLabelsInPackage,
      SymbolGenerator<?> symbolGenerator,
      @Nullable Label analysisRuleLabel,
      @Nullable Label networkAllowlistForTests) {
    this.phase = Preconditions.checkNotNull(phase);
    this.toolsRepository = toolsRepository;
    this.fragmentNameToClass = fragmentNameToClass;
    this.convertedLabelsInPackage = convertedLabelsInPackage;
    this.symbolGenerator = Preconditions.checkNotNull(symbolGenerator);
    this.analysisRuleLabel = analysisRuleLabel;
    this.networkAllowlistForTests = networkAllowlistForTests;
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

  /**
   * Returns a map of {@code RepositoryName}s where the keys are repository names that are written
   * in the BUILD files and the values are new repository names chosen by the main repository.
   */
  @Override
  public RepositoryMapping getRepoMappingForCurrentBzlFile(StarlarkThread thread) {
    // TODO(b/200024947): Find a better place for this. We don't want Label to have to depend on
    //   StarlarkModuleContext, but having the logic in BazelStarlarkContext is purely a historical
    //   misstep.
    return BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread)).repoMapping();
  }

  /**
   * Returns a String -> Label map of all the Strings that have already been converted to Labels
   * during package loading of the current package.
   *
   * <p>This is used for a performance optimization during package loading, and unused otherwise.
   */
  public HashMap<String, Label> getConvertedLabelsInPackage() {
    return convertedLabelsInPackage;
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
