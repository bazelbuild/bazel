// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An Analysis phase interface for an {@link Action} or Action-like object, containing only
 * side-effect-free query methods for information needed during action analysis.
 */
public interface ActionAnalysisMetadata {

  /**
   * Return this key from {@link #getKey} to signify a failed key computation.
   *
   * <p>Actions that return this value should fail to execute.
   *
   * <p>Consumers must either gracefully handle multiple failed actions having the same key,
   * (recommended), or check against this value explicitly.
   */
  String KEY_ERROR = "1ea50e01-0349-4552-80cf-76cf520e8592";

  /**
   * Returns the owner of this executable if this executable can supply verbose information. This is
   * typically the rule that constructed it; see ActionOwner class comment for details. Returns
   * {@code null} if no owner can be determined.
   *
   * <p>If this executable does not supply verbose information, this function may throw an
   * IllegalStateException.
   */
  ActionOwner getOwner();

  /**
   * Returns true if the action can be shared, i.e. multiple configured targets can create the same
   * action.
   *
   * <p>In theory, these should not exist, but in practice, they do.
   */
  boolean isShareable();

  /**
   * Returns a mnemonic (string constant) for this kind of action; written into
   * the master log so that the appropriate parser can be invoked for the output
   * of the action. Effectively a public method as the value is used by the
   * extra_action feature to match actions.
   */
  String getMnemonic();

  /**
   * Returns a string encoding all of the significant behaviour of this Action that might affect the
   * output. The general contract of <code>getKey</code> is this: if the work to be performed by the
   * execution of this action changes, the key must change.
   *
   * <p>As a corollary, the build system is free to omit the execution of an Action <code>a1</code>
   * if (a) at some time in the past, it has already executed an Action <code>a0</code> with the
   * same key as <code>a1</code>, (b) the names and contents of the input files listed by <code>
   * a1.getInputs()</code> are identical to the names and contents of the files listed by <code>
   * a0.getInputs()</code>, and (c) the names and values in the client environment of the variables
   * listed by <code>a1.getClientEnvironmentVariables()</code> are identical to those listed by
   * <code>a0.getClientEnvironmentVariables()</code>.
   *
   * <p>Examples of changes that should affect the key are:
   *
   * <ul>
   *   <li>Changes to the BUILD file that materially affect the rule which gave rise to this Action.
   *   <li>Changes to the command-line options, environment, or other global configuration resources
   *       which affect the behaviour of this kind of Action (other than changes to the names of the
   *       input/output files, which are handled externally).
   *   <li>An upgrade to the build tools which changes the program logic of this kind of Action
   *       (typically this is achieved by incorporating a UUID into the key, which is changed each
   *       time the program logic of this action changes).
   * </ul>
   *
   * <p>Note the following exception: for actions that discover inputs, the key must change if any
   * input names change or else action validation may falsely validate.
   */
  String getKey(ActionKeyContext actionKeyContext);

  /**
   * Returns a pretty string representation of this action, suitable for use in
   * progress messages or error messages.
   */
  String prettyPrint();

  /**
   * Returns the tool Artifacts that this Action depends upon. May be empty. This is a subset of
   * getInputs().
   *
   * <p>This may be used by spawn strategies to determine whether an external tool has not changed
   * since the last time it was used and could thus be reused, or whether it has to be restarted.
   *
   * <p>See {@link AbstractAction#getTools()} for an explanation of why it's important that this set
   * contains exactly the right set of artifacts in order for the build to stay correct and the
   * worker strategy to work.
   */
  NestedSet<Artifact> getTools();

  /**
   * Returns the input Artifacts that this Action depends upon. May be empty.
   *
   * <p>During execution, the {@link Iterable} returned by {@code getInputs} <em>must not</em> be
   * concurrently modified before the value is fully read in {@code JavaDistributorDriver#exec} (via
   * the {@code Iterable<ActionInput>} argument there). Violating this would require somewhat
   * pathological behavior by the {@link Action}, since it would have to modify its inputs, as a
   * list, say, without reassigning them. This should never happen with any Action subclassing
   * AbstractAction, since AbstractAction's implementation of getInputs() returns an immutable
   * iterable.
   */
  NestedSet<Artifact> getInputs();

  /**
   * Returns the environment variables from the client environment that this action depends on. May
   * be empty.
   *
   * <p>Warning: For optimization reasons, the available environment variables are restricted to
   * those white-listed on the command line. If actions want to specify additional client
   * environment variables to depend on, that restriction must be lifted in
   * {@link com.google.devtools.build.lib.runtime.CommandEnvironment}.
   */
  Iterable<String> getClientEnvironmentVariables();

  /**
   * Returns the (unordered, immutable) set of output Artifacts that
   * this action generates.  (It would not make sense for this to be empty.)
   */
  ImmutableSet<Artifact> getOutputs();

  /**
   * Returns input files that need to be present to allow extra_action rules to shadow this action
   * correctly when run remotely. This is at least the normal inputs of the action, but may include
   * other files as well. For example C(++) compilation may perform include file header scanning.
   * This needs to be mirrored by the extra_action rule. Called by
   * {@link com.google.devtools.build.lib.analysis.extra.ExtraAction} at execution time for actions
   * that return true for {link #discoversInputs()}.
   *
   * @param actionExecutionContext Services in the scope of the action, like the Out/Err streams.
   * @throws ActionExecutionException only when code called from this method
   *     throws that exception.
   * @throws InterruptedException if interrupted
   */
  Iterable<Artifact> getInputFilesForExtraAction(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException;

  /**
   * Returns the set of output Artifacts that are required to be saved. This is
   * used to identify items that would otherwise be potentially identified as
   * orphaned (not consumed by any downstream {@link Action}s and potentially
   * discarded during the build process.
   */
  ImmutableSet<Artifact> getMandatoryOutputs();

  /**
   * Returns the "primary" input of this action, if applicable.
   *
   * <p>For example, a C++ compile action would return the .cc file which is being compiled,
   * irrespective of the other inputs.
   *
   * <p>May return null.
   */
  Artifact getPrimaryInput();

  /**
   * Returns the "primary" output of this action.
   *
   * <p>For example, the linked library would be the primary output of a LinkAction.
   *
   * <p>Never returns null.
   */
  Artifact getPrimaryOutput();

  /**
   * Returns an iterable of input Artifacts that MUST exist prior to executing an action. In other
   * words, in case when action is scheduled for execution, builder will ensure that all artifacts
   * returned by this method are present in the filesystem (artifact.getPath().exists() is true) or
   * action execution will be aborted with an error that input file does not exist. While in
   * majority of cases this method will return all action inputs, for some actions (e.g.
   * CppCompileAction) it can return a subset of inputs because that not all action inputs might be
   * mandatory for action execution to succeed (e.g. header files retrieved from *.d file from the
   * previous build).
   */
  NestedSet<Artifact> getMandatoryInputs();

  /**
   * @return true iff path prefix conflict (conflict where two actions generate
   *         two output artifacts with one of the artifact's path being the
   *         prefix for another) between this action and another action should
   *         be reported.
   */
  boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action);

  /** Returns the action type. Must not be {@code null}. */
  MiddlemanType getActionType();

  /** The action type. */
  enum MiddlemanType {

    /** A normal action. */
    NORMAL,

    /** A normal middleman, which just encapsulates a list of artifacts. */
    AGGREGATING_MIDDLEMAN,

    /**
     * A middleman that denotes a scheduling dependency.
     *
     * <p>If an action has dependencies through scheduling dependency middleman, those dependencies
     * will get built before the action is run and the build will error out if they cannot be built,
     * but the dependencies will not be considered inputs of the action.
     *
     * <p>This is useful in cases when an action <em>might</em> need some inputs, but that is only
     * found out right before it gets executed. The most salient case is C++ compilation where all
     * files that can possibly be included need to be built before the action is executed, but if
     * include scanning is used, only a subset of them will end up as inputs.
     */
    SCHEDULING_DEPENDENCY_MIDDLEMAN,

    /**
     * A runfiles middleman, which is validated by the dependency checker, but is not expanded
     * in blaze. Instead, the runfiles manifest is sent to remote execution client, which
     * performs the expansion.
     */
    RUNFILES_MIDDLEMAN;

    public boolean isMiddleman() {
      return this != NORMAL;
    }
  }

  /**
   * Indicates whether this action has loose headers, or if this is an {@link ActionTemplate},
   * whether the expanded action(s) will have loose headers.
   *
   * <p>If this is true, top-down evaluation considers an action changed if any source files in
   * package have changed.
   */
  default boolean hasLooseHeaders() {
    return false;
  }

  /** Returns a String to String map containing the execution properties of this action. */
  ImmutableMap<String, String> getExecProperties();

  /**
   * Returns the {@link PlatformInfo} platform this action should be executed on. If the execution
   * platform is {@code null}, then the host platform is assumed.
   */
  @Nullable
  PlatformInfo getExecutionPlatform();

  /**
   * Returns the execution requirements for this action, or null if the action type does not have
   * access to execution requirements.
   */
  @Nullable
  default Map<String, String> getExecutionInfo() {
    return null;
  }
}
