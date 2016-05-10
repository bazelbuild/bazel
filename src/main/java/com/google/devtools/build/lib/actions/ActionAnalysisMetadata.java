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

import com.google.common.collect.ImmutableSet;

/**
 * An Analysis phase interface for an {@link Action} or Action-like object, containing only
 * side-effect-free query methods for information needed during action analysis.
 */
public interface ActionAnalysisMetadata {
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
   * Returns a mnemonic (string constant) for this kind of action; written into
   * the master log so that the appropriate parser can be invoked for the output
   * of the action. Effectively a public method as the value is used by the
   * extra_action feature to match actions.
   */
  String getMnemonic();

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
   * <p>See {@link AbstractAction#getTools()} for an explanation of why it's important that this
   * set contains exactly the right set of artifacts in order for the build to stay correct and the
   * worker strategy to work.
   */
  Iterable<Artifact> getTools();

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
  Iterable<Artifact> getInputs();

  /**
   * Returns the (unordered, immutable) set of output Artifacts that
   * this action generates.  (It would not make sense for this to be empty.)
   */
  ImmutableSet<Artifact> getOutputs();

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
  Iterable<Artifact> getMandatoryInputs();

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
  public enum MiddlemanType {

    /** A normal action. */
    NORMAL,

    /** A normal middleman, which just encapsulates a list of artifacts. */
    AGGREGATING_MIDDLEMAN,

    /**
     * A middleman that enforces action ordering, is not validated by the dependency checker, but
     * allows errors to be propagated.
     */
    ERROR_PROPAGATING_MIDDLEMAN,

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
}
