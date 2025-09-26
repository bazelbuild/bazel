// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableListMultimap.toImmutableListMultimap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.events.EventHandler;
import javax.annotation.Nullable;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

/**
 * A placeholder action that, at execution time, expands into a list of {@link Action}s to be
 * executed.
 *
 * <p>ActionTemplate is for users who want to dynamically register Actions operating on individual
 * {@link TreeFileArtifact} inside input and output TreeArtifacts at execution time.
 *
 * <p>It takes in one or more input TreeArtifacts and generates one or more output TreeArtifacts.
 * The following happens at execution time for ActionTemplate:
 *
 * <ol>
 *   <li>Input TreeArtifact(s) are resolved.
 *   <li>Given the set of {@link TreeFileArtifact}s inside each input TreeArtifact, generate actions
 *       with outputs inside output TreeArtifact(s).
 *   <li>All expanded {@link Action}s are executed and their output {@link TreeFileArtifact}s
 *       collected.
 *   <li>Output TreeArtifact(s) are resolved.
 * </ol>
 *
 * <p>Implementations of ActionTemplate must follow the contract of this interface and also make
 * sure:
 *
 * <ol>
 *   <li>ActionTemplate instances should be immutable and side-effect free.
 *   <li>ActionTemplate inputs and outputs are supersets of the inputs and outputs of expanded
 *       actions, excluding inputs discovered at execution time. This ensures the ActionTemplate can
 *       properly represent the expanded actions at analysis time, and the action graph at analysis
 *       time is correct. This is important because the action graph is walked in a lot of places
 *       for correctness checks and build analysis.
 *   <li>The outputs of expanded actions must be under one of the output TreeArtifact(s) and must
 *       not have artifact or artifact path prefix conflicts.
 * </ol>
 */
public interface ActionTemplate<T extends Action> extends ActionAnalysisMetadata, StarlarkValue {
  /**
   * Given a list of input TreeFileArtifacts resolved at execution time, returns a list of expanded
   * actions to be executed.
   *
   * <p>Each of the expanded actions' outputs must be a {@link TreeFileArtifact} owned by {@code
   * artifactOwner} with a parent in {@link #getOutputs}. This is generally satisfied by calling
   * {@link TreeFileArtifact#createTemplateExpansionOutput}.
   *
   * @param inputTreeFileArtifacts a list of {@link TreeFileArtifact}s from the input
   *     TreeArtifact(s). Use {@link TreeFileArtifact#getParent()} to identify which input {@link
   *     TreeArtifact} the tree file artifact is from.
   * @param artifactOwner the {@link ArtifactOwner} of the generated output {@link
   *     TreeFileArtifact}s
   * @param eventHandler the {@link EventHandler} to report events to.
   * @return a list of expanded {@link Action}s to execute
   */
  ImmutableList<T> generateActionsForInputArtifacts(
      ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
      ActionLookupKey artifactOwner,
      EventHandler eventHandler)
      throws ActionConflictException, ActionExecutionException, InterruptedException;

  /** Returns the input TreeArtifacts. */
  ImmutableList<SpecialArtifact> getInputTreeArtifacts();

  /**
   * Helper method to partition/denormalize the flattened list of input {@link TreeFileArtifact}s
   * into a list multimap of input {@link SpecialArtifact} -> children {@link TreeFileArtifact}s.
   */
  public static ImmutableListMultimap<SpecialArtifact, TreeFileArtifact>
      getInputTreeArtifactsToChildren(ImmutableList<TreeFileArtifact> inputTreeArtifacts) {
    return inputTreeArtifacts.stream()
        .collect(toImmutableListMultimap(TreeFileArtifact::getParent, x -> x));
  }

  @Override
  default SpecialArtifact getPrimaryInput() {
    return getInputTreeArtifacts().get(0);
  }

  @Override
  default Artifact getPrimaryOutput() {
    return getOutputs().iterator().next();
  }

  @Override
  default ImmutableMap<String, String> getExecProperties() {
    return ImmutableMap.of();
  }

  @Override
  @Nullable
  default PlatformInfo getExecutionPlatform() {
    return null;
  }

  @Override
  default void repr(Printer printer) {
    printer.append(prettyPrint());
  }

  @Override
  default boolean isImmutable() {
    return true;
  }
}
