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
package com.google.devtools.build.lib.analysis.actions;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;

/**
 * A placeholder action that, at execution time, expands into a list of {@link Action}s to be
 * executed.
 *
 * <p>ActionTemplate is for users who want to dynamically register Actions operating on
 * individual {@link TreeFileArtifact} inside input and output TreeArtifacts at execution time.
 *
 * <p>It takes in one TreeArtifact and generates one TreeArtifact. The following happens at
 * execution time for ActionTemplate:
 * <ol>
 *   <li>Input TreeArtifact is resolved.
 *   <li>For each individual {@link TreeFileArtifact} inside input TreeArtifact, generate an output
 *       {@link TreeFileArtifact} inside output TreeArtifact.
 *   <li>For each pair of input and output {@link TreeFileArtifact}s, generate an associated
 *       {@link Action}.
 *   <li>All expanded {@link Action}s are executed and their output {@link TreeFileArtifact}s
 *       collected.
 *   <li>Output TreeArtifact is resolved.
 * </ol>
 *
 * <p>Implementations of ActionTemplate must follow the contract of this interface and also make
 * sure:
 * <ol>
 *   <li>ActionTemplate instances should be immutable and side-effect free.
 *   <li>ActionTemplate inputs and outputs are supersets of the inputs and outputs of expanded
 *       actions, excluding inputs discovered at execution time. This ensures the ActionTemplate
 *       can properly represent the expanded actions at analysis time, and the action graph
 *       at analysis time is correct. This is important because the action graph is walked in a lot
 *       of places for correctness checks and build analysis.
 *   <li>The outputs of expanded actions must be under the output TreeArtifact and must not have
 *       artifact or artifact path prefix conflicts.
 * </ol>
 */
public interface ActionTemplate<T extends Action> extends ActionAnalysisMetadata {
  /**
   * Given a list of input TreeFileArtifacts resolved at execution time, returns a list of expanded
   * SpawnActions to be executed.
   *
   * @param inputTreeFileArtifacts the list of {@link TreeFileArtifact}s inside input TreeArtifact
   *     resolved at execution time
   * @param artifactOwner the {@link ArtifactOwner} of the generated output
   *     {@link TreeFileArtifact}s
   * @return a list of expanded {@link Action}s to execute, one for each input
   *     {@link TreeFileArtifact}
   */
  Iterable<T> generateActionForInputArtifacts(
      Iterable<TreeFileArtifact> inputTreeFileArtifacts, ArtifactOwner artifactOwner);

  /** Returns the input TreeArtifact. */
  Artifact getInputTreeArtifact();

  /** Returns the output TreeArtifact. */
  Artifact getOutputTreeArtifact();
}
