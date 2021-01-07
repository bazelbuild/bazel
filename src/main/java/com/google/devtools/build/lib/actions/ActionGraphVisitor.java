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

/**
 * An abstract visitor for the action graph.  Specializes {@link BipartiteVisitor} for artifacts and
 * actions, and takes care of visiting the complete transitive closure.
 */
public abstract class ActionGraphVisitor extends
    BipartiteVisitor<ActionAnalysisMetadata, Artifact> {

  private final ActionGraph actionGraph;

  public ActionGraphVisitor(ActionGraph actionGraph) {
    this.actionGraph = actionGraph;
  }

  /**
   * Called for all artifacts in the visitation.  Hook for subclasses. 
   *
   * @param artifact
   */
  protected void visitArtifact(Artifact artifact) {}

  /**
   * Called for all actions in the visitation. Hook for subclasses.
   *
   * @param action
   */
  protected void visitAction(ActionAnalysisMetadata action) throws InterruptedException {}

  /**
   * Whether the given action should be visited. If this returns false, the visitation stops here,
   * so the dependencies of this action are also not visited.
   *
   * @param action  
   */
  protected boolean shouldVisit(ActionAnalysisMetadata action) {
    return true;
  }

  /**
   * Whether the given artifact should be visited. If this returns false, the visitation stops here,
   * so dependencies of this artifact (if it is a generated one) are also not visited.
   *
   * @param artifact
   */
  protected boolean shouldVisit(Artifact artifact) {
    return true;
  }

  @SuppressWarnings("unused")
  protected final void visitArtifacts(Iterable<Artifact> artifacts) {
    for (Artifact artifact : artifacts) {
      visitArtifact(artifact);
    }
  }

  @Override
  protected void white(Artifact artifact) throws InterruptedException {
    ActionAnalysisMetadata action = actionGraph.getGeneratingAction(artifact);
    visitArtifact(artifact);
    if (action != null && shouldVisit(action)) {
      visitBlackNode(action);
    }
  }

  @Override
  protected void black(ActionAnalysisMetadata action) throws InterruptedException {
    visitAction(action);
    for (Artifact input : action.getInputs().toList()) {
      if (shouldVisit(input)) {
        visitWhiteNode(input);
      }
    }
  }
}
