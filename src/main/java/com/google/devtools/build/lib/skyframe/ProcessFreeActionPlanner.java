// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.Set;

/** Builds a fail-closed plan for materializing Bazel-owned process-free actions. */
final class ProcessFreeActionPlanner {
  record Plan(
      ImmutableList<ActionLookupData> actionKeys,
      int visitedActions,
      int deferredActions,
      int unresolvedArtifacts,
      ImmutableList<ActionInventory> actionInventory) {}

  record ActionInventory(
      String state,
      String actionClass,
      String mnemonic,
      int inputEdges,
      int schedulingDependencyEdges,
      int outputs) {}

  private enum State {
    VISITING,
    ELIGIBLE,
    DEFERRED
  }

  private static final class ActionFrame {
    private final ActionAnalysisMetadata action;
    private final ImmutableList<Artifact> inputs;
    private final ImmutableList<Artifact> schedulingDependencies;
    private int nextInput;
    private int nextSchedulingDependency;
    private boolean inputsEligible = true;

    private ActionFrame(ActionAnalysisMetadata action) {
      this.action = action;
      this.inputs = action.getInputs().toList();
      this.schedulingDependencies = action.getSchedulingDependencies().toList();
    }

    private boolean hasNextDependency() {
      return nextInput < inputs.size() || nextSchedulingDependency < schedulingDependencies.size();
    }

    private Artifact nextDependency() {
      if (nextInput < inputs.size()) {
        return inputs.get(nextInput++);
      }
      return schedulingDependencies.get(nextSchedulingDependency++);
    }
  }

  private final ActionGraph actionGraph;
  private final boolean collectInventory;
  private final IdentityHashMap<ActionAnalysisMetadata, State> states = new IdentityHashMap<>();
  private final Set<Artifact> visitedArtifacts =
      Collections.newSetFromMap(new IdentityHashMap<>());
  private final LinkedHashSet<ActionLookupData> actionKeys = new LinkedHashSet<>();
  private int unresolvedArtifacts;
  private final ImmutableList.Builder<ActionInventory> actionInventory = ImmutableList.builder();

  private ProcessFreeActionPlanner(ActionGraph actionGraph, boolean collectInventory) {
    this.actionGraph = actionGraph;
    this.collectInventory = collectInventory;
  }

  static Plan create(
      Iterable<Artifact> roots, ActionGraph actionGraph, boolean collectInventory) {
    ProcessFreeActionPlanner planner =
        new ProcessFreeActionPlanner(actionGraph, collectInventory);
    for (Artifact root : roots) {
      planner.visitArtifact(root);
    }
    int deferred = 0;
    for (State state : planner.states.values()) {
      if (state == State.DEFERRED) {
        deferred++;
      }
    }
    return new Plan(
        ImmutableList.copyOf(planner.actionKeys),
        planner.states.size(),
        deferred,
        planner.unresolvedArtifacts,
        planner.actionInventory.build());
  }

  private boolean visitArtifact(Artifact artifact) {
    ArrayDeque<ActionFrame> stack = new ArrayDeque<>();
    Artifact nextArtifact = artifact;
    boolean eligible;

    traversal:
    while (true) {
      if (!visitedArtifacts.add(nextArtifact)) {
        if (nextArtifact.isSourceArtifact()) {
          eligible = true;
        } else {
          ActionAnalysisMetadata generatingAction =
              actionGraph.getGeneratingAction(nextArtifact);
          eligible = generatingAction != null && states.get(generatingAction) == State.ELIGIBLE;
        }
      } else {
        ActionAnalysisMetadata generatingAction = actionGraph.getGeneratingAction(nextArtifact);
        if (generatingAction == null) {
          eligible = nextArtifact.isSourceArtifact();
          if (!eligible) {
            unresolvedArtifacts++;
          }
        } else {
          State prior = states.get(generatingAction);
          if (prior != null) {
            eligible = prior == State.ELIGIBLE;
          } else {
            states.put(generatingAction, State.VISITING);
            ActionFrame frame = new ActionFrame(generatingAction);
            if (frame.hasNextDependency()) {
              stack.push(frame);
              nextArtifact = frame.nextDependency();
              continue;
            }
            eligible = finish(frame);
          }
        }
      }

      while (!stack.isEmpty()) {
        ActionFrame frame = stack.peek();
        frame.inputsEligible &= eligible;
        if (frame.hasNextDependency()) {
          nextArtifact = frame.nextDependency();
          continue traversal;
        }
        stack.pop();
        eligible = finish(frame);
      }
      return eligible;
    }
  }

  private boolean finish(ActionFrame frame) {
    ActionAnalysisMetadata action = frame.action;
    boolean declaredProcessFree = action.isProcessFree();
    // Input-discovering actions can add generated dependencies only during execution, after this
    // plan has established its source-rooted closure. Defer them even if a future implementation
    // accidentally declares the capability.
    boolean discoversInputs =
        action instanceof Action executableAction && executableAction.discoversInputs();
    boolean eligible = declaredProcessFree && !discoversInputs && frame.inputsEligible;
    states.put(action, eligible ? State.ELIGIBLE : State.DEFERRED);
    if (collectInventory) {
      actionInventory.add(
          new ActionInventory(
              eligible ? "selected" : declaredProcessFree ? "blocked" : "deferred",
              action.getClass().getName(),
              action.getMnemonic(),
              frame.inputs.size(),
              frame.schedulingDependencies.size(),
              action.getOutputs().size()));
    }
    if (eligible) {
      actionKeys.add(
          ((Artifact.DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey());
    }
    return eligible;
  }
}
