// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.EquivalenceRelation;
import com.google.devtools.build.lib.syntax.Label;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Class for debugging, that dumps the Action graph to a GraphViz file for easy visualisation.
 * For example:
 *
 * <pre>new ActionGraphDumper(configuredTarget.getFilesToBuild()).dump(System.out);</pre>
 */
public class ActionGraphDumper implements Dumper {

  private final Collection<Artifact> roots;

  private final ActionGraph actionGraph;

  // Maps artifacts to the set of actions that consume them:
  private final Multimap<Artifact, Action> artifactConsumers = HashMultimap.create();

  // Maps artifacts to the set of topologically-equivalent ones:
  private final Map<Artifact, Set<Artifact>> artifactToClass = new HashMap<>();

  // Options for filtering the graph
  private final Set<String> packagesToKeep;
  private final boolean showMiddlemen;

  /**
   * An equivalence relation for Artifacts that considers them equal iff they
   * have equal topology (predecessors/successors).
   */
  private final EquivalenceRelation<Artifact> EQUIVALENT_TOPOLOGY =
      new EquivalenceRelation<Artifact>() {
        @Override
        public int compare(Artifact x, Artifact y) {
          boolean equal =
              actionGraph.getGeneratingAction(x) == actionGraph.getGeneratingAction(y) &&
              artifactConsumers.get(x).equals(artifactConsumers.get(y));
          return equal ? 0 : -1;
        }
      };

  /**
   * @param roots The Artifacts from which the traversal of the action graph should start.
   */
  public ActionGraphDumper(Collection<Artifact> roots, ActionGraph actionGraph,
      List<String> packagesToKeep, boolean showMiddlemen) {
    this.roots = Preconditions.checkNotNull(roots);
    this.actionGraph = Preconditions.checkNotNull(actionGraph);
    this.packagesToKeep = ImmutableSet.copyOf(packagesToKeep);
    this.showMiddlemen = showMiddlemen;
  }

  @Override
  public void dump(PrintStream out) {
    Preconditions.checkNotNull(out);

    new ComputeConsumersVisitor(actionGraph).computeConsumers();
    new PrintFactoredGraphVisitor(out).printFactoredGraph();
  }

  @Override
  public String getFileName() {
    return "BlazeActionGraph.dot";
  }

  @Override
  public String getName() {
    return "Action Graph";
  }

  // Pass 1: compute the consumers relation for artifacts and the sets of
  // topologically-equivalent artifacts.
  private class ComputeConsumersVisitor extends ActionGraphVisitor {

    private final Predicate<String> packagesToKeepPrediate;

    public ComputeConsumersVisitor(ActionGraph actionGraph) {
      super(actionGraph);
      if (packagesToKeep.isEmpty()) {
        packagesToKeepPrediate = Predicates.alwaysTrue();
      } else {
        packagesToKeepPrediate = Predicates.in(packagesToKeep);
      }
    }

    @Override
    protected boolean shouldVisit(Action action) {
      if (!showMiddlemen && action.getActionType().isMiddleman()) {
        return false;
      }
      ActionOwner owner = action.getOwner();
      if (owner != null) {
        Label label = owner.getLabel();
        if (label != null) {
          if (packagesToKeepPrediate.apply("//" + label.getPackageName())) {
            return true;
          }
        }
      }
      return false;
    }

    @Override
    protected boolean shouldVisit(Artifact artifact) {
      if (!showMiddlemen && artifact.isMiddlemanArtifact()) {
        return false;
      }
      Action action = actionGraph.getGeneratingAction(artifact);
      if ((action == null) || !shouldVisit(action)) {
        return false;
      }
      return true;
    }

    @Override
    protected void visitAction(Action action) {
      for (Artifact input : action.getInputs()) {
        artifactConsumers.put(input, action); // Record that this action consumes 'input'
      }
    }

    public void computeConsumers() {
      for (Artifact root : roots) {
        if (shouldVisit(root)) {
          visitWhiteNode(root);
        }
      }
      // Merge all artifacts of equivalent topology:
      for (Set<Artifact> eqClass :
               CollectionUtils.partition(visitedWhiteNodes.keySet(),
                                         EQUIVALENT_TOPOLOGY)) {
        // Record the mapping from each artifact to its class
        for (Artifact artifact : eqClass) {
          artifactToClass.put(artifact, eqClass);
        }
      }
    }
  }

  // Pass 2: Print the factored graph, using a single node for each set of
  // topologically-equivalent artifacts.
  private class PrintFactoredGraphVisitor
      extends BipartiteVisitor<Set<Artifact>, Action> {

    private final PrintStream out;

    public PrintFactoredGraphVisitor(PrintStream out) {
      this.out = out;
    }

    // In GraphViz file, "eq"-nodes are Artifact equivalence classes, "a"-nodes
    // are Actions.

    @Override
    protected void black(Set<Artifact> eqClass) {
      int eq_id = visitedBlackNodes.get(eqClass);
      out.println("  eq" + eq_id + " [shape=box,label=\""
                  + truncate(prettyArtifacts(eqClass)) + "\"];");
      Action generator = actionGraph.getGeneratingAction(eqClass.iterator().next());
      if (generator != null) {
        visitWhiteNode(generator);
        int a_id = visitedWhiteNodes.get(generator);
        out.println("  eq" + eq_id + " -> a" + a_id + ";");
      }
    }

    @Override
    protected void white(Action action) {
      int a_id = visitedWhiteNodes.get(action);
      out.println("  a" + a_id + " [label=\"" + truncate(prettyAction(action)) + "\"];");
      for (Artifact input : action.getInputs()) {
        Set<Artifact> eqClass = artifactToClass.get(input);
        if ((eqClass != null) && visitBlackNode(eqClass)) {
          int eq_id = visitedBlackNodes.get(eqClass);
          out.println("  a" + a_id + " -> eq" + eq_id + ";");
        }
      }
    }

    public void printFactoredGraph() {
      out.println("digraph action_graph {");
      out.println("  rankdir=LR;");
      for (Artifact root : roots) {
        Set<Artifact> eqClass = artifactToClass.get(root);
        if (eqClass != null) {
          visitBlackNode(eqClass);
        }
      }
      out.println("}");
    }
  }

  // Helpers

  private static String truncate(String s) {
    // GraphViz chokes on labels longer than 1KB.
    return s.length() >= 1024
        ? s.substring(0, 1020) + "..."
        : s;
  }

  private static String prettyAction(Action action) {
    return action.getMnemonic();
  }

  private static String prettyArtifacts(Collection<Artifact> artifacts) {
    List<String> basenames = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      basenames.add(artifact.getExecPath().getBaseName());
    }
    Collections.sort(basenames);
    return Joiner.on("\\n").join(basenames);
  }

}
