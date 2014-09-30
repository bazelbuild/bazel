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

package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.syntax.Label;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * This class finds the subset of labels whose transitive closure is
 * error-free.
 *
 * <p> The graph is assumed to be potentially cyclic (such errors are reported later,
 * in the analysis phase).
 *
 * <p> This is a required phase in --keep_going builds when loading
 * is unsuccessful. In this case, we'd like to know the subset
 * of labels whose transitive closure are error free. Only these
 * transitively error-free labels may be passed to the analysis phase.
 *
 * The methods which implements TargetEdgeObserver are thread-safe.
 * Only after the visitation is complete is it safe to call
 * {@link #getRootCauses}.
 */
@ConditionallyThreadSafe // condition: only call getTransitivelyErrorFreeTargets
                         // and getRootCauses once the visitation is complete.
public class TransitivelyErrorFreeTargetEdgeObserver implements TargetEdgeObserver {

  /** A representation of the target graph. May be cyclic. */
  private final Digraph<Label> targetGraph = new Digraph<>();

  private final Set<Label> errorTargets = new HashSet<>();

  /** See {@link TargetEdgeObserver#edge}. */
  @Override
  public synchronized void edge(Target from, Attribute attribute, Target to) {
    targetGraph.addEdge(from.getLabel(), to.getLabel());
    checkTarget(from);
    checkTarget(to);
  }

  /** See {@link TargetEdgeObserver#node}. */
  @Override
  public synchronized void node(Target node) {
    targetGraph.createNode(node.getLabel());
    checkTarget(node);
  }

  private final void checkTarget(Target target) {
    if (TargetUtils.containsErrors(target)) {
      errorTargets.add(target.getLabel());
    }
  }

  /** See {@link TargetEdgeObserver#missingEdge}. */
  @Override
  public synchronized void missingEdge(Target from, Label to, NoSuchThingException e) {
    targetGraph.createNode(to);
    errorTargets.add(to);
    if (from != null) {
      errorTargets.add(from.getLabel());
      targetGraph.addEdge(from.getLabel(), to);
    }
  }

  private static final String ROOT_CAUSE_SUFFIX = "____root_cause_analysis___";
  private static Label rootCauseLabel(Label label) {
    try {
      return Label.parseAbsolute(label.toString()
          + ROOT_CAUSE_SUFFIX);
    } catch (Label.SyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Return a mapping between targets and root causes. Note that targets in the
   * input that are transitively error free will not be in the output map.
   *
   * @param originalTargets the set of targets to be checked
   * @return a mapping of targets to root causes
   */
  public Multimap<Label, Label> getRootCauses(Collection<Label> originalTargets) {
    // Root causes for a certain input node are those error nodes which are
    // reachable from the input node, but from which no other error node is
    // reachable.
    Multimap<Label, Label> result = ArrayListMultimap.create();

    // 1. Copy the original target graph, and duplicate the original target
    // nodes in it. This is needed so that we do not confuse the original nodes
    // with error nodes during the graph algorithm.

    Digraph<Label> graph1 = targetGraph.clone();
    Set<Label> copyLabels = new HashSet<>();
    for (Label originalTarget : originalTargets) {
      Label newLabel = rootCauseLabel(originalTarget);
      graph1.createNode(newLabel);
      graph1.addEdge(newLabel, originalTarget);
      copyLabels.add(newLabel);
    }

    // 2. Create a subgraph that contains the copy of the original target nodes
    // and the error nodes.

    Digraph<Label> graph2 = graph1.extractSubgraph(
        Sets.union(errorTargets, copyLabels));

    // 3. graph2 now contains information about which error is reachable from
    // which label. Select the leaf nodes in that graph because they are the
    // root causes and compute another subgraph.

    Set<Label> rootLabels = new HashSet<>();
    for (Node<Label> rootLeaf : graph2.getLeaves()) {
      rootLabels.add(rootLeaf.getLabel());
    }

    Digraph<Label> graph3 = graph2.extractSubgraph(
        Sets.union(rootLabels, copyLabels));

    // 4. graph3 now contains the copies of original target nodes (which are
    // guaranteed to be roots) and the root causes. Thus, the set of root causes
    // for a target is the set of nodes reachable from it. Furthermore, since
    // we only included labels in this subgraph that are leaves in graph2, the
    // length of maximum path from a root to a leaf is one.

    for (Label target : originalTargets) {
      for (Node<Label> rootCause : graph3.getNode(rootCauseLabel(target)).getSuccessors()) {
        result.put(target, rootCause.getLabel());
      }
    }

    return result;
  }
}
