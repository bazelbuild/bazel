// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;

/**
 * An utility class for gathering non-phony input dependencies of phony {@link NinjaTarget} objects
 * from some Ninja file.
 *
 * <p>Cycles are detected and {@link GenericParsingException} is thrown in that case.
 */
public class NinjaPhonyTargetsUtil {
  private NinjaPhonyTargetsUtil() {}

  @VisibleForTesting
  public static <T> ImmutableSortedMap<PathFragment, NestedSet<T>> getPhonyPathsMap(
      ImmutableSortedMap<PathFragment, NinjaTarget> phonyTargets,
      InputArtifactCreator<T> artifactsHelper)
      throws GenericParsingException {
    // There is always a DAG (or forest) of phony targets (as item can be included into several
    // phony targets).
    // This gives us the idea that we can compute any subgraph in the DAG independently, and
    // later include that subgraph into the parent DAG.
    // In the list 'topoOrderedTargets' we will put all the NinjaTargets from the phonyTargets,
    // in the following order: all nodes from a node's subtree are preceding it in a list.
    // This way we can later visit the list "topoOrderedTargets", computing the input files
    // for targets, and for each target K the results for all the phony targets from a subgraph of K
    // will be already computed.
    // The sorting is linear, as we are only checking each input of each node once (we use already).
    List<NinjaTarget> topoOrderedTargets = Lists.newArrayListWithCapacity(phonyTargets.size());
    SortedSet<NinjaTarget> alreadyVisited =
        Sets.newTreeSet(Comparator.comparing(t -> Iterables.getOnlyElement(t.getAllOutputs())));
    for (Map.Entry<PathFragment, NinjaTarget> entry : phonyTargets.entrySet()) {
      NinjaTarget target = entry.getValue();
      topoOrderedTargets.addAll(topoOrderSubGraph(phonyTargets, alreadyVisited, target));
    }

    checkState(topoOrderedTargets.size() == phonyTargets.size());

    SortedMap<PathFragment, NestedSet<T>> result = Maps.newTreeMap();
    for (NinjaTarget target : topoOrderedTargets) {
      NestedSetBuilder<T> builder = new NestedSetBuilder<>(Order.STABLE_ORDER);
      for (PathFragment input : target.getAllInputs()) {
        NinjaTarget phonyInput = phonyTargets.get(input);
        if (phonyInput != null) {
          // The input is the other phony target.
          // Add the corresponding already computed NestedSet as transitive.
          // Phony target must have only one output (alias); it is checked during parsing.
          PathFragment phonyName = Iterables.getOnlyElement(phonyInput.getAllOutputs());
          NestedSet<T> alreadyComputedSet = result.get(phonyName);
          if (alreadyComputedSet == null) {
            // If the target's paths were not computed, then the topo sorting was not successful,
            // which means that there are cycles in phony targets dependencies.
            throw new GenericParsingException(
                String.format(
                    "Detected a dependency cycle involving the phony target '%s'", phonyName));
          }
          builder.addTransitive(alreadyComputedSet);
        } else {
          // The input is the usual file.
          // We do not check for the duplicates, this would make NestedSet optimization senseless.
          builder.add(artifactsHelper.createArtifact(input));
        }
      }
      result.put(Iterables.getOnlyElement(target.getAllOutputs()), builder.build());
    }

    return ImmutableSortedMap.copyOf(result);
  }

  /**
   * For the given phony NinjaTarget, return a list of all phony NinjaTargets, composing its subtree
   * (direct and transitive inputs). The list is ordered from leaves to their dependents; for any
   * node all its direct and transitive inputs are preceding it in the list.
   *
   * <p>Function does BFS starting from the NinjaTarget, records the order of visiting, and reverses
   * the result. The nodes that were already visited in previous iterations are skipped, because
   * they have already been added to the resulting aggregated list.
   */
  private static List<NinjaTarget> topoOrderSubGraph(
      ImmutableSortedMap<PathFragment, NinjaTarget> phonyTargets,
      SortedSet<NinjaTarget> alreadyVisited,
      NinjaTarget target) {
    List<NinjaTarget> fragment = Lists.newArrayList();
    ArrayDeque<NinjaTarget> queue = new ArrayDeque<>();
    queue.add(target);
    while (!queue.isEmpty()) {
      NinjaTarget currentTarget = queue.remove();
      if (alreadyVisited.add(currentTarget)) {
        // If not visited, put all phony inputs into the queue.
        fragment.add(currentTarget);
        for (PathFragment input : currentTarget.getAllInputs()) {
          NinjaTarget phonyInput = phonyTargets.get(input);
          if (phonyInput != null) {
            queue.add(phonyInput);
          }
        }
      }
    }
    // Preconditions were added after their dependants -> reverse the list to get the topo order.
    Collections.reverse(fragment);
    return fragment;
  }

  /**
   * Helper interface for artifact creation. We do not pass NinjaArtifactsHelper directly to keep
   * tests simpler.
   */
  public interface InputArtifactCreator<T> {
    T createArtifact(PathFragment pathFragment) throws GenericParsingException;
  }
}
