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
import com.google.common.base.Preconditions;
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
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import javax.annotation.Nullable;

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
    Set<NinjaTarget> alreadyVisited = Sets.newHashSet();
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
          Preconditions.checkNotNull(alreadyComputedSet);
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
   * <p>Function does DFS starting from the NinjaTarget, with two phases: in initial processing: 1)
   * if the target was already computed, nothing happens 2) the target is checked for cycle and
   * marked in cycleProtection set, its phony inputs are queued (put in the beginning of the queue)
   * for initial processing 3) the target is queued after its inputs for post-processing in
   * post-processing, the target is recorded into resulting list; all its inputs should have been
   * already written to that list on the previous steps
   */
  private static List<NinjaTarget> topoOrderSubGraph(
      ImmutableSortedMap<PathFragment, NinjaTarget> phonyTargets,
      Set<NinjaTarget> alreadyVisited,
      NinjaTarget target)
      throws GenericParsingException {
    Set<NinjaTarget> cycleProtection = Sets.newHashSet();
    List<NinjaTarget> fragment = Lists.newArrayList();
    ArrayDeque<Pair<NinjaTarget, Boolean>> queue = new ArrayDeque<>();
    queue.add(Pair.of(target, true));
    while (!queue.isEmpty()) {
      Pair<NinjaTarget, Boolean> pair = queue.remove();
      NinjaTarget currentTarget = pair.getFirst();
      if (pair.getSecond()) {
        // Initial processing: checking all the phony inputs of the current target.
        if (alreadyVisited.contains(currentTarget)) {
          continue;
        }
        if (!cycleProtection.add(currentTarget)) {
          throw new GenericParsingException(
              String.format(
                  "Detected a dependency cycle involving the phony target '%s'",
                  Iterables.getOnlyElement(currentTarget.getAllOutputs())));
        }
        // Adding <phony-inputs-of-current-target> for initial processing in front of
        // <current-target>
        // for post-processing into the queue.
        queue.addFirst(Pair.of(currentTarget, false));
        for (PathFragment input : currentTarget.getAllInputs()) {
          NinjaTarget phonyInput = phonyTargets.get(input);
          if (phonyInput != null) {
            queue.addFirst(Pair.of(phonyInput, true));
          }
        }
      } else {
        // Post processing: all inputs should have been processed and added to fragment.
        cycleProtection.remove(currentTarget);
        alreadyVisited.add(currentTarget);
        fragment.add(currentTarget);
      }
    }
    return fragment;
  }

  /**
   * Helper interface for artifact creation. We do not pass NinjaArtifactsHelper directly to keep
   * tests simpler.
   */
  public interface InputArtifactCreator<T> {
    @Nullable
    T createArtifact(PathFragment pathFragment) throws GenericParsingException;
  }
}
