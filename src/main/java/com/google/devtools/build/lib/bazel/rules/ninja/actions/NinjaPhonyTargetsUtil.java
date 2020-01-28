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
 * Cycles are detected and {@link GenericParsingException} is thrown in that case.
 */
public class NinjaPhonyTargetsUtil {
  private final ImmutableSortedMap<PathFragment, NinjaTarget> phonyTargets;

  @VisibleForTesting
  public NinjaPhonyTargetsUtil(
      ImmutableSortedMap<PathFragment, NinjaTarget> phonyTargets) {
    this.phonyTargets = phonyTargets;
  }

  @VisibleForTesting
  public ImmutableSortedMap<PathFragment, NestedSet<PathFragment>> getPhonyPathsMap()
      throws GenericParsingException {
    List<NinjaTarget> topoOrderedTargets = Lists.newArrayListWithCapacity(this.phonyTargets.size());
    SortedSet<NinjaTarget> alreadyQueued = Sets.newTreeSet(
        Comparator.comparing(t -> Iterables.getOnlyElement(t.getAllOutputs())));
    for (Map.Entry<PathFragment, NinjaTarget> entry : phonyTargets.entrySet()) {
      NinjaTarget target = entry.getValue();
      // Topo-ordered phony targets needed to compute 'target' in 'fragment'
      // (if they are not already queued into queue).
      List<NinjaTarget> fragment = topoOrderSubGraph(alreadyQueued, target);
      topoOrderedTargets.addAll(fragment);
    }

    Preconditions.checkState(topoOrderedTargets.size() == this.phonyTargets.size());

    SortedMap<PathFragment, NestedSet<PathFragment>> result = Maps.newTreeMap();
    for (NinjaTarget target : topoOrderedTargets) {
      NestedSetBuilder<PathFragment> builder = new NestedSetBuilder<>(Order.STABLE_ORDER);
      for (PathFragment input : target.getAllInputs()) {
        NinjaTarget innerTarget = phonyTargets.get(input);
        if (innerTarget != null) {
          PathFragment innerOutput = Iterables.getOnlyElement(innerTarget.getAllOutputs());
          // Because of topological sort, must be already computed.
          NestedSet<PathFragment> alreadyComputedIncluded = result.get(innerOutput);
          if (alreadyComputedIncluded == null) {
            // If the target's paths were not computed, then the topo sorting was not successful,
            // which means that there are cycles in phony targets dependencies.
            throw new GenericParsingException(String.format(
                "Detected a dependency cycle involving the phony target '%s'", innerOutput));
          }
          builder.addTransitive(alreadyComputedIncluded);
        } else {
          // We do not check for the duplicates, this would make NestedSet optimization senseless.
          builder.add(input);
        }
      }
      result.put(Iterables.getOnlyElement(target.getAllOutputs()), builder.build());
    }

    return ImmutableSortedMap.copyOf(result);
  }

  private List<NinjaTarget> topoOrderSubGraph(
      SortedSet<NinjaTarget> alreadyQueued, NinjaTarget target) {
    List<NinjaTarget> fragment = Lists.newArrayList();
    ArrayDeque<NinjaTarget> innerQueue = new ArrayDeque<>();
    innerQueue.add(target);
    while (!innerQueue.isEmpty()) {
      NinjaTarget innerTarget = innerQueue.remove();
      if (alreadyQueued.add(innerTarget)) {
        fragment.add(innerTarget);
        for (PathFragment input : innerTarget.getAllInputs()) {
          NinjaTarget innerPhony = phonyTargets.get(input);
          if (innerPhony != null) {
            innerQueue.add(innerPhony);
          }
        }
      }
    }
    // Preconditions were added after their dependants -> reverse the list to get the topo order.
    Collections.reverse(fragment);
    return fragment;
  }
}
