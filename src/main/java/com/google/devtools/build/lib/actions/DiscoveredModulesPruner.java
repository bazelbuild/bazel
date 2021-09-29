// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Prunes discovered CPP modules by filtering out modules which are already accounted for
 * transitively.
 *
 * <p>{@link #DEFAULT} should be used except when special handling is required for {@link NestedSet}
 * data backed by remote storage.
 */
public interface DiscoveredModulesPruner {

  /**
   * Computes top-level only modules, i.e. used modules that aren't also dependencies of other used
   * modules.
   *
   * <p>The returned set's iteration order should match that of {@code usedModules}.
   *
   * @param action the action requesting module pruning
   * @param usedModules set of all modules used by {@code action}
   * @param transitivelyUsedModules map from module to its transitive module dependencies
   * @return a subset of {@code usedModules} without elements that are already accounted for via
   *     transitive dependencies
   * @throws InterruptedException if {@link NestedSet} data in {@code transitivelyUsedModules} is
   *     backed by remote storage and an interruption occurs during retrieval
   * @throws LostInputsActionExecutionException if {@link NestedSet} data in {@code
   *     transitivelyUsedModules} is backed by remote storage and retrieval fails (e.g. due to
   *     timeout)
   */
  Set<Artifact> computeTopLevelModules(
      Action action,
      Set<? extends Artifact> usedModules,
      ImmutableMap<Artifact, NestedSet<Artifact>> transitivelyUsedModules)
      throws InterruptedException, LostInputsActionExecutionException;

  /** Default implementation of module pruning for in-memory {@link NestedSet} data. */
  @SuppressWarnings("SetRemoveAll") // See comment on topLevel.remove().
  DiscoveredModulesPruner DEFAULT =
      (action, usedModules, transitivelyUsedModules) -> {
        Set<Artifact> topLevel = new LinkedHashSet<>(usedModules);

        // It is better to iterate over each nested set here instead of creating a joint one and
        // iterating over it, as this makes use of NestedSet's memoization (each of them has likely
        // been iterated over before).
        for (Map.Entry<Artifact, NestedSet<Artifact>> entry : transitivelyUsedModules.entrySet()) {
          Artifact directDep = entry.getKey();
          if (!topLevel.contains(directDep)) {
            // If this module was removed from topLevel because it is a dependency of another
            // module, we can safely ignore it now as all of its dependants have also been removed.
            continue;
          }
          List<Artifact> transitiveDeps = entry.getValue().toList();

          // Don't use Set.removeAll() here as that iterates over the smaller set (topLevel, which
          // would support efficient lookup) and looks up in the larger one (transitiveDeps, which
          // is a linear scan).
          for (Artifact module : transitiveDeps) {
            topLevel.remove(module);
          }
        }

        return topLevel;
      };
}
