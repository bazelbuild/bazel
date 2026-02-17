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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static java.util.Comparator.naturalOrder;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Comparators;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.bzlmod.InterimModule.DepSpec;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import java.util.ArrayDeque;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;

import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Runs module selection. This step of module resolution reads the output of {@link Discovery} and
 * applies the Minimal Version Selection algorithm to it, removing unselected modules from the
 * dependency graph and rewriting dependencies to point to the selected versions. It also returns an
 * un-pruned version of the dep graph for inspection purpose.
 *
 * <p>Minimal Version Selection (MVS) is used to select a single version for each module.
 *
  * <ul>
 *   <li>In the most basic case, only one version of each module is selected (ie. remains in the dep
 *       graph). The selected version is simply the highest among all existing versions in the dep
 *       graph. In other words, each module name forms a "selection group". If foo@1.5 is selected,
 *       then any other foo@X is removed from the dep graph, and any module depending on foo@X will
 *       depend on foo@1.5 instead.
 *   <li>As an extension of the above, we also remove any module that becomes unreachable from the
 *       root module because of the removal of some other module.
 *   <li>Things get more complicated with multiple-version overrides. If module foo has a
 *       multiple-version override which allows versions [1.3, 1.5, 2.0], then we further split the
 *       selection groups by the target allowed version (keep in mind that versions are upgraded to
 *       the nearest higher-or-equal allowed version). If, for example, some module depends on
 *       foo@1.0, then it'll depend on foo@1.3 post-selection instead (and foo@1.0 will be removed).
 *       If any of foo@2.2, or foo@3.0 exist in the dependency graph before selection, they must be
 *       removed before the end of selection (by becoming unreachable, for example), otherwise it'll
 *       be an error since they're not allowed by the override (these versions are in selection
 *       groups that have no valid target allowed version).
 * </ul>

 */
final class Selection {
  private Selection() {}

  /**
   * The result of selection.
   *
   * @param resolvedDepGraph Final dep graph sorted in BFS iteration order, with unused modules
   *     removed.
   * @param unprunedDepGraph Un-pruned dep graph, with updated dep keys, and additionally containing
   *     the unused modules which were initially discovered (and their MODULE.bazel files loaded).
   *     Does not contain modules overridden by {@code single_version_override} or {@link
   *     NonRegistryOverride}, only by {@code multiple_version_override}.
   */
  record Result(
      ImmutableMap<ModuleKey, InterimModule> resolvedDepGraph,
      ImmutableMap<ModuleKey, InterimModule> unprunedDepGraph) {}

  /**
   * During selection, a version is selected for each distinct "selection group".
   *
   * @param targetAllowedVersion This is only used for modules with multiple-version overrides.
   */
  private record SelectionGroup(String moduleName, Version targetAllowedVersion) {}



  /**
   * For the given module, compute its selection group. Versions of the same module that fall into
   * different "allowed version buckets" (defined by {@code multiple_version_override}) belong to
   * different selection groups.
   */
  private static SelectionGroup computeSelectionGroup(
      InterimModule module, ImmutableMap<String, ImmutableSortedSet<Version>> allowedVersionSets) {
    return computeSelectionGroup(
        module.getKey().name(), module.getKey().version(), allowedVersionSets);
  }

  private static SelectionGroup computeSelectionGroup(
      String name, Version version, ImmutableMap<String, ImmutableSortedSet<Version>> allowedVersionSets) {
    ImmutableSortedSet<Version> allowedVersions = allowedVersionSets.get(name);
    Version target = Version.EMPTY;
    if (allowedVersions != null) {
      target = allowedVersions.ceiling(version);
      if (target == null) {
        target = Version.EMPTY;
      }
    }
    return new SelectionGroup(name, target);
  }

  private static ImmutableMap<String, ImmutableSortedSet<Version>> computeAllowedVersionSets(
      ImmutableMap<String, ModuleOverride> overrides,
      ImmutableMap<ModuleKey, InterimModule> depGraph)
      throws ExternalDepsException {
    Map<String, Set<Version>> moduleToVersionsInDepGraph = new HashMap<>();
    for (ModuleKey key : depGraph.keySet()) {
      moduleToVersionsInDepGraph.computeIfAbsent(key.name(), k -> new HashSet<>()).add(key.version());
    }

    ImmutableMap.Builder<String, ImmutableSortedSet<Version>> allowedVersionSets =
        ImmutableMap.builder();
    for (Map.Entry<String, ModuleOverride> entry : overrides.entrySet()) {
      if (!(entry.getValue() instanceof MultipleVersionOverride override)) {
        continue;
      }
      String moduleName = entry.getKey();
      for (Version v : override.versions()) {
        if (!moduleToVersionsInDepGraph.getOrDefault(moduleName, ImmutableSet.of()).contains(v)) {
          throw ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR,
              "multiple_version_override for module %s contains version %s, but it doesn't exist"
                  + " in the dependency graph",
              moduleName,
              v);
        }
      }
      allowedVersionSets.put(moduleName, ImmutableSortedSet.copyOf(override.versions()));
    }
    return allowedVersionSets.buildOrThrow();
  }

  /** Runs module selection (aka version resolution). */
  public static Result run(
      ImmutableMap<ModuleKey, InterimModule> depGraph,
      ImmutableMap<String, ModuleOverride> overrides)
      throws ExternalDepsException, InterruptedException {
    // Compute the allowed version sets for each module, and check that all versions listed in
    // multiple-version overrides exist in the dep graph.
    ImmutableMap<String, ImmutableSortedSet<Version>> allowedVersionSets =
        computeAllowedVersionSets(overrides, depGraph);

    // For each module in the dep graph, pre-compute its selection group.
    ImmutableMap<ModuleKey, SelectionGroup> selectionGroups =
        ImmutableMap.copyOf(
            Maps.transformValues(
                depGraph, module -> computeSelectionGroup(module, allowedVersionSets)));

    // Figure out the version to select for every selection group.
    Map<SelectionGroup, Version> selectedVersions = new HashMap<>();
    for (Map.Entry<ModuleKey, SelectionGroup> entry : selectionGroups.entrySet()) {
      ModuleKey key = entry.getKey();
      SelectionGroup selectionGroup = entry.getValue();
      selectedVersions.merge(selectionGroup, key.version(), Comparators::max);
    }

    Function<DepSpec, Version> resolutionStrategy =
        depSpec ->
            selectedVersions.get(
                computeSelectionGroup(depSpec.name(), depSpec.version(), allowedVersionSets));

    DepGraphWalker depGraphWalker = new DepGraphWalker(depGraph, overrides, selectionGroups);

    // Walk the graph taking nodep edges into account.
    // If we selected a version that doesn't exist (e.g. because of multiple_version_override
    // snapping to a non-existent version), the walker will throw.
    var unused = depGraphWalker.walk(resolutionStrategy, /* ignoreNodeps= */ false);

    // Walk the graph again, this time ignoring nodeps, so that we don't end up with modules that
    // are only reachable via nodep edges.
    ImmutableMap<ModuleKey, InterimModule> prunedDepGraph =
        depGraphWalker.walk(resolutionStrategy, /* ignoreNodeps= */ true);

    ImmutableMap<ModuleKey, InterimModule> unprunedDepGraph =
        ImmutableMap.copyOf(
            Maps.transformValues(
                depGraph,
                module ->
                    module.withDepsTransformed(
                        depSpec -> depSpec.withVersion(resolutionStrategy.apply(depSpec)))));

    return new Result(prunedDepGraph, unprunedDepGraph);
  }

  /**
   * Walks the dependency graph from the root node, collecting any reachable nodes through deps into
   * a new dep graph and checking that nothing conflicts.
   */
  static class DepGraphWalker {
    private static final Joiner JOINER = Joiner.on(", ");
    private final ImmutableMap<ModuleKey, InterimModule> oldDepGraph;
    private final ImmutableMap<String, ModuleOverride> overrides;
    private final ImmutableMap<ModuleKey, SelectionGroup> selectionGroups;

    DepGraphWalker(
        ImmutableMap<ModuleKey, InterimModule> oldDepGraph,
        ImmutableMap<String, ModuleOverride> overrides,
        ImmutableMap<ModuleKey, SelectionGroup> selectionGroups) {
      this.oldDepGraph = oldDepGraph;
      this.overrides = overrides;
      this.selectionGroups = selectionGroups;
    }

    /**
     * Walks the old dep graph and builds a new dep graph containing only deps reachable from the
     * root module. The returned map has a guaranteed breadth-first iteration order.
     */
    ImmutableMap<ModuleKey, InterimModule> walk(
        Function<DepSpec, Version> resolutionStrategy, boolean ignoreNodeps)
        throws ExternalDepsException {
      HashMap<String, ExistingModule> moduleByName = new HashMap<>();
      ImmutableMap.Builder<ModuleKey, InterimModule> newDepGraph = ImmutableMap.builder();
      Set<ModuleKey> known = new HashSet<>();
      Queue<ModuleKeyAndDependent> toVisit = new ArrayDeque<>();
      toVisit.add(new ModuleKeyAndDependent(ModuleKey.ROOT, null));
      known.add(ModuleKey.ROOT);
      while (!toVisit.isEmpty()) {
        ModuleKeyAndDependent moduleKeyAndDependent = toVisit.remove();
        ModuleKey key = moduleKeyAndDependent.moduleKey();
        InterimModule oldModule = oldDepGraph.get(key);
        if (oldModule == null) {
          // This should only happen if we selected a version that doesn't exist. This could happen
          // if MultipleVersionOverride snapped to something that's not in the graph, or if
          // selectedVersions.get(group) returned something weird. But we validated that
          // targetAllowedVersion is in the graph, and ROOT is the only thing with EMPTY version.
          throw new RuntimeException("Unexpected error: missing key " + key + " in old dep graph");
        }
        InterimModule module =
            oldModule.withDepsTransformed(
                depSpec -> depSpec.withVersion(resolutionStrategy.apply(depSpec)));
        visit(key, module, moduleKeyAndDependent.dependent(), moduleByName);

        for (DepSpec depSpec :
            ignoreNodeps
                ? module.getDeps().values()
                : Iterables.concat(module.getDeps().values(), module.getNodepDeps())) {
          if (known.add(depSpec.toModuleKey())) {
            toVisit.add(new ModuleKeyAndDependent(depSpec.toModuleKey(), key));
          }
        }
        newDepGraph.put(key, module);
      }
      return newDepGraph.buildOrThrow();
    }

    void visit(
        ModuleKey key,
        InterimModule module,
        @Nullable ModuleKey from,
        HashMap<String, ExistingModule> moduleByName)
        throws ExternalDepsException {
      if (overrides.get(key.name()) instanceof MultipleVersionOverride override) {
        if (selectionGroups.get(key).targetAllowedVersion().isEmpty()) {
          // This module has no target allowed version, which means that there's no allowed version
          // higher than its version in the allowlist.
          Preconditions.checkState(
              from != null, "the root module cannot have a multiple version override");
          throw ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR,
              "%s depends on %s which is not allowed by the multiple_version_override on %s,"
                  + " which allows only [%s]",
              from,
              key,
              key.name(),
              JOINER.join(override.versions()));
        }
      } else {
        ExistingModule existingModuleWithSameName =
            moduleByName.put(module.getName(), new ExistingModule(key, from));
        if (existingModuleWithSameName != null) {
          // This should only happen if there's a multiple-version override.
          Preconditions.checkState(
              overrides.get(module.getName()) instanceof MultipleVersionOverride);
        }
      }

      // Make sure that we don't have `module` depending on the same dependency version twice.
      HashMap<ModuleKey, String> depKeyToRepoName = new HashMap<>();
      for (Map.Entry<String, DepSpec> depEntry : module.getDeps().entrySet()) {
        String repoName = depEntry.getKey();
        DepSpec depSpec = depEntry.getValue();
        String previousRepoName = depKeyToRepoName.put(depSpec.toModuleKey(), repoName);
        if (previousRepoName != null) {
          throw ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR,
              "%s depends on %s at least twice (with repo names %s and %s). Consider adding a"
                  + " multiple_version_override if you want to depend on multiple versions of"
                  + " %s simultaneously",
              key,
              depSpec.toModuleKey(),
              repoName,
              previousRepoName,
              depSpec.name());
        }
      }
    }

    record ModuleKeyAndDependent(ModuleKey moduleKey, @Nullable ModuleKey dependent) {}

    record ExistingModule(ModuleKey moduleKey, @Nullable ModuleKey dependent) {}
  }
}
