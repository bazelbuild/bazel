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

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Comparators;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Runs module selection. This step of module resolution reads the output of {@link Discovery} and
 * applies the Minimal Version Selection algorithm to it, removing unselected modules from the
 * dependency graph and rewriting dependencies to point to the selected versions. It also returns an
 * un-pruned version of the dep graph for inspection purpose.
 *
 * <p>Essentially, what needs to happen is:
 *
 * <ul>
 *   <li>In the most basic case, only one version of each module is selected (ie. remains in the dep
 *       graph). The selected version is simply the highest among all existing versions in the dep
 *       graph. In other words, each module name forms a "selection group". If foo@1.5 is selected,
 *       then any other foo@X is removed from the dep graph, and any module depending on foo@X will
 *       depend on foo@1.5 instead.
 *   <li>As an extension of the above, we also remove any module that becomes unreachable from the
 *       root module because of the removal of some other module.
 *   <li>If, however, versions of the same module but with different compatibility levels exist in
 *       the dep graph, then one version is selected for each compatibility level (ie. we split the
 *       selection groups by compatibility level). In the end, though, still only one version can
 *       remain in the dep graph after the removal of unselected and unreachable modules.
 *   <li>Things get more complicated with multiple-version overrides. If module foo has a
 *       multiple-version override which allows versions [1.3, 1.5, 2.0] (using the major version as
 *       the compatibility level), then we further split the selection groups by the target allowed
 *       version (keep in mind that versions are upgraded to the nearest higher-or-equal allowed
 *       version at the same compatibility level). If, for example, some module depends on foo@1.0,
 *       then it'll depend on foo@1.3 post-selection instead (and foo@1.0 will be removed). If any
 *       of foo@1.7, foo@2.2, or foo@3.0 exist in the dependency graph before selection, they must
 *       be removed before the end of selection (by becoming unreachable, for example), otherwise
 *       it'll be an error since they're not allowed by the override (these versions are in
 *       selection groups that have no valid target allowed version).
 * </ul>
 */
final class Selection {
  private Selection() {}

  /** During selection, a version is selected for each distinct "selection group". */
  @AutoValue
  abstract static class SelectionGroup {
    static SelectionGroup create(
        String moduleName, int compatibilityLevel, Version targetAllowedVersion) {
      return new AutoValue_Selection_SelectionGroup(
          moduleName, compatibilityLevel, targetAllowedVersion);
    }

    abstract String getModuleName();

    abstract int getCompatibilityLevel();

    /** This is only used for modules with multiple-version overrides. */
    abstract Version getTargetAllowedVersion();
  }

  @AutoValue
  abstract static class ModuleNameAndCompatibilityLevel {
    static ModuleNameAndCompatibilityLevel create(String moduleName, int compatibilityLevel) {
      return new AutoValue_Selection_ModuleNameAndCompatibilityLevel(
          moduleName, compatibilityLevel);
    }

    abstract String getModuleName();

    abstract int getCompatibilityLevel();
  }

  /**
   * Computes a mapping from (moduleName, compatibilityLevel) to the set of allowed versions. This
   * is only performed for modules with multiple-version overrides.
   */
  private static ImmutableMap<ModuleNameAndCompatibilityLevel, ImmutableSortedSet<Version>>
      computeAllowedVersionSets(
          ImmutableMap<String, ModuleOverride> overrides,
          ImmutableMap<ModuleKey, UnresolvedModule> unresolvedModules)
          throws ExternalDepsException {
    Map<ModuleNameAndCompatibilityLevel, ImmutableSortedSet.Builder<Version>> allowedVersionSets =
        new HashMap<>();
    for (Map.Entry<String, ModuleOverride> overrideEntry : overrides.entrySet()) {
      String moduleName = overrideEntry.getKey();
      ModuleOverride override = overrideEntry.getValue();
      if (!(override instanceof MultipleVersionOverride)) {
        continue;
      }
      ImmutableList<Version> allowedVersions = ((MultipleVersionOverride) override).getVersions();
      for (Version allowedVersion : allowedVersions) {
        UnresolvedModule allowedVersionModule =
            unresolvedModules.get(ModuleKey.create(moduleName, allowedVersion));
        if (allowedVersionModule == null) {
          throw ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR,
              "multiple_version_override for module %s contains version %s, but it doesn't"
                  + " exist in the dependency graph",
              moduleName,
              allowedVersion);
        }
        ImmutableSortedSet.Builder<Version> allowedVersionSet =
            allowedVersionSets.computeIfAbsent(
                ModuleNameAndCompatibilityLevel.create(
                    moduleName, allowedVersionModule.getCompatibilityLevel()),
                // Remember that the empty version compares greater than any other version, so we
                // can use it as a sentinel value.
                k -> ImmutableSortedSet.<Version>naturalOrder().add(Version.EMPTY));
        allowedVersionSet.add(allowedVersion);
      }
    }
    return ImmutableMap.copyOf(
        Maps.transformValues(allowedVersionSets, ImmutableSortedSet.Builder::build));
  }

  /**
   * Computes the {@link SelectionGroup} for the given module. If the module has a multiple-version
   * override (which would be reflected in the allowedVersionSets), information in there will be
   * used to compute its targetAllowedVersion.
   */
  private static SelectionGroup computeSelectionGroup(
      UnresolvedModule module,
      ImmutableMap<ModuleNameAndCompatibilityLevel, ImmutableSortedSet<Version>>
          allowedVersionSets) {
    ImmutableSortedSet<Version> allowedVersionSet =
        allowedVersionSets.get(
            ModuleNameAndCompatibilityLevel.create(
                module.getName(), module.getCompatibilityLevel()));
    if (allowedVersionSet == null) {
      // This means that this module has no multiple-version override.
      return SelectionGroup.create(module.getName(), module.getCompatibilityLevel(), Version.EMPTY);
    }
    return SelectionGroup.create(
        module.getName(),
        module.getCompatibilityLevel(),
        // We use the `ceiling` method here to quickly locate the lowest allowed version that's
        // still no lower than this module's version.
        // If this module's version is higher than any allowed version (in which case EMPTY is
        // returned), it should result in an error. We don't immediately throw here because it might
        // still become unreferenced later.
        allowedVersionSet.ceiling(module.getVersion()));
  }

  /**
   * Computes a mapping from {@link UnresolvedModuleKey} to a list of {@link UnresolvedModule}s, for
   * any {@link UnresolvedModuleKey} that has multiple potential {@link Module}s it can resolve to.
   * {@link UnresolvedModule}s that can only resolve to a single {@link Module} will not be included
   * in the map.
   */
  private static ImmutableMap<UnresolvedModuleKey, ImmutableList<UnresolvedModule>>
      computeCompatibilityLevelRangeDeps(
          ImmutableMap<UnresolvedModuleKey, UnresolvedModule> depGraph,
          ImmutableMap<ModuleKey, UnresolvedModule> unresolvedModules,
          ImmutableMap<ModuleKey, SelectionGroup> selectionGroups,
          Map<SelectionGroup, Version> selectedVersions) {
    // Compute the minimum version available for each (module name, compatibility level). We will
    // need these to determine initial versions for higher compatibility levels. They will get
    // upgraded per `selectedVersions` next.
    Map<ModuleNameAndCompatibilityLevel, Version> minimumVersions = new HashMap<>();
    for (UnresolvedModule module : unresolvedModules.values()) {
      Version keyVersion = module.getKey().getVersion();
      if (keyVersion.isEmpty()) {
        continue;
      }
      minimumVersions.merge(
          ModuleNameAndCompatibilityLevel.create(module.getName(), module.getCompatibilityLevel()),
          keyVersion,
          Comparators::min);
    }

    // Create a mapping of module name to sorted map of compatibility level to `UnresolvedModule`.
    // These modules are upgraded to the selected version per `selectedVersions`. The sorted map
    // will allow us to select the list of modules that a given `UnresolvedModuleKey` can possibly
    // resolve to.
    Map<String, ImmutableSortedMap.Builder<Integer, UnresolvedModule>>
        unbuiltSelectedModulesPerCompatibilityLevel = new HashMap<>();
    for (Map.Entry<ModuleNameAndCompatibilityLevel, Version> entry : minimumVersions.entrySet()) {
      ModuleNameAndCompatibilityLevel moduleNameAndCompatibilityLevel = entry.getKey();
      String moduleName = moduleNameAndCompatibilityLevel.getModuleName();
      ModuleKey minLevelModuleKey = ModuleKey.create(moduleName, entry.getValue());
      Version selectedVersion = selectedVersions.get(selectionGroups.get(minLevelModuleKey));
      ImmutableSortedMap.Builder<Integer, UnresolvedModule> selectedModulePerCompatibilityLevel =
          unbuiltSelectedModulesPerCompatibilityLevel.computeIfAbsent(
              moduleName, k -> ImmutableSortedMap.<Integer, UnresolvedModule>naturalOrder());
      selectedModulePerCompatibilityLevel.put(
          moduleNameAndCompatibilityLevel.getCompatibilityLevel(),
          // Upgrade the minimum version to the selected version.
          unresolvedModules.get(ModuleKey.create(moduleName, selectedVersion)));
    }
    Map<String, ImmutableSortedMap<Integer, UnresolvedModule>>
        selectedModulesPerCompatibilityLevel =
            Maps.transformValues(
                unbuiltSelectedModulesPerCompatibilityLevel, ImmutableSortedMap.Builder::build);

    // Finally compute the mapping from `UnresolvedModuleKey` to a list of `UnresolvedModule`s, but
    // only for
    // `UnresolvedModuleKey`s that have multiple potential `UnresolvedModule`s.
    ImmutableMap.Builder<UnresolvedModuleKey, ImmutableList<UnresolvedModule>>
        compatibilityLevelRangeModules = ImmutableMap.builder();
    for (Map.Entry<UnresolvedModuleKey, UnresolvedModule> entry : depGraph.entrySet()) {
      UnresolvedModuleKey key = entry.getKey();
      UnresolvedModule module = entry.getValue();
      int minCompatibilityLevel = module.getCompatibilityLevel();
      int maxCompatibilityLevel = key.getMaxCompatibilityLevel();

      // If the maximum compatibility level is less than or equal to the minimum compatibility
      // level, this key can only resolve to the base module it refers to.
      if (maxCompatibilityLevel <= minCompatibilityLevel) {
        continue;
      }

      // Select all modules that are > minCompatibilityLevel, but <= maxCompatibilityLevel.
      String moduleName = module.getName();
      ImmutableCollection<UnresolvedModule> extraModules =
          selectedModulesPerCompatibilityLevel
              .get(moduleName)
              .subMap(
                  minCompatibilityLevel,
                  /* fromInclusive= */ false,
                  maxCompatibilityLevel,
                  /* toInclusive= */ true)
              .values();

      // This key is only a compatibility level range key if there are multiple modules resolved.
      if (extraModules.isEmpty()) {
        continue;
      }

      ImmutableList.Builder<UnresolvedModule> modules = ImmutableList.builder();
      modules.add(
          unresolvedModules.get(
              ModuleKey.create(
                  moduleName,
                  // Upgrade the base module to the selected version.
                  selectedVersions.get(selectionGroups.get(module.getKey())))));
      modules.addAll(extraModules);
      compatibilityLevelRangeModules.put(key, modules.build());
    }

    return compatibilityLevelRangeModules.buildOrThrow();
  }

  /**
   * Runs module selection (aka version resolution). Returns a {@link BazelModuleResolutionValue}.
   */
  public static BazelModuleResolutionValue run(
      ImmutableMap<UnresolvedModuleKey, UnresolvedModule> unresolvedDepGraph,
      ImmutableMap<String, ModuleOverride> overrides)
      throws ExternalDepsException {
    Map<ModuleKey, UnresolvedModule> rawUnresolvedModules = new HashMap<>();
    for (UnresolvedModule module : unresolvedDepGraph.values()) {
      rawUnresolvedModules.put(module.getKey(), module);
    }
    ImmutableMap<ModuleKey, UnresolvedModule> unresolvedModules =
        ImmutableMap.copyOf(rawUnresolvedModules);

    // For any multiple-version overrides, build a mapping from (moduleName, compatibilityLevel) to
    // the set of allowed versions.
    ImmutableMap<ModuleNameAndCompatibilityLevel, ImmutableSortedSet<Version>> allowedVersionSets =
        computeAllowedVersionSets(overrides, unresolvedModules);

    // For each module in the dep graph, pre-compute its selection group. For most modules this is
    // simply its (moduleName, compatibilityLevel) tuple; for modules with multiple-version
    // overrides, it additionally includes the targetAllowedVersion, which denotes the version to
    // "snap" to during selection.
    ImmutableMap<ModuleKey, SelectionGroup> selectionGroups =
        ImmutableMap.copyOf(
            Maps.transformValues(
                unresolvedModules, module -> computeSelectionGroup(module, allowedVersionSets)));

    // Figure out the version to select for every selection group.
    Map<SelectionGroup, Version> selectedVersions = new HashMap<>();
    for (Map.Entry<ModuleKey, SelectionGroup> entry : selectionGroups.entrySet()) {
      ModuleKey key = entry.getKey();
      SelectionGroup selectionGroup = entry.getValue();
      selectedVersions.merge(selectionGroup, key.getVersion(), Comparators::max);
    }

    ImmutableMap<UnresolvedModuleKey, ImmutableList<UnresolvedModule>> compatibilityLevelRangeDeps =
        computeCompatibilityLevelRangeDeps(
            unresolvedDepGraph, unresolvedModules, selectionGroups, selectedVersions);

    // Update deps to use selected versions. `UnresolvedModuleKey`s that haven't been resolved to a
    // specific compatibility level won't be adjusted (they are resolved a little later).
    Map<UnresolvedModuleKey, UnresolvedModule> versionsSelectedDepGraph =
        Maps.transformValues(
            unresolvedDepGraph,
            module ->
                module.withUnresolvedDepKeysTransformed(
                    depKey -> {
                      if (compatibilityLevelRangeDeps.containsKey(depKey)) {
                        // Keep original dep key for compatibility level range keys.
                        return depKey;
                      }
                      // Use selected version for non-range keys.
                      return UnresolvedModuleKey.create(
                          depKey.getName(),
                          selectedVersions.get(
                              selectionGroups.get(depKey.getMinCompatibilityModuleKey())),
                          0);
                    }));

    // Create a "walkable" dependency graph, that allows for resolution of compatibility level range
    // keys.
    ImmutableMap.Builder<UnresolvedModuleKey, ImmutableList<UnresolvedModule>> walkableDepGraph =
        new ImmutableMap.Builder<>();
    for (Map.Entry<UnresolvedModuleKey, UnresolvedModule> entry :
        versionsSelectedDepGraph.entrySet()) {
      UnresolvedModuleKey key = entry.getKey();
      UnresolvedModule module = entry.getValue();

      ImmutableList<UnresolvedModule> nonRangeModules = ImmutableList.of(module);
      ImmutableList<UnresolvedModule> modules =
          compatibilityLevelRangeDeps.getOrDefault(key, nonRangeModules);
      walkableDepGraph.put(key, modules);

      // Also include minimum compatibility level modules for compatibility level range keys, so
      // resolution for upgraded versions work.
      UnresolvedModuleKey minCompatibilityLevelKey =
          UnresolvedModuleKey.create(key.getName(), key.getVersion(), 0);
      if (!versionsSelectedDepGraph.containsKey(minCompatibilityLevelKey)) {
        walkableDepGraph.put(minCompatibilityLevelKey, nonRangeModules);
      }
    }

    // Further, removes unreferenced modules from the graph. We can find out which modules are
    // referenced by collecting deps transitively from the root.
    // We can also take this opportunity to check that none of the remaining modules conflict with
    // each other (e.g. same module name but different compatibility levels, or not satisfying
    // multiple_version_override).
    ImmutableMap<UnresolvedModuleKey, ModuleKey> resolvedModuleKeys =
        new DepGraphResolver(walkableDepGraph.buildOrThrow(), overrides, selectionGroups).walk();

    // Now that we have module keys resolved, we can generate `Module`s with resolved deps.
    HashMap<ModuleKey, Module> resolvedModules = new HashMap<>();
    for (UnresolvedModule module : versionsSelectedDepGraph.values()) {
      resolvedModules.put(
          module.getKey(),
          module.withResolvedDepKeys(
              depKey -> {
                ModuleKey resolvedModuleKey = resolvedModuleKeys.get(depKey);
                if (resolvedModuleKey != null) {
                  return resolvedModuleKey;
                } else {
                  return versionsSelectedDepGraph.get(depKey).getKey();
                }
              }));
    }

    // Build a new dep graph where deps with unselected versions are removed.
    ImmutableMap.Builder<ModuleKey, Module> prunedDepGraph = ImmutableMap.builder();

    // Also keep a version of the full dep graph with updated deps.
    ImmutableMap.Builder<ModuleKey, Module> unprunedDepGraph = ImmutableMap.builder();

    // Build up in `resolvedModuleKeys` order, to ensure breath-first ordering.
    for (ModuleKey key : resolvedModuleKeys.values()) {
      Module module = resolvedModules.remove(key);

      if (module == null) {
        // Multiple unresolved keys can resolve to the same key, so we may have already processed
        // this module.
        continue;
      }

      // Make sure that we don't have `module` depending on the same dependency version twice.
      HashMap<ModuleKey, String> depKeyToRepoName = new HashMap<>();
      for (Map.Entry<String, ModuleKey> depEntry : module.getDeps().entrySet()) {
        String repoName = depEntry.getKey();
        ModuleKey depKey = depEntry.getValue();
        String previousRepoName = depKeyToRepoName.put(depKey, repoName);
        if (previousRepoName != null) {
          throw ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR,
              "%s depends on %s at least twice (with repo names %s and %s). Consider adding a"
                  + " multiple_version_override if you want to depend on multiple versions of"
                  + " %s simultaneously",
              key,
              depKey,
              repoName,
              previousRepoName,
              key.getName());
        }
      }

      prunedDepGraph.put(key, module);
      unprunedDepGraph.put(key, module);
    }

    // Then add the rest of the modules to `unprunedDepGraph`.
    for (Module module : resolvedModules.values()) {
      unprunedDepGraph.put(module.getKey(), module);
    }

    // Return the result containing both the pruned and un-pruned dep graphs
    return BazelModuleResolutionValue.create(
        prunedDepGraph.buildOrThrow(), unprunedDepGraph.buildOrThrow());
  }

  /**
   * Walks the dependency graph from the root node, collecting any reachable nodes through deps into
   * an `UnresolvedModuleKey` -> `ModuleKey` map and checking that nothing conflicts.
   */
  static class DepGraphResolver {
    private static final Joiner JOINER = Joiner.on(", ");
    private final ImmutableMap<UnresolvedModuleKey, ImmutableList<UnresolvedModule>>
        unresolvedDepGraph;
    private final ImmutableMap<String, ModuleOverride> overrides;
    private final ImmutableMap<ModuleKey, SelectionGroup> selectionGroups;
    private final HashMap<String, ExistingModule> moduleByName;

    DepGraphResolver(
        ImmutableMap<UnresolvedModuleKey, ImmutableList<UnresolvedModule>> unresolvedDepGraph,
        ImmutableMap<String, ModuleOverride> overrides,
        ImmutableMap<ModuleKey, SelectionGroup> selectionGroups) {
      this(unresolvedDepGraph, overrides, selectionGroups, new HashMap<>());
    }

    private DepGraphResolver(
        ImmutableMap<UnresolvedModuleKey, ImmutableList<UnresolvedModule>> unresolvedDepGraph,
        ImmutableMap<String, ModuleOverride> overrides,
        ImmutableMap<ModuleKey, SelectionGroup> selectionGroups,
        HashMap<String, ExistingModule> moduleByName) {
      this.unresolvedDepGraph = unresolvedDepGraph;
      this.overrides = overrides;
      this.selectionGroups = selectionGroups;
      this.moduleByName = moduleByName;
    }

    /**
     * Walks the dependency graph and builds an `UnresolvedModuleKey` -> `ModuleKey` map containing
     * only deps reachable from the root module. The returned map has a guaranteed breadth-first
     * iteration order.
     */
    ImmutableMap<UnresolvedModuleKey, ModuleKey> walk() throws ExternalDepsException {
      ImmutableMap.Builder<UnresolvedModuleKey, ModuleKey> resolvedModuleKeys =
          ImmutableMap.builder();
      Set<UnresolvedModuleKey> known = new HashSet<>();
      ArrayDeque<UnresolvedModuleKeyAndDependent> toVisit = new ArrayDeque<>();
      toVisit.add(UnresolvedModuleKeyAndDependent.create(UnresolvedModuleKey.ROOT, null));
      known.add(UnresolvedModuleKey.ROOT);

      WalkResult result = walk(resolvedModuleKeys, known, toVisit);
      if (result.getConflictingModules() != null) {
        throwConflictingModulesException(result.getConflictingModules());
      }
      return result.getResolvedModuleKeys();
    }

    private WalkResult walk(
        ImmutableMap.Builder<UnresolvedModuleKey, ModuleKey> resolvedModuleKeys,
        Set<UnresolvedModuleKey> known,
        ArrayDeque<UnresolvedModuleKeyAndDependent> toVisit)
        throws ExternalDepsException {
      while (!toVisit.isEmpty()) {
        UnresolvedModuleKeyAndDependent moduleKeyAndDependent = toVisit.remove();
        UnresolvedModuleKey unresolvedKey = moduleKeyAndDependent.getUnresolvedModuleKey();
        ModuleKey dependent = moduleKeyAndDependent.getDependent();
        List<UnresolvedModule> modules = unresolvedDepGraph.get(unresolvedKey);

        if (modules.size() > 1) {
          // Multiple modules means we need to try each one, and use the first that doesn't have
          // compatibility level conflicts. If all have conflicts, we report up the first conflict,
          // which will be thrown in `walk()`.

          WalkResult subGraphError = null;
          for (UnresolvedModule module : modules) {
            ModuleKey subKey = module.getKey();
            // Create an `UnresolvedModuleKey` without `maxCompatibilityLevel`.
            UnresolvedModuleKey subUnresolvedKey =
                UnresolvedModuleKey.create(subKey.getName(), subKey.getVersion(), 0);

            if (known.contains(subUnresolvedKey)) {
              // We've already visited this module, so skip all the versions (below).
              resolvedModuleKeys.put(unresolvedKey, subKey);
              subGraphError = null;
              break;
            }

            // Get ready to visit this module in a sub-walk.
            Set<UnresolvedModuleKey> subKnown = new HashSet<>(known);
            subKnown.add(subUnresolvedKey);
            ArrayDeque<UnresolvedModuleKeyAndDependent> subToVisit = toVisit.clone();
            subToVisit.addFirst(
                UnresolvedModuleKeyAndDependent.create(subUnresolvedKey, dependent));

            ImmutableMap.Builder<UnresolvedModuleKey, ModuleKey> subResolvedModuleKeys =
                new ImmutableMap.Builder<>();
            subResolvedModuleKeys.putAll(resolvedModuleKeys.build());
            subResolvedModuleKeys.put(unresolvedKey, subKey);

            WalkResult subResult =
                new DepGraphResolver(
                        unresolvedDepGraph, overrides, selectionGroups, new HashMap(moduleByName))
                    .walk(subResolvedModuleKeys, subKnown, subToVisit);
            if (subResult.getResolvedModuleKeys() != null) {
              return subResult;
            }
            if (subGraphError == null) {
              subGraphError = subResult;
            }
          }

          if (subGraphError == null) {
            // One of the module options was previously known, so we can skip checking any of the
            // others.
            continue;
          }

          // If we've reached here, we have an unresolvable module conflict.
          return subGraphError;
        }

        // Non compatibility level range keys here.
        UnresolvedModule module = modules.get(0);
        ModuleKey moduleKey = module.getKey();

        Pair<ExistingModule, ExistingModule> conflictingModules =
            visit(moduleKey, module, dependent);
        if (conflictingModules != null) {
          return WalkResult.conflictingModules(conflictingModules);
        }

        for (UnresolvedModuleKey depKey : module.getUnresolvedDeps().values()) {
          if (known.add(depKey)) {
            toVisit.add(UnresolvedModuleKeyAndDependent.create(depKey, moduleKey));
          }
        }
        resolvedModuleKeys.put(unresolvedKey, moduleKey);
      }
      return WalkResult.resolvedModuleKeys(resolvedModuleKeys.buildOrThrow());
    }

    Pair<ExistingModule, ExistingModule> visit(
        ModuleKey key, UnresolvedModule module, @Nullable ModuleKey from)
        throws ExternalDepsException {
      ModuleOverride override = overrides.get(key.getName());
      if (override instanceof MultipleVersionOverride) {
        if (selectionGroups.get(key).getTargetAllowedVersion().isEmpty()) {
          // This module has no target allowed version, which means that there's no allowed version
          // higher than its version at the same compatibility level.
          Preconditions.checkState(
              from != null, "the root module cannot have a multiple version override");
          throw ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR,
              "%s depends on %s which is not allowed by the multiple_version_override on %s,"
                  + " which allows only [%s]",
              from,
              key,
              key.getName(),
              JOINER.join(((MultipleVersionOverride) override).getVersions()));
        }
      } else {
        ExistingModule existingModule =
            ExistingModule.create(key, module.getCompatibilityLevel(), from);
        ExistingModule existingModuleWithSameName =
            moduleByName.put(module.getName(), existingModule);
        if (existingModuleWithSameName != null) {
          // This has to mean that a module with the same name but a different compatibility level
          // was also selected.
          Preconditions.checkState(
              from != null && existingModuleWithSameName.getDependent() != null,
              "the root module cannot possibly exist more than once in the dep graph");
          return Pair.of(existingModule, existingModuleWithSameName);
        }
      }

      return null;
    }

    void throwConflictingModulesException(Pair<ExistingModule, ExistingModule> conflictingModules)
        throws ExternalDepsException {
      ExistingModule firstExistingModule = conflictingModules.first;
      ExistingModule secondExistingModule = conflictingModules.second;
      throw ExternalDepsException.withMessage(
          Code.VERSION_RESOLUTION_ERROR,
          "%s depends on %s with compatibility level %d, but %s depends on %s with compatibility"
              + " level %d which is different",
          firstExistingModule.getDependent(),
          firstExistingModule.getModuleKey(),
          firstExistingModule.getCompatibilityLevel(),
          secondExistingModule.getDependent(),
          secondExistingModule.getModuleKey(),
          secondExistingModule.getCompatibilityLevel());
    }

    @AutoValue
    abstract static class WalkResult {
      @Nullable
      abstract ImmutableMap<UnresolvedModuleKey, ModuleKey> getResolvedModuleKeys();

      @Nullable
      abstract Pair<ExistingModule, ExistingModule> getConflictingModules();

      static WalkResult resolvedModuleKeys(
          ImmutableMap<UnresolvedModuleKey, ModuleKey> resolvedModuleKeys) {
        return new AutoValue_Selection_DepGraphResolver_WalkResult(resolvedModuleKeys, null);
      }

      static WalkResult conflictingModules(
          Pair<ExistingModule, ExistingModule> conflictingModules) {
        return new AutoValue_Selection_DepGraphResolver_WalkResult(null, conflictingModules);
      }
    }

    @AutoValue
    abstract static class UnresolvedModuleKeyAndDependent {
      abstract UnresolvedModuleKey getUnresolvedModuleKey();

      @Nullable
      abstract ModuleKey getDependent();

      static UnresolvedModuleKeyAndDependent create(
          UnresolvedModuleKey depKey, @Nullable ModuleKey dependent) {
        return new AutoValue_Selection_DepGraphResolver_UnresolvedModuleKeyAndDependent(
            depKey, dependent);
      }
    }

    @AutoValue
    abstract static class ExistingModule {
      abstract ModuleKey getModuleKey();

      abstract int getCompatibilityLevel();

      @Nullable
      abstract ModuleKey getDependent();

      static ExistingModule create(ModuleKey depKey, int compatibilityLevel, ModuleKey dependent) {
        return new AutoValue_Selection_DepGraphResolver_ExistingModule(
            depKey, compatibilityLevel, dependent);
      }
    }
  }
}
