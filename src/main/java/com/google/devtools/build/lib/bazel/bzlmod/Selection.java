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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
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
   * Runs module selection (aka version resolution). Returns a {@link BazelModuleResolutionValue}.
   */
  public static BazelModuleResolutionValue run(
      ImmutableMap<UnresolvedModuleKey, UnresolvedModule> unresolvedDepGraph,
      ImmutableMap<String, ModuleOverride> overrides)
      throws ExternalDepsException {
    ImmutableMap.Builder<ModuleKey, UnresolvedModule> unresolvedModulesBuilder =
        ImmutableMap.builder();
    for (UnresolvedModule module : unresolvedDepGraph.values()) {
      unresolvedModulesBuilder.put(module.getKey(), module);
    }
    ImmutableMap<ModuleKey, UnresolvedModule> unresolvedModules = unresolvedModulesBuilder.build();

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

    // Update deps to use selected versions.
    ImmutableMap<UnresolvedModuleKey, UnresolvedModule> versionsSelectedDepGraph =
        ImmutableMap.copyOf(
            Maps.transformValues(
                unresolvedDepGraph,
                module ->
                    module.withUnresolvedDepKeysTransformed(
                        depKey ->
                            UnresolvedModuleKey.create(
                                depKey.getName(),
                                selectedVersions.get(
                                    selectionGroups.get(depKey.getModuleKey()))))));

    // Further, removes unreferenced modules from the graph. We can find out which modules are
    // referenced by collecting deps transitively from the root.
    // We can also take this opportunity to check that none of the remaining modules conflict with
    // each other (e.g. same module name but different compatibility levels, or not satisfying
    // multiple_version_override).
    ImmutableMap<UnresolvedModuleKey, ModuleKey> resolvedModuleKeys =
        new DepGraphResolver(versionsSelectedDepGraph, overrides, selectionGroups).walk();

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
    private final ImmutableMap<UnresolvedModuleKey, UnresolvedModule> unresolvedDepGraph;
    private final ImmutableMap<String, ModuleOverride> overrides;
    private final ImmutableMap<ModuleKey, SelectionGroup> selectionGroups;
    private final HashMap<String, ExistingModule> moduleByName;

    DepGraphResolver(
        ImmutableMap<UnresolvedModuleKey, UnresolvedModule> unresolvedDepGraph,
        ImmutableMap<String, ModuleOverride> overrides,
        ImmutableMap<ModuleKey, SelectionGroup> selectionGroups) {
      this.unresolvedDepGraph = unresolvedDepGraph;
      this.overrides = overrides;
      this.selectionGroups = selectionGroups;
      this.moduleByName = new HashMap<>();
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
      Queue<UnresolvedModuleKeyAndDependent> toVisit = new ArrayDeque<>();
      toVisit.add(UnresolvedModuleKeyAndDependent.create(UnresolvedModuleKey.ROOT, null));
      known.add(UnresolvedModuleKey.ROOT);
      while (!toVisit.isEmpty()) {
        UnresolvedModuleKeyAndDependent moduleKeyAndDependent = toVisit.remove();
        UnresolvedModuleKey unresolvedKey = moduleKeyAndDependent.getUnresolvedModuleKey();
        ModuleKey dependent = moduleKeyAndDependent.getDependent();
        UnresolvedModule module = unresolvedDepGraph.get(unresolvedKey);
        ModuleKey moduleKey = unresolvedModule.getKey();
        visit(moduleKey, unresolvedModule, dependent);

        for (UnresolvedModuleKey depKey : module.getUnresolvedDeps().values()) {
          if (known.add(depKey)) {
            toVisit.add(UnresolvedModuleKeyAndDependent.create(depKey, moduleKey));
          }
        }
        resolvedModuleKeys.put(unresolvedKey, moduleKey);
      }
      return resolvedModuleKeys.buildOrThrow();
    }

    void visit(ModuleKey key, UnresolvedModule module, @Nullable ModuleKey from)
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
          throw ExternalDepsException.withMessage(
              Code.VERSION_RESOLUTION_ERROR,
              "%s depends on %s with compatibility level %d, but %s depends on %s with"
                  + " compatibility level %d which is different",
              existingModule.getDependent(),
              existingModule.getModuleKey(),
              existingModule.getCompatibilityLevel(),
              existingModuleWithSameName.getDependent(),
              existingModuleWithSameName.getModuleKey(),
              existingModuleWithSameName.getCompatibilityLevel());
        }
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
