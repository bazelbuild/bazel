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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Comparators;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Runs module selection. This step of module resolution reads the output of {@link
 * DiscoveryFunction} and applies the Minimal Version Selection algorithm to it, removing unselected
 * modules from the dependency graph and rewriting dependencies to point to the selected versions.
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
 *       multiple-version override which allows version [1.3, 1.5, 2.0] (using the major version as
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
public class SelectionFunction implements SkyFunction {

  /** During selection, a version is selected for each distinct "selection group". */
  @AutoValue
  abstract static class SelectionGroup {
    static SelectionGroup create(
        String moduleName, int compatibilityLevel, Version targetAllowedVersion) {
      return new AutoValue_SelectionFunction_SelectionGroup(
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
      return new AutoValue_SelectionFunction_ModuleNameAndCompatibilityLevel(
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
          ImmutableMap<String, ModuleOverride> overrides, ImmutableMap<ModuleKey, Module> depGraph)
          throws SelectionException {
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
        Module allowedVersionModule = depGraph.get(ModuleKey.create(moduleName, allowedVersion));
        if (allowedVersionModule == null) {
          throw new SelectionException(
              String.format(
                  "multiple_version_override for module %s contains version %s, but it doesn't"
                      + " exist in the dependency graph",
                  moduleName, allowedVersion));
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
      Module module,
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

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    DiscoveryValue discovery = (DiscoveryValue) env.getValue(DiscoveryValue.KEY);
    if (discovery == null) {
      return null;
    }
    ImmutableMap<ModuleKey, Module> depGraph = discovery.getDepGraph();
    RootModuleFileValue rootModule =
        (RootModuleFileValue) env.getValue(ModuleFileValue.keyForRootModule());
    if (rootModule == null) {
      return null;
    }
    ImmutableMap<String, ModuleOverride> overrides = rootModule.getOverrides();

    // For any multiple-version overrides, build a mapping from (moduleName, compatibilityLevel) to
    // the set of allowed versions.
    ImmutableMap<ModuleNameAndCompatibilityLevel, ImmutableSortedSet<Version>> allowedVersionSets;
    try {
      allowedVersionSets = computeAllowedVersionSets(overrides, depGraph);
    } catch (SelectionException e) {
      throw new SelectionFunctionException(e);
    }

    // For each module in the dep graph, pre-compute its selection group. For most modules this is
    // simply its (moduleName, compatibilityLevel) tuple; for modules with multiple-version
    // overrides, it additionally includes the targetAllowedVersion, which denotes the version to
    // "snap" to during selection.
    ImmutableMap<ModuleKey, SelectionGroup> selectionGroups =
        ImmutableMap.copyOf(
            Maps.transformValues(
                depGraph, module -> computeSelectionGroup(module, allowedVersionSets)));

    // Figure out the version to select for every selection group.
    Map<SelectionGroup, Version> selectedVersions = new HashMap<>();
    for (Map.Entry<ModuleKey, SelectionGroup> entry : selectionGroups.entrySet()) {
      ModuleKey key = entry.getKey();
      SelectionGroup selectionGroup = entry.getValue();
      selectedVersions.merge(selectionGroup, key.getVersion(), Comparators::max);
    }

    // Build a new dep graph where deps with unselected versions are removed.
    ImmutableMap.Builder<ModuleKey, Module> newDepGraphBuilder = new ImmutableMap.Builder<>();
    for (Map.Entry<ModuleKey, Module> entry : depGraph.entrySet()) {
      ModuleKey moduleKey = entry.getKey();
      Module module = entry.getValue();

      // Remove any dep whose version isn't selected.
      Version selectedVersion = selectedVersions.get(selectionGroups.get(moduleKey));
      if (!moduleKey.getVersion().equals(selectedVersion)) {
        continue;
      }

      // Rewrite deps to point to the selected version.
      newDepGraphBuilder.put(
          moduleKey,
          module.withDepKeysTransformed(
              depKey ->
                  ModuleKey.create(
                      depKey.getName(), selectedVersions.get(selectionGroups.get(depKey)))));
    }
    ImmutableMap<ModuleKey, Module> newDepGraph = newDepGraphBuilder.build();

    // Further remove unreferenced modules from the graph. We can find out which modules are
    // referenced by collecting deps transitively from the root.
    // We can also take this opportunity to check that none of the remaining modules conflict with
    // each other (e.g. same module name but different compatibility levels, or not satisfying
    // multiple_version_override).
    DepGraphWalker walker = new DepGraphWalker(newDepGraph, overrides, selectionGroups);
    try {
      walker.walk(ModuleKey.create(discovery.getRootModuleName(), Version.EMPTY), null);
    } catch (SelectionException e) {
      throw new SelectionFunctionException(e);
    }

    newDepGraph = walker.getNewDepGraph();
    ImmutableMap<String, ModuleKey> canonicalRepoNameLookup =
        depGraph.keySet().stream()
            .collect(toImmutableMap(ModuleKey::getCanonicalRepoName, key -> key));
    ImmutableMap<String, ModuleKey> moduleNameLookup =
        Maps.filterKeys(
                newDepGraph,
                key -> !(overrides.get(key.getName()) instanceof MultipleVersionOverride))
            .keySet()
            .stream()
            .collect(toImmutableMap(ModuleKey::getName, key -> key));
    return SelectionValue.create(
        discovery.getRootModuleName(),
        walker.getNewDepGraph(),
        canonicalRepoNameLookup,
        moduleNameLookup);
  }

  /**
   * Walks the dependency graph from the root node, collecting any reachable nodes through deps into
   * a new dep graph and checking that nothing conflicts.
   */
  static class DepGraphWalker {
    private static final Joiner JOINER = Joiner.on(", ");
    private final ImmutableMap<ModuleKey, Module> oldDepGraph;
    private final ImmutableMap<String, ModuleOverride> overrides;
    private final ImmutableMap<ModuleKey, SelectionGroup> selectionGroups;
    private final HashMap<ModuleKey, Module> newDepGraph;
    private final HashMap<String, ExistingModule> moduleByName;

    DepGraphWalker(
        ImmutableMap<ModuleKey, Module> oldDepGraph,
        ImmutableMap<String, ModuleOverride> overrides,
        ImmutableMap<ModuleKey, SelectionGroup> selectionGroups) {
      this.oldDepGraph = oldDepGraph;
      this.overrides = overrides;
      this.selectionGroups = selectionGroups;
      this.newDepGraph = new HashMap<>();
      this.moduleByName = new HashMap<>();
    }

    ImmutableMap<ModuleKey, Module> getNewDepGraph() {
      return ImmutableMap.copyOf(newDepGraph);
    }

    void walk(ModuleKey key, @Nullable ModuleKey from) throws SelectionException {
      if (newDepGraph.containsKey(key)) {
        return;
      }
      Module module = oldDepGraph.get(key);
      newDepGraph.put(key, module);

      ModuleOverride override = overrides.get(key.getName());
      if (override instanceof MultipleVersionOverride) {
        if (selectionGroups.get(key).getTargetAllowedVersion().isEmpty()) {
          // This module has no target allowed version, which means that there's no allowed version
          // higher than its version at the same compatibility level.
          Preconditions.checkState(
              from != null, "the root module cannot have a multiple version override");
          throw new SelectionException(
              String.format(
                  "%s depends on %s which is not allowed by the multiple_version_override on %s,"
                      + " which allows only [%s]",
                  from,
                  key,
                  key.getName(),
                  JOINER.join(((MultipleVersionOverride) override).getVersions())));
        }
      } else {
        ExistingModule existingModuleWithSameName =
            moduleByName.put(
                module.getName(), ExistingModule.create(key, module.getCompatibilityLevel(), from));
        if (existingModuleWithSameName != null) {
          // This has to mean that a module with the same name but a different compatibility level
          // was also selected.
          Preconditions.checkState(
              from != null && existingModuleWithSameName.getDependent() != null,
              "the root module cannot possibly exist more than once in the dep graph");
          throw new SelectionException(
              String.format(
                  "%s depends on %s with compatibility level %d, but %s depends on %s with"
                      + " compatibility level %d which is different",
                  from,
                  key,
                  module.getCompatibilityLevel(),
                  existingModuleWithSameName.getDependent(),
                  existingModuleWithSameName.getModuleKey(),
                  existingModuleWithSameName.getCompatibilityLevel()));
        }
      }

      // Make sure that we don't have `module` depending on the same dependency version twice.
      HashMap<ModuleKey, String> depKeyToRepoName = new HashMap<>();
      for (Map.Entry<String, ModuleKey> depEntry : module.getDeps().entrySet()) {
        String repoName = depEntry.getKey();
        ModuleKey depKey = depEntry.getValue();
        String previousRepoName = depKeyToRepoName.put(depKey, repoName);
        if (previousRepoName != null) {
          throw new SelectionException(
              String.format(
                  "%s depends on %s at least twice (with repo names %s and %s). Consider adding a"
                      + " multiple_version_override if you want to depend on multiple versions of"
                      + " %s simultaneously",
                  key, depKey, repoName, previousRepoName, key.getName()));
        }
      }

      // Now visit our dependencies.
      for (ModuleKey depKey : module.getDeps().values()) {
        walk(depKey, key);
      }
    }

    @AutoValue
    abstract static class ExistingModule {
      abstract ModuleKey getModuleKey();

      abstract int getCompatibilityLevel();

      @Nullable
      abstract ModuleKey getDependent();

      static ExistingModule create(
          ModuleKey moduleKey, int compatibilityLevel, ModuleKey dependent) {
        return new AutoValue_SelectionFunction_DepGraphWalker_ExistingModule(
            moduleKey, compatibilityLevel, dependent);
      }
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  static final class SelectionFunctionException extends SkyFunctionException {
    SelectionFunctionException(Exception cause) {
      super(cause, Transience.PERSISTENT);
    }
  }

  // TODO(wyv): Replace this with a DetailedException (possibly named ModuleException or
  //   ExternalDepsException) and use it consistently across the space.
  static final class SelectionException extends Exception {
    SelectionException(String message) {
      super(message);
    }
  }
}
