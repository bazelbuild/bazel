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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
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
 */
public class SelectionFunction implements SkyFunction {

  /** During selection, a version is selected for each distinct "selection group". */
  @AutoValue
  abstract static class SelectionGroup {
    static SelectionGroup of(Module module) {
      return new AutoValue_SelectionFunction_SelectionGroup(
          module.getName(), module.getCompatibilityLevel());
    }

    abstract String getModuleName();

    abstract int getCompatibilityLevel();
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    DiscoveryValue discovery = (DiscoveryValue) env.getValue(DiscoveryValue.KEY);
    if (discovery == null) {
      return null;
    }

    // TODO(wyv): multiple_version_override

    // First figure out the version to select for every selection group.
    ImmutableMap<ModuleKey, Module> depGraph = discovery.getDepGraph();
    Map<SelectionGroup, ParsedVersion> selectedVersions = new HashMap<>();
    for (Map.Entry<ModuleKey, Module> entry : depGraph.entrySet()) {
      ModuleKey key = entry.getKey();
      Module module = entry.getValue();

      ParsedVersion parsedVersion;
      try {
        parsedVersion = ParsedVersion.parse(key.getVersion());
      } catch (ParsedVersion.ParseException e) {
        throw new SelectionFunctionException(e);
      }
      selectedVersions.merge(SelectionGroup.of(module), parsedVersion, ParsedVersion::max);
    }

    // Now build a new dep graph where deps with unselected versions are removed.
    ImmutableMap.Builder<ModuleKey, Module> newDepGraphBuilder = new ImmutableMap.Builder<>();
    for (Map.Entry<ModuleKey, Module> entry : depGraph.entrySet()) {
      ModuleKey moduleKey = entry.getKey();
      Module module = entry.getValue();

      // Remove any dep whose version isn't selected.
      String selectedVersion = selectedVersions.get(SelectionGroup.of(module)).getOriginal();
      if (!moduleKey.getVersion().equals(selectedVersion)) {
        continue;
      }

      // Rewrite deps to point to the selected version.
      newDepGraphBuilder.put(
          moduleKey,
          module.withDepKeysTransformed(
              depKey ->
                  ModuleKey.create(
                      depKey.getName(),
                      selectedVersions
                          .get(SelectionGroup.of(depGraph.get(depKey)))
                          .getOriginal())));
    }
    ImmutableMap<ModuleKey, Module> newDepGraph = newDepGraphBuilder.build();

    // Further remove unreferenced modules from the graph. We can find out which modules are
    // referenced by collecting deps transitively from the root.
    // We can also take this opportunity to check that none of the remaining modules conflict with
    // each other (e.g. same module name but different compatibility levels, or not satisfying
    // multiple_version_override).
    DepGraphWalker walker = new DepGraphWalker(newDepGraph);
    try {
      walker.walk(ModuleKey.create(discovery.getRootModuleName(), ""), null);
    } catch (SelectionException e) {
      throw new SelectionFunctionException(e);
    }

    return SelectionValue.create(
        discovery.getRootModuleName(), walker.getNewDepGraph(), discovery.getOverrides());
  }

  /**
   * Walks the dependency graph from the root node, collecting any reachable nodes through deps into
   * a new dep graph and checking that nothing conflicts.
   */
  static class DepGraphWalker {
    private final ImmutableMap<ModuleKey, Module> oldDepGraph;
    private final HashMap<ModuleKey, Module> newDepGraph;
    private final HashMap<String, ExistingModule> moduleByName;

    DepGraphWalker(ImmutableMap<ModuleKey, Module> oldDepGraph) {
      this.oldDepGraph = oldDepGraph;
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

      ExistingModule existingModuleWithSameName =
          moduleByName.put(
              module.getName(), ExistingModule.create(key, module.getCompatibilityLevel(), from));
      if (existingModuleWithSameName != null) {
        // This has to mean that a module with the same name but a different compatibility level was
        // also selected.
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
