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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;
import java.util.Map;

/**
 * Runs module selection. This step of module resolution reads the output of {@link
 * DiscoveryFunction} and applies the Minimal Version Selection algorithm to it, removing unselected
 * modules from the dependency graph and rewriting dependencies to point to the selected versions.
 */
public class SelectionFunction implements SkyFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    DiscoveryValue discovery = (DiscoveryValue) env.getValue(DiscoveryValue.KEY);
    if (discovery == null) {
      return null;
    }

    // TODO(wyv): compatibility_level, multiple_version_override

    // First figure out the version to select for every module.
    ImmutableMap<ModuleKey, Module> depGraph = discovery.getDepGraph();
    Map<String, ParsedVersion> selectedVersionForEachModule = new HashMap<>();
    for (ModuleKey key : depGraph.keySet()) {
      try {
        ParsedVersion parsedVersion = ParsedVersion.parse(key.getVersion());
        selectedVersionForEachModule.merge(key.getName(), parsedVersion, ParsedVersion::max);
      } catch (ParsedVersion.ParseException e) {
        throw new SelectionFunctionException(e);
      }
    }

    // Now build a new dep graph where deps with unselected versions are removed.
    ImmutableMap.Builder<ModuleKey, Module> newDepGraphBuilder = new ImmutableMap.Builder<>();
    for (Map.Entry<ModuleKey, Module> entry : depGraph.entrySet()) {
      ModuleKey moduleKey = entry.getKey();
      Module module = entry.getValue();
      // Remove any dep whose version isn't selected.
      String selectedVersion = selectedVersionForEachModule.get(moduleKey.getName()).getOriginal();
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
                      selectedVersionForEachModule.get(depKey.getName()).getOriginal())));
    }
    ImmutableMap<ModuleKey, Module> newDepGraph = newDepGraphBuilder.build();

    // Further remove unreferenced modules from the graph. We can find out which modules are
    // referenced by collecting deps transitively from the root.
    HashMap<ModuleKey, Module> finalDepGraph = new HashMap<>();
    collectDeps(ModuleKey.create(discovery.getRootModuleName(), ""), newDepGraph, finalDepGraph);
    return SelectionValue.create(
        discovery.getRootModuleName(),
        ImmutableMap.copyOf(finalDepGraph),
        discovery.getOverrides());
  }

  private void collectDeps(
      ModuleKey key,
      ImmutableMap<ModuleKey, Module> oldDepGraph,
      HashMap<ModuleKey, Module> newDepGraph) {
    if (newDepGraph.containsKey(key)) {
      return;
    }
    Module module = oldDepGraph.get(key);
    newDepGraph.put(key, module);
    for (ModuleKey depKey : module.getDeps().values()) {
      collectDeps(depKey, oldDepGraph, newDepGraph);
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
}
