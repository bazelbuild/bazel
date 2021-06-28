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
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import javax.annotation.Nullable;

/**
 * Runs module discovery. This step of module resolution reads the module file of the root module
 * (i.e. the current workspace), adds its direct {@code bazel_dep}s to the dependency graph, and
 * repeats the step for any added dependencies until the entire graph is discovered.
 */
public class DiscoveryFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    ModuleFileValue root = (ModuleFileValue) env.getValue(ModuleFileValue.keyForRootModule());
    if (root == null) {
      return null;
    }
    ModuleKey rootModuleKey = ModuleKey.create(root.getModule().getName(), "");
    ImmutableMap<String, ModuleOverride> overrides = root.getOverrides();
    Map<ModuleKey, Module> depGraph = new HashMap<>();
    depGraph.put(
        rootModuleKey, rewriteDepKeys(root.getModule(), overrides, rootModuleKey.getName()));
    Queue<ModuleKey> unexpanded = new ArrayDeque<>();
    unexpanded.add(rootModuleKey);
    // TODO(wyv): currently we expand the "unexpanded" keys one by one. We should try to expand them
    //   all at once, using `env.getValues`.
    while (!unexpanded.isEmpty()) {
      Module module = depGraph.get(unexpanded.remove());
      for (ModuleKey depKey : module.getDeps().values()) {
        if (depGraph.containsKey(depKey)) {
          continue;
        }
        ModuleFileValue dep =
            (ModuleFileValue)
                env.getValue(ModuleFileValue.key(depKey, overrides.get(depKey.getName())));
        if (dep == null) {
          // Don't return yet. Try to expand any other unexpanded nodes before returning.
          depGraph.put(depKey, null);
        } else {
          depGraph.put(depKey, rewriteDepKeys(dep.getModule(), overrides, rootModuleKey.getName()));
          unexpanded.add(depKey);
        }
      }
    }
    if (env.valuesMissing()) {
      return null;
    }
    return DiscoveryValue.create(
        root.getModule().getName(), ImmutableMap.copyOf(depGraph), overrides);
  }

  private static Module rewriteDepKeys(
      Module module, ImmutableMap<String, ModuleOverride> overrides, String rootModuleName) {
    return module.withDepKeysTransformed(
        depKey -> {
          String newVersion = depKey.getVersion();

          @Nullable ModuleOverride override = overrides.get(depKey.getName());
          if (override instanceof NonRegistryOverride || rootModuleName.equals(depKey.getName())) {
            newVersion = "";
          } else if (override instanceof SingleVersionOverride) {
            String overrideVersion = ((SingleVersionOverride) override).getVersion();
            if (!overrideVersion.isEmpty()) {
              newVersion = overrideVersion;
            }
          }

          return ModuleKey.create(depKey.getName(), newVersion);
        });
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
