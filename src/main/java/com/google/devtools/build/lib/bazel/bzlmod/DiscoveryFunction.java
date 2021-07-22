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
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
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
    RootModuleFileValue root =
        (RootModuleFileValue) env.getValue(ModuleFileValue.keyForRootModule());
    if (root == null) {
      return null;
    }
    ModuleKey rootModuleKey = ModuleKey.create(root.getModule().getName(), Version.EMPTY);
    ImmutableMap<String, ModuleOverride> overrides = root.getOverrides();
    Map<ModuleKey, Module> depGraph = new HashMap<>();
    depGraph.put(
        rootModuleKey, rewriteDepKeys(root.getModule(), overrides, rootModuleKey.getName()));
    Queue<ModuleKey> unexpanded = new ArrayDeque<>();
    unexpanded.add(rootModuleKey);
    while (!unexpanded.isEmpty()) {
      Set<SkyKey> unexpandedSkyKeys = new HashSet<>();
      while (!unexpanded.isEmpty()) {
        Module module = depGraph.get(unexpanded.remove());
        for (ModuleKey depKey : module.getDeps().values()) {
          if (depGraph.containsKey(depKey)) {
            continue;
          }
          unexpandedSkyKeys.add(ModuleFileValue.key(depKey, overrides.get(depKey.getName())));
        }
      }
      Map<SkyKey, SkyValue> result = env.getValues(unexpandedSkyKeys);
      for (Map.Entry<SkyKey, SkyValue> entry : result.entrySet()) {
        ModuleKey depKey = ((ModuleFileValue.Key) entry.getKey()).getModuleKey();
        ModuleFileValue moduleFileValue = (ModuleFileValue) entry.getValue();
        if (moduleFileValue == null) {
          // Don't return yet. Try to expand any other unexpanded nodes before returning.
          depGraph.put(depKey, null);
        } else {
          depGraph.put(
              depKey,
              rewriteDepKeys(moduleFileValue.getModule(), overrides, rootModuleKey.getName()));
          unexpanded.add(depKey);
        }
      }
    }
    if (env.valuesMissing()) {
      return null;
    }
    return DiscoveryValue.create(root.getModule().getName(), ImmutableMap.copyOf(depGraph));
  }

  private static Module rewriteDepKeys(
      Module module, ImmutableMap<String, ModuleOverride> overrides, String rootModuleName) {
    return module.withDepKeysTransformed(
        depKey -> {
          Version newVersion = depKey.getVersion();

          @Nullable ModuleOverride override = overrides.get(depKey.getName());
          if (override instanceof NonRegistryOverride || rootModuleName.equals(depKey.getName())) {
            newVersion = Version.EMPTY;
          } else if (override instanceof SingleVersionOverride) {
            Version overrideVersion = ((SingleVersionOverride) override).getVersion();
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
