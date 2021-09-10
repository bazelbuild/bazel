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
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.BuildType.LabelConversionContext;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;

/**
 * Runs Bazel module resolution. This function produces the dependency graph containing all Bazel
 * modules, along with a few lookup maps that help with further usage. By this stage, module
 * extensions are not evaluated yet.
 */
public class BazelModuleResolutionFunction implements SkyFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RootModuleFileValue root =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (root == null) {
      return null;
    }
    ImmutableMap<ModuleKey, Module> initialDepGraph = Discovery.run(env, root);
    if (initialDepGraph == null) {
      return null;
    }
    ImmutableMap<String, ModuleOverride> overrides = root.getOverrides();
    ImmutableMap<ModuleKey, Module> resolvedDepGraph;
    try {
      resolvedDepGraph = Selection.run(initialDepGraph, overrides);
    } catch (ExternalDepsException e) {
      throw new BazelModuleResolutionFunctionException(e, Transience.PERSISTENT);
    }
    return createValue(resolvedDepGraph, overrides);
  }

  @VisibleForTesting
  static BazelModuleResolutionValue createValue(
      ImmutableMap<ModuleKey, Module> depGraph, ImmutableMap<String, ModuleOverride> overrides)
      throws BazelModuleResolutionFunctionException {
    // Build some reverse lookups for later use.
    ImmutableMap<String, ModuleKey> canonicalRepoNameLookup =
        depGraph.keySet().stream()
            .collect(toImmutableMap(ModuleKey::getCanonicalRepoName, key -> key));
    ImmutableMap<String, ModuleKey> moduleNameLookup =
        depGraph.keySet().stream()
            // The root module is not meaningfully used by this lookup so we skip it (it's
            // guaranteed to be the first in iteration order).
            .skip(1)
            .filter(key -> !(overrides.get(key.getName()) instanceof MultipleVersionOverride))
            .collect(toImmutableMap(ModuleKey::getName, key -> key));

    // For each extension usage, we resolve (i.e. canonicalize) its bzl file label. Then we can
    // group all usages by the label + name (the ModuleExtensionId).
    ImmutableTable.Builder<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
        extensionUsagesTableBuilder = ImmutableTable.builder();
    for (Module module : depGraph.values()) {
      LabelConversionContext labelConversionContext =
          new LabelConversionContext(
              StarlarkBazelModule.createModuleRootLabel(module.getCanonicalRepoName()),
              module.getRepoMappingWithBazelDepsOnly(),
              new HashMap<>());
      for (ModuleExtensionUsage usage : module.getExtensionUsages()) {
        try {
          ModuleExtensionId moduleExtensionId =
              ModuleExtensionId.create(
                  labelConversionContext.convert(usage.getExtensionBzlFile()),
                  usage.getExtensionName());
          extensionUsagesTableBuilder.put(moduleExtensionId, module.getKey(), usage);
        } catch (LabelSyntaxException e) {
          throw new BazelModuleResolutionFunctionException(
              ExternalDepsException.withCauseAndMessage(
                  Code.BAD_MODULE,
                  e,
                  "invalid label for module extension found at %s",
                  usage.getLocation()),
              Transience.PERSISTENT);
        }
      }
    }
    ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById =
        extensionUsagesTableBuilder.build();

    // Calculate a unique name for each used extension id.
    BiMap<String, ModuleExtensionId> extensionUniqueNames = HashBiMap.create();
    for (ModuleExtensionId id : extensionUsagesById.rowKeySet()) {
      String bestName =
          id.getBzlFileLabel().getRepository().strippedName() + "." + id.getExtensionName();
      if (extensionUniqueNames.putIfAbsent(bestName, id) == null) {
        continue;
      }
      int suffix = 2;
      while (extensionUniqueNames.putIfAbsent(bestName + suffix, id) != null) {
        suffix++;
      }
    }

    return BazelModuleResolutionValue.create(
        depGraph,
        canonicalRepoNameLookup,
        moduleNameLookup,
        depGraph.values().stream().map(AbridgedModule::from).collect(toImmutableList()),
        extensionUsagesById,
        ImmutableMap.copyOf(extensionUniqueNames.inverse()));
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  static class BazelModuleResolutionFunctionException extends SkyFunctionException {
    BazelModuleResolutionFunctionException(ExternalDepsException e, Transience transience) {
      super(e, transience);
    }
  }
}
