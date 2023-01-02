// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction.BazelModuleResolutionFunctionException;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * This function runs Bazel module resolution, extracts the dependency graph from it and creates a
 * value containing all Bazel modules, along with a few lookup maps that help with further usage. By
 * this stage, module extensions are not evaluated yet.
 */
public class BazelDepGraphFunction implements SkyFunction {

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    BazelModuleResolutionValue selectionResult =
        (BazelModuleResolutionValue) env.getValue(BazelModuleResolutionValue.KEY);
    if (env.valuesMissing()) {
      return null;
    }

    ImmutableMap<ModuleKey, Module> depGraph = selectionResult.getResolvedDepGraph();
    ImmutableMap<RepositoryName, ModuleKey> canonicalRepoNameLookup =
        depGraph.keySet().stream()
            .collect(toImmutableMap(ModuleKey::getCanonicalRepoName, key -> key));

    // For each extension usage, we resolve (i.e. canonicalize) its bzl file label. Then we can
    // group all usages by the label + name (the ModuleExtensionId).
    ImmutableTable.Builder<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
        extensionUsagesTableBuilder = ImmutableTable.builder();
    for (Module module : depGraph.values()) {
      LabelConverter labelConverter =
          new LabelConverter(
              PackageIdentifier.create(module.getCanonicalRepoName(), PathFragment.EMPTY_FRAGMENT),
              module.getRepoMappingWithBazelDepsOnly());
      for (ModuleExtensionUsage usage : module.getExtensionUsages()) {
        try {
          ModuleExtensionId moduleExtensionId =
              ModuleExtensionId.create(
                  labelConverter.convert(usage.getExtensionBzlFile()), usage.getExtensionName());
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
        extensionUsagesTableBuilder.buildOrThrow();

    // Calculate a unique name for each used extension id.
    BiMap<String, ModuleExtensionId> extensionUniqueNames = HashBiMap.create();
    for (ModuleExtensionId id : extensionUsagesById.rowKeySet()) {
      // Ensure that the resulting extension name (and thus the repository names derived from it) do
      // not start with a tilde.
      RepositoryName repository = id.getBzlFileLabel().getRepository();
      String nonEmptyRepoPart;
      if (repository.isMain()) {
        nonEmptyRepoPart = "_main";
      } else {
        nonEmptyRepoPart = repository.getName();
      }
      String bestName = nonEmptyRepoPart + "~" + id.getExtensionName();
      if (extensionUniqueNames.putIfAbsent(bestName, id) == null) {
        continue;
      }
      int suffix = 2;
      while (extensionUniqueNames.putIfAbsent(bestName + suffix, id) != null) {
        suffix++;
      }
    }

    return BazelDepGraphValue.create(
        depGraph,
        canonicalRepoNameLookup,
        depGraph.values().stream().map(AbridgedModule::from).collect(toImmutableList()),
        extensionUsagesById,
        ImmutableMap.copyOf(extensionUniqueNames.inverse()));
  }
}
