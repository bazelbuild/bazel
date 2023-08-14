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
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Precomputes an augmented version of the un-pruned dep graph that is used for dep graph
 * inspection. By this stage, the Bazel module resolution should have been completed.
 */
public class BazelModuleInspectorFunction implements SkyFunction {

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    RootModuleFileValue root =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (root == null) {
      return null;
    }
    BazelDepGraphValue depGraphValue = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (depGraphValue == null) {
      return null;
    }
    BazelModuleResolutionValue resolutionValue =
        (BazelModuleResolutionValue) env.getValue(BazelModuleResolutionValue.KEY);
    if (resolutionValue == null) {
      return null;
    }
    ImmutableMap<String, ModuleOverride> overrides = root.getOverrides();
    ImmutableMap<ModuleKey, InterimModule> unprunedDepGraph = resolutionValue.getUnprunedDepGraph();
    ImmutableMap<ModuleKey, Module> resolvedDepGraph = resolutionValue.getResolvedDepGraph();

    ImmutableMap<ModuleKey, AugmentedModule> depGraph =
        computeAugmentedGraph(unprunedDepGraph, resolvedDepGraph.keySet(), overrides);

    ImmutableSetMultimap<ModuleExtensionId, String> extensionToRepoInternalNames =
        computeExtensionToRepoInternalNames(depGraphValue, env);
    if (extensionToRepoInternalNames == null) {
      return null;
    }

    // Group all ModuleKeys seen by their module name for easy lookup
    ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex =
        ImmutableMap.copyOf(
            depGraph.values().stream()
                .collect(
                    Collectors.groupingBy(
                        AugmentedModule::getName,
                        Collectors.mapping(AugmentedModule::getKey, toImmutableSet()))));

    return BazelModuleInspectorValue.create(depGraph, modulesIndex, extensionToRepoInternalNames);
  }

  public static ImmutableMap<ModuleKey, AugmentedModule> computeAugmentedGraph(
      ImmutableMap<ModuleKey, InterimModule> unprunedDepGraph,
      ImmutableSet<ModuleKey> usedModules,
      ImmutableMap<String, ModuleOverride> overrides) {
    Map<ModuleKey, AugmentedModule.Builder> depGraphAugmentBuilder = new HashMap<>();

    // For all Modules in the un-pruned dep graph, inspect their dependencies and add themselves
    // to their children AugmentedModule as dependant. Also fill in their own AugmentedModule
    // with a map from their dependencies to the resolution reason that was applied to each.
    // The newly created graph will also contain ModuleAugments for non-loaded modules.
    for (Entry<ModuleKey, InterimModule> e : unprunedDepGraph.entrySet()) {
      ModuleKey parentKey = e.getKey();
      InterimModule parentModule = e.getValue();

      AugmentedModule.Builder parentBuilder =
          depGraphAugmentBuilder
              .computeIfAbsent(
                  parentKey, k -> AugmentedModule.builder(k).setName(parentModule.getName()))
              .setVersion(parentModule.getVersion())
              .setLoaded(true);

      for (String childDep : parentModule.getDeps().keySet()) {
        ModuleKey originalKey = parentModule.getOriginalDeps().get(childDep).toModuleKey();
        InterimModule originalModule = unprunedDepGraph.get(originalKey);
        ModuleKey key = parentModule.getDeps().get(childDep).toModuleKey();
        InterimModule module = unprunedDepGraph.get(key);

        AugmentedModule.Builder originalChildBuilder =
            depGraphAugmentBuilder.computeIfAbsent(originalKey, AugmentedModule::builder);
        if (originalModule != null) {
          originalChildBuilder
              .setName(originalModule.getName())
              .setVersion(originalModule.getVersion())
              .setLoaded(true);
        }

        AugmentedModule.Builder newChildBuilder =
            depGraphAugmentBuilder.computeIfAbsent(
                key,
                k ->
                    AugmentedModule.builder(k)
                        .setName(module.getName())
                        .setVersion(module.getVersion())
                        .setLoaded(true));

        // originalDependants and dependants can differ because
        // parentModule could have had originalChild in the unresolved graph, but in the resolved
        // graph the originalChild could have become orphan due to an override or selection
        originalChildBuilder.addOriginalDependant(parentKey);
        // also, even if the dep has not changed, the parentModule may not be referenced
        // anymore in the resolved graph, so parentModule will only be added above
        if (usedModules.contains(parentKey)) {
          newChildBuilder.addDependant(parentKey);
        }

        ResolutionReason reason = ResolutionReason.ORIGINAL;
        if (!key.getVersion().equals(originalKey.getVersion())) {
          ModuleOverride override = overrides.get(key.getName());
          if (override != null) {
            if (override instanceof SingleVersionOverride) {
              reason = ResolutionReason.SINGLE_VERSION_OVERRIDE;
            } else if (override instanceof MultipleVersionOverride) {
              reason = ResolutionReason.MULTIPLE_VERSION_OVERRIDE;
            } else {
              // There is no other possible override
              Preconditions.checkArgument(override instanceof NonRegistryOverride);
              reason = ((NonRegistryOverride) override).getResolutionReason();
            }
          } else {
            reason = ResolutionReason.MINIMAL_VERSION_SELECTION;
          }
        }

        if (!reason.equals(ResolutionReason.ORIGINAL)) {
          parentBuilder.addUnusedDep(childDep, originalKey);
        }
        parentBuilder.addDep(childDep, key);
        parentBuilder.addDepReason(childDep, reason);
      }
    }

    return depGraphAugmentBuilder.entrySet().stream()
        .collect(toImmutableMap(Entry::getKey, e -> e.getValue().build()));
  }

  @Nullable
  private ImmutableSetMultimap<ModuleExtensionId, String> computeExtensionToRepoInternalNames(
      BazelDepGraphValue depGraphValue, Environment env) throws InterruptedException {
    ImmutableSet<ModuleExtensionId> extensionEvalKeys =
        depGraphValue.getExtensionUsagesTable().rowKeySet();
    ImmutableList<SingleExtensionEvalValue.Key> singleEvalKeys =
        extensionEvalKeys.stream().map(SingleExtensionEvalValue::key).collect(toImmutableList());
    SkyframeLookupResult singleEvalValues = env.getValuesAndExceptions(singleEvalKeys);

    ImmutableSetMultimap.Builder<ModuleExtensionId, String> extensionToRepoInternalNames =
        ImmutableSetMultimap.builder();
    for (SingleExtensionEvalValue.Key singleEvalKey : singleEvalKeys) {
      SingleExtensionEvalValue singleEvalValue =
          (SingleExtensionEvalValue) singleEvalValues.get(singleEvalKey);
      if (singleEvalValue == null) {
        return null;
      }
      extensionToRepoInternalNames.putAll(
          singleEvalKey.argument(), singleEvalValue.getGeneratedRepoSpecs().keySet());
    }
    return extensionToRepoInternalNames.build();
  }
}
