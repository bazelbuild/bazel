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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeIterableResult;
import java.util.Map;

/** Resolves module extension repos by evaluating all module extensions. */
public class ModuleExtensionResolutionFunction implements SkyFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    BazelModuleResolutionValue bazelModuleResolutionValue =
        (BazelModuleResolutionValue) env.getValue(BazelModuleResolutionValue.KEY);
    if (bazelModuleResolutionValue == null) {
      return null;
    }

    // Individually evaluate each module extension.
    ImmutableList<SingleExtensionEvalValue.Key> singleEvalKeys =
        bazelModuleResolutionValue.getExtensionUsagesTable().rowKeySet().stream()
            .map(SingleExtensionEvalValue::key)
            .collect(toImmutableList());
    SkyframeIterableResult singleEvalValues = env.getOrderedValuesAndExceptions(singleEvalKeys);
    if (env.valuesMissing()) {
      return null;
    }

    // Collect information from all single extension evaluations.
    // Note that we generate one package per repo. If we generated one package containing all repos,
    // this package will be stored in the BzlmodRepoRuleValue for all repos, which in turn means
    // that any change in any module-extension-generated repo will cause all other repos to go
    // through RepositoryDelegatorFunction again. This isn't a normally a problem, but since we
    // always refetch "local" (aka "always fetch") repos, this could be an unnecessary performance
    // hit.
    ImmutableMap.Builder<RepositoryName, Package> canonicalRepoNameToPackage =
        ImmutableMap.builder();
    ImmutableMap.Builder<RepositoryName, ModuleExtensionId> canonicalRepoNameToExtensionId =
        ImmutableMap.builder();
    ImmutableListMultimap.Builder<ModuleExtensionId, String> extensionIdToRepoInternalNames =
        ImmutableListMultimap.builder();
    ImmutableSet<ModuleExtensionId> moduleEvalKeys =
        bazelModuleResolutionValue.getExtensionUsagesTable().rowKeySet();
    for (ModuleExtensionId extensionId : moduleEvalKeys) {
      SkyValue value = singleEvalValues.next();
      if (value == null) {
        return null;
      }
      ImmutableMap<String, Package> generatedRepos =
          ((SingleExtensionEvalValue) value).getGeneratedRepos();
      String repoPrefix =
          bazelModuleResolutionValue.getExtensionUniqueNames().get(extensionId) + '.';
      for (Map.Entry<String, Package> entry : generatedRepos.entrySet()) {
        RepositoryName canonicalRepoName =
            RepositoryName.createUnvalidated(repoPrefix + entry.getKey());
        canonicalRepoNameToPackage.put(canonicalRepoName, entry.getValue());
        canonicalRepoNameToExtensionId.put(canonicalRepoName, extensionId);
      }
      extensionIdToRepoInternalNames.putAll(extensionId, generatedRepos.keySet());
    }
    return ModuleExtensionResolutionValue.create(
        canonicalRepoNameToPackage.buildOrThrow(),
        canonicalRepoNameToExtensionId.buildOrThrow(),
        extensionIdToRepoInternalNames.build());
  }
}
