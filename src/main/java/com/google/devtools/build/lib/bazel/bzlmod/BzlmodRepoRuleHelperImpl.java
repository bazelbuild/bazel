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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.io.IOException;
import java.util.Optional;

/** A helper class to get {@link RepoSpec} for Bzlmod generated repositories. */
public final class BzlmodRepoRuleHelperImpl implements BzlmodRepoRuleHelper {

  @Override
  public Optional<RepoSpec> getRepoSpec(Environment env, String repositoryName)
      throws InterruptedException, IOException {

    ModuleFileValue root = (ModuleFileValue) env.getValue(ModuleFileValue.keyForRootModule());
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableMap<String, ModuleOverride> overrides = root.getOverrides();

    // Step 1: Look for repositories defined by non-registry overrides.
    Optional<RepoSpec> repoSpec = checkRepoFromNonRegistryOverrides(overrides, repositoryName);
    if (repoSpec.isPresent()) {
      return repoSpec;
    }

    // SelectionValue is affected by repos found in Step 1, therefore it should NOT be asked
    // in Step 1 to avoid cycle dependency.
    SelectionValue selectionValue = (SelectionValue) env.getValue(SelectionValue.KEY);
    if (env.valuesMissing()) {
      return null;
    }

    // Step 2: Look for repositories derived from Bazel Modules.
    repoSpec =
        checkRepoFromBazelModules(selectionValue, overrides, env.getListener(), repositoryName);
    if (repoSpec.isPresent()) {
      return repoSpec;
    }

    // Step 3: Look for repositories derived from module rules.
    return checkRepoFromModuleRules();
  }

  private static Optional<RepoSpec> checkRepoFromNonRegistryOverrides(
      ImmutableMap<String, ModuleOverride> overrides, String repositoryName) {
    if (overrides.containsKey(repositoryName)) {
      ModuleOverride override = overrides.get(repositoryName);
      if (override instanceof NonRegistryOverride) {
        return Optional.of(((NonRegistryOverride) override).getRepoSpec(repositoryName));
      }
    }
    return Optional.empty();
  }

  private static Optional<RepoSpec> checkRepoFromBazelModules(
      SelectionValue selectionValue,
      ImmutableMap<String, ModuleOverride> overrides,
      ExtendedEventHandler eventlistener,
      String repositoryName)
      throws InterruptedException, IOException {
    for (ModuleKey moduleKey : selectionValue.getDepGraph().keySet()) {
      // TODO(pcloudy): Support multiple version override.
      // Currently we assume there is only one version for each module, therefore the module name is
      // the repository name, but that's not the case if multiple version of the same module are
      // allowed.
      if (moduleKey.getName().equals(repositoryName)) {
        Module module = selectionValue.getDepGraph().get(moduleKey);
        Registry registry = checkNotNull(module.getRegistry());
        RepoSpec repoSpec = registry.getRepoSpec(moduleKey, repositoryName, eventlistener);
        repoSpec = maybeAppendAdditionalPatches(repoSpec, overrides.get(moduleKey.getName()));
        return Optional.of(repoSpec);
      }
    }
    return Optional.empty();
  }

  private static RepoSpec maybeAppendAdditionalPatches(RepoSpec repoSpec, ModuleOverride override) {
    if (!(override instanceof SingleVersionOverride)) {
      return repoSpec;
    }
    SingleVersionOverride singleVersion = (SingleVersionOverride) override;
    if (singleVersion.getPatches().isEmpty()) {
      return repoSpec;
    }
    ImmutableMap.Builder<String, Object> attrBuilder = ImmutableMap.builder();
    attrBuilder.putAll(repoSpec.attributes());
    attrBuilder.put("patches", singleVersion.getPatches());
    attrBuilder.put("patch_args", ImmutableList.of("-p" + singleVersion.getPatchStrip()));
    return RepoSpec.builder()
        .setBzlFile(repoSpec.bzlFile())
        .setRuleClassName(repoSpec.ruleClassName())
        .setAttributes(attrBuilder.build())
        .build();
  }

  private static Optional<RepoSpec> checkRepoFromModuleRules() {
    // TODO(pcloudy): Implement calculating RepoSpec from module rules.
    return Optional.empty();
  }
}
