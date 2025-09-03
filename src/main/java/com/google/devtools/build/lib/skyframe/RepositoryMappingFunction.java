// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionRepoMappingEntriesValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/** {@link SkyFunction} for {@link RepositoryMappingValue}s. */
public class RepositoryMappingFunction implements SkyFunction {
  public static final PrecomputedValue.Precomputed<Map<RepositoryName, PathFragment>>
      REPOSITORY_OVERRIDES = new PrecomputedValue.Precomputed<>("repository_overrides");
  private final RuleClassProvider ruleClassProvider;

  public RepositoryMappingFunction(RuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = ruleClassProvider;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RepositoryMappingValue.Key key = (RepositoryMappingValue.Key) skyKey;
    RepositoryMappingValue repositoryMappingValue = computeInternal(key, env);
    if (repositoryMappingValue == null) {
      return null;
    }
    if (repositoryMappingValue == RepositoryMappingValue.NOT_FOUND_VALUE
        && REPOSITORY_OVERRIDES.get(env).containsKey(key.repoName())) {
      throw new RepositoryMappingFunctionException(
          String.format(
              "the repository %s does not exist, but has been specified as overridden with"
                  + " --override_repository. Use --inject_repository instead to add a new"
                  + " repository.",
              key.repoName()));
    }
    return repositoryMappingValue;
  }

  @Nullable
  private RepositoryMappingValue computeInternal(RepositoryMappingValue.Key skyKey, Environment env)
      throws InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    RepositoryName repositoryName = skyKey.repoName();

    if (StarlarkBuiltinsValue.isBuiltinsRepo(repositoryName)) {
      // If tools repo is not set, use the default empty mapping.
      if (ruleClassProvider.getToolsRepository() == null) {
        return RepositoryMappingValue.DEFAULT_VALUE_FOR_BUILTINS_REPO;
      }
      // Builtins .bzl files should use the repo mapping of @bazel_tools, to get access to repos
      // such as @platforms.
      RepositoryMappingValue bazelToolsMapping =
          (RepositoryMappingValue)
              env.getValue(RepositoryMappingValue.key(ruleClassProvider.getToolsRepository()));
      if (bazelToolsMapping == null) {
        return null;
      }
      return RepositoryMappingValue.DEFAULT_VALUE_FOR_BUILTINS_REPO.withAdditionalMappings(
          bazelToolsMapping.repositoryMapping());
    }

    BazelDepGraphValue bazelDepGraphValue =
        (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (bazelDepGraphValue == null) {
      return null;
    }

    // Try and see if this is a repo generated from a Bazel module.
    Optional<RepositoryMappingValue> mappingValue =
        computeForBazelModuleRepo(repositoryName, bazelDepGraphValue);
    if (mappingValue.isPresent()) {
      return repositoryName.isMain()
          ? mappingValue.get().withCachedInverseMap()
          : mappingValue.get();
    }

    // Now try and see if this is a repo generated from a module extension.
    Optional<ModuleExtensionId> moduleExtensionId =
        maybeGetModuleExtensionForRepo(repositoryName, bazelDepGraphValue);

    if (moduleExtensionId.isPresent()) {
      var repoMappingEntriesValue =
          (ModuleExtensionRepoMappingEntriesValue)
              env.getValue(ModuleExtensionRepoMappingEntriesValue.key(moduleExtensionId.get()));
      if (repoMappingEntriesValue == null) {
        return null;
      }
      return RepositoryMappingValue.create(
          RepositoryMapping.create(repoMappingEntriesValue.entries(), repositoryName),
          repoMappingEntriesValue.moduleKey().name(),
          repoMappingEntriesValue.moduleKey().version());
    }

    return RepositoryMappingValue.NOT_FOUND_VALUE;
  }

  /**
   * Calculates repo mappings for a repo generated from a Bazel module. Such a repo can see all its
   * {@code bazel_dep}s, as well as any repos generated by an extension it has a {@code use_repo}
   * clause for.
   *
   * @return the repo mappings for the repo if it's generated from a Bazel module, otherwise return
   *     Optional.empty().
   */
  private Optional<RepositoryMappingValue> computeForBazelModuleRepo(
      RepositoryName repositoryName, BazelDepGraphValue bazelDepGraphValue) {
    ModuleKey moduleKey = bazelDepGraphValue.getCanonicalRepoNameLookup().get(repositoryName);
    if (moduleKey == null) {
      return Optional.empty();
    }
    Module module = bazelDepGraphValue.getDepGraph().get(moduleKey);
    return Optional.of(
        RepositoryMappingValue.create(
            bazelDepGraphValue.getFullRepoMapping(moduleKey),
            module.getName(),
            module.getVersion()));
  }

  private static Optional<ModuleExtensionId> maybeGetModuleExtensionForRepo(
      RepositoryName repositoryName, BazelDepGraphValue bazelDepGraphValue) {
    return bazelDepGraphValue.getExtensionUniqueNames().entrySet().stream()
        .filter(e -> repositoryName.getName().startsWith(e.getValue() + "+"))
        .map(Entry::getKey)
        .findFirst();
  }

  private static class RepositoryMappingFunctionException extends SkyFunctionException {
    RepositoryMappingFunctionException(String message) {
      super(
          new BuildFileContainsErrorsException(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, message),
          Transience.PERSISTENT);
    }
  }
}
