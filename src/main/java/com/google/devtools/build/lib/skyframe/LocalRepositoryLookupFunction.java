// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.LocalPathRepoSpecs;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RepoRuleId;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionValue;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionFunction;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import javax.annotation.Nullable;

/** SkyFunction for {@link LocalRepositoryLookupValue}s. */
public class LocalRepositoryLookupFunction implements SkyFunction {
  private static final RepoRuleId NEW_LOCAL_REPOSITORY =
      new RepoRuleId(LocalPathRepoSpecs.LOCAL_REPOSITORY.bzlFileLabel(), "new_local_repository");

  private final Path workspaceRoot;

  public LocalRepositoryLookupFunction(Path workspaceRoot) {
    this.workspaceRoot = workspaceRoot;
  }

  // Implementation note: Although LocalRepositoryLookupValue.NOT_FOUND exists, it should never be
  // returned from this method.
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RootedPath directory = (RootedPath) skyKey.argument();

    if (directory.getRootRelativePath().equals(PathFragment.EMPTY_FRAGMENT)) {
      return LocalRepositoryLookupValue.mainRepository();
    }

    LocalRepositoryLookupValue repository = maybeCheckDirectoryForRepository(env, directory);
    if (repository == null) {
      return null;
    }
    if (repository.exists()) {
      return repository;
    }

    return env.getValue(LocalRepositoryLookupValue.key(directory.getParentDirectory()));
  }

  @Nullable
  private LocalRepositoryLookupValue maybeCheckDirectoryForRepository(Environment env, RootedPath directory)
      throws InterruptedException {
    RepositoryMappingValue repositoryMappingValue =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (repositoryMappingValue == null) {
      return null;
    }
    RootModuleFileValue rootModuleFileValue =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (rootModuleFileValue == null) {
      return null;
    }
    BazelDepGraphValue bazelDepGraphValue = (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (bazelDepGraphValue == null) {
      return null;
    }
    RepositoryMapping repositoryMapping = repositoryMappingValue.repositoryMapping();
    if (repositoryMapping == null) {
      return LocalRepositoryLookupValue.notFound();
    }

    LocalRepositoryLookupValue repository =
        maybeMatchCommandLineOverrides(directory, repositoryMapping, env);
    if (repository != null) {
      return repository;
    }

    repository = maybeMatchNonRegistryOverrides(directory, rootModuleFileValue);
    if (repository != null) {
      return repository;
    }

    repository = maybeMatchModuleRepositories(directory, bazelDepGraphValue);
    if (repository != null) {
      return repository;
    }

    return maybeMatchExtensionRepositories(directory, bazelDepGraphValue, env);
  }

  @Nullable
  private LocalRepositoryLookupValue maybeMatchCommandLineOverrides(
      RootedPath directory, RepositoryMapping repositoryMapping, Environment env)
      throws InterruptedException {
    var repositoryOverrides = RepoDefinitionFunction.REPOSITORY_OVERRIDES.get(env);
    if (repositoryOverrides == null) {
      return null;
    }
    for (var entry : repositoryOverrides.entrySet()) {
      RepositoryName repositoryName = repositoryMapping.get(entry.getKey());
      if (!repositoryName.isVisible()) {
        repositoryName = RepositoryName.createUnvalidated(entry.getKey());
      }
      if (repositoryName.isMain()) {
        continue;
      }
      LocalRepositoryLookupValue repository =
          maybeMatchPath(directory, repositoryName, entry.getValue());
      if (repository != null) {
        return repository;
      }
    }
    return null;
  }

  @Nullable
  private LocalRepositoryLookupValue maybeMatchNonRegistryOverrides(
      RootedPath directory, RootModuleFileValue rootModuleFileValue) {
    for (var entry : rootModuleFileValue.nonRegistryOverrideCanonicalRepoToModuleName().entrySet()) {
      var override = rootModuleFileValue.overrides().get(entry.getValue());
      if (!(override instanceof NonRegistryOverride nonRegistryOverride)
          || override == NonRegistryOverride.BAZEL_TOOLS_OVERRIDE) {
        continue;
      }
      LocalRepositoryLookupValue repository =
          maybeMatchRepoSpec(directory, entry.getKey(), nonRegistryOverride.repoSpec());
      if (repository != null) {
        return repository;
      }
    }
    return null;
  }

  @Nullable
  private LocalRepositoryLookupValue maybeMatchModuleRepositories(
      RootedPath directory, BazelDepGraphValue bazelDepGraphValue) {
    for (var entry : bazelDepGraphValue.getCanonicalRepoNameLookup().entrySet()) {
      if (entry.getKey().isMain()) {
        continue;
      }
      Module module = bazelDepGraphValue.getDepGraph().get(entry.getValue());
      if (module == null || module.getRepoSpec() == null) {
        continue;
      }
      LocalRepositoryLookupValue repository =
          maybeMatchRepoSpec(directory, entry.getKey(), module.getRepoSpec());
      if (repository != null) {
        return repository;
      }
    }
    return null;
  }

  @Nullable
  private LocalRepositoryLookupValue maybeMatchExtensionRepositories(
      RootedPath directory, BazelDepGraphValue bazelDepGraphValue, Environment env)
      throws InterruptedException {
    ImmutableSet<ModuleExtensionId> extensionIds =
        ImmutableSet.copyOf(bazelDepGraphValue.getExtensionUniqueNames().keySet());
    if (extensionIds.isEmpty()) {
      return LocalRepositoryLookupValue.notFound();
    }
    SkyframeLookupResult extensionValues =
        env.getValuesAndExceptions(
            extensionIds.stream().map(SingleExtensionValue::key).collect(ImmutableSet.toImmutableSet()));
    for (ModuleExtensionId extensionId : extensionIds) {
      SingleExtensionValue extensionValue =
          (SingleExtensionValue) extensionValues.get(SingleExtensionValue.key(extensionId));
      if (extensionValue == null) {
        return null;
      }
      for (var entry : extensionValue.canonicalRepoNameToInternalNames().entrySet()) {
        RepoSpec repoSpec = extensionValue.generatedRepoSpecs().get(entry.getValue());
        if (repoSpec == null) {
          continue;
        }
        LocalRepositoryLookupValue repository =
            maybeMatchRepoSpec(directory, entry.getKey(), repoSpec);
        if (repository != null) {
          return repository;
        }
      }
    }
    return LocalRepositoryLookupValue.notFound();
  }

  @Nullable
  private LocalRepositoryLookupValue maybeMatchRepoSpec(
      RootedPath directory, RepositoryName repositoryName, RepoSpec repoSpec) {
    if (!repoSpec.repoRuleId().equals(LocalPathRepoSpecs.LOCAL_REPOSITORY)
        && !repoSpec.repoRuleId().equals(NEW_LOCAL_REPOSITORY)) {
      return null;
    }
    Object pathAttr = repoSpec.attributes().attributes().get("path");
    if (!(pathAttr instanceof String path)) {
      return null;
    }
    return maybeMatchPath(directory, repositoryName, PathFragment.create(path));
  }

  @Nullable
  private LocalRepositoryLookupValue maybeMatchPath(
      RootedPath directory, RepositoryName repositoryName, PathFragment repoPath) {
    Path localRepositoryPath = workspaceRoot.getRelative(repoPath);
    if (!directory.asPath().equals(localRepositoryPath)) {
      return null;
    }
    return LocalRepositoryLookupValue.success(repositoryName, repoPath);
  }
}
