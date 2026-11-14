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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RepoRuleId;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionInputs.ResolvedRepo;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.spelling.SpellChecker;
import net.starlark.java.syntax.Location;

/** Looks up the definition of a repo with the given name. */
public final class RepoDefinitionFunction implements SkyFunction {
  public static final PrecomputedValue.Precomputed<Map<String, PathFragment>> REPOSITORY_OVERRIDES =
      new PrecomputedValue.Precomputed<>("repository_overrides");
  private final BlazeDirectories directories;

  private RepoDefinitionFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  /** Returns the repository definition function and the functions that compute its inputs. */
  public static ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions(
      BlazeDirectories directories) {
    return ImmutableMap.of(
        RepoDefinitionValue.REPO_DEFINITION, new RepoDefinitionFunction(directories),
        SkyFunctions.ROOT_REPO_DEFINITION_INPUTS, RepoDefinitionInputs::computeRoot,
        SkyFunctions.RESOLVED_REPO_DEFINITION_INPUTS, RepoDefinitionInputs::computeResolved,
        SkyFunctions.RESOLVED_REPO_DEFINITION, RepoDefinitionInputs::computeResolvedRepo,
        SkyFunctions.EXTENSION_REPO_SPEC, new ExtensionRepoSpecFunction());
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    var root = (RepoDefinitionInputs.Root) env.getValue(RepoDefinitionInputs.ROOT_KEY);
    if (root == null) {
      return null;
    }
    RepositoryMapping basicMainRepoMapping = root.basicMainRepoMapping();

    RepositoryName repositoryName = ((RepoDefinitionValue.Key) skyKey).argument();

    // Step 0: Look for repository overrides via canonical names. Requesting the main repo mapping
    // at this point would result in a cycle, so any lookup by apparent name requires special
    // handling.
    Map<String, PathFragment> overrides = REPOSITORY_OVERRIDES.get(env);
    if (Preconditions.checkNotNull(overrides).containsKey(repositoryName.getName())) {
      return new RepoDefinitionValue.RepoOverride(overrides.get(repositoryName.getName()));
    }
    if (repositoryName.equals(RepositoryName.BAZEL_TOOLS)) {
      return new RepoDefinitionValue.RepoOverride(
          directories.getEmbeddedBinariesRoot().getRelative("embedded_tools").asFragment());
    }

    // Step 1: Look for repositories defined by non-registry overrides.
    Optional<RepoDefinitionValue> nonRegistryOverride =
        checkRepoFromNonRegistryOverrides(
            root, repositoryName, overrides, basicMainRepoMapping, env);
    if (env.valuesMissing()) {
      return null;
    }
    if (nonRegistryOverride.isPresent()) {
      return nonRegistryOverride.get();
    }

    // The resolved graph can depend on repos found in Step 1.
    var resolved =
        (ResolvedRepo) env.getValue(RepoDefinitionInputs.ResolvedRepoKey.create(repositoryName));
    if (resolved == null) {
      return null;
    }

    // Step 2: Look for repository overrides via apparent names.
    ImmutableMap<RepositoryName, PathFragment> apparentNameOverrides =
        getApparentNameOverrides(overrides, env);
    if (env.valuesMissing()) {
      return null;
    }
    if (apparentNameOverrides.containsKey(repositoryName)) {
      return new RepoDefinitionValue.RepoOverride(apparentNameOverrides.get(repositoryName));
    }

    // Step 3: Look for repositories derived from Bazel modules or module extensions.
    return switch (resolved) {
      case ResolvedRepo.Module(RepoSpec spec) ->
          createRepoDefinitionFromSpec(
              spec, repositoryName, /* originalName= */ null, basicMainRepoMapping, env);
      case ResolvedRepo.Extension(ModuleExtensionId extensionId) ->
          checkRepoFromExtension(extensionId, repositoryName, basicMainRepoMapping, env);
      case ResolvedRepo.Missing unused -> RepoDefinitionValue.NOT_FOUND;
    };
  }

  @Nullable
  private RepoDefinitionValue checkRepoFromExtension(
      ModuleExtensionId extensionId,
      RepositoryName repositoryName,
      RepositoryMapping basicMainRepoMapping,
      Environment env)
      throws RepoDefinitionFunctionException, InterruptedException {
    // Depend on the selected spec so changes to other specs don't invalidate this definition.
    var extensionRepo =
        (ExtensionRepoSpecFunction.Value)
            env.getValue(ExtensionRepoSpecFunction.Key.create(extensionId, repositoryName));
    return switch (extensionRepo) {
      case null -> null;
      case ExtensionRepoSpecFunction.NotFound unused -> RepoDefinitionValue.NOT_FOUND;
      case ExtensionRepoSpecFunction.Found(String internalName, RepoSpec spec) ->
          createRepoDefinitionFromSpec(spec, repositoryName, internalName, basicMainRepoMapping, env);
    };
  }

  // Callers must check env.valuesMissing() and ignore the result if true.
  private Optional<RepoDefinitionValue> checkRepoFromNonRegistryOverrides(
      RepoDefinitionInputs.Root root,
      RepositoryName repositoryName,
      Map<String, PathFragment> overrides,
      RepositoryMapping basicMainRepoMapping,
      Environment env)
      throws RepoDefinitionFunctionException, InterruptedException {
    var override = root.nonRegistryOverrides().get(repositoryName);
    if (override == null) {
      return Optional.empty();
    }
    // If this module is a direct dep of the root module, its apparent name may be named by the
    // --override_repository flag, which takes precedence over the non-registry override.
    if (override.apparentName() != null) {
      PathFragment commandLineOverride = overrides.get(override.apparentName());
      if (commandLineOverride != null) {
        return Optional.of(new RepoDefinitionValue.RepoOverride(commandLineOverride));
      }
    }
    return Optional.ofNullable(
        createRepoDefinitionFromSpec(
            override.spec(), repositoryName, /* originalName= */ null, basicMainRepoMapping, env));
  }

  @Nullable
  private ImmutableMap<RepositoryName, PathFragment> getApparentNameOverrides(
      Map<String, PathFragment> overrides, Environment env)
      throws InterruptedException, RepoDefinitionFunctionException {
    if (overrides.isEmpty()) {
      return ImmutableMap.of();
    }
    var mainRepoMapping =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (mainRepoMapping == null) {
      return null;
    }
    var apparentNameOverrides = ImmutableMap.<RepositoryName, PathFragment>builder();
    for (Map.Entry<String, PathFragment> entry : overrides.entrySet()) {
      if (!RepositoryName.isApparent(entry.getKey())) {
        continue;
      }
      String apparentName = entry.getKey();
      RepositoryName canonicalName = mainRepoMapping.repositoryMapping().get(apparentName);
      if (!canonicalName.isVisible()) {
        throw new RepoDefinitionFunctionException(
            ExternalDepsException.withMessage(
                Code.BAD_MODULE,
                "no repository visible as '@%s'%s from the main repository, but overridden with"
                    + " --override_repository. Use --inject_repository to add new repositories.",
                apparentName,
                SpellChecker.didYouMean(
                    apparentName,
                    Iterables.transform(
                        mainRepoMapping.repositoryMapping().entries().keySet(),
                        name -> "@" + name))),
            Transience.PERSISTENT);
      }
      apparentNameOverrides.put(canonicalName, entry.getValue());
    }
    return apparentNameOverrides.buildOrThrow();
  }

  @Nullable
  private RepoDefinitionValue createRepoDefinitionFromSpec(
      RepoSpec repoSpec,
      RepositoryName repositoryName,
      @Nullable String originalName,
      RepositoryMapping basicMainRepoMapping,
      Environment env)
      throws RepoDefinitionFunctionException, InterruptedException {
    RepoRule repoRule = loadRepoRule(repoSpec.repoRuleId(), env);
    if (repoRule == null) {
      return null;
    }

    try {
      RepoSpec typeCheckedRepoSpec =
          repoRule.instantiate(
              repoSpec.attributes().attributes(),
              // Use a completely fake call stack. This should never be user-visible anyway, since
              // the repo spec here should have already been typechecked (or generated directly by
              // a Registry#getRepoSpec call -- in which case it better be correct already...).
              ImmutableList.of(StarlarkThread.callStackEntry("<toplevel>", Location.BUILTIN)),
              new LabelConverter(PackageIdentifier.EMPTY_PACKAGE_ID, basicMainRepoMapping),
              env.getListener(),
              "to the root module");
      var repoDefinition =
          new RepoDefinition(
              repoRule, typeCheckedRepoSpec.attributes(), repositoryName.getName(), originalName);
      return new RepoDefinitionValue.Found(repoDefinition);
    } catch (ExternalDepsException e) {
      throw new RepoDefinitionFunctionException(e, Transience.PERSISTENT);
    }
  }

  @Nullable
  private RepoRule loadRepoRule(RepoRuleId repoRuleId, Environment env)
      throws InterruptedException, RepoDefinitionFunctionException {
    SkyKey key;
    if (NonRegistryOverride.BOOTSTRAP_REPO_RULES.contains(repoRuleId)) {
      key = BzlLoadValue.keyForBzlmodBootstrap(repoRuleId.bzlFileLabel());
    } else {
      key = BzlLoadValue.keyForBzlmod(repoRuleId.bzlFileLabel());
    }

    // Load the .bzl file pointed to by the label.
    BzlLoadValue bzlLoadValue;
    try {
      bzlLoadValue = (BzlLoadValue) env.getValueOrThrow(key, BzlLoadFailedException.class);
    } catch (BzlLoadFailedException e) {
      // No need for a super detailed error message, since errors here can basically only happen
      // when something is horribly wrong. (The labels to load are either hardcoded or already
      // sanity-checked somewhere else.)
      throw new RepoDefinitionFunctionException(e, Transience.PERSISTENT);
    }
    if (bzlLoadValue == null) {
      return null;
    }

    Object object = bzlLoadValue.getModule().getGlobal(repoRuleId.ruleName());
    if (object instanceof RepoRule.Supplier repoRuleSupplier) {
      return repoRuleSupplier.getRepoRule();
    } else if (object == null) {
      throw new RepoDefinitionFunctionException(
          ExternalDepsException.withMessage(
              Code.EXTENSION_EVAL_ERROR,
              "repository rule %s does not exist (no such symbol in that file)",
              repoRuleId),
          Transience.PERSISTENT);
    } else {
      throw new RepoDefinitionFunctionException(
          ExternalDepsException.withMessage(
              Code.EXTENSION_EVAL_ERROR,
              "invalid repository rule: %s, expected type repository_rule, got type %s",
              repoRuleId,
              Starlark.type(object)),
          Transience.PERSISTENT);
    }
  }

  private static final class RepoDefinitionFunctionException extends SkyFunctionException {
    RepoDefinitionFunctionException(BzlLoadFailedException e, Transience transience) {
      super(e, transience);
    }

    RepoDefinitionFunctionException(ExternalDepsException e, Transience transience) {
      super(e, transience);
    }
  }
}
