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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RepoRuleId;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionValue;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map.Entry;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** Looks up the definition of a repo with the given name. */
public final class RepoDefinitionFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RootModuleFileValue root =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (env.valuesMissing()) {
      return null;
    }

    // Sometimes, the attributes in the repo specs contain label strings instead of label objects.
    // This really only happens for attributes like `patches` that come from an `archive_override`
    // or `git_override` in MODULE.bazel. (This is because those overrides just store all attributes
    // unparsed and unvalidated in the repo spec.) In other words, this can only happen for Step 1
    // below -- but for consistency's sake, we pass this mapping for all 3 cases.
    //
    // In such cases, we need to provide a "basic repo mapping", so that we can properly turn those
    // label strings into label objects. Since we only accept patches from the main repo anyway, we
    // only need the two simple entries pointing into the main repo itself.
    RepositoryMapping basicMainRepoMapping =
        RepositoryMapping.create(
            ImmutableMap.<String, RepositoryName>builder()
                .put("", RepositoryName.MAIN)
                .put(root.module().getRepoName(), RepositoryName.MAIN)
                .buildKeepingLast(),
            RepositoryName.MAIN);

    RepositoryName repositoryName = ((RepoDefinitionValue.Key) skyKey).argument();

    // Step 1: Look for repositories defined by non-registry overrides.
    Optional<RepoSpec> repoSpec = checkRepoFromNonRegistryOverrides(root, repositoryName);
    if (repoSpec.isPresent()) {
      return createRepoDefinitionFromSpec(
          repoSpec.get(), repositoryName, /* originalName= */ null, basicMainRepoMapping, env);
    }

    // BazelDepGraphValue is affected by repos found in Step 1, therefore it should NOT
    // be requested in Step 1 to avoid cycle dependency.
    BazelDepGraphValue bazelDepGraphValue =
        (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (env.valuesMissing()) {
      return null;
    }

    // Step 2: Look for repositories derived from Bazel Modules.
    repoSpec = checkRepoFromBazelModules(bazelDepGraphValue, repositoryName);
    if (repoSpec.isPresent()) {
      return createRepoDefinitionFromSpec(
          repoSpec.get(), repositoryName, /* originalName= */ null, basicMainRepoMapping, env);
    }

    // Step 3: look for the repo from module extension evaluation results.
    Optional<ModuleExtensionId> extensionId =
        bazelDepGraphValue.getExtensionUniqueNames().entrySet().stream()
            .filter(e -> repositoryName.getName().startsWith(e.getValue() + "+"))
            .map(Entry::getKey)
            .findFirst();

    if (extensionId.isEmpty()) {
      return RepoDefinitionValue.NOT_FOUND;
    }

    SingleExtensionValue extensionValue =
        (SingleExtensionValue) env.getValue(SingleExtensionValue.key(extensionId.get()));
    if (extensionValue == null) {
      return null;
    }

    String internalRepo = extensionValue.canonicalRepoNameToInternalNames().get(repositoryName);
    if (internalRepo == null) {
      return RepoDefinitionValue.NOT_FOUND;
    }
    RepoSpec extRepoSpec = extensionValue.generatedRepoSpecs().get(internalRepo);
    return createRepoDefinitionFromSpec(
        extRepoSpec, repositoryName, internalRepo, basicMainRepoMapping, env);
  }

  private static Optional<RepoSpec> checkRepoFromNonRegistryOverrides(
      RootModuleFileValue root, RepositoryName repositoryName) {
    String moduleName = root.nonRegistryOverrideCanonicalRepoNameLookup().get(repositoryName);
    if (moduleName == null) {
      return Optional.empty();
    }
    NonRegistryOverride override = (NonRegistryOverride) root.overrides().get(moduleName);
    return Optional.of(override.repoSpec());
  }

  private Optional<RepoSpec> checkRepoFromBazelModules(
      BazelDepGraphValue bazelDepGraphValue, RepositoryName repositoryName) {
    ModuleKey moduleKey = bazelDepGraphValue.getCanonicalRepoNameLookup().get(repositoryName);
    if (moduleKey == null) {
      return Optional.empty();
    }
    return Optional.ofNullable(bazelDepGraphValue.getDepGraph().get(moduleKey).getRepoSpec());
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
