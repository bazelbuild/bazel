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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.ArchiveRepoSpecBuilder;
import com.google.devtools.build.lib.bazel.bzlmod.AttributeValues;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleCreator;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.bzlmod.GitRepoSpecBuilder;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileValue.RootModuleFileValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.RuleFunction;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;

/**
 * Looks up the {@link RepoSpec} of a given repository name and create its repository rule instance.
 */
public final class BzlmodRepoRuleFunction implements SkyFunction {

  private final RuleClassProvider ruleClassProvider;
  private final BlazeDirectories directories;

  /**
   * An empty repo mapping anchored to the main repo. Label strings in {@link RepoSpec}s are always
   * in unambiguous canonical form and thus require no mapping, except instances read from old
   * lockfiles.
   */
  // TODO(fmeum): Make this mapping truly empty after bumping LOCK_FILE_VERSION.
  private static final RepositoryMapping EMPTY_MAIN_REPO_MAPPING =
      RepositoryMapping.create(
          ImmutableMap.of("", RepositoryName.MAIN, "bazel_tools", RepositoryName.BAZEL_TOOLS),
          RepositoryName.MAIN);

  public BzlmodRepoRuleFunction(RuleClassProvider ruleClassProvider, BlazeDirectories directories) {
    this.ruleClassProvider = ruleClassProvider;
    this.directories = directories;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    RootModuleFileValue root =
        (RootModuleFileValue) env.getValue(ModuleFileValue.KEY_FOR_ROOT_MODULE);
    if (env.valuesMissing()) {
      return null;
    }

    RepositoryName repositoryName = ((BzlmodRepoRuleValue.Key) skyKey).argument();

    // Step 1: Look for repositories defined by non-registry overrides.
    Optional<RepoSpec> repoSpec = checkRepoFromNonRegistryOverrides(root, repositoryName);
    if (repoSpec.isPresent()) {
      return createRuleFromSpec(repoSpec.get(), repositoryName, starlarkSemantics, env);
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
      return createRuleFromSpec(repoSpec.get(), repositoryName, starlarkSemantics, env);
    }

    // Step 3: look for the repo from module extension evaluation results.
    Optional<ModuleExtensionId> extensionId =
        bazelDepGraphValue.getExtensionUniqueNames().entrySet().stream()
            .filter(e -> repositoryName.getName().startsWith(e.getValue() + "~"))
            .map(Entry::getKey)
            .findFirst();

    if (extensionId.isEmpty()) {
      return BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE;
    }

    SingleExtensionValue extensionValue =
        (SingleExtensionValue) env.getValue(SingleExtensionValue.key(extensionId.get()));
    if (extensionValue == null) {
      return null;
    }

    String internalRepo = extensionValue.getCanonicalRepoNameToInternalNames().get(repositoryName);
    if (internalRepo == null) {
      return BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE;
    }
    RepoSpec extRepoSpec = extensionValue.getGeneratedRepoSpecs().get(internalRepo);
    return createRuleFromSpec(extRepoSpec, repositoryName, starlarkSemantics, env);
  }

  private static Optional<RepoSpec> checkRepoFromNonRegistryOverrides(
      RootModuleFileValue root, RepositoryName repositoryName) {
    String moduleName = root.getNonRegistryOverrideCanonicalRepoNameLookup().get(repositoryName);
    if (moduleName == null) {
      return Optional.empty();
    }
    NonRegistryOverride override = (NonRegistryOverride) root.getOverrides().get(moduleName);
    return Optional.of(override.getRepoSpec());
  }

  private Optional<RepoSpec> checkRepoFromBazelModules(
      BazelDepGraphValue bazelDepGraphValue, RepositoryName repositoryName) {
    ModuleKey moduleKey = bazelDepGraphValue.getCanonicalRepoNameLookup().get(repositoryName);
    if (moduleKey == null) {
      return Optional.empty();
    }
    return Optional.of(bazelDepGraphValue.getDepGraph().get(moduleKey).getRepoSpec());
  }

  @Nullable
  private BzlmodRepoRuleValue createRuleFromSpec(
      RepoSpec repoSpec,
      RepositoryName repositoryName,
      StarlarkSemantics starlarkSemantics,
      Environment env)
      throws BzlmodRepoRuleFunctionException, InterruptedException {
    RuleClass ruleClass;
    if (repoSpec.isNativeRepoRule()) {
      if (!ruleClassProvider.getRuleClassMap().containsKey(repoSpec.ruleClassName())) {
        InvalidRuleException e =
            new InvalidRuleException(
                "Unrecognized native repository rule: " + repoSpec.getRuleClass());
        throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
      }
      ruleClass = ruleClassProvider.getRuleClassMap().get(repoSpec.ruleClassName());
    } else {
      ImmutableMap<String, Module> loadedModules =
          loadBzlModules(env, repoSpec.bzlFile(), repoSpec.getRuleClass(), starlarkSemantics);
      if (env.valuesMissing()) {
        return null;
      }
      ruleClass = getStarlarkRuleClass(repoSpec, loadedModules);
    }

    var attributes =
        ImmutableMap.<String, Object>builder()
            .putAll(resolveRemotePatchesUrl(repoSpec).attributes())
            .put("name", repositoryName.getName())
            .buildOrThrow();
    try {
      Rule rule =
          BzlmodRepoRuleCreator.createRule(
              PackageIdentifier.EMPTY_PACKAGE_ID,
              EMPTY_MAIN_REPO_MAPPING,
              directories,
              starlarkSemantics,
              env.getListener(),
              "BzlmodRepoRuleFunction.createRule",
              ruleClass,
              attributes);
      return new BzlmodRepoRuleValue(rule.getPackage(), rule.getName());
    } catch (InvalidRuleException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    } catch (NoSuchPackageException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    } catch (EvalException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }
  }

  /* Resolves repo specs containing remote patches that are stored with %workspace% place holder*/
  @SuppressWarnings("unchecked")
  private AttributeValues resolveRemotePatchesUrl(RepoSpec repoSpec) {
    if (repoSpec
        .getRuleClass()
        .equals(ArchiveRepoSpecBuilder.HTTP_ARCHIVE_PATH + "%http_archive")) {
      return AttributeValues.create(
          repoSpec.attributes().attributes().entrySet().stream()
              .collect(
                  toImmutableMap(
                      Map.Entry::getKey,
                      e -> {
                        if (e.getKey().equals("remote_patches")) {
                          Map<String, String> remotePatches = (Map<String, String>) e.getValue();
                          return remotePatches.keySet().stream()
                              .collect(
                                  toImmutableMap(
                                      key ->
                                          key.replace(
                                              "%workspace%",
                                              directories.getWorkspace().getPathString()),
                                      remotePatches::get));
                        }
                        return e.getValue();
                      })));
    }
    return repoSpec.attributes();
  }

  // Starlark rules loaded from bazel_tools that may define Bazel module repositories and thus must
  // be loaded without relying on any other modules.
  private static final Set<String> BOOTSTRAP_RULE_CLASSES =
      ImmutableSet.of(
          ArchiveRepoSpecBuilder.HTTP_ARCHIVE_PATH + "%http_archive",
          GitRepoSpecBuilder.GIT_REPO_PATH + "%git_repository");

  /** Loads modules from the given bzl file. */
  private ImmutableMap<String, Module> loadBzlModules(
      Environment env, String bzlFile, String ruleClass, StarlarkSemantics starlarkSemantics)
      throws InterruptedException, BzlmodRepoRuleFunctionException {
    ImmutableList<Pair<String, Location>> programLoads =
        ImmutableList.of(Pair.of(bzlFile, Location.BUILTIN));

    ImmutableList<Label> loadLabels =
        BzlLoadFunction.getLoadLabels(
            env.getListener(),
            programLoads,
            PackageIdentifier.EMPTY_PACKAGE_ID,
            EMPTY_MAIN_REPO_MAPPING,
            starlarkSemantics);
    if (loadLabels == null) {
      NoSuchPackageException e =
          PackageFunction.PackageFunctionException.builder()
              .setType(PackageFunction.PackageFunctionException.Type.BUILD_FILE_CONTAINS_ERRORS)
              .setPackageIdentifier(PackageIdentifier.EMPTY_PACKAGE_ID)
              .setMessage("malformed load statements")
              .setPackageLoadingCode(PackageLoading.Code.IMPORT_STARLARK_FILE_ERROR)
              .buildCause();
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }

    Preconditions.checkArgument(loadLabels.size() == 1);
    ImmutableList<BzlLoadValue.Key> keys;
    if (BOOTSTRAP_RULE_CLASSES.contains(ruleClass)) {
      keys = ImmutableList.of(BzlLoadValue.keyForBzlmodBootstrap(loadLabels.get(0)));
    } else {
      keys = ImmutableList.of(BzlLoadValue.keyForBzlmod(loadLabels.get(0)));
    }

    // Load the .bzl module.
    try {
      // No need to check visibility for an extension repospec that is always public
      return PackageFunction.loadBzlModules(
          env,
          PackageIdentifier.EMPTY_PACKAGE_ID,
          "Bzlmod system",
          programLoads,
          keys,
          starlarkSemantics,
          null,
          /* checkVisibility= */ false);
    } catch (NoSuchPackageException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }
  }

  private RuleClass getStarlarkRuleClass(
      RepoSpec repoSpec, ImmutableMap<String, Module> loadedModules)
      throws BzlmodRepoRuleFunctionException {
    Object object = loadedModules.get(repoSpec.bzlFile()).getGlobal(repoSpec.ruleClassName());
    if (object instanceof RuleFunction) {
      return ((RuleFunction) object).getRuleClass();
    } else {
      InvalidRuleException e =
          new InvalidRuleException("Invalid repository rule: " + repoSpec.getRuleClass());
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }
  }

  private static final class BzlmodRepoRuleFunctionException extends SkyFunctionException {

    BzlmodRepoRuleFunctionException(InvalidRuleException e, Transience transience) {
      super(e, transience);
    }

    BzlmodRepoRuleFunctionException(NoSuchPackageException e, Transience transience) {
      super(e, transience);
    }

    BzlmodRepoRuleFunctionException(EvalException e, Transience transience) {
      super(e, transience);
    }
  }
}
