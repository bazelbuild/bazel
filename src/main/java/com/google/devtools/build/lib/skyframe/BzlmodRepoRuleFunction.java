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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleCreator;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleHelper;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionResolutionValue;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
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
import java.io.IOException;
import java.util.Optional;
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
  private final BzlmodRepoRuleHelper bzlmodRepoRuleHelper;
  /**
   * An empty repo mapping anchored to the main repo.
   *
   * <p>None of the labels present in RepoSpecs can point to any repo other than the main repo
   * or @bazel_tools, because at this point we don't know how any other repo is defined yet. The
   * RepoSpecs processed by this class can only contain labels from the MODULE.bazel file (from
   * overrides). In the future, they might contain labels from the lockfile, but those will need to
   * be canonical label literals, which bypass repo mapping anyway.
   */
  private static final RepositoryMapping EMPTY_MAIN_REPO_MAPPING =
      RepositoryMapping.create(
          ImmutableMap.of("", RepositoryName.MAIN, "bazel_tools", RepositoryName.BAZEL_TOOLS),
          RepositoryName.MAIN);

  public BzlmodRepoRuleFunction(
      RuleClassProvider ruleClassProvider,
      BlazeDirectories directories,
      BzlmodRepoRuleHelper bzlmodRepoRuleHelper) {
    this.ruleClassProvider = ruleClassProvider;
    this.directories = directories;
    this.bzlmodRepoRuleHelper = bzlmodRepoRuleHelper;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    RepositoryName repositoryName = ((BzlmodRepoRuleValue.Key) skyKey).argument();

    // Look for the repo from Bazel module generated repos.
    try {
      Optional<RepoSpec> result = bzlmodRepoRuleHelper.getRepoSpec(env, repositoryName);
      if (env.valuesMissing()) {
        return null;
      }
      if (result.isPresent()) {
        return createRuleFromSpec(result.get(), starlarkSemantics, env);
      }
    } catch (IOException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }

    // Otherwise, look for the repo from module extension evaluation results.
    ModuleExtensionResolutionValue extensionResolution =
        (ModuleExtensionResolutionValue) env.getValue(ModuleExtensionResolutionValue.KEY);
    if (extensionResolution == null) {
      return null;
    }
    Package pkg = extensionResolution.getCanonicalRepoNameToPackage().get(repositoryName);
    if (pkg != null) {
      return new BzlmodRepoRuleValue(pkg, repositoryName.getName());
    }

    return BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE;
  }

  @Nullable
  private BzlmodRepoRuleValue createRuleFromSpec(
      RepoSpec repoSpec, StarlarkSemantics starlarkSemantics, Environment env)
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
      ImmutableMap<String, Module> loadedModules = loadBzlModules(env, repoSpec.bzlFile().get());
      if (env.valuesMissing()) {
        return null;
      }
      ruleClass = getStarlarkRuleClass(repoSpec, loadedModules);
    }
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
              repoSpec.attributes());
      return new BzlmodRepoRuleValue(rule.getPackage(), rule.getName());
    } catch (InvalidRuleException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    } catch (NoSuchPackageException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    } catch (EvalException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }
  }

  /** Loads modules from the given bzl file. */
  private ImmutableMap<String, Module> loadBzlModules(Environment env, String bzlFile)
      throws InterruptedException, BzlmodRepoRuleFunctionException {
    ImmutableList<Pair<String, Location>> programLoads =
        ImmutableList.of(Pair.of(bzlFile, Location.BUILTIN));

    ImmutableList<Label> loadLabels =
        BzlLoadFunction.getLoadLabels(
            env.getListener(),
            programLoads,
            PackageIdentifier.EMPTY_PACKAGE_ID,
            EMPTY_MAIN_REPO_MAPPING);
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
    ImmutableList<BzlLoadValue.Key> keys =
        ImmutableList.of(BzlLoadValue.keyForBzlmod(loadLabels.get(0)));

    // Load the .bzl module.
    try {
      // TODO(b/22193153, wyv): Determine whether bzl-visibility should apply at all to this type of
      // bzl load. As it stands, this call checks that bzlFile is visible to package @//.
      return PackageFunction.loadBzlModules(
          env, PackageIdentifier.EMPTY_PACKAGE_ID, "Bzlmod system", programLoads, keys, null);
    } catch (NoSuchPackageException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }
  }

  private RuleClass getStarlarkRuleClass(
      RepoSpec repoSpec, ImmutableMap<String, Module> loadedModules)
      throws BzlmodRepoRuleFunctionException {
    Object object = loadedModules.get(repoSpec.bzlFile().get()).getGlobal(repoSpec.ruleClassName());
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

    BzlmodRepoRuleFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }

    BzlmodRepoRuleFunctionException(EvalException e, Transience transience) {
      super(e, transience);
    }
  }
}
