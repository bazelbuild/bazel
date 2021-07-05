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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleHelper;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.RuleFactory;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;
import net.starlark.java.syntax.Location;

/**
 * Looks up the {@link RepoSpec} of a given repository name and create its repository rule instance.
 */
public final class BzlmodRepoRuleFunction implements SkyFunction {

  private static final String TOOLS_REPO = "bazel_tools";

  private final PackageFactory packageFactory;
  private final RuleClassProvider ruleClassProvider;
  private final BlazeDirectories directories;
  private final BzlmodRepoRuleHelper bzlmodRepoRuleHelper;

  public BzlmodRepoRuleFunction(
      PackageFactory packageFactory,
      RuleClassProvider ruleClassProvider,
      BlazeDirectories directories,
      BzlmodRepoRuleHelper bzlmodRepoRuleHelper) {
    this.packageFactory = packageFactory;
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

    String repositoryName = ((BzlmodRepoRuleValue.Key) skyKey).argument();

    // @bazel_tools is a special repo that we pull from the extracted install dir.
    if (repositoryName.equals(TOOLS_REPO)) {
      RepoSpec repoSpec =
          RepoSpec.builder()
              .setRuleClassName("local_repository")
              .setAttributes(
                  ImmutableMap.of(
                      "name",
                      "bazel_tools",
                      "path",
                      directories
                          .getEmbeddedBinariesRoot()
                          .getChild("embedded_tools")
                          .getPathString()))
              .build();
      return createRuleFromSpec(repoSpec, starlarkSemantics, env);
    }

    Optional<RepoSpec> result = bzlmodRepoRuleHelper.getRepoSpec(env, repositoryName);
    if (env.valuesMissing()) {
      return null;
    }
    if (result.isPresent()) {
      return createRuleFromSpec(result.get(), starlarkSemantics, env);
    }

    return BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE;
  }

  private BzlmodRepoRuleValue createRuleFromSpec(
      RepoSpec repoSpec, StarlarkSemantics starlarkSemantics, Environment env)
      throws BzlmodRepoRuleFunctionException, InterruptedException {
    if (repoSpec.isNativeRepoRule()) {
      return createNativeRepoRule(repoSpec, starlarkSemantics, env);
    }
    // TODO(pcloudy): Implement creating starlark repository rule
    return BzlmodRepoRuleValue.REPO_RULE_NOT_FOUND_VALUE;
  }

  private BzlmodRepoRuleValue createNativeRepoRule(
      RepoSpec repoSpec, StarlarkSemantics semantics, Environment env)
      throws InterruptedException, BzlmodRepoRuleFunctionException {
    RuleClass ruleClass = ruleClassProvider.getRuleClassMap().get(repoSpec.ruleClassName());
    BuildLangTypedAttributeValuesMap attributeValues =
        new BuildLangTypedAttributeValuesMap(repoSpec.attributes());
    ImmutableList.Builder<CallStackEntry> callStack = ImmutableList.builder();
    callStack.add(new CallStackEntry("BzlmodRepoRuleFunction.getNativeRepoRule", Location.BUILTIN));
    // TODO(bazel-team): Don't use the {@link Rule} class for repository rule.
    // Currently, the repository rule is represented with the {@link Rule} class that's designed
    // for build rules. Therefore, we have to create a package instance for it, which doesn't make
    // sense. We should migrate away from this implementation so that we don't refer to any build
    // rule specific things in repository rule.
    Rule rule;
    Package.Builder pkg = createExternalPackageBuilder(semantics);
    try {
      rule =
          RuleFactory.createRule(
              pkg, ruleClass, attributeValues, env.getListener(), semantics, callStack.build());
      // We need to actually build the package so that the rule has the correct package reference.
      pkg.build();
    } catch (InvalidRuleException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    } catch (NoSuchPackageException e) {
      throw new BzlmodRepoRuleFunctionException(e, Transience.PERSISTENT);
    }
    return new BzlmodRepoRuleValue(rule);
  }

  /**
   * Create the external package builder, which is only for the convenience of creating repository
   * rules.
   */
  private Package.Builder createExternalPackageBuilder(StarlarkSemantics semantics) {
    RootedPath bzlmodFile =
        RootedPath.toRootedPath(
            Root.fromPath(directories.getWorkspace()), LabelConstants.MODULE_DOT_BAZEL_FILE_NAME);

    return packageFactory.newExternalPackageBuilder(
        bzlmodFile, ruleClassProvider.getRunfilesPrefix(), semantics);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    // default behaviour
    return null;
  }

  private static final class BzlmodRepoRuleFunctionException extends SkyFunctionException {

    BzlmodRepoRuleFunctionException(InvalidRuleException e, Transience transience) {
      super(e, transience);
    }

    BzlmodRepoRuleFunctionException(NoSuchPackageException e, Transience transience) {
      super(e, transience);
    }
  }
}
