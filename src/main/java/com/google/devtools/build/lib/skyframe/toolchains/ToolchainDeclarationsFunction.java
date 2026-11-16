// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.transitions.NoConfigTransition;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.rules.platform.ConstraintValueRule;
import com.google.devtools.build.lib.rules.platform.ToolchainRule;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil.InvalidTargetPatternException;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.RegisteredToolchainsFunctionException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} that expands all registered toolchains and analyzes those declarations that
 * don't depend on the target configuration.
 *
 * <p>A native {@code toolchain} target yields the same {@link DeclaredToolchainInfo} in every
 * configuration if neither it nor its {@code toolchain_type} use {@code select()} or have
 * dependencies that are analyzed in their configuration, other than the native {@code
 * toolchain_type} and {@code constraint_value} targets that its {@code toolchain_type}, {@code
 * exec_compatible_with} and {@code target_compatible_with} attributes point directly at: {@code
 * target_settings} is not a dependency and is only analyzed, validated and evaluated by {@link
 * RegisteredToolchainsFunction} in each target configuration. Such targets are analyzed once
 * without a configuration, which reduces the number of configured toolchain targets from {@code
 * O(toolchains * configurations)} to {@code O(toolchains)}. All other targets are left to {@link
 * RegisteredToolchainsFunction} to analyze in each target configuration.
 */
public class ToolchainDeclarationsFunction implements SkyFunction {

  private static final String TOOLCHAIN_TYPE_RULE_NAME = "toolchain_type";

  /** The attributes of {@code toolchain} whose dependencies are checked individually. */
  private static final ImmutableSet<String> TOOLCHAIN_DEP_ATTRS =
      ImmutableSet.of(
          ToolchainRule.TOOLCHAIN_TYPE_ATTR,
          ToolchainRule.EXEC_COMPATIBLE_WITH_ATTR,
          ToolchainRule.TARGET_COMPATIBLE_WITH_ATTR);

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws RegisteredToolchainsFunctionException, InterruptedException {
    ToolchainDeclarationsValue.Key key = (ToolchainDeclarationsValue.Key) skyKey;
    RepositoryMappingValue mainRepoMapping =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (mainRepoMapping == null) {
      return null;
    }

    TargetPattern.Parser mainRepoParser =
        new TargetPattern.Parser(
            PathFragment.EMPTY_FRAGMENT, RepositoryName.MAIN, mainRepoMapping.repositoryMapping());
    ImmutableList.Builder<SignedTargetPattern> targetPatternBuilder = new ImmutableList.Builder<>();

    // Get the toolchains from the configuration.
    // Reverse the list so the last one defined takes precedences.
    try {
      targetPatternBuilder.addAll(
          TargetPatternUtil.parseAllSigned(key.extraToolchains().reverse(), mainRepoParser));
    } catch (InvalidTargetPatternException e) {
      throw new RegisteredToolchainsFunctionException(
          new InvalidToolchainLabelException(e), Transience.PERSISTENT);
    }

    // Get registered toolchains from the external dep graph.
    ImmutableList<TargetPattern> bzlmodToolchains = getBzlmodToolchains(env);
    if (bzlmodToolchains == null) {
      return null;
    }
    targetPatternBuilder.addAll(TargetPatternUtil.toSigned(bzlmodToolchains));

    // Expand target patterns.
    ImmutableSet<Label> toolchainLabels;
    try {
      toolchainLabels =
          TargetPatternUtil.expandTargetPatterns(
              env, targetPatternBuilder.build(), FilteringPolicies.ruleTypeExplicit("toolchain"));
      if (env.valuesMissing()) {
        return null;
      }
    } catch (TargetPatternUtil.InvalidTargetPatternException e) {
      throw new RegisteredToolchainsFunctionException(
          new InvalidToolchainLabelException(e), Transience.PERSISTENT);
    }

    ImmutableSet<Label> configIndependentLabels =
        findConfigIndependentDeclarations(env, toolchainLabels);
    if (configIndependentLabels == null) {
      return null;
    }

    Map<Label, DeclaredToolchainInfo> infos =
        RegisteredToolchainsFunction.configureRegisteredToolchains(
            env, BuildConfigurationKey.create(key.noConfigOptions()), configIndependentLabels);
    if (infos == null) {
      return null;
    }

    return new ToolchainDeclarationsValue(toolchainLabels.asList(), ImmutableMap.copyOf(infos));
  }

  @Nullable
  private static ImmutableList<TargetPattern> getBzlmodToolchains(Environment env)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    BazelDepGraphValue bazelDepGraphValue =
        (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (bazelDepGraphValue == null) {
      return null;
    }
    ImmutableList.Builder<TargetPattern> toolchains = ImmutableList.builder();
    for (Module module : bazelDepGraphValue.getDepGraph().values()) {
      if (module.getToolchainsToRegister().isEmpty()) {
        continue;
      }
      RepositoryName repoName =
          bazelDepGraphValue.getCanonicalRepoNameLookup().inverse().get(module.getKey());
      RepositoryMappingValue repoMapping =
          (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(repoName));
      if (repoMapping == null) {
        continue;
      }
      TargetPattern.Parser parser =
          new TargetPattern.Parser(
              PathFragment.EMPTY_FRAGMENT, repoName, repoMapping.repositoryMapping());
      for (String pattern : module.getToolchainsToRegister()) {
        try {
          toolchains.add(parser.parse(pattern));
        } catch (TargetParsingException e) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(pattern, e), Transience.PERSISTENT);
        }
      }
    }
    return env.valuesMissing() ? null : toolchains.build();
  }

  /**
   * Returns the subset of {@code labels} that can be analyzed without a target configuration, or
   * {@code null} if Skyframe values are missing.
   *
   * <p>Anything unexpected excludes a label so that analysis in the target configuration reports
   * the error. The only exception is a dependency in a package that can't be loaded, which fails
   * the analysis in every configuration and is thus reported right away.
   */
  @Nullable
  private static ImmutableSet<Label> findConfigIndependentDeclarations(
      Environment env, ImmutableSet<Label> labels)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    // These packages have already been loaded during target pattern expansion.
    SkyframeLookupResult toolchainPackages =
        env.getValuesAndExceptions(
            labels.stream().map(Label::getPackageIdentifier).collect(toImmutableSet()));
    if (env.valuesMissing()) {
      return null;
    }

    // Collect the candidates along with their toolchain type and constraints. Aliases and other
    // rules are configuration-dependent in general as they can contain a select(), which can't be
    // resolved without a configuration.
    Map<Label, ImmutableList<Label>> candidates = new LinkedHashMap<>();
    for (Label label : labels) {
      if (getTarget(toolchainPackages, label) instanceof Rule rule
          && isNativeRule(rule, ToolchainRule.RULE_NAME)
          && !dependsOnConfiguration(rule, TOOLCHAIN_DEP_ATTRS)) {
        RawAttributeMapper attributes = RawAttributeMapper.of(rule);
        ImmutableList.Builder<Label> deps = ImmutableList.builder();
        for (String attribute : TOOLCHAIN_DEP_ATTRS) {
          attributes.visitLabels(attribute, deps::add);
        }
        candidates.put(label, deps.build());
      }
    }

    SkyframeLookupResult depPackages =
        env.getValuesAndExceptions(
            candidates.values().stream()
                .flatMap(ImmutableList::stream)
                .map(Label::getPackageIdentifier)
                .collect(toImmutableSet()));
    // Report an error for any toolchain before returning due to missing values so that it is
    // attributed to the toolchain even if the loading of other packages is aborted.
    for (Map.Entry<Label, ImmutableList<Label>> candidate : candidates.entrySet()) {
      for (Label dep : candidate.getValue()) {
        try {
          var unused =
              depPackages.getOrThrow(dep.getPackageIdentifier(), NoSuchPackageException.class);
        } catch (NoSuchPackageException e) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(candidate.getKey(), e), Transience.PERSISTENT);
        }
      }
    }
    if (env.valuesMissing()) {
      return null;
    }

    ImmutableSet.Builder<Label> result = ImmutableSet.builder();
    for (Map.Entry<Label, ImmutableList<Label>> candidate : candidates.entrySet()) {
      if (candidate.getValue().stream()
          .allMatch(dep -> isConfigIndependentDep(getTarget(depPackages, dep)))) {
        result.add(candidate.getKey());
      }
    }
    return result.build();
  }

  /**
   * Returns whether the given dependency of a toolchain that is analyzed without a configuration
   * is a toolchain type or constraint value that is analyzed in the same way as in any other
   * configuration.
   */
  private static boolean isConfigIndependentDep(@Nullable Target dep) {
    return dep instanceof Rule rule
        // Constraint values are always analyzed without a configuration, whereas toolchain types
        // are analyzed in the configuration of the toolchain.
        && (isNativeRule(rule, ConstraintValueRule.RULE_NAME)
            || (isNativeRule(rule, TOOLCHAIN_TYPE_RULE_NAME)
                && !dependsOnConfiguration(rule, ImmutableSet.of())));
  }

  /**
   * Returns whether the analysis of the given rule may depend on its configuration through a
   * {@code select()} or a dependency that is analyzed in its configuration, ignoring the
   * dependencies in the given attributes.
   */
  private static boolean dependsOnConfiguration(Rule rule, ImmutableSet<String> ignoredAttributes) {
    RawAttributeMapper attributes = RawAttributeMapper.of(rule);
    for (Attribute attribute : rule.getRuleClassObject().getAttributeProvider().getAttributes()) {
      if (attributes.isConfigurable(attribute.getName())) {
        return true;
      }
    }
    boolean[] hasConfiguredDep = {false};
    attributes.visitAllLabels(
        (attribute, label) ->
            hasConfiguredDep[0] |=
                !ignoredAttributes.contains(attribute.getName())
                    && !NoConfigTransition.isInstance(attribute.getTransitionFactory()));
    return hasConfiguredDep[0];
  }

  @Nullable
  private static Target getTarget(SkyframeLookupResult packages, Label label) {
    return ((PackageValue) packages.get(label.getPackageIdentifier()))
        .getPackage()
        .getTargetOrNull(label.getName());
  }

  private static boolean isNativeRule(Rule rule, String ruleClassName) {
    return !rule.getRuleClassObject().isStarlark() && rule.getRuleClass().equals(ruleClassName);
  }
}
