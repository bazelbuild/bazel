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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.rules.platform.ConstraintValueRule;
import com.google.devtools.build.lib.rules.platform.ToolchainRule;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil.InvalidTargetPatternException;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.RegisteredToolchainsFunctionException;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainDeclarationsValue.Declaration;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} that expands all registered toolchains and analyzes those declarations that
 * don't depend on the target configuration.
 *
 * <p>A native {@code toolchain} target whose attributes contain no {@code select()} and whose
 * {@code toolchain_type}, {@code exec_compatible_with} and {@code target_compatible_with}
 * attributes point directly at native {@code toolchain_type} and {@code constraint_value} targets
 * yields the same {@link DeclaredToolchainInfo} in every configuration: {@code target_settings} is
 * not a dependency and is only analyzed, validated and evaluated by {@link
 * RegisteredToolchainsFunction} in each target configuration. Such targets are analyzed once
 * without a configuration, which reduces the number of configured toolchain targets from {@code
 * O(toolchains * configurations)} to {@code O(toolchains)}. All other targets are left to {@link
 * RegisteredToolchainsFunction} to analyze in each target configuration.
 */
public class ToolchainDeclarationsFunction implements SkyFunction {

  private static final String TOOLCHAIN_TYPE_RULE_NAME = "toolchain_type";

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
        analyzeWithoutConfiguration(env, configIndependentLabels);
    if (infos == null) {
      return null;
    }

    return new ToolchainDeclarationsValue(
        toolchainLabels.stream()
            .map(label -> new Declaration(label, infos.get(label)))
            .collect(ImmutableList.toImmutableList()));
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
      TargetPattern.Parser parser =
          new TargetPattern.Parser(
              PathFragment.EMPTY_FRAGMENT,
              bazelDepGraphValue.getCanonicalRepoNameLookup().inverse().get(module.getKey()),
              bazelDepGraphValue.getFullRepoMapping(module.getKey()));
      for (String pattern : module.getToolchainsToRegister()) {
        try {
          toolchains.add(parser.parse(pattern));
        } catch (TargetParsingException e) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(pattern, e), Transience.PERSISTENT);
        }
      }
    }
    return toolchains.build();
  }

  /**
   * Returns the subset of {@code labels} that can be analyzed without a target configuration, or
   * {@code null} if Skyframe values are missing.
   *
   * <p>Anything unexpected, including loading errors, excludes a label so that analysis in the
   * target configuration reports errors exactly as before.
   */
  @Nullable
  private static ImmutableSet<Label> findConfigIndependentDeclarations(
      Environment env, ImmutableSet<Label> labels) throws InterruptedException {
    Map<PackageIdentifier, PackageValue> toolchainPackages = getPackages(env, labels);
    if (toolchainPackages == null) {
      return null;
    }

    // Collect the candidates along with the rule class each of their dependencies must have.
    // Aliases and other rules are configuration-dependent in general as they can contain a
    // select(), which can't be resolved without a configuration.
    Map<Label, ImmutableMap<Label, String>> candidates = new HashMap<>();
    for (Label label : labels) {
      if (!(getTarget(toolchainPackages, label) instanceof Rule rule)
          || !isNativeRule(rule, ToolchainRule.RULE_NAME)
          || hasSelect(rule)) {
        continue;
      }
      RawAttributeMapper attributes = RawAttributeMapper.of(rule);
      Label toolchainType = attributes.get(ToolchainRule.TOOLCHAIN_TYPE_ATTR, BuildType.LABEL);
      if (toolchainType == null) {
        continue;
      }
      Map<Label, String> deps = new HashMap<>();
      deps.put(toolchainType, TOOLCHAIN_TYPE_RULE_NAME);
      for (Label constraint :
          Iterables.concat(
              attributes.get(ToolchainRule.EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST),
              attributes.get(ToolchainRule.TARGET_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST))) {
        deps.put(constraint, ConstraintValueRule.RULE_NAME);
      }
      candidates.put(label, ImmutableMap.copyOf(deps));
    }

    Map<PackageIdentifier, PackageValue> depPackages =
        getPackages(
            env,
            candidates.values().stream()
                .flatMap(deps -> deps.keySet().stream())
                .collect(toImmutableSet()));
    if (depPackages == null) {
      return null;
    }

    ImmutableSet.Builder<Label> result = ImmutableSet.builder();
    for (Label label : labels) {
      ImmutableMap<Label, String> deps = candidates.get(label);
      if (deps != null
          && deps.entrySet().stream()
              .allMatch(
                  dep ->
                      getTarget(depPackages, dep.getKey()) instanceof Rule depRule
                          && isNativeRule(depRule, dep.getValue()))) {
        result.add(label);
      }
    }
    return result.build();
  }

  @Nullable
  private static Map<Label, DeclaredToolchainInfo> analyzeWithoutConfiguration(
      Environment env, ImmutableSet<Label> labels)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    BuildConfigurationKey noConfig = BuildConfigurationKey.create(CommonOptions.EMPTY_OPTIONS);
    ImmutableSet<ConfiguredTargetKey> keys =
        labels.stream()
            .map(
                label ->
                    ConfiguredTargetKey.builder()
                        .setLabel(label)
                        .setConfigurationKey(noConfig)
                        .build())
            .collect(toImmutableSet());
    SkyframeLookupResult values = env.getValuesAndExceptions(keys);
    Map<Label, DeclaredToolchainInfo> infos = new HashMap<>();
    boolean valuesMissing = false;
    for (ConfiguredTargetKey key : keys) {
      Label toolchainLabel = key.getLabel();
      try {
        SkyValue value = values.getOrThrow(key, ConfiguredValueCreationException.class);
        if (value == null) {
          valuesMissing = true;
          continue;
        }
        ConfiguredTarget target = ((ConfiguredTargetValue) value).getConfiguredTarget();
        DeclaredToolchainInfo toolchainInfo = PlatformProviderUtils.declaredToolchainInfo(target);
        if (toolchainInfo == null) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(toolchainLabel), Transience.PERSISTENT);
        }
        infos.put(toolchainLabel, toolchainInfo);
      } catch (ConfiguredValueCreationException e) {
        throw new RegisteredToolchainsFunctionException(
            new InvalidToolchainLabelException(toolchainLabel, e), Transience.PERSISTENT);
      }
    }
    return valuesMissing ? null : infos;
  }

  @Nullable
  private static Map<PackageIdentifier, PackageValue> getPackages(
      Environment env, ImmutableSet<Label> labels) throws InterruptedException {
    ImmutableSet<PackageIdentifier> packageIds =
        labels.stream().map(Label::getPackageIdentifier).collect(toImmutableSet());
    SkyframeLookupResult values = env.getValuesAndExceptions(packageIds);
    if (env.valuesMissing()) {
      return null;
    }
    Map<PackageIdentifier, PackageValue> packages = new HashMap<>();
    for (PackageIdentifier packageId : packageIds) {
      try {
        PackageValue value =
            (PackageValue) values.getOrThrow(packageId, NoSuchPackageException.class);
        if (value != null) {
          packages.put(packageId, value);
        }
      } catch (NoSuchPackageException e) {
        // Leave the error to be reported by analysis in the target configuration.
      }
    }
    return packages;
  }

  @Nullable
  private static Target getTarget(Map<PackageIdentifier, PackageValue> packages, Label label) {
    PackageValue value = packages.get(label.getPackageIdentifier());
    return value == null ? null : value.getPackage().getTargetOrNull(label.getName());
  }

  private static boolean isNativeRule(Rule rule, String ruleClassName) {
    return !rule.getRuleClassObject().isStarlark() && rule.getRuleClass().equals(ruleClassName);
  }

  private static boolean hasSelect(Rule rule) {
    RawAttributeMapper attributes = RawAttributeMapper.of(rule);
    for (Attribute attribute : rule.getRuleClassObject().getAttributeProvider().getAttributes()) {
      if (attributes.isConfigurable(attribute.getName())) {
        return true;
      }
    }
    return false;
  }
}
