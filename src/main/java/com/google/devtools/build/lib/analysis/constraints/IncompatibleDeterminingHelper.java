// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.constraints;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.IncompatiblePlatformProvider;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.test.TestActionBuilder;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageGroupsRuleVisibility;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContext;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Helpers for creating configured targets that are incompatible.
 *
 * A target is considered incompatible if any of the following applies:
 * <ol>
 *   <li>The target's <code>target_compatible_with</code> attribute specifies a constraint that is
 *       not present in the target platform.
 *   <li>One or more of the target's dependencies is incompatible.
 * </ol>
 *
 * The intent of these helpers is that they get called as early in the analysis phase as possible.
 * Doing this as early as possible allows us to skip analysing unused dependencies and ignore unused
 * toolchains.
 *
 * See https://bazel.build/docs/platforms#skipping-incompatible-targets for more information on
 * incompatible target skipping.
 */
public class IncompatibleDeterminingHelper {
  /**
   * Creates an incompatible configured target if it is "directly incompatible".
   *
   * In other words, this function checks if a target is incompatible because of its
   * "target_compatible_with" attribute.
   *
   * After this function returns, the caller must immediatly check if "env.valuesMissing()" returns
   * true. If that check succeeds, the return value of this function is guaranteed to be null. The
   * caller must request a skyframe restart. If the check fails, then the return value is either
   * null or an incompatible configured target. If null is returned, the target is not directly
   * incompatible and analysis of this target should proceed.
   */
  @Nullable
  public static RuleConfiguredTargetValue createDirectlyIncompatibleTarget(
      TargetAndConfiguration targetAndConfiguration,
      ConfigConditions configConditions,
      Environment env,
      ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts,
      NestedSetBuilder<Package> transitivePackagesForPackageRootResolution)
      throws InterruptedException {
    Target target = targetAndConfiguration.getTarget();
    Rule rule = target.getAssociatedRule();
    PlatformInfo platformInfo =
        unloadedToolchainContexts != null ? unloadedToolchainContexts.getTargetPlatform() : null;

    if (rule == null || rule.getRuleClass().equals("toolchain") || platformInfo == null) {
      return null;
    }

    // Retrieve the label list for the target_compatible_with attribute.
    BuildConfigurationValue configuration = targetAndConfiguration.getConfiguration();
    ConfiguredAttributeMapper attrs =
        ConfiguredAttributeMapper.of(rule, configConditions.asProviders(), configuration);
    if (!attrs.has("target_compatible_with", BuildType.LABEL_LIST)) {
      return null;
    }

    // Resolve the constraint labels.
    List<Label> labels = attrs.get("target_compatible_with", BuildType.LABEL_LIST);
    ImmutableList<ConfiguredTargetKey> constraintKeys =
        labels.stream()
            .map(
                label ->
                    Dependency.builder()
                        .setLabel(label)
                        .setConfiguration(configuration)
                        .build()
                        .getConfiguredTargetKey())
            .collect(toImmutableList());

    SkyframeLookupResult constraintValues = env.getValuesAndExceptions(constraintKeys);
    if (env.valuesMissing()) {
      return null;
    }

    // Find the constraints that don't satisfy the target platform.
    ImmutableList<ConstraintValueInfo> invalidConstraintValues =
        constraintKeys.stream()
            .map(key -> (ConfiguredTargetValue) constraintValues.get(key))
            .map(ctv -> PlatformProviderUtils.constraintValue(ctv.getConfiguredTarget()))
            .filter(cv -> cv != null && !platformInfo.constraints().hasConstraintValue(cv))
            .collect(toImmutableList());
    if (invalidConstraintValues.isEmpty()) {
      return null;
    }

    return createIncompatibleRuleConfiguredTarget(
            target,
            configuration,
            configConditions,
            IncompatiblePlatformProvider.incompatibleDueToConstraints(
                platformInfo.label(), invalidConstraintValues),
            rule.getRuleClass(),
            transitivePackagesForPackageRootResolution);
  }

  /**
   * Creates an incompatible target if it is "indirectly incompatible".
   *
   * In other words, this function checks if a target is incompatible because of one of its
   * dependencies. If a dependency is incompatible, then this target is also incompatible.
   *
   * This function returns either null or an incompatible configured target. If null is returned,
   * the target is not indirectly incompatible and analysis of this target should proceed.
   */
  @Nullable
  public static RuleConfiguredTargetValue createIndirectlyIncompatibleTarget(
      TargetAndConfiguration targetAndConfiguration,
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depValueMap,
      ConfigConditions configConditions,
      ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts,
      NestedSetBuilder<Package> transitivePackagesForPackageRootResolution) {
    Target target = targetAndConfiguration.getTarget();
    Rule rule = target.getAssociatedRule();

    if (rule == null || rule.getRuleClass().equals("toolchain")) {
      return null;
    }

    // Find all the incompatible dependencies.
    ImmutableList<ConfiguredTarget> incompatibleDeps =
        depValueMap.values().stream()
            .map(ConfiguredTargetAndData::getConfiguredTarget)
            .filter(
                dep -> RuleContextConstraintSemantics.checkForIncompatibility(dep).isIncompatible())
            .collect(toImmutableList());
    if (incompatibleDeps.isEmpty()) {
      return null;
    }

    BuildConfigurationValue configuration = targetAndConfiguration.getConfiguration();
    PlatformInfo platformInfo =
        unloadedToolchainContexts != null ? unloadedToolchainContexts.getTargetPlatform() : null;
    Label platformLabel = platformInfo != null ? platformInfo.label() : null;
    return createIncompatibleRuleConfiguredTarget(
        target,
        configuration,
        configConditions,
        IncompatiblePlatformProvider.incompatibleDueToTargets(platformLabel, incompatibleDeps),
        rule.getRuleClass(),
        transitivePackagesForPackageRootResolution);
  }

  /**
   * Creates an incompatible target.
   */
  private static RuleConfiguredTargetValue createIncompatibleRuleConfiguredTarget(
      Target target,
      BuildConfigurationValue configuration,
      ConfigConditions configConditions,
      IncompatiblePlatformProvider incompatiblePlatformProvider,
      String ruleClassString,
      NestedSetBuilder<Package> transitivePackagesForPackageRootResolution) {
    // Create dummy instances of the necessary data for a configured target. None of this data will
    // actually be used because actions associated with incompatible targets must not be evaluated.
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    FileProvider fileProvider = new FileProvider(filesToBuild);
    FilesToRunProvider filesToRunProvider = new FilesToRunProvider(filesToBuild, null, null);

    TransitiveInfoProviderMapBuilder providerBuilder =
        new TransitiveInfoProviderMapBuilder()
            .put(incompatiblePlatformProvider)
            .add(RunfilesProvider.simple(Runfiles.EMPTY))
            .add(fileProvider)
            .add(filesToRunProvider);
    if (configuration.hasFragment(TestConfiguration.class)) {
      // Create a dummy TestProvider instance so that other parts of the code base stay happy. Even
      // though this test will never execute, some code still expects the provider.
      TestProvider.TestParams testParams = TestActionBuilder.createEmptyTestParams();
      providerBuilder.put(TestProvider.class, new TestProvider(testParams));
    }

    RuleConfiguredTarget configuredTarget =
        new RuleConfiguredTarget(
            target.getLabel(),
            configuration.getKey(),
            convertVisibility(target),
            providerBuilder.build(),
            configConditions.asProviders(),
            ruleClassString);
    return new RuleConfiguredTargetValue(
        configuredTarget,
        transitivePackagesForPackageRootResolution == null
            ? null
            : transitivePackagesForPackageRootResolution.build());
  }

  // Taken from ConfiguredTargetFactory.convertVisibility().
  // TODO(phil): Figure out how to de-duplicate somehow.
  private static NestedSet<PackageGroupContents> convertVisibility(Target target) {
    RuleVisibility ruleVisibility = target.getVisibility();
    if (ruleVisibility instanceof ConstantRuleVisibility) {
      return ((ConstantRuleVisibility) ruleVisibility).isPubliclyVisible()
          ? NestedSetBuilder.create(
              Order.STABLE_ORDER,
              PackageGroupContents.create(ImmutableList.of(PackageSpecification.everything())))
          : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    } else if (ruleVisibility instanceof PackageGroupsRuleVisibility) {
      throw new IllegalStateException("unsupported PackageGroupsRuleVisibility");
    } else {
      throw new IllegalStateException("unknown visibility");
    }
  }
}
