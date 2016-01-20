// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Joiner;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageGroupsRuleVisibility;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.SkylarkRuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.rules.fileset.FilesetProvider;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * This class creates {@link ConfiguredTarget} instances using a given {@link
 * ConfiguredRuleClassProvider}.
 */
@ThreadSafe
public final class ConfiguredTargetFactory {
  // This class is not meant to be outside of the analysis phase machinery and is only public
  // in order to be accessible from the .view.skyframe package.

  private final ConfiguredRuleClassProvider ruleClassProvider;

  public ConfiguredTargetFactory(ConfiguredRuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = ruleClassProvider;
  }

  /**
   * Returns the visibility of the given target. Errors during package group resolution are reported
   * to the {@code AnalysisEnvironment}.
   */
  private NestedSet<PackageSpecification> convertVisibility(
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap, EventHandler reporter,
      Target target, BuildConfiguration packageGroupConfiguration) {
    RuleVisibility ruleVisibility = target.getVisibility();
    if (ruleVisibility instanceof ConstantRuleVisibility) {
      return ((ConstantRuleVisibility) ruleVisibility).isPubliclyVisible()
          ? NestedSetBuilder.<PackageSpecification>create(
              Order.STABLE_ORDER, PackageSpecification.EVERYTHING)
          : NestedSetBuilder.<PackageSpecification>emptySet(Order.STABLE_ORDER);
    } else if (ruleVisibility instanceof PackageGroupsRuleVisibility) {
      PackageGroupsRuleVisibility packageGroupsVisibility =
          (PackageGroupsRuleVisibility) ruleVisibility;

      NestedSetBuilder<PackageSpecification> packageSpecifications =
          NestedSetBuilder.stableOrder();
      for (Label groupLabel : packageGroupsVisibility.getPackageGroups()) {
        // PackageGroupsConfiguredTargets are always in the package-group configuration.
        ConfiguredTarget group =
            findPrerequisite(prerequisiteMap, groupLabel, packageGroupConfiguration);
        PackageSpecificationProvider provider = null;
        // group == null can only happen if the package group list comes
        // from a default_visibility attribute, because in every other case,
        // this missing link is caught during transitive closure visitation or
        // if the RuleConfiguredTargetGraph threw out a visibility edge
        // because if would have caused a cycle. The filtering should be done
        // in a single place, ConfiguredTargetGraph, but for now, this is the
        // minimally invasive way of providing a sane error message in case a
        // cycle is created by a visibility attribute.
        if (group != null) {
          provider = group.getProvider(PackageSpecificationProvider.class);
        }
        if (provider != null) {
          packageSpecifications.addTransitive(provider.getPackageSpecifications());
        } else {
          reporter.handle(Event.error(target.getLocation(),
              String.format("Label '%s' does not refer to a package group", groupLabel)));
        }
      }

      packageSpecifications.addAll(packageGroupsVisibility.getDirectPackages());
      return packageSpecifications.build();
    } else {
      throw new IllegalStateException("unknown visibility");
    }
  }

  private ConfiguredTarget findPrerequisite(
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap, Label label,
      BuildConfiguration config) {
    for (ConfiguredTarget prerequisite : prerequisiteMap.get(null)) {
      if (prerequisite.getLabel().equals(label) && (prerequisite.getConfiguration() == config)) {
        return prerequisite;
      }
    }
    return null;
  }

  private Artifact getOutputArtifact(OutputFile outputFile, BuildConfiguration configuration,
      boolean isFileset, ArtifactFactory artifactFactory) {
    Rule rule = outputFile.getAssociatedRule();
    Root root = rule.hasBinaryOutput()
        ? configuration.getBinDirectory()
        : configuration.getGenfilesDirectory();
    ArtifactOwner owner =
        new ConfiguredTargetKey(rule.getLabel(), configuration.getArtifactOwnerConfiguration());
    PathFragment rootRelativePath =
        outputFile.getLabel().getPackageIdentifier().getPathFragment().getRelative(
            outputFile.getLabel().getName());
    Artifact result = isFileset
        ? artifactFactory.getFilesetArtifact(rootRelativePath, root, owner)
        : artifactFactory.getDerivedArtifact(rootRelativePath, root, owner);
    // The associated rule should have created the artifact.
    Preconditions.checkNotNull(result, "no artifact for %s", rootRelativePath);
    return result;
  }

  /**
   * Invokes the appropriate constructor to create a {@link ConfiguredTarget} instance.
   * <p>For use in {@code ConfiguredTargetFunction}.
   *
   * <p>Returns null if Skyframe deps are missing or upon certain errors.
   */
  @Nullable
  public final ConfiguredTarget createConfiguredTarget(AnalysisEnvironment analysisEnvironment,
      ArtifactFactory artifactFactory, Target target, BuildConfiguration config,
      BuildConfiguration hostConfig, ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      Set<ConfigMatchingProvider> configConditions)
      throws InterruptedException {
    if (target instanceof Rule) {
      return createRule(analysisEnvironment, (Rule) target, config, hostConfig,
          prerequisiteMap, configConditions);
    }

    // Visibility, like all package groups, doesn't have a configuration
    NestedSet<PackageSpecification> visibility = convertVisibility(
        prerequisiteMap, analysisEnvironment.getEventHandler(), target, null);
    TargetContext targetContext = new TargetContext(analysisEnvironment, target, config,
        prerequisiteMap.get(null), visibility);
    if (target instanceof OutputFile) {
      OutputFile outputFile = (OutputFile) target;
      boolean isFileset = outputFile.getGeneratingRule().getRuleClass().equals("Fileset");
      Artifact artifact = getOutputArtifact(outputFile, config, isFileset, artifactFactory);
      TransitiveInfoCollection rule = targetContext.findDirectPrerequisite(
          outputFile.getGeneratingRule().getLabel(), config);
      if (isFileset) {
        return new FilesetOutputConfiguredTarget(
            targetContext,
            outputFile,
            rule,
            artifact,
            rule.getProvider(FilesetProvider.class).getTraversals());
      } else {
        return new OutputFileConfiguredTarget(targetContext, outputFile, rule, artifact);
      }
    } else if (target instanceof InputFile) {
      InputFile inputFile = (InputFile) target;
      Artifact artifact = artifactFactory.getSourceArtifact(
          inputFile.getExecPath(),
          Root.asSourceRoot(inputFile.getPackage().getSourceRoot()),
          new ConfiguredTargetKey(target.getLabel(), config));

      return new InputFileConfiguredTarget(targetContext, inputFile, artifact);
    } else if (target instanceof PackageGroup) {
      PackageGroup packageGroup = (PackageGroup) target;
      return new PackageGroupConfiguredTarget(targetContext, packageGroup);
    } else if (target instanceof EnvironmentGroup) {
      return new EnvironmentGroupConfiguredTarget(targetContext, (EnvironmentGroup) target);
    } else {
      throw new AssertionError("Unexpected target class: " + target.getClass().getName());
    }
  }

  /**
   * Factory method: constructs a RuleConfiguredTarget of the appropriate class, based on the rule
   * class. May return null if an error occurred.
   */
  @Nullable
  private ConfiguredTarget createRule(
      AnalysisEnvironment env, Rule rule, BuildConfiguration configuration,
      BuildConfiguration hostConfiguration,
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      Set<ConfigMatchingProvider> configConditions) throws InterruptedException {
    // Visibility computation and checking is done for every rule.
    RuleContext ruleContext = new RuleContext.Builder(env, rule, null,
        configuration, hostConfiguration,
        ruleClassProvider.getPrerequisiteValidator(),
        rule.getRuleClassObject().getConfigurationFragmentPolicy())
        .setVisibility(convertVisibility(prerequisiteMap, env.getEventHandler(), rule, null))
        .setPrerequisites(prerequisiteMap)
        .setConfigConditions(configConditions)
        .setUniversalFragment(ruleClassProvider.getUniversalFragment())
        .build();
    if (ruleContext.hasErrors()) {
      return null;
    }
    ConfigurationFragmentPolicy configurationFragmentPolicy =
        rule.getRuleClassObject().getConfigurationFragmentPolicy();

    MissingFragmentPolicy missingFragmentPolicy =
        configurationFragmentPolicy.getMissingFragmentPolicy();
    if (missingFragmentPolicy != MissingFragmentPolicy.IGNORE
        && !configuration.hasAllFragments(
            configurationFragmentPolicy.getRequiredConfigurationFragments())) {
      if (missingFragmentPolicy == MissingFragmentPolicy.FAIL_ANALYSIS) {
        ruleContext.ruleError(missingFragmentError(ruleContext, configurationFragmentPolicy));
        return null;
      }
      // Otherwise missingFragmentPolicy == MissingFragmentPolicy.CREATE_FAIL_ACTIONS:
      return createFailConfiguredTarget(ruleContext);
    }

    if (rule.getRuleClassObject().isSkylarkExecutable()) {
      // TODO(bazel-team): maybe merge with RuleConfiguredTargetBuilder?
      return SkylarkRuleConfiguredTargetBuilder.buildRule(
          ruleContext, rule.getRuleClassObject().getConfiguredTargetFunction());
    } else {
      RuleClass.ConfiguredTargetFactory<ConfiguredTarget, RuleContext> factory =
          rule.getRuleClassObject().<ConfiguredTarget, RuleContext>getConfiguredTargetFactory();
      Preconditions.checkNotNull(factory, rule.getRuleClassObject());
      return factory.create(ruleContext);
    }
  }

  private String missingFragmentError(
      RuleContext ruleContext, ConfigurationFragmentPolicy configurationFragmentPolicy) {
    RuleClass ruleClass = ruleContext.getRule().getRuleClassObject();
    Set<Class<?>> missingFragments = new LinkedHashSet<>();
    for (Class<?> fragment : configurationFragmentPolicy.getRequiredConfigurationFragments()) {
      if (!ruleContext.getConfiguration().hasFragment(fragment.asSubclass(Fragment.class))) {
        missingFragments.add(fragment);
      }
    }
    Preconditions.checkState(!missingFragments.isEmpty());
    StringBuilder result = new StringBuilder();
    result.append("all rules of type " + ruleClass.getName() + " require the presence of ");
    List<String> names = new ArrayList<>();
    for (Class<?> fragment : missingFragments) {
      // TODO(bazel-team): Using getSimpleName here is sub-optimal, but we don't have anything
      // better right now.
      names.add(fragment.getSimpleName());
    }
    result.append("all of [");
    Joiner.on(",").appendTo(result, names);
    result.append("], but these were all disabled");
    return result.toString();
  }

  /**
   * Constructs an {@link ConfiguredAspect}. Returns null if an error occurs; in that case,
   * {@code aspectFactory} should call one of the error reporting methods of {@link RuleContext}.
   */
  public ConfiguredAspect createAspect(
      AnalysisEnvironment env,
      RuleConfiguredTarget associatedTarget,
      ConfiguredAspectFactory aspectFactory,
      Aspect aspect,
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      Set<ConfigMatchingProvider> configConditions,
      BuildConfiguration hostConfiguration)
      throws InterruptedException {
    ConfigurationFragmentPolicy aspectPolicy =
        aspect.getDefinition().getConfigurationFragmentPolicy();
    ConfigurationFragmentPolicy rulePolicy =
        ((Rule) associatedTarget.getTarget()).getRuleClassObject().getConfigurationFragmentPolicy();
    RuleContext.Builder builder = new RuleContext.Builder(env,
        associatedTarget.getTarget(),
        aspect.getAspectClass().getName(),
        associatedTarget.getConfiguration(),
        hostConfiguration,
        ruleClassProvider.getPrerequisiteValidator(),
        // TODO(mstaib): When AspectDefinition can no longer have null ConfigurationFragmentPolicy,
        // remove this conditional.
        aspectPolicy != null ? aspectPolicy : rulePolicy);
    RuleContext ruleContext =
        builder
            .setVisibility(
                convertVisibility(
                    prerequisiteMap, env.getEventHandler(), associatedTarget.getTarget(), null))
            .setPrerequisites(prerequisiteMap)
            .setAspectAttributes(aspect.getDefinition().getAttributes())
            .setConfigConditions(configConditions)
            .setUniversalFragment(ruleClassProvider.getUniversalFragment())
            .build();
    if (ruleContext.hasErrors()) {
      return null;
    }

    return aspectFactory.create(associatedTarget, ruleContext, aspect.getParameters());
  }

  /**
   * A pseudo-implementation for configured targets that creates fail actions for all declared
   * outputs, both implicit and explicit.
   */
  private static ConfiguredTarget createFailConfiguredTarget(RuleContext ruleContext) {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    if (!ruleContext.getOutputArtifacts().isEmpty()) {
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          ruleContext.getOutputArtifacts(), "Can't build this"));
    }
    builder.add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));
    return builder.build();
  }
}
