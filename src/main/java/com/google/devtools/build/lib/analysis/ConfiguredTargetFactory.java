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

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.EnvironmentGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.FilesetOutputConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.fileset.FilesetProvider;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleConfiguredTargetUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
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
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.memory.CurrentRuleTracker;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
  private NestedSet<PackageGroupContents> convertVisibility(
      OrderedSetMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      EventHandler reporter,
      Target target,
      BuildConfiguration packageGroupConfiguration) {
    RuleVisibility ruleVisibility = target.getVisibility();
    if (ruleVisibility instanceof ConstantRuleVisibility) {
      return ((ConstantRuleVisibility) ruleVisibility).isPubliclyVisible()
          ? NestedSetBuilder.create(
              Order.STABLE_ORDER,
              PackageGroupContents.create(ImmutableList.of(PackageSpecification.everything())))
          : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    } else if (ruleVisibility instanceof PackageGroupsRuleVisibility) {
      PackageGroupsRuleVisibility packageGroupsVisibility =
          (PackageGroupsRuleVisibility) ruleVisibility;

      NestedSetBuilder<PackageGroupContents> result = NestedSetBuilder.stableOrder();
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
          result.addTransitive(provider.getPackageSpecifications());
        } else {
          reporter.handle(Event.error(target.getLocation(),
              String.format("Label '%s' does not refer to a package group", groupLabel)));
        }
      }

      result.add(packageGroupsVisibility.getDirectPackages());
      return result.build();
    } else {
      throw new IllegalStateException("unknown visibility");
    }
  }

  private ConfiguredTarget findPrerequisite(
      OrderedSetMultimap<Attribute, ConfiguredTarget> prerequisiteMap, Label label,
      BuildConfiguration config) {
    for (ConfiguredTarget prerequisite : prerequisiteMap.get(null)) {
      if (prerequisite.getLabel().equals(label) && (prerequisite.getConfiguration() == config)) {
        return prerequisite;
      }
    }
    return null;
  }

  /**
   * Returns the output artifact for the given file, or null if Skyframe deps are missing.
   */
  private Artifact getOutputArtifact(AnalysisEnvironment analysisEnvironment, OutputFile outputFile,
      BuildConfiguration configuration, boolean isFileset, ArtifactFactory artifactFactory)
      throws InterruptedException {
    Rule rule = outputFile.getAssociatedRule();
    Root root = rule.hasBinaryOutput()
        ? configuration.getBinDirectory(rule.getRepository())
        : configuration.getGenfilesDirectory(rule.getRepository());
    ArtifactOwner owner = new ConfiguredTargetKey(rule.getLabel(),
        getArtifactOwnerConfiguration(analysisEnvironment.getSkyframeEnv(), configuration));
    if (analysisEnvironment.getSkyframeEnv().valuesMissing()) {
      return null;
    }
    PathFragment rootRelativePath =
        outputFile.getLabel().getPackageIdentifier().getSourceRoot().getRelative(
            outputFile.getLabel().getName());
    Artifact result = isFileset
        ? artifactFactory.getFilesetArtifact(rootRelativePath, root, owner)
        : artifactFactory.getDerivedArtifact(rootRelativePath, root, owner);
    // The associated rule should have created the artifact.
    Preconditions.checkNotNull(result, "no artifact for %s", rootRelativePath);
    return result;
  }

  /**
   * Returns the configuration's artifact owner (which may be null). Also returns null if the
   * owning configuration isn't yet available from Skyframe.
   */
  public static BuildConfiguration getArtifactOwnerConfiguration(SkyFunction.Environment env,
      BuildConfiguration fromConfig) throws InterruptedException {
    if (fromConfig == null) {
      return null;
    }
    PatchTransition ownerTransition = fromConfig.getArtifactOwnerTransition();
    if (ownerTransition == null) {
      return fromConfig;
    }
    try {
      BuildConfigurationValue ownerConfig = (BuildConfigurationValue) env.getValueOrThrow(
          BuildConfigurationValue.key(
              fromConfig.fragmentClasses(), ownerTransition.apply(fromConfig.getOptions())),
          InvalidConfigurationException.class);
      return ownerConfig == null ? null : ownerConfig.getConfiguration();
    } catch (InvalidConfigurationException e) {
      // We don't expect to have to handle an invalid configuration because in practice the owning
      // configuration should already exist. For example, the main user of this feature, the LIPO
      // context collector, expects the owning configuration to be the top-level target config.
      throw new IllegalStateException(
          "this method should only return a pre-existing valid configuration");
    }
  }

  /**
   * Invokes the appropriate constructor to create a {@link ConfiguredTarget} instance.
   *
   * <p>For use in {@code ConfiguredTargetFunction}.
   *
   * <p>Returns null if Skyframe deps are missing or upon certain errors.
   */
  @Nullable
  public final ConfiguredTarget createConfiguredTarget(
      AnalysisEnvironment analysisEnvironment,
      ArtifactFactory artifactFactory,
      Target target,
      BuildConfiguration config,
      BuildConfiguration hostConfig,
      OrderedSetMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext)
      throws InterruptedException {
    if (target instanceof Rule) {
      try {
        CurrentRuleTracker.beginConfiguredTarget(((Rule) target).getRuleClassObject());
        return createRule(
            analysisEnvironment,
            (Rule) target,
            config,
            hostConfig,
            prerequisiteMap,
            configConditions,
            toolchainContext);
      } finally {
        CurrentRuleTracker.endConfiguredTarget();
      }
    }

    // Visibility, like all package groups, doesn't have a configuration
    NestedSet<PackageGroupContents> visibility =
        convertVisibility(prerequisiteMap, analysisEnvironment.getEventHandler(), target, null);
    TargetContext targetContext = new TargetContext(analysisEnvironment, target, config,
        prerequisiteMap.get(null), visibility);
    if (target instanceof OutputFile) {
      OutputFile outputFile = (OutputFile) target;
      boolean isFileset = outputFile.getGeneratingRule().getRuleClass().equals("Fileset");
      Artifact artifact =
          getOutputArtifact(analysisEnvironment, outputFile, config, isFileset, artifactFactory);
      if (analysisEnvironment.getSkyframeEnv().valuesMissing()) {
        return null;
      }
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
          Root.asSourceRoot(inputFile.getPackage().getSourceRoot(),
              inputFile.getPackage().getPackageIdentifier().getRepository().isMain()),
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
      AnalysisEnvironment env,
      Rule rule,
      BuildConfiguration configuration,
      BuildConfiguration hostConfiguration,
      OrderedSetMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext)
      throws InterruptedException {

    // Load the requested toolchains into the ToolchainContext.
    if (toolchainContext != null) {
      toolchainContext.resolveToolchains(prerequisiteMap);
    }

    // Visibility computation and checking is done for every rule.
    RuleContext ruleContext =
        new RuleContext.Builder(
                env,
                rule,
                ImmutableList.of(),
                configuration,
                hostConfiguration,
                ruleClassProvider.getPrerequisiteValidator(),
                rule.getRuleClassObject().getConfigurationFragmentPolicy())
            .setVisibility(convertVisibility(prerequisiteMap, env.getEventHandler(), rule, null))
            .setPrerequisites(prerequisiteMap)
            .setConfigConditions(configConditions)
            .setUniversalFragment(ruleClassProvider.getUniversalFragment())
            .setToolchainContext(toolchainContext)
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

    if (rule.getRuleClassObject().isSkylark()) {
      // TODO(bazel-team): maybe merge with RuleConfiguredTargetBuilder?
      return SkylarkRuleConfiguredTargetUtil.buildRule(
          ruleContext,
          rule.getRuleClassObject().getConfiguredTargetFunction(),
          env.getSkylarkSemantics());
    } else {
      RuleClass.ConfiguredTargetFactory<ConfiguredTarget, RuleContext> factory =
          rule.getRuleClassObject().<ConfiguredTarget, RuleContext>getConfiguredTargetFactory();
      Preconditions.checkNotNull(factory, rule.getRuleClassObject());
      try {
        return factory.create(ruleContext);
      } catch (RuleErrorException ruleErrorException) {
        // Returning null in this method is an indication an error occurred. Exceptions are not
        // propagated, as this would show a nasty stack trace to users, and only provide info
        // on one specific failure with poor messaging. By returning null, the caller can
        // inspect ruleContext for multiple errors and output thorough messaging on each.
        return null;
      }
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

  private static final Function<Aspect, AspectDescriptor> ASPECT_TO_DESCRIPTOR =
      new Function<Aspect, AspectDescriptor>() {
        @Override
        public AspectDescriptor apply(Aspect aspect) {
          return aspect.getDescriptor();
        }
      };

  /**
   * Constructs an {@link ConfiguredAspect}. Returns null if an error occurs; in that case, {@code
   * aspectFactory} should call one of the error reporting methods of {@link RuleContext}.
   */
  public ConfiguredAspect createAspect(
      AnalysisEnvironment env,
      ConfiguredTarget associatedTarget,
      ImmutableList<Aspect> aspectPath,
      ConfiguredAspectFactory aspectFactory,
      Aspect aspect,
      OrderedSetMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext,
      BuildConfiguration aspectConfiguration,
      BuildConfiguration hostConfiguration)
      throws InterruptedException {

    // Load the requested toolchains into the ToolchainContext.
    if (toolchainContext != null) {
      toolchainContext.resolveToolchains(prerequisiteMap);
    }

    RuleContext.Builder builder =
        new RuleContext.Builder(
            env,
            associatedTarget.getTarget().getAssociatedRule(),
            aspectPath,
            aspectConfiguration,
            hostConfiguration,
            ruleClassProvider.getPrerequisiteValidator(),
            aspect.getDefinition().getConfigurationFragmentPolicy());

    Map<String, Attribute> aspectAttributes = mergeAspectAttributes(aspectPath);

    RuleContext ruleContext =
        builder
            .setVisibility(
                convertVisibility(
                    prerequisiteMap, env.getEventHandler(), associatedTarget.getTarget(), null))
            .setPrerequisites(prerequisiteMap)
            .setAspectAttributes(aspectAttributes)
            .setConfigConditions(configConditions)
            .setUniversalFragment(ruleClassProvider.getUniversalFragment())
            .setToolchainContext(toolchainContext)
            .build();
    if (ruleContext.hasErrors()) {
      return null;
    }

    ConfiguredAspect configuredAspect = aspectFactory
        .create(associatedTarget, ruleContext, aspect.getParameters());
    if (configuredAspect != null) {
      validateAdvertisedProviders(
          configuredAspect, aspect.getDefinition().getAdvertisedProviders(),
          associatedTarget.getTarget(),
          env.getEventHandler()
      );
    }
    return configuredAspect;
  }

  private Map<String, Attribute> mergeAspectAttributes(ImmutableList<Aspect> aspectPath) {
    if (aspectPath.isEmpty()) {
      return ImmutableMap.of();
    } else if (aspectPath.size() == 1) {
      return aspectPath.get(0).getDefinition().getAttributes();
    } else {

      LinkedHashMap<String, Attribute> aspectAttributes = new LinkedHashMap<>();
      for (Aspect underlyingAspect : aspectPath) {
        ImmutableMap<String, Attribute> currentAttributes = underlyingAspect.getDefinition()
            .getAttributes();
        for (Entry<String, Attribute> kv : currentAttributes.entrySet()) {
          if (!aspectAttributes.containsKey(kv.getKey())) {
            aspectAttributes.put(kv.getKey(), kv.getValue());
          }
        }
      }
      return aspectAttributes;
    }
  }

  private void validateAdvertisedProviders(
      ConfiguredAspect configuredAspect,
      AdvertisedProviderSet advertisedProviders, Target target,
      EventHandler eventHandler) {
    if (advertisedProviders.canHaveAnyProvider()) {
      return;
    }
    for (Class<?> aClass : advertisedProviders.getNativeProviders()) {
      if (configuredAspect.getProvider(aClass.asSubclass(TransitiveInfoProvider.class)) == null) {
        eventHandler.handle(Event.error(
            target.getLocation(),
            String.format(
                "Aspect '%s', applied to '%s', does not provide advertised provider '%s'",
                configuredAspect.getName(),
                target.getLabel(),
                aClass.getSimpleName()
            )));
      }
    }

    for (SkylarkProviderIdentifier providerId : advertisedProviders.getSkylarkProviders()) {
      if (configuredAspect.getProvider(providerId) == null) {
        eventHandler.handle(Event.error(
            target.getLocation(),
            String.format(
                "Aspect '%s', applied to '%s', does not provide advertised provider '%s'",
                configuredAspect.getName(),
                target.getLabel(),
                providerId
            )));
      }
    }
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
