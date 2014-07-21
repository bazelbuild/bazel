// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageGroupsRuleVisibility;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.skyframe.LabelAndConfiguration;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

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
   * Returns a list of build-info factories that are needed for the supported languages.
   */
  public ImmutableList<BuildInfoFactory> getBuildInfoFactories() {
    return ruleClassProvider.getBuildInfoFactories();
  }

  /**
   * Returns the visibility of the given target. Errors during package group resolution are reported
   * to the {@code AnalysisEnvironment}.
   */
  private NestedSet<PackageSpecification> convertVisibility(
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap, ErrorEventListener reporter,
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
          reporter.error(target.getLocation(),
              String.format("Label '%s' does not refer to a package group", groupLabel));
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
    ArtifactOwner owner = new LabelAndConfiguration(rule.getLabel(), configuration);
    PathFragment rootRelativePath = Util.getWorkspaceRelativePath(outputFile);
    Artifact result = isFileset
        ? artifactFactory.getExistingFilesetArtifact(rootRelativePath, root, owner)
        : artifactFactory.getExistingDerivedArtifact(rootRelativePath, root, owner);
    // The associated rule should have created the artifact.
    Preconditions.checkNotNull(result, "no artifact for %s", rootRelativePath);
    return result;
  }

  /**
   * Invokes the appropriate constructor to create a {@link ConfiguredTarget} instance.
   */
  @Nullable
  public final ConfiguredTarget createAndInitialize(
      AnalysisEnvironment analysisEnvironment, ArtifactFactory artifactFactory,
      Target target, BuildConfiguration config,
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap)
      throws InterruptedException {
    if (target instanceof Rule) {
      return createRule(
          analysisEnvironment, (Rule) target, config, prerequisiteMap);
    }

    // Visibility, like all package groups, doesn't have a configuration
    NestedSet<PackageSpecification> visibility = convertVisibility(
        prerequisiteMap, analysisEnvironment.getReporter(), target, null);
    TargetContext targetContext = new TargetContext(analysisEnvironment, target, config,
        prerequisiteMap.get(null), visibility);
    if (target instanceof OutputFile) {
      OutputFile outputFile = (OutputFile) target;
      boolean isFileset = outputFile.getGeneratingRule().getRuleClass().equals("Fileset");
      Artifact artifact = getOutputArtifact(outputFile, config, isFileset, artifactFactory);
      TransitiveInfoCollection rule = targetContext.findDirectPrerequisite(
          outputFile.getGeneratingRule().getLabel(), config);
      if (isFileset) {
        return new FilesetOutputConfiguredTarget(targetContext, outputFile, rule, artifact);
      } else {
        return new OutputFileConfiguredTarget(targetContext, outputFile, rule, artifact);
      }
    } else if (target instanceof InputFile) {
      InputFile inputFile = (InputFile) target;
      Artifact artifact = artifactFactory.getSourceArtifact(
          inputFile.getExecPath(),
          Root.asSourceRoot(inputFile.getPackage().getSourceRoot()),
          new LabelAndConfiguration(target.getLabel(), config));

      return new InputFileConfiguredTarget(targetContext, inputFile, artifact);
    } else if (target instanceof PackageGroup) {
      PackageGroup packageGroup = (PackageGroup) target;
      return new PackageGroupConfiguredTarget(targetContext, packageGroup);
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
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap) throws InterruptedException {
    // Visibility computation and checking is done for every rule.
    RuleContext ruleContext = new RuleContext.Builder(env, rule, configuration,
        ruleClassProvider.getPrerequisiteValidator())
        .setVisibility(convertVisibility(prerequisiteMap, env.getReporter(), rule, null))
        .setPrerequisites(prerequisiteMap)
        .build();
    if (ruleContext.hasErrors()) {
      return null;
    }
    if (rule.getRuleClassObject().isSkylarkExecutable()) {
      // TODO(bazel-team): maybe merge with RuleConfiguredTargetBuilder?
      return RuleConfiguredTargetBuilder.buildRule(
          ruleContext, rule.getRuleClassObject().getConfiguredTargetFunction());
    } else {
      return ruleClassProvider.createConfiguredTarget(rule, ruleContext);
    }
  }
}
