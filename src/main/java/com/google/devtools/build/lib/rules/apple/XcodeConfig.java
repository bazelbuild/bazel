// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;

import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Implementation for the {@code xcode_config} rule.
 */
public class XcodeConfig implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .build();
  }
  
  /**
   * Uses the {@link AppleCommandLineOptions#xcodeVersion} and
   * {@link AppleCommandLineOptions#xcodeVersionConfig} command line options to determine and
   * return the effective xcode version.
   *
   * @param env the current configuration environment
   * @param xcodeConfigLabel the label for the xcode_config target to parse
   * @param xcodeVersionOverrideFlag the value of the command line flag to override the default
   *     xcode version, absent if unspecified
   * @param errorDescription a description of the origin of {@code #xcodeConfigLabel} for messaging
   *     parse errors
   * @throws InvalidConfigurationException if the options given (or configuration targets) were
   *     malformed and thus the xcode version could not be determined
   */
  public static Optional<DottedVersion> resolveXcodeVersion(ConfigurationEnvironment env,
      Label xcodeConfigLabel, Optional<DottedVersion> xcodeVersionOverrideFlag,
      String errorDescription) throws InvalidConfigurationException {
    Rule xcodeConfigRule =
        getRuleForLabel(xcodeConfigLabel, "xcode_config", env, errorDescription);

    DottedVersion dottedVersion =
        resolveExplicitlyDefinedVersion(env, xcodeConfigRule, xcodeVersionOverrideFlag);
    
    if (dottedVersion != null) {
      return Optional.of(dottedVersion);
    } else {
      return xcodeVersionOverrideFlag;
    }
  }

  /**
   * Returns the xcode version number corresponding to the {@code --xcode_version} flag, if there
   * is an available {@code xcode_version} target which recognizes the flag value as either
   * an official version or an alias. Returns null if no such target is found.
   */
  @Nullable private static DottedVersion resolveExplicitlyDefinedVersion(
      ConfigurationEnvironment env, Rule xcodeConfigTarget,
      Optional<DottedVersion> versionOverrideFlag) throws InvalidConfigurationException {
    if (versionOverrideFlag.isPresent()) {
      // The version override flag is not necessarily an actual version - it may be a version
      // alias.
      DottedVersion explicitVersion =
          aliasesToVersionMap(env, xcodeConfigTarget).get(versionOverrideFlag.get().toString());
      if (explicitVersion != null) {
        return explicitVersion;
      }
    } else { // No override specified. Use default.
      DottedVersion defaultVersion = getDefaultVersion(env, xcodeConfigTarget);
      
      if (defaultVersion != null) {
        return defaultVersion;
      }
    }
    
    boolean requireDefinedVersions = NonconfigurableAttributeMapper.of(xcodeConfigTarget)
        .get(XcodeConfigRule.REQUIRE_DEFINED_VERSIONS_ATTR_NAME, Type.BOOLEAN);
    if (requireDefinedVersions) {
      throw new InvalidConfigurationException(
          "xcode version config required an explicitly defined version, but none was available");
    }

    return null;
  }

  /**
   * Returns the default xcode version to use, if no {@code --xcode_version} command line flag
   * was specified.
   */
  @Nullable private static DottedVersion getDefaultVersion(ConfigurationEnvironment env,
      Rule xcodeConfigTarget) throws InvalidConfigurationException {
    Label defaultVersionLabel = NonconfigurableAttributeMapper.of(xcodeConfigTarget)
        .get(XcodeConfigRule.DEFAULT_ATTR_NAME, BuildType.LABEL);
    if (defaultVersionLabel != null) {
      Rule defaultVersionRule = getRuleForLabel(
          defaultVersionLabel, "xcode_version", env, "default xcode version");
      return new XcodeVersionRuleData(defaultVersionLabel, defaultVersionRule).getVersion();
    } else {
      return null;
    }
  }

  private static Map<String, DottedVersion> aliasesToVersionMap(ConfigurationEnvironment env,
      Rule xcodeConfigTarget) throws InvalidConfigurationException {
    List<Label> xcodeVersionLabels = NonconfigurableAttributeMapper.of(xcodeConfigTarget)
        .get(XcodeConfigRule.VERSIONS_ATTR_NAME, BuildType.LABEL_LIST);
    ImmutableList.Builder<XcodeVersionRuleData> xcodeVersionRuleListBuilder =
        ImmutableList.builder();
    for (Label label : xcodeVersionLabels) {
      Rule xcodeVersionRule = getRuleForLabel(label, "xcode_version", env, "xcode_version");
      xcodeVersionRuleListBuilder.add(new XcodeVersionRuleData(label, xcodeVersionRule));
    }
    ImmutableList<XcodeVersionRuleData> xcodeVersionRules = xcodeVersionRuleListBuilder.build();

    Map<String, DottedVersion> aliasesToVersionMap = Maps.newLinkedHashMap();
    for (XcodeVersionRuleData xcodeVersionRule : xcodeVersionRules) {
      for (String alias : xcodeVersionRule.getAliases()) {
        if (aliasesToVersionMap.put(alias, xcodeVersionRule.getVersion()) != null) {
          configErrorDuplicateAlias(alias, xcodeVersionRules);
        }
      }
      if (aliasesToVersionMap.put(
          xcodeVersionRule.getVersion().toString(), xcodeVersionRule.getVersion()) != null) {
        configErrorDuplicateAlias(xcodeVersionRule.getVersion().toString(), xcodeVersionRules);
      }
    }
    return aliasesToVersionMap;
  }
  
  /**
   * Convenience method for throwing an {@link InvalidConfigurationException} due to presence
   * of duplicate aliases in an {@code xcode_config} target definition. 
   */
  private static void configErrorDuplicateAlias(String alias,
      List<XcodeVersionRuleData> xcodeVersionRules) throws InvalidConfigurationException {

    ImmutableList.Builder<Label> labelsContainingAlias = ImmutableList.builder();
    for (XcodeVersionRuleData xcodeVersionRule : xcodeVersionRules) {
      if (xcodeVersionRule.getAliases().contains(alias)
          || xcodeVersionRule.getVersion().toString().equals(alias)) {
        labelsContainingAlias.add(xcodeVersionRule.getLabel());
      }
    }

    throw new InvalidConfigurationException(
        String.format("'%s' is registered to multiple labels (%s) in a single xcode_config rule",
            alias, Joiner.on(", ").join(labelsContainingAlias.build())));
  }

  /**
   * If the given label (following redirects) is a target for a rule of type {@code type},
   * then returns the {@link Rule} representing that target. Otherwise, throws a
   * {@link InvalidConfigurationException}.
   */
  private static Rule getRuleForLabel(Label label, String type, ConfigurationEnvironment env,
      String description) throws InvalidConfigurationException {
    label = RedirectChaser.followRedirects(env, label, description);

    if (label == null) {
      throw new InvalidConfigurationException(String.format(
          "Expected value of %s (%s) to resolve to a target of type %s",
          description, label, type));
    }

    try {
      Target target = env.getTarget(label);
      
      if (target instanceof Rule && ((Rule) target).getRuleClass().equals(type)) {
        return (Rule) target;
      } else {
        throw new InvalidConfigurationException(String.format(
            "Expected value of %s (%s) to resolve to a target of type %s",
            description, label, type));
      }
    } catch (NoSuchPackageException | NoSuchTargetException exception) {
      env.getEventHandler().handle(Event.error(exception.getMessage()));
      throw new InvalidConfigurationException(exception);
    }
  }
}