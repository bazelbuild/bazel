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
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Implementation for the {@code xcode_config} rule.
 */
public class XcodeConfig implements RuleConfiguredTargetFactory {
  private static ImmutableList<XcodeVersionRuleData> getAvailableVersions(
      ConfigurationEnvironment env, Rule xcodeConfigTarget)
      throws InvalidConfigurationException, InterruptedException {
    List<Label> xcodeVersionLabels = NonconfigurableAttributeMapper.of(xcodeConfigTarget)
        .get(XcodeConfigRule.VERSIONS_ATTR_NAME, BuildType.LABEL_LIST);
    ImmutableList.Builder<XcodeVersionRuleData> xcodeVersionRuleListBuilder =
        ImmutableList.builder();
    for (Label label : xcodeVersionLabels) {
      Rule xcodeVersionRule = getRuleForLabel(label, "xcode_version", env, "xcode_version");
      xcodeVersionRuleListBuilder.add(new XcodeVersionRuleData(label, xcodeVersionRule));
    }
    return xcodeVersionRuleListBuilder.build();
  }

  /**
   * Uses the {@link AppleCommandLineOptions#xcodeVersion} and {@link
   * AppleCommandLineOptions#xcodeVersionConfig} command line options to determine and return the
   * effective xcode version properties. Returns absent if no explicit xcode version is declared,
   * and host system defaults should be used.
   *
   * @param env the current configuration environment
   * @param appleOptions the command line options
   * @throws InvalidConfigurationException if the options given (or configuration targets) were
   *     malformed and thus the xcode version could not be determined
   */
  static XcodeVersionProperties getXcodeVersionProperties(
      ConfigurationEnvironment env, AppleCommandLineOptions appleOptions)
      throws InvalidConfigurationException, InterruptedException {
    Label xcodeVersionConfigLabel = appleOptions.xcodeVersionConfig;

    Rule xcodeConfigRule = getRuleForLabel(
        xcodeVersionConfigLabel, "xcode_config", env, "xcode_version_config");

    ImmutableList<XcodeVersionRuleData> versions = getAvailableVersions(env, xcodeConfigRule);
    XcodeVersionRuleData defaultVersion = getDefaultVersion(env, xcodeConfigRule);

    try {
      return resolveXcodeVersion(appleOptions.xcodeVersion, versions, defaultVersion);
    } catch (XcodeConfigException e) {
      throw new InvalidConfigurationException(e.getMessage());
    }
  }

  /**
   * An exception that signals that an Xcode config setup was invalid.
   */
  public static class XcodeConfigException extends Exception {
    XcodeConfigException(String reason) {
      super(reason);
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    AppleCommandLineOptions appleOptions =
        ruleContext.getFragment(AppleConfiguration.class).getOptions();
    XcodeVersionRuleData defaultVersion = ruleContext.getPrerequisite(
        XcodeConfigRule.DEFAULT_ATTR_NAME, RuleConfiguredTarget.Mode.TARGET,
        XcodeVersionRuleData.class);
    Iterable<XcodeVersionRuleData> availableVersions = ruleContext.getPrerequisites(
        XcodeConfigRule.VERSIONS_ATTR_NAME, RuleConfiguredTarget.Mode.TARGET,
        XcodeVersionRuleData.class);
    XcodeVersionProperties xcodeVersionProperties;
    try {
      xcodeVersionProperties = resolveXcodeVersion(
          appleOptions.xcodeVersion,
          availableVersions,
          defaultVersion);
    } catch (XcodeConfigException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    }

    DottedVersion iosSdkVersion = (appleOptions.iosSdkVersion != null)
        ? appleOptions.iosSdkVersion : xcodeVersionProperties.getDefaultIosSdkVersion();
    DottedVersion iosMinimumOsVersion = (appleOptions.iosMinimumOs != null)
        ? appleOptions.iosMinimumOs : iosSdkVersion;
    DottedVersion watchosSdkVersion = (appleOptions.watchOsSdkVersion != null)
        ? appleOptions.watchOsSdkVersion : xcodeVersionProperties.getDefaultWatchosSdkVersion();
    DottedVersion watchosMinimumOsVersion = (appleOptions.watchosMinimumOs != null)
        ? appleOptions.watchosMinimumOs : watchosSdkVersion;
    DottedVersion tvosSdkVersion = (appleOptions.tvOsSdkVersion != null)
        ? appleOptions.tvOsSdkVersion : xcodeVersionProperties.getDefaultTvosSdkVersion();
    DottedVersion tvosMinimumOsVersion = (appleOptions.tvosMinimumOs != null)
        ? appleOptions.tvosMinimumOs : tvosSdkVersion;
    DottedVersion macosSdkVersion = (appleOptions.macOsSdkVersion != null)
        ? appleOptions.macOsSdkVersion : xcodeVersionProperties.getDefaultMacosSdkVersion();
    DottedVersion macosMinimumOsVersion = (appleOptions.macosMinimumOs != null)
        ? appleOptions.macosMinimumOs : macosSdkVersion;

    XcodeConfigProvider xcodeVersions = new XcodeConfigProvider(
        iosSdkVersion, iosMinimumOsVersion,
        watchosSdkVersion, watchosMinimumOsVersion,
        tvosSdkVersion, tvosMinimumOsVersion,
        macosSdkVersion, macosMinimumOsVersion,
        xcodeVersionProperties.getXcodeVersion().orNull());

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addNativeDeclaredProvider(xcodeVersions)
        .addNativeDeclaredProvider(xcodeVersionProperties)
        .build();
  }
  
  /**
   * Uses the {@link AppleCommandLineOptions#xcodeVersion} and {@link
   * AppleCommandLineOptions#xcodeVersionConfig} command line options to determine and return the
   * effective xcode version and its properties.
   *
   * @param requireDefinedVersions whether the version config requires an explicitly defined version
   * @param xcodeVersionOverrideFlag the value of the {@code --xcode_version} command line flag
   * @param xcodeVersions the Xcode versions listed in the {@code xcode_config} rule
   * @param defaultVersion the default Xcode version in the {@code xcode_config} rule.
   * @throws XcodeConfigException if the options given (or configuration targets) were
   *     malformed and thus the xcode version could not be determined
   */
  static XcodeVersionProperties resolveXcodeVersion(
      String xcodeVersionOverrideFlag,
      Iterable<XcodeVersionRuleData> xcodeVersions,
      XcodeVersionRuleData defaultVersion)
      throws XcodeConfigException {
    if (defaultVersion != null
        && Iterables.isEmpty(Iterables.filter(
              xcodeVersions,
              ruleData -> ruleData.getLabel().equals(defaultVersion.getLabel())))) {
      throw new XcodeConfigException(
          String.format("default label '%s' must be contained in versions attribute",
              defaultVersion.getLabel()));
    }
    if (Iterables.isEmpty(xcodeVersions)) {
      if (defaultVersion != null) {
        throw new XcodeConfigException(
            "default label must be contained in versions attribute");
      }
      return XcodeVersionProperties.unknownXcodeVersionProperties();
    }
    if (defaultVersion == null) {
      throw new XcodeConfigException(
          "if any versions are specified, a default version must be specified");
    }

    XcodeVersionRuleData xcodeVersion = resolveExplicitlyDefinedVersion(
        xcodeVersions, defaultVersion, xcodeVersionOverrideFlag);

    return xcodeVersion.getXcodeVersionProperties();
  }

  /**
   * Returns the {@link XcodeVersionRuleData} associated with the {@code xcode_version} target
   * explicitly defined in the {@code --xcode_version_config} build flag and selected by the {@code
   * --xcode_version} flag. If {@code --xcode_version} is unspecified, then this will return the
   * default rule data as specified in the {@code --xcode_version_config} target.
   */
  private static XcodeVersionRuleData resolveExplicitlyDefinedVersion(
      Iterable<XcodeVersionRuleData> xcodeVersionRules,
      XcodeVersionRuleData defaultVersion,
      String versionOverrideFlag)
      throws XcodeConfigException {

    Map<String, XcodeVersionRuleData> aliasesToVersionMap = aliasesToVersionMap(xcodeVersionRules);

    if (!Strings.isNullOrEmpty(versionOverrideFlag)) {
      // The version override flag is not necessarily an actual version - it may be a version
      // alias.
      XcodeVersionRuleData explicitVersion =
          aliasesToVersionMap.get(versionOverrideFlag);
      if (explicitVersion != null) {
        return explicitVersion;
      } else {
        throw new XcodeConfigException(
            String.format("%s matches no alias in the config", versionOverrideFlag));
      }
    }

    return defaultVersion;
  }

  /**
   * Returns the default xcode version to use, if no {@code --xcode_version} command line flag was
   * specified.
   */
  @Nullable
  private static XcodeVersionRuleData getDefaultVersion(
      ConfigurationEnvironment env, Rule xcodeConfigTarget)
      throws InvalidConfigurationException, InterruptedException {
    Label defaultVersionLabel = NonconfigurableAttributeMapper.of(xcodeConfigTarget)
        .get(XcodeConfigRule.DEFAULT_ATTR_NAME, BuildType.LABEL);
    if (defaultVersionLabel != null) {
      Rule defaultVersionRule = getRuleForLabel(
          defaultVersionLabel, "xcode_version", env, "default xcode version");
      return new XcodeVersionRuleData(defaultVersionLabel, defaultVersionRule);
    } else {
      return null;
    }
  }

  /**
   * Returns a map where keys are "names" of xcode versions as defined by the configuration target,
   * and values are the rule data objects which contain information regarding that xcode version.
   *
   * @throws XcodeConfigException if there are duplicate aliases (if two xcode versions
   *     were registered to the same alias)
   */
  private static Map<String, XcodeVersionRuleData> aliasesToVersionMap(
      Iterable<XcodeVersionRuleData> xcodeVersionRules)
      throws XcodeConfigException {
    Map<String, XcodeVersionRuleData> aliasesToXcodeRules = Maps.newLinkedHashMap();
    for (XcodeVersionRuleData xcodeVersionRule : xcodeVersionRules) {
      for (String alias : xcodeVersionRule.getAliases()) {
        if (aliasesToXcodeRules.put(alias, xcodeVersionRule) != null) {
          configErrorDuplicateAlias(alias, xcodeVersionRules);
        }
      }
      // Only add the version as an alias if it's not included in this xcode_version target's
      // aliases (in which case it would have just been added. This offers some leniency in target
      // definition, as it's silly to error if a version is aliased to its own version.
      if (!xcodeVersionRule.getAliases().contains(xcodeVersionRule.getVersion().toString())) {
        if (aliasesToXcodeRules.put(
            xcodeVersionRule.getVersion().toString(), xcodeVersionRule) != null) {
          configErrorDuplicateAlias(xcodeVersionRule.getVersion().toString(), xcodeVersionRules);
        }
      }
    }
    return aliasesToXcodeRules;
  }
  
  /**
   * Convenience method for throwing an {@link XcodeConfigException} due to presence
   * of duplicate aliases in an {@code xcode_config} target definition. 
   */
  private static void configErrorDuplicateAlias(String alias,
      Iterable<XcodeVersionRuleData> xcodeVersionRules) throws XcodeConfigException {

    ImmutableList.Builder<Label> labelsContainingAlias = ImmutableList.builder();
    for (XcodeVersionRuleData xcodeVersionRule : xcodeVersionRules) {
      if (xcodeVersionRule.getAliases().contains(alias)
          || xcodeVersionRule.getVersion().toString().equals(alias)) {
        labelsContainingAlias.add(xcodeVersionRule.getLabel());
      }
    }

    throw new XcodeConfigException(
        String.format("'%s' is registered to multiple labels (%s) in a single xcode_config rule",
            alias, Joiner.on(", ").join(labelsContainingAlias.build())));
  }

  /**
   * If the given label (following redirects) is a target for a rule of type {@code type}, then
   * returns the {@link Rule} representing that target. Otherwise, throws a {@link
   * InvalidConfigurationException}.
   */
  static Rule getRuleForLabel(
      Label label, String type, ConfigurationEnvironment env, String description)
      throws InvalidConfigurationException, InterruptedException {
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

  /**
   * Returns the minimum compatible OS version for target simulator and devices for a particular
   * platform type.
   */
  public static DottedVersion getMinimumOsForPlatformType(
      RuleContext ruleContext, ApplePlatform.PlatformType platformType) {
    AppleConfiguration config = ruleContext.getFragment(AppleConfiguration.class);
    XcodeConfigProvider versions = ruleContext.getPrerequisite(
        XcodeConfigRule.XCODE_CONFIG_ATTR_NAME,
        RuleConfiguredTarget.Mode.TARGET,
        XcodeConfigProvider.PROVIDER);
    DottedVersion fromProvider = versions.getMinimumOsForPlatformType(platformType);
    DottedVersion fromConfig = config.getMinimumOsForPlatformType(platformType);
    // This sanity check is there to keep this provider in sync with AppleConfiguration until the
    // latter can be removed. Tracking bug: https://github.com/bazelbuild/bazel/issues/3424
    Preconditions.checkState(fromProvider.equals(fromConfig));
    return fromProvider;
  }

  /**
   * Returns the SDK version for a platform (whether they be for simulator or device). This is
   * directly derived from command line args.
   */
  public static DottedVersion getSdkVersionForPlatform(
      RuleContext ruleContext, ApplePlatform platform) {
    XcodeConfigProvider versions = ruleContext.getPrerequisite(
        XcodeConfigRule.XCODE_CONFIG_ATTR_NAME,
        RuleConfiguredTarget.Mode.TARGET,
        XcodeConfigProvider.PROVIDER);
    return versions.getSdkVersionForPlatform(platform);
  }

  /**
   * Returns the value of the xcode version, if available. This is determined based on a combination
   * of the {@code --xcode_version} build flag and the {@code xcode_config} target defined in the
   * {@code --xcode_version_config} flag. Returns null if no xcode is available.
   */
  public static DottedVersion getXcodeVersion(RuleContext ruleContext) {
    XcodeConfigProvider versions = ruleContext.getPrerequisite(
        XcodeConfigRule.XCODE_CONFIG_ATTR_NAME,
        RuleConfiguredTarget.Mode.TARGET,
        XcodeConfigProvider.PROVIDER);
    return versions.getXcodeVersion();
  }
}
