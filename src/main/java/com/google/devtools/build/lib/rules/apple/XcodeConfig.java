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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import java.util.Map;

/**
 * Implementation for the {@code xcode_config} rule.
 */
public class XcodeConfig implements RuleConfiguredTargetFactory {
  private static final DottedVersion MINIMUM_BITCODE_XCODE_VERSION = DottedVersion.fromString("7");

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
      throws InterruptedException, RuleErrorException, ActionConflictException {
    AppleConfiguration appleConfig = ruleContext.getFragment(AppleConfiguration.class);
    AppleCommandLineOptions appleOptions = appleConfig.getOptions();
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
        ? DottedVersion.maybeUnwrap(appleOptions.iosSdkVersion)
        : xcodeVersionProperties.getDefaultIosSdkVersion();
    DottedVersion iosMinimumOsVersion = (appleOptions.iosMinimumOs != null)
        ? DottedVersion.maybeUnwrap(appleOptions.iosMinimumOs) : iosSdkVersion;
    DottedVersion watchosSdkVersion = (appleOptions.watchOsSdkVersion != null)
        ? DottedVersion.maybeUnwrap(appleOptions.watchOsSdkVersion)
        : xcodeVersionProperties.getDefaultWatchosSdkVersion();
    DottedVersion watchosMinimumOsVersion = (appleOptions.watchosMinimumOs != null)
        ? DottedVersion.maybeUnwrap(appleOptions.watchosMinimumOs) : watchosSdkVersion;
    DottedVersion tvosSdkVersion = (appleOptions.tvOsSdkVersion != null)
        ? DottedVersion.maybeUnwrap(appleOptions.tvOsSdkVersion)
        : xcodeVersionProperties.getDefaultTvosSdkVersion();
    DottedVersion tvosMinimumOsVersion = (appleOptions.tvosMinimumOs != null)
        ? DottedVersion.maybeUnwrap(appleOptions.tvosMinimumOs) : tvosSdkVersion;
    DottedVersion macosSdkVersion = (appleOptions.macOsSdkVersion != null)
        ? DottedVersion.maybeUnwrap(appleOptions.macOsSdkVersion)
        : xcodeVersionProperties.getDefaultMacosSdkVersion();
    DottedVersion macosMinimumOsVersion = (appleOptions.macosMinimumOs != null)
        ? DottedVersion.maybeUnwrap(appleOptions.macosMinimumOs) : macosSdkVersion;

    XcodeConfigProvider xcodeVersions = new XcodeConfigProvider(
        iosSdkVersion, iosMinimumOsVersion,
        watchosSdkVersion, watchosMinimumOsVersion,
        tvosSdkVersion, tvosMinimumOsVersion,
        macosSdkVersion, macosMinimumOsVersion,
        xcodeVersionProperties.getXcodeVersion().orNull());

    AppleBitcodeMode bitcodeMode = appleConfig.getBitcodeMode();
    DottedVersion xcodeVersion = xcodeVersions.getXcodeVersion();
    if (bitcodeMode != AppleBitcodeMode.NONE
        && xcodeVersion != null
        && xcodeVersion.compareTo(MINIMUM_BITCODE_XCODE_VERSION) < 0) {
      ruleContext.throwWithRuleError(String.format(
          "apple_bitcode mode '%s' is unsupported for xcode version '%s'",
          bitcodeMode, xcodeVersion));
    }

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
   * @param xcodeVersionOverrideFlag the value of the {@code --xcode_version} command line flag
   * @param xcodeVersions the Xcode versions listed in the {@code xcode_config} rule
   * @param defaultVersion the default Xcode version in the {@code xcode_config} rule.
   * @throws XcodeConfigException if the options given (or configuration targets) were malformed and
   *     thus the xcode version could not be determined
   */
  static XcodeVersionProperties resolveXcodeVersion(
      String xcodeVersionOverrideFlag,
      Iterable<XcodeVersionRuleData> xcodeVersions,
      XcodeVersionRuleData defaultVersion)
      throws XcodeConfigException {
    if (defaultVersion != null
        && Iterables.isEmpty(
            Iterables.filter(
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
            String.format(
                "--xcode_version=%1$s specified, but '%1$s' is not an available Xcode version. "
                    + "available versions: [%2$s]. If you believe you have '%1$s' installed, try "
                    + "running \"bazel shutdown\", and then re-run your command.",
                versionOverrideFlag, printableXcodeVersions(xcodeVersionRules)));
      }
    }

    return defaultVersion;
  }

  private static String printableXcodeVersions(Iterable<XcodeVersionRuleData> xcodeVersions) {
    return Streams.stream(xcodeVersions)
        .map(versionData -> versionData.getVersion().toString())
        .collect(joining(", "));
  }

  /**
   * Returns a map where keys are "names" of xcode versions as defined by the configuration target,
   * and values are the rule data objects which contain information regarding that xcode version.
   *
   * @throws XcodeConfigException if there are duplicate aliases (if two xcode versions were
   *     registered to the same alias)
   */
  private static Map<String, XcodeVersionRuleData> aliasesToVersionMap(
      Iterable<XcodeVersionRuleData> xcodeVersionRules) throws XcodeConfigException {
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
   * Convenience method for throwing an {@link XcodeConfigException} due to presence of duplicate
   * aliases in an {@code xcode_config} target definition.
   */
  private static void configErrorDuplicateAlias(
      String alias, Iterable<XcodeVersionRuleData> xcodeVersionRules) throws XcodeConfigException {

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

  public static XcodeConfigProvider getXcodeConfigProvider(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(
        XcodeConfigRule.XCODE_CONFIG_ATTR_NAME,
        RuleConfiguredTarget.Mode.TARGET,
        XcodeConfigProvider.PROVIDER);
  }
}
