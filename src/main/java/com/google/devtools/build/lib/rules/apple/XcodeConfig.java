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

import static com.google.common.base.Preconditions.checkState;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
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
import com.google.devtools.build.lib.rules.apple.XcodeConfigInfo.Availability;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implementation for the {@code xcode_config} rule.
 */
public class XcodeConfig implements RuleConfiguredTargetFactory {
  private static final DottedVersion MINIMUM_BITCODE_XCODE_VERSION =
      DottedVersion.fromStringUnchecked("7");

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
    XcodeVersionRuleData explicitDefaultVersion =
        ruleContext.getPrerequisite(
            XcodeConfigRule.DEFAULT_ATTR_NAME,
            RuleConfiguredTarget.Mode.TARGET,
            XcodeVersionRuleData.class);

    List<XcodeVersionRuleData> explicitVersions =
        (List<XcodeVersionRuleData>)
            ruleContext.getPrerequisites(
                XcodeConfigRule.VERSIONS_ATTR_NAME,
                RuleConfiguredTarget.Mode.TARGET,
                XcodeVersionRuleData.class);
    AvailableXcodesInfo remoteVersions =
        ruleContext.getPrerequisite(
            XcodeConfigRule.REMOTE_VERSIONS_ATTR_NAME,
            RuleConfiguredTarget.Mode.TARGET,
            AvailableXcodesInfo.PROVIDER);
    AvailableXcodesInfo localVersions =
        ruleContext.getPrerequisite(
            XcodeConfigRule.LOCAL_VERSIONS_ATTR_NAME,
            RuleConfiguredTarget.Mode.TARGET,
            AvailableXcodesInfo.PROVIDER);

    XcodeVersionProperties xcodeVersionProperties;
    XcodeConfigInfo.Availability availability = null;
    if (useAvailableXcodesMode(
        explicitVersions, explicitDefaultVersion, localVersions, remoteVersions, ruleContext)) {
      Map.Entry<XcodeVersionRuleData, Availability> xcode =
          resolveXcodeFromLocalAndRemote(
              localVersions, remoteVersions, ruleContext, appleOptions.xcodeVersion);
      xcodeVersionProperties = xcode.getKey().getXcodeVersionProperties();
      availability = xcode.getValue();
    } else {
      xcodeVersionProperties =
          resolveExplicitlyDefinedVersion(
              explicitVersions, explicitDefaultVersion, appleOptions.xcodeVersion, ruleContext);
      availability = XcodeConfigInfo.Availability.UNKNOWN;
    }
    DottedVersion iosSdkVersion =
        (appleOptions.iosSdkVersion != null)
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

    XcodeConfigInfo xcodeVersions =
        new XcodeConfigInfo(
            iosSdkVersion,
            iosMinimumOsVersion,
            watchosSdkVersion,
            watchosMinimumOsVersion,
            tvosSdkVersion,
            tvosMinimumOsVersion,
            macosSdkVersion,
            macosMinimumOsVersion,
            xcodeVersionProperties.getXcodeVersion().orNull(),
            availability);

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
   * Returns {@code true} if the xcode version will be determined from {@code local_versions} and
   * {@code remote_versions}.
   *
   * @throws RuleErrorException if attributes from both modes have been set.
   */
  private static boolean useAvailableXcodesMode(
      List<XcodeVersionRuleData> explicitVersions,
      XcodeVersionRuleData explicitDefaultVersion,
      AvailableXcodesInfo localVersions,
      AvailableXcodesInfo remoteVersions,
      RuleContext ruleContext)
      throws RuleErrorException {
    if ((remoteVersions != null && !Iterables.isEmpty(remoteVersions.getAvailableVersions()))) {
      if (!explicitVersions.isEmpty()) {
        ruleContext.ruleError("'versions' may not be set if '[local,remote]_versions' is set.");
      }
      if (explicitDefaultVersion != null) {
        ruleContext.ruleError("'default' may not be set if '[local,remote]_versions' is set.");
      }
      if (localVersions == null || Iterables.isEmpty(localVersions.getAvailableVersions())) {
        ruleContext.throwWithRuleError(
            "if 'remote_versions' are set, you must also set 'local_versions'");
      }
      return true;
    }
    return false;
  }
  /**
   * Returns the {@link XcodeVersionProperties} selected by the {@code--xcode_version} flag from the
   * {@code versions} attribute of the {@code xcode_config} target explicitly defined in the {@code
   * --xcode_version_config} build flag. This is not used if the {@code local_versions} or {@code
   * remote_versions} attributes are set. If {@code --xcode_version} is unspecified, then this will
   * return the default rule data as specified in the {@code --xcode_version_config} target.
   *
   * @throws RuleErrorException if required dependencies are missing or ill formatted.
   */
  private static XcodeVersionProperties resolveExplicitlyDefinedVersion(
      List<XcodeVersionRuleData> explicitVersions,
      XcodeVersionRuleData explicitDefaultVersion,
      String versionOverrideFlag,
      RuleContext ruleContext)
      throws RuleErrorException {
    if (explicitDefaultVersion != null
        && !Iterables.any(
            explicitVersions,
            ruleData -> ruleData.getLabel().equals(explicitDefaultVersion.getLabel()))) {
      ruleContext.throwWithRuleError(
          String.format(
              "default label '%s' must be contained in versions attribute",
              explicitDefaultVersion.getLabel()));
    }
    if (explicitVersions.isEmpty()) {
      if (explicitDefaultVersion != null) {
        ruleContext.throwWithRuleError("default label must be contained in versions attribute");
      }
      return XcodeVersionProperties.unknownXcodeVersionProperties();
    }
    if (explicitDefaultVersion == null) {
      ruleContext.throwWithRuleError(
          "if any versions are specified, a default version must be specified");
    }
    Map<String, XcodeVersionRuleData> aliasesToVersionMap = null;
    try {
      aliasesToVersionMap = aliasesToVersionMap(explicitVersions);
    } catch (XcodeConfigException e) {
      ruleContext.throwWithRuleError(e.getMessage());
    }
    if (!Strings.isNullOrEmpty(versionOverrideFlag)) {
      // The version override flag is not necessarily an actual version - it may be a version
      // alias.
      XcodeVersionRuleData explicitVersion =
          aliasesToVersionMap.get(versionOverrideFlag);
      if (explicitVersion != null) {
        return explicitVersion.getXcodeVersionProperties();
      } else {
        ruleContext.throwWithRuleError(
            String.format(
                "--xcode_version=%1$s specified, but '%1$s' is not an available Xcode version. "
                    + "available versions: [%2$s]. If you believe you have '%1$s' installed, try "
                    + "running \"bazel shutdown\", and then re-run your command.",
                versionOverrideFlag, printableXcodeVersions(explicitVersions)));
      }
    }

    return explicitDefaultVersion.getXcodeVersionProperties();
  }

  /**
   * Returns the {@link XcodeVersionRuleData} and availability associated with the {@code
   * xcode_version} target determined from its {@code remote_xcodes} and {@code local_xcodes}
   * dependencies and selected by the {@code --xcode_version} flag. The version specified by {@code
   * --xcode_version} will be used if it's specified and is available locally or remotely (or both).
   * If {@code --xcode_version} is unspecified, then this will return the newest mutually available
   * version if possibls, otherwise the default local version.
   */
  private static Map.Entry<XcodeVersionRuleData, Availability> resolveXcodeFromLocalAndRemote(
      AvailableXcodesInfo localVersions,
      AvailableXcodesInfo remoteVersions,
      RuleContext ruleContext,
      String versionOverrideFlag)
      throws RuleErrorException {

    // Mutually available Xcode versions are versions that are available both locally and remotely,
    // but are referred to by the aliases listed in remote_xcodes.
    Set<XcodeVersionRuleData> mutuallyAvailableVersions =
        Sets.newHashSet(remoteVersions.getAvailableVersions());
    mutuallyAvailableVersions.retainAll(Sets.newHashSet(localVersions.getAvailableVersions()));
    Map<String, XcodeVersionRuleData> localAliasesToVersionMap;
    Map<String, XcodeVersionRuleData> remoteAliasesToVersionMap;
    try {
      localAliasesToVersionMap = aliasesToVersionMap(localVersions.getAvailableVersions());
      remoteAliasesToVersionMap = aliasesToVersionMap(remoteVersions.getAvailableVersions());
    } catch (XcodeConfigException e) {
      throw ruleContext.throwWithRuleError(e.getMessage());
    }
    if (!Strings.isNullOrEmpty(versionOverrideFlag)) {
      XcodeVersionRuleData specifiedVersionFromRemote =
          remoteAliasesToVersionMap.get(versionOverrideFlag);
      XcodeVersionRuleData specifiedVersionFromLocal =
          localAliasesToVersionMap.get(versionOverrideFlag);
      if (specifiedVersionFromLocal != null && specifiedVersionFromRemote != null) {
        return Maps.immutableEntry(specifiedVersionFromRemote, XcodeConfigInfo.Availability.BOTH);
      } else if (specifiedVersionFromLocal != null) {
        String error =
            String.format(
                "--xcode_version=%1$s specified, but it is not available remotely. Actions"
                    + " requiring Xcode will be run locally, which could make your build"
                    + " slower.",
                versionOverrideFlag);
        if (!mutuallyAvailableVersions.isEmpty()) {
          error =
              error
                  + String.format(
                      " Consider using one of [%s].",
                      printableXcodeVersions(mutuallyAvailableVersions));
        }
        ruleContext.ruleWarning(error);
        return Maps.immutableEntry(specifiedVersionFromLocal, XcodeConfigInfo.Availability.LOCAL);
      } else if (specifiedVersionFromRemote != null) {
        ruleContext.ruleWarning(
            String.format(
                "--xcode_version=%1$s specified, but it is not available locally. Your build"
                    + " will fail if any actions require a local Xcode. If you believe you have"
                    + " '%1$s' installed, try running \"blaze shutdown\", and then re-run your"
                    + " command.  localy available versions: [%2$s]. remotely available"
                    + " versions: [%3$s]",
                versionOverrideFlag,
                printableXcodeVersions(localVersions.getAvailableVersions()),
                printableXcodeVersions(remoteVersions.getAvailableVersions())));
        return Maps.immutableEntry(specifiedVersionFromRemote, XcodeConfigInfo.Availability.REMOTE);
      } else { // if (specifiedVersionFromRemote == null && specifiedVersionFromLocal == null)
        ruleContext.throwWithRuleError(
            String.format(
                "--xcode_version=%1$s specified, but '%1$s' is not an available Xcode version."
                    + " localy available versions: [%2$s]. remotely available versions:"
                    + " [%3$s]. If you believe you have '%1$s' installed, try running \"blaze"
                    + " shutdown\", and then re-run your command.",
                versionOverrideFlag,
                printableXcodeVersions(localVersions.getAvailableVersions()),
                printableXcodeVersions(remoteVersions.getAvailableVersions())));
      }
    }
    if (!mutuallyAvailableVersions.isEmpty()) {
      DottedVersion newestVersionNumber = DottedVersion.fromStringUnchecked("0.0");
      XcodeVersionRuleData defaultVersion = null;
      for (XcodeVersionRuleData versionRuleData : mutuallyAvailableVersions) {
        if (versionRuleData.getVersion().compareTo(newestVersionNumber) > 0) {
          defaultVersion = versionRuleData;
          newestVersionNumber = defaultVersion.getVersion();
        }
      }
      // This should never occur. All input versions should be above 0.0.
      checkState(defaultVersion != null);
      return Maps.immutableEntry(defaultVersion, XcodeConfigInfo.Availability.BOTH);
    } else {
      ruleContext.ruleWarning(
          String.format(
              "Using a local Xcode version, '%s', since there are no"
                  + " remotely available Xcodes on this machine. Consider downloading one of the"
                  + " remotely available Xcode versions (%s) in order to get the best build"
                  + " performance.",
              localVersions.getDefaultVersion().getVersion(),
              printableXcodeVersions(remoteVersions.getAvailableVersions())));
      return Maps.immutableEntry(
          localVersions.getDefaultVersion(), XcodeConfigInfo.Availability.LOCAL);
    }
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
    if (xcodeVersionRules == null) {
      return aliasesToXcodeRules;
    }
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

  public static XcodeConfigInfo getXcodeConfigInfo(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(
        XcodeConfigRule.XCODE_CONFIG_ATTR_NAME,
        RuleConfiguredTarget.Mode.TARGET,
        XcodeConfigInfo.PROVIDER);
  }
}
