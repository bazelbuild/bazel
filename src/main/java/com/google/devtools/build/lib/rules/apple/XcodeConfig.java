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

import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * Implementation for the {@code xcode_config} rule.
 */
public class XcodeConfig implements RuleConfiguredTargetFactory {

  private static final DottedVersion MINIMUM_BITCODE_XCODE_VERSION = DottedVersion.fromString("7");

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    AppleConfiguration configuration = ruleContext.getFragment(AppleConfiguration.class);
    Optional<DottedVersion> versionOverrideFlag = configuration.getXcodeVersionOverrideFlag();

    DottedVersion targetVersion = resolveExplicitlyDefinedVersion(ruleContext);

    XcodeConfigProvider xcodeConfigProvider;
    if (targetVersion == null) {
      if (versionOverrideFlag.isPresent()) {
        xcodeConfigProvider = new XcodeConfigProvider(versionOverrideFlag.get());
      } else {
        xcodeConfigProvider = XcodeConfigProvider.hostSystemDefault();
      }
    } else {
      xcodeConfigProvider = new XcodeConfigProvider(targetVersion);
    }

    validateXcodeConfig(ruleContext, configuration, xcodeConfigProvider);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(XcodeConfigProvider.class, xcodeConfigProvider)
        .build();
  }

  private void validateXcodeConfig(RuleContext ruleContext, AppleConfiguration config,
      XcodeConfigProvider xcodeConfigProvider) {
    Optional<DottedVersion> xcodeVersion = xcodeConfigProvider.getXcodeVersion();
    if (config.getBitcodeMode() != AppleBitcodeMode.NONE
        && xcodeVersion.isPresent()
        && xcodeVersion.get().compareTo(MINIMUM_BITCODE_XCODE_VERSION) < 0) {
      ruleContext.ruleError(
          String.format("apple_bitcode mode '%s' is unsupported for xcode version '%s'",
              config.getBitcodeMode(), xcodeVersion.get()));
    }
  }

  /**
   * Returns the xcode version number corresponding to the {@code --xcode_version} flag, if there
   * is an available {@code xcode_version} target which recognizes the flag value as either
   * an official version or an alias. Returns null if no such target is found.
   */
  @Nullable private DottedVersion resolveExplicitlyDefinedVersion(RuleContext ruleContext) {
    AppleConfiguration configuration = ruleContext.getFragment(AppleConfiguration.class);
    Optional<DottedVersion> versionOverrideFlag = configuration.getXcodeVersionOverrideFlag();
    if (versionOverrideFlag.isPresent()) {
      // The version override flag is not necessarily an actual version - it may be a version
      // alias.
      DottedVersion explicitVerison =
          aliasesToVersionMap(ruleContext).get(versionOverrideFlag.get().toString());
      if (explicitVerison != null) {
        return explicitVerison;
      }
    } else { // No override specified. Use default.
      XcodeVersionProvider defaultProvider = ruleContext.getPrerequisite(
          XcodeConfigRule.DEFAULT_ATTR_NAME, Mode.TARGET, XcodeVersionProvider.class);

      if (defaultProvider != null) {
        return defaultProvider.getVersion();
      }
    }

    boolean requireDefinedVersions = ruleContext.attributes().get(
        XcodeConfigRule.REQUIRE_DEFINED_VERSIONS_ATTR_NAME, BOOLEAN);
    if (requireDefinedVersions) {
      ruleContext.ruleError(
          "xcode version config required an explicitly defined version, but none was available");
    }

    return null;
  }

  private static Map<String, DottedVersion> aliasesToVersionMap(RuleContext ruleContext) {
    Iterable<XcodeVersionProvider> xcodeVersionProviders =
        ruleContext.getPrerequisites(XcodeConfigRule.VERSIONS_ATTR_NAME, Mode.TARGET,
            XcodeVersionProvider.class);

    Map<String, DottedVersion> aliasesToVersionMap = Maps.newLinkedHashMap();
    for (XcodeVersionProvider versionProvider : xcodeVersionProviders) {
      for (String alias : versionProvider.getAliases()) {
        if (aliasesToVersionMap.put(alias, versionProvider.getVersion()) != null) {
          ruleErrorDuplicateAlias(alias, ruleContext);
        }
      }
      if (aliasesToVersionMap.put(
          versionProvider.getVersion().toString(), versionProvider.getVersion()) != null) {
        ruleErrorDuplicateAlias(versionProvider.getVersion().toString(), ruleContext);
      }
    }
    return aliasesToVersionMap;
  }
  
  private static void ruleErrorDuplicateAlias(String alias, RuleContext ruleContext) {
    Iterable<XcodeVersionProvider> xcodeVersionProviders =
        ruleContext.getPrerequisites(XcodeConfigRule.VERSIONS_ATTR_NAME, Mode.TARGET,
            XcodeVersionProvider.class);
    
    ImmutableList.Builder<Label> labelsContainingAlias = ImmutableList.builder();
    for (XcodeVersionProvider versionProvider : xcodeVersionProviders) {
      if (versionProvider.getAliases().contains(alias)
          || versionProvider.getVersion().toString().equals(alias)) {
        labelsContainingAlias.add(versionProvider.getLabel());
      }
    }
    ruleContext.ruleError(String.format(
        "'%s' is registered to multiple labels (%s) in a single xcode version "
        + "configuration", alias, Joiner.on(", ").join(labelsContainingAlias.build())));
  }
}