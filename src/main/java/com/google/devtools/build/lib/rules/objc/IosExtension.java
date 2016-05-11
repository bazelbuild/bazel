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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition;

import java.io.Serializable;

/**
 * Implementation for {@code ios_extension}.
 */
public class IosExtension extends ReleaseBundlingTargetFactory {

  // Apple only accepts extensions starting at 8.0.
  @VisibleForTesting
  static final DottedVersion EXTENSION_MINIMUM_OS_VERSION = DottedVersion.fromString("8.0");

  /**
   * Transition that when applied to a target generates a configured target for each value in
   * {@code --ios_multi_cpus}, such that {@code --ios_cpu} is set to a different one of those values
   * in the configured targets.
   *
   * <p>Also ensures that, no matter whether {@code --ios_multi_cpus} is set, {@code
   * --ios_minimum_os} is at least {@code 8.0} as Apple requires this for extensions.
   */
  static final SplitTransition<BuildOptions> MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION =
      new ExtensionSplitArchTransition(EXTENSION_MINIMUM_OS_VERSION,
          ConfigurationDistinguisher.IOS_EXTENSION);

  public IosExtension() {
    super(ReleaseBundlingSupport.EXTENSION_BUNDLE_DIR_FORMAT, XcodeProductType.EXTENSION,
        ImmutableSet.of(new Attribute("binary", Mode.SPLIT)),
        ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Override
  protected DottedVersion bundleMinimumOsVersion(RuleContext ruleContext) {
    return determineMinimumOsVersion(ObjcRuleClasses.objcConfiguration(ruleContext).getMinimumOs(),
        EXTENSION_MINIMUM_OS_VERSION);
  }

  @Override
  protected ObjcProvider exposedObjcProvider(
      RuleContext ruleContext, ReleaseBundlingSupport releaseBundlingSupport)
      throws InterruptedException {
    ObjcProvider.Builder builder =
        new ObjcProvider.Builder()
            // Nest this target's bundle under final IPA
            .add(MERGE_ZIP, ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA));

    releaseBundlingSupport.addExportedDebugArtifacts(builder, DsymOutputType.APP);
    return builder.build();
  }

  /**
   * Overrides (if necessary) any flag-set minimum iOS version for extensions only with given
   * minimum OS version.
   *
   * Extensions are not accepted by Apple below given mininumOSVersion. While applications built
   * with a minimum iOS version of less than give version may contain extensions in their bundle,
   * the extension itself needs to be built with given version or higher.
   *
   * @param fromFlag the minimum OS version from command line flag
   * @param minimumOSVersion the minumum OS version the extension should be built with
   */
  private static DottedVersion determineMinimumOsVersion(DottedVersion fromFlag,
      DottedVersion minimumOSVersion) {
    return Ordering.natural().max(fromFlag, minimumOSVersion);
  }

  /**
   * Split transition that configures the minimum iOS version in addition to architecture splitting.
   */
  static class ExtensionSplitArchTransition extends SplitArchTransition
      implements Serializable {

    private final DottedVersion minimumOSVersion;
    private final ConfigurationDistinguisher configurationDistinguisher;

    ExtensionSplitArchTransition(DottedVersion minimumOSVersion,
        ConfigurationDistinguisher configurationDistinguisher) {
      this.minimumOSVersion = minimumOSVersion;
      this.configurationDistinguisher = configurationDistinguisher;
    }

    @Override
    protected ImmutableList<BuildOptions> defaultOptions(BuildOptions originalOptions) {
      ObjcCommandLineOptions objcOptions = originalOptions.get(ObjcCommandLineOptions.class);
      DottedVersion newMinimumVersion = determineMinimumOsVersion(objcOptions.iosMinimumOs,
          minimumOSVersion);

      if (newMinimumVersion.equals(objcOptions.iosMinimumOs)) {
        return ImmutableList.of();
      }

      BuildOptions splitOptions = originalOptions.clone();
      setMinimumOsVersion(splitOptions, newMinimumVersion);
      splitOptions.get(AppleCommandLineOptions.class).configurationDistinguisher =
          getConfigurationDistinguisher();
      return ImmutableList.of(splitOptions);
    }

    @Override
    protected void setAdditionalOptions(BuildOptions splitOptions, BuildOptions originalOptions) {
      DottedVersion fromFlag = originalOptions.get(ObjcCommandLineOptions.class).iosMinimumOs;
      setMinimumOsVersion(splitOptions, determineMinimumOsVersion(fromFlag, minimumOSVersion));
    }

    @Override
    protected ConfigurationDistinguisher getConfigurationDistinguisher() {
      return configurationDistinguisher;
    }

    private void setMinimumOsVersion(BuildOptions splitOptions, DottedVersion newMinimumVersion) {
      splitOptions.get(ObjcCommandLineOptions.class).iosMinimumOs = newMinimumVersion;
    }
  }
}
