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
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition.ConfigurationDistinguisher;

import java.io.Serializable;

/**
 * Implementation for {@code ios_extension}.
 */
public class IosExtension extends ReleaseBundlingTargetFactory {

  /**
   * Transition that when applied to a target generates a configured target for each value in
   * {@code --ios_multi_cpus}, such that {@code --ios_cpu} is set to a different one of those values
   * in the configured targets.
   *
   * <p>Also ensures that, no matter whether {@code --ios_multi_cpus} is set, {@code
   * --ios_minimum_os} is at least {@code 8.0} as Apple requires this for extensions.
   */
  static final SplitTransition<BuildOptions> MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION =
      new ExtensionSplitArchTransition();

  // Apple only accepts extensions starting at 8.0.
  @VisibleForTesting
  static final DottedVersion EXTENSION_MINIMUM_OS_VERSION = DottedVersion.fromString("8.0");

  public IosExtension() {
    super(ReleaseBundlingSupport.EXTENSION_BUNDLE_DIR_FORMAT, XcodeProductType.EXTENSION,
        ImmutableSet.of(new Attribute("binary", Mode.SPLIT)),
        ConfigurationDistinguisher.IOS_EXTENSION);
  }

  @Override
  protected DottedVersion bundleMinimumOsVersion(RuleContext ruleContext) {
    return determineMinimumOsVersion(ObjcRuleClasses.objcConfiguration(ruleContext).getMinimumOs());
  }

  @Override
  protected ObjcProvider exposedObjcProvider(RuleContext ruleContext) throws InterruptedException {
    // Nest this target's bundle under final IPA
    return new ObjcProvider.Builder()
        .add(MERGE_ZIP, ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA))
        .build();
  }

  private static DottedVersion determineMinimumOsVersion(DottedVersion fromFlag) {
    // Extensions are not accepted by Apple below version 8.0. While applications built with a
    // minimum iOS version of less than 8.0 may contain extensions in their bundle, the extension
    // itself needs to be built with 8.0 or higher. This logic overrides (if necessary) any
    // flag-set minimum iOS version for extensions only so that this requirement is not violated.
    return Ordering.natural().max(fromFlag, EXTENSION_MINIMUM_OS_VERSION);
  }

  /**
   * Split transition that configures the minimum iOS version in addition to architecture splitting.
   */
  private static class ExtensionSplitArchTransition extends SplitArchTransition
      implements Serializable {

    @Override
    protected ImmutableList<BuildOptions> defaultOptions(BuildOptions originalOptions) {
      ObjcCommandLineOptions objcOptions = originalOptions.get(ObjcCommandLineOptions.class);
      DottedVersion newMinimumVersion = determineMinimumOsVersion(objcOptions.iosMinimumOs);

      if (newMinimumVersion.equals(objcOptions.iosMinimumOs)) {
        return ImmutableList.of();
      }

      BuildOptions splitOptions = originalOptions.clone();
      setMinimumOsVersion(splitOptions, newMinimumVersion);
      splitOptions.get(ObjcCommandLineOptions.class).configurationDistinguisher =
          getConfigurationDistinguisher();
      return ImmutableList.of(splitOptions);
    }

    @Override
    protected void setAdditionalOptions(BuildOptions splitOptions, BuildOptions originalOptions) {
      DottedVersion fromFlag = originalOptions.get(ObjcCommandLineOptions.class).iosMinimumOs;
      setMinimumOsVersion(splitOptions, determineMinimumOsVersion(fromFlag));
    }

    @Override
    protected ConfigurationDistinguisher getConfigurationDistinguisher() {
      return ConfigurationDistinguisher.IOS_EXTENSION;
    }

    private void setMinimumOsVersion(BuildOptions splitOptions, DottedVersion newMinimumVersion) {
      splitOptions.get(ObjcCommandLineOptions.class).iosMinimumOs = newMinimumVersion;
    }
  }
}
