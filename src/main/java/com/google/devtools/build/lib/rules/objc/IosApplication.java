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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition;

/**
 * Implementation for {@code ios_application}.
 *
 * @deprecated The native bundling rules have been deprecated. This class will be removed in the
 *     future.
 */
@Deprecated
public class IosApplication extends ReleaseBundlingTargetFactory {

  /**
   * Transition that when applied to a target generates a configured target for each value in
   * {@code --ios_multi_cpus}, such that {@code --ios_cpu} is set to a different one of those values
   * in the configured targets.
   */
  public static final SplitTransition<BuildOptions> SPLIT_ARCH_TRANSITION =
      new SplitArchTransition();

  private static final ImmutableSet<Attribute> DEPENDENCY_ATTRIBUTES =
      ImmutableSet.of(
          new Attribute("binary", Mode.SPLIT),
          new Attribute("extensions", Mode.TARGET));

  public IosApplication() {
    super(ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT, DEPENDENCY_ATTRIBUTES);
  }
  
  /**
   * Validates that there is exactly one watch extension for each OS version.
   */
  @Override
  protected void validateAttributes(RuleContext ruleContext) {
    Iterable<ObjcProvider> extensionProviders = ruleContext.getPrerequisites(
        "extensions", Mode.TARGET, ObjcProvider.class);
    if (hasMoreThanOneWatchExtension(extensionProviders, Flag.HAS_WATCH1_EXTENSION)
        || hasMoreThanOneWatchExtension(extensionProviders, Flag.HAS_WATCH2_EXTENSION)) {
      ruleContext.attributeError("extensions", "An iOS application can contain exactly one "
          + "watch extension for each watch OS version");
    }
  }

  private boolean hasMoreThanOneWatchExtension(
      Iterable<ObjcProvider> objcProviders, final Flag watchExtensionVersionFlag) {
    return Streams.stream(objcProviders)
            .filter(objcProvider -> objcProvider.is(watchExtensionVersionFlag))
            .count()
        > 1;
  }

  @Override
  protected void configureTarget(RuleConfiguredTargetBuilder target, RuleContext ruleContext,
      ReleaseBundlingSupport releaseBundlingSupport) throws InterruptedException {
    // If this is an application built for the simulator, make it runnable.
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    if (appleConfiguration.getMultiArchPlatform(PlatformType.IOS) == ApplePlatform.IOS_SIMULATOR) {
      Artifact runnerScript = ObjcRuleClasses.intermediateArtifacts(ruleContext).runnerScript();
      Artifact ipaFile = ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
      releaseBundlingSupport.registerGenerateRunnerScriptAction(runnerScript, ipaFile);
      target.setRunfilesSupport(releaseBundlingSupport.runfilesSupport(runnerScript), runnerScript);
    }
  }
}
