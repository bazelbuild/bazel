// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;

/**
 * Implementation for {@code ios_application}.
 */
public class IosApplication extends ReleaseBundlingTargetFactory {

  private static final ImmutableSet<Attribute> DEPENDENCY_ATTRIBUTES =
      ImmutableSet.of(
          new Attribute("binary", Mode.SPLIT),
          new Attribute("extensions", Mode.TARGET));

  public IosApplication() {
    super(ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT, XcodeProductType.APPLICATION,
        ExposeAsNestedBundle.NO, DEPENDENCY_ATTRIBUTES);
  }

  @Override
  protected OptionsProvider optionsProvider(RuleContext ruleContext) {
    return new OptionsProvider.Builder()
        .addInfoplists(ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET).list())
        .addTransitive(
            Optional.fromNullable(
                ruleContext.getPrerequisite("options", Mode.TARGET, OptionsProvider.class)))
        .build();
  }

  @Override
  protected void configureTarget(RuleConfiguredTargetBuilder target, RuleContext ruleContext,
      ReleaseBundlingSupport releaseBundlingSupport) {
    // If this is an application built for the simulator, make it runnable.
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    if (objcConfiguration.getBundlingPlatform() == Platform.SIMULATOR) {
      Artifact runnerScript = ObjcRuleClasses.intermediateArtifacts(ruleContext).runnerScript();
      Artifact ipaFile = ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA);
      releaseBundlingSupport.registerGenerateRunnerScriptAction(runnerScript, ipaFile);
      target.setRunfilesSupport(releaseBundlingSupport.runfilesSupport(runnerScript), runnerScript);
    }
  }
}
