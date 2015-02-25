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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/**
 * Implementation for {@code objc_xcodeproj}.
 */
public class ObjcXcodeproj implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    XcodeProvider.Project project = XcodeProvider.Project.fromTopLevelTargets(
        ruleContext.getPrerequisites("deps", Mode.TARGET, XcodeProvider.class));
    Artifact pbxproj = ruleContext.getImplicitOutputArtifact(XcodeSupport.PBXPROJ);

    ObjcActionsBuilder actionsBuilder = ObjcRuleClasses.actionsBuilder(ruleContext);
    actionsBuilder.registerXcodegenActions(
        new ObjcRuleClasses.Tools(ruleContext), pbxproj, project);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.create(Order.STABLE_ORDER, pbxproj))
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .build();
  }
}
