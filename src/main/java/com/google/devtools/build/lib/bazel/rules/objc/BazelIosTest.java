// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.objc.IosTest;
import com.google.devtools.build.lib.rules.objc.ObjcCommon;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses;
import com.google.devtools.build.lib.rules.objc.XcodeProvider;

/**
 * Implementation for ios_test rule in Bazel.
 */
public final class BazelIosTest extends IosTest {
  static final String IOS_TEST_ON_BAZEL_ATTR = "$ios_test_on_bazel";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext, ObjcCommon common,
      XcodeProvider xcodeProvider, NestedSet<Artifact> filesToBuild) throws InterruptedException {
    Artifact testRunner = ruleContext.getPrerequisiteArtifact(IOS_TEST_ON_BAZEL_ATTR, Mode.TARGET);

    Artifact testScript =
        ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, "_test_script");

    ruleContext.registerAction(
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            testRunner,
            testScript,
            ImmutableList.<TemplateExpansionAction.Substitution>of(),
            /*executable=*/ true));

    Runfiles runfiles = new Runfiles.Builder().addArtifact(testScript).build();
    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, runfiles, testScript);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(filesToBuild)
            .add(testRunner)
            .build())
        .add(XcodeProvider.class, xcodeProvider)
        .add(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .setRunfilesSupport(runfilesSupport, testRunner)
        .build();
  }
}
