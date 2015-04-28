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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;

/**
 * Implementation for {$code experimental_ios_test} rule in Bazel.
 *
 * <p>Note that this will be renamed to ${code ios_test}, and {@link
 * com.google.devtools.build.lib.bazel.rules.objc.BazelIosTest} will be removed when it is slightly
 * more feature complete.
 */
public final class ExperimentalIosTest extends IosTest {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext, ObjcCommon common,
      XcodeProvider xcodeProvider, NestedSet<Artifact> filesToBuild) throws InterruptedException {

    Runfiles.Builder runfilesBuilder = new Runfiles.Builder();
    NestedSetBuilder<Artifact> filesToBuildBuilder = NestedSetBuilder.<Artifact>stableOrder();
    filesToBuildBuilder.addTransitive(filesToBuild);

    TestSupport testSupport = new TestSupport(ruleContext)
        .registerTestRunnerActionsForSimulator()
        .addRunfiles(runfilesBuilder)
        .addFilesToBuild(filesToBuildBuilder);

    Artifact executable = testSupport.generatedTestScript();

    Runfiles runfiles = runfilesBuilder.build();
    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, runfiles, executable);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuildBuilder.build())
        .add(XcodeProvider.class, xcodeProvider)
        .add(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .setRunfilesSupport(runfilesSupport, executable)
        .build();
  }
}
