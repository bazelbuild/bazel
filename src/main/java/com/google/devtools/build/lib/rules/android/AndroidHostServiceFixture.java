// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;

/** An implementation of the {@code android_host_service_fixture} rule. */
public class AndroidHostServiceFixture implements RuleConfiguredTargetFactory {

  private final AndroidSemantics androidSemantics;

  protected AndroidHostServiceFixture(AndroidSemantics androidSemantics) {
    this.androidSemantics = androidSemantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);
    RuleConfiguredTargetBuilder ruleBuilder = new RuleConfiguredTargetBuilder(ruleContext);
    NestedSet<Artifact> supportApks = AndroidCommon.getSupportApks(ruleContext);
    FilesToRunProvider executable =
        ruleContext.getExecutablePrerequisite("executable", TransitionMode.HOST);

    NestedSet<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(supportApks)
            .addTransitive(executable.getFilesToRun())
            .build();
    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addTransitiveArtifacts(supportApks)
            .merge(executable.getRunfilesSupport())
            .build();
    return ruleBuilder
        .setFilesToBuild(filesToBuild)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .addNativeDeclaredProvider(
            new AndroidHostServiceFixtureInfoProvider(
                executable.getExecutable(),
                ruleContext.getExpander().withDataLocations().tokenized("service_names"),
                AndroidCommon.getSupportApks(ruleContext),
                ruleContext.attributes().get("provides_test_args", Type.BOOLEAN),
                ruleContext.attributes().get("daemon", Type.BOOLEAN)))
        .build();
  }
}
