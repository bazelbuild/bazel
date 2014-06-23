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
package com.google.devtools.build.lib.bazel.rules.sh;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder.StatelessRunfilesProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesCollector;
import com.google.devtools.build.lib.view.RunfilesProvider;

/**
 * Implementation for the sh_library rule.
 */
public class ShLibrary implements RuleConfiguredTargetFactory {

  @Override
  public RuleConfiguredTarget create(RuleContext ruleContext) {
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET))
        .addAll(ruleContext.getPrerequisiteArtifacts("deps", Mode.TARGET))
        .addAll(ruleContext.getPrerequisiteArtifacts("data", Mode.DATA))
        .build();
    Runfiles runfiles = new Runfiles.Builder()
        .addArtifacts(filesToBuild)
        // When visiting proto_library rules, collect the source files for the benefit of jobs that
        // parse .proto files at runtime.
        .addRunfiles(RunfilesCollector.State.INTERPRETED, ruleContext)
        .build();
    return new GenericRuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .addProvider(RunfilesProvider.class, new StatelessRunfilesProvider(runfiles))
        .build();
  }
}
