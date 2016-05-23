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
package com.google.devtools.build.lib.bazel.rules.sh;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/**
 * Implementation for the sh_library rule.
 */
public class ShLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws RuleErrorException {
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list())
        .addAll(ruleContext.getPrerequisiteArtifacts("deps", Mode.TARGET).list())
        .addAll(ruleContext.getPrerequisiteArtifacts("data", Mode.DATA).list())
        .build();
    Runfiles runfiles = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
        .addTransitiveArtifacts(filesToBuild)
        .build();
    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .build();
  }
}
