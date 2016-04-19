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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.ExecutableSymlinkAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/**
 * Implementation for the sh_binary rule.
 */
public class ShBinary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) {
    ImmutableList<Artifact> srcs = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();
    if (srcs.size() != 1) {
      ruleContext.attributeError("srcs", "you must specify exactly one file in 'srcs'");
      return null;
    }

    Artifact symlink = ruleContext.createOutputArtifact();
    // Note that src is used as the executable script too
    Artifact src = srcs.get(0);
    // The interpretation of this deceptively simple yet incredibly generic rule is complicated
    // by the distinction between targets and (not properly encapsulated) artifacts. It depends
    // on the notion of other rule's "files-to-build" sets, which are undocumented, making it
    // impossible to give a precise definition of what this rule does in all cases (e.g. what
    // happens when srcs = ['x', 'y'] but 'x' is an empty filegroup?). This is a pervasive
    // problem in Blaze.
    ruleContext.registerAction(
        new ExecutableSymlinkAction(ruleContext.getActionOwner(), src, symlink));

    NestedSet<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .add(src)
        .add(symlink)
        .build();
    Runfiles runfiles = new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
        .addTransitiveArtifacts(filesToBuild)
        .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
        .build();
    RunfilesSupport runfilesSupport = RunfilesSupport.withExecutable(
        ruleContext, runfiles, symlink);
    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .setRunfilesSupport(runfilesSupport, symlink)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .build();
  }
}
