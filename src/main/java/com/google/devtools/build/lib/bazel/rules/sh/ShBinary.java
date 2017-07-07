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
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.bazel.rules.BazelConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.util.OS;

/**
 * Implementation for the sh_binary rule.
 */
public class ShBinary implements RuleConfiguredTargetFactory {
  private static final Template STUB_SCRIPT_WINDOWS =
      Template.forResource(ShBinary.class, "sh_stub_template_windows.txt");

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws RuleErrorException {
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

    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().add(src).add(symlink);
    Runfiles.Builder runfilesBuilder =
        new Runfiles.Builder(
            ruleContext.getWorkspaceName(),
            ruleContext.getConfiguration().legacyExternalRunfiles());

    Artifact mainExecutable =
        (OS.getCurrent() == OS.WINDOWS) ? wrapperForWindows(ruleContext, symlink, src) : symlink;
    if (symlink != mainExecutable) {
      filesToBuildBuilder.add(mainExecutable);
      runfilesBuilder.addArtifact(symlink);
    }
    NestedSet<Artifact> filesToBuild = filesToBuildBuilder.build();
    Runfiles runfiles =
        runfilesBuilder
            .addTransitiveArtifacts(filesToBuild)
            .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
            .build();

    // Create the RunfilesSupport with the symlink's name, even on Windows. This way the runfiles
    // directory's name is derived from the symlink (yielding "%{name}.runfiles) and not from the
    // wrapper script (yielding "%{name}.cmd.runfiles").
    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, runfiles, symlink);
    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .setRunfilesSupport(runfilesSupport, mainExecutable)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .build();
  }

  private static Artifact wrapperForWindows(
      RuleContext ruleContext, Artifact primaryOutput, Artifact mainFile) {
    if (primaryOutput.getFilename().endsWith(".exe")
        || primaryOutput.getFilename().endsWith(".bat")
        || primaryOutput.getFilename().endsWith(".cmd")) {
      String suffix =
          primaryOutput.getFilename().substring(primaryOutput.getFilename().length() - 4);
      if (mainFile.getFilename().endsWith(suffix)) {
        return primaryOutput;
      }
    }

    Artifact wrapper =
        ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".cmd");
    ruleContext.registerAction(
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            wrapper,
            STUB_SCRIPT_WINDOWS,
            ImmutableList.of(
                Substitution.of(
                    "%bash_exe_path%",
                    ruleContext
                        .getFragment(BazelConfiguration.class)
                        .getShellExecutable()
                        .getPathString())),
            true));
    return wrapper;
  }
}
