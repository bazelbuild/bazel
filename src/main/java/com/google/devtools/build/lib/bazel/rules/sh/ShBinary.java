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
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.ExecutableSymlinkAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.bazel.rules.BazelConfiguration;
import com.google.devtools.build.lib.bazel.rules.NativeLauncherUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.OS;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

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
        (OS.getCurrent() == OS.WINDOWS) ? launcherForWindows(ruleContext, symlink, src) : symlink;
    if (!symlink.equals(mainExecutable)) {
      filesToBuildBuilder.add(mainExecutable);
      runfilesBuilder.addArtifact(symlink);
    }
    NestedSet<Artifact> filesToBuild = filesToBuildBuilder.build();
    Runfiles runfiles =
        runfilesBuilder
            .addTransitiveArtifacts(filesToBuild)
            .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
            .build();

    // Create the RunfilesSupport with the mainExecutable's name. On Windows, this way the runfiles
    // directory's name is derived from the launcher (yielding "%{name}.cmd.runfiles" or
    // "%{name}.exe.runfiles").
    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, runfiles, mainExecutable);
    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .setRunfilesSupport(runfilesSupport, mainExecutable)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .build();
  }

  private static boolean isWindowsExecutable(Artifact artifact) {
    return artifact.getExtension().equals("exe")
        || artifact.getExtension().equals("cmd")
        || artifact.getExtension().equals("bat");
  }

  private static Artifact createWindowsExeLauncher(RuleContext ruleContext)
      throws RuleErrorException {
    Artifact bashLauncher =
        ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");

    ByteArrayOutputStream launchInfo = new ByteArrayOutputStream();
    try {
      NativeLauncherUtil.writeLaunchInfo(launchInfo, "binary_type", "Bash");
      NativeLauncherUtil.writeLaunchInfo(
          launchInfo, "workspace_name", ruleContext.getWorkspaceName());
      NativeLauncherUtil.writeLaunchInfo(
          launchInfo,
          "bash_bin_path",
          ruleContext.getFragment(BazelConfiguration.class).getShellExecutable().getPathString());
      NativeLauncherUtil.writeDataSize(launchInfo);
    } catch (IOException e) {
      ruleContext.ruleError(e.getMessage());
      throw new RuleErrorException();
    }

    NativeLauncherUtil.createNativeLauncherActions(ruleContext, bashLauncher, launchInfo);

    return bashLauncher;
  }

  private static Artifact launcherForWindows(
      RuleContext ruleContext, Artifact primaryOutput, Artifact mainFile)
      throws RuleErrorException {
    if (isWindowsExecutable(mainFile)) {
      if (mainFile.getExtension().equals(primaryOutput.getExtension())) {
        return primaryOutput;
      } else {
        // If the extensions don't match, we should always respect mainFile's extension.
        ruleContext.ruleError(
            "Source file is a Windows executable file,"
                + " target name extension should match source file extension");
        throw new RuleErrorException();
      }
    }

    if (ruleContext.getConfiguration().enableWindowsExeLauncher()) {
      return createWindowsExeLauncher(ruleContext);
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
