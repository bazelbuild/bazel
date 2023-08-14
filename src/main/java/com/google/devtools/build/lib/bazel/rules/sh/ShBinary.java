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
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction.LaunchInfo;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Implementation for the sh_binary rule.
 */
public class ShBinary implements RuleConfiguredTargetFactory {

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ImmutableList<Artifact> srcs = ruleContext.getPrerequisiteArtifacts("srcs").list();
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
        SymlinkAction.toExecutable(
            ruleContext.getActionOwner(), src, symlink, "Symlinking " + ruleContext.getLabel()));

    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().add(src).add(symlink);
    Runfiles.Builder runfilesBuilder =
        new Runfiles.Builder(
            ruleContext.getWorkspaceName(),
            ruleContext.getConfiguration().legacyExternalRunfiles());

    Artifact mainExecutable =
        ruleContext.isExecutedOnWindows() ? launcherForWindows(ruleContext, symlink, src) : symlink;
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
        .addNativeDeclaredProvider(
            InstrumentedFilesCollector.collect(ruleContext, ShCoverage.INSTRUMENTATION_SPEC))
        .build();
  }

  private static boolean isWindowsExecutable(Artifact artifact) {
    return artifact.getExtension().equals("exe")
        || artifact.getExtension().equals("cmd")
        || artifact.getExtension().equals("bat");
  }

  private static Artifact createWindowsExeLauncher(
      RuleContext ruleContext, PathFragment shExecutable) throws RuleErrorException {
    Artifact bashLauncher =
        ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");

    LaunchInfo launchInfo =
        LaunchInfo.builder()
            .addKeyValuePair("binary_type", "Bash")
            .addKeyValuePair("workspace_name", ruleContext.getWorkspaceName())
            .addKeyValuePair(
                "symlink_runfiles_enabled",
                ruleContext.getConfiguration().runfilesEnabled() ? "1" : "0")
            .addKeyValuePair("bash_bin_path", shExecutable.getPathString())
            .build();

    LauncherFileWriteAction.createAndRegister(ruleContext, bashLauncher, launchInfo);

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
        throw ruleContext.throwWithRuleError(
            "Source file is a Windows executable file,"
                + " target name extension should match source file extension");
      }
    }

    // TODO(b/234923262): Take exec_group into consideration when selecting sh tools
    PathFragment shExecutable =
        ShToolchain.getPathForPlatform(
            ruleContext.getConfiguration(), ruleContext.getExecutionPlatform());
    return createWindowsExeLauncher(ruleContext, shExecutable);
  }
}
