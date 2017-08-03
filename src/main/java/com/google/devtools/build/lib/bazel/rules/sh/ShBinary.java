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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ExecutableSymlinkAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.bazel.rules.BazelConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.util.OS;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Implementation for the sh_binary rule.
 */
public class ShBinary implements RuleConfiguredTargetFactory {

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

  // Write launch info to buffer, return the number of bytes written.
  private static int writeLaunchInfo(ByteArrayOutputStream buffer, String key, String value)
      throws IOException {
    byte[] keyBytes = key.getBytes(UTF_8);
    byte[] valueBytes = value.getBytes(UTF_8);
    buffer.write(keyBytes);
    buffer.write('=');
    buffer.write(valueBytes);
    buffer.write('\0');
    return keyBytes.length + valueBytes.length + 2;
  }

  private static boolean isWindowsExecutable(Artifact artifact) {
    return artifact.getExtension().equals("exe")
        || artifact.getExtension().equals("cmd")
        || artifact.getExtension().equals("bat");
  }

  private static Artifact launcherForWindows(
      RuleContext ruleContext, Artifact primaryOutput, Artifact mainFile)
      throws RuleErrorException {
    if (isWindowsExecutable(mainFile)) {
      // If the extensions don't match, we should always respect mainFile's extension.
      if (mainFile.getExtension().equals(primaryOutput.getExtension())) {
        return primaryOutput;
      } else {
        ruleContext.ruleError(
            "Source file is a Windows executable file,"
                + " target name extension should match source file extension");
        throw new RuleErrorException();
      }
    }

    // The launcher file consists of a base launcher binary and the launch information appended to
    // the binary. The length of launch info is a signed 64-bit integer written at the end of
    // the binary in little endian.
    Artifact launcher = ruleContext.getPrerequisiteArtifact("$launcher", Mode.HOST);
    Artifact bashLauncher =
        ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");
    Artifact launchInfoFile =
        ruleContext.getRelatedArtifact(bashLauncher.getRootRelativePath(), ".launch_info");

    ByteArrayOutputStream launchInfo = new ByteArrayOutputStream();
    Long dataSize = 0L;
    try {
      dataSize += writeLaunchInfo(launchInfo, "binary_type", "Bash");
      dataSize += writeLaunchInfo(launchInfo, "workspace_name", ruleContext.getWorkspaceName());
      dataSize +=
          writeLaunchInfo(
              launchInfo,
              "bash_bin_path",
              ruleContext
                  .getFragment(BazelConfiguration.class)
                  .getShellExecutable()
                  .getPathString());
      dataSize += writeLaunchInfo(launchInfo, "bash_main_file", mainFile.getExecPathString());

      ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);
      // All Windows versions are little endian.
      buffer.order(ByteOrder.LITTLE_ENDIAN);
      buffer.putLong(dataSize);

      launchInfo.write(buffer.array());
    } catch (IOException e) {
      ruleContext.ruleError(e.getMessage());
      throw new RuleErrorException();
    }

    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(),
            launchInfoFile,
            ByteSource.wrap(launchInfo.toByteArray()),
            /*makeExecutable=*/ false));
    String path = ruleContext.getConfiguration().getActionEnvironment().getFixedEnv().get("PATH");
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(launcher)
            .addInput(launchInfoFile)
            .addOutput(bashLauncher)
            .setShellCommand(
                "cmd.exe /c \"copy /Y /B "
                    + launcher.getExecPathString().replace('/', '\\')
                    + "+"
                    + launchInfoFile.getExecPathString().replace('/', '\\')
                    + " "
                    + bashLauncher.getExecPathString().replace('/', '\\')
                    + " > nul\"")
            .setEnvironment(ImmutableMap.of("PATH", path))
            .setMnemonic("BuildBashLauncher")
            .build(ruleContext));

    return bashLauncher;
  }
}
