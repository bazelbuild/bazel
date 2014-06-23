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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.RunfilesCollector;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.RunfilesSupport;
import com.google.devtools.build.lib.view.actions.SpawnAction;

/**
 * Helper class to create shell archive actions.
 */
final class ShHelper {
  /**
   * Initialization code for the implicit .sar (script package) output.
   */
  public static void initializeSarOutput(RuleContext ruleContext, Artifact executable,
      Artifact executableScript, RunfilesSupport runfiles) {
    boolean randomTmpDir =
        ruleContext.attributes().get("random_tmpdir", Type.BOOLEAN).booleanValue();
    String bashVersion = ruleContext.attributes().get("bash_version", Type.STRING);

    // The executable that generates the .sar file.
    Artifact packageTool = ruleContext
        .getPrerequisite("$package_tool", Mode.HOST, FilesToRunProvider.class)
        .getExecutable();

    // Create the sar package, but don't add it to filesToBuild; it's only built when requested.
    Artifact sarOutput = ruleContext.getImplicitOutputArtifact(ShRuleClasses.SAR_PACKAGE_FILENAME);

    String bashPath = ShRuleClasses.BASH_BINARY_BINDINGS.get(bashVersion).execPath;

    // Generate the creating command.
    StringBuilder cmd = new StringBuilder();
    cmd.append(packageTool.getExecPathString());
    cmd.append(" " + executable.getExecPathString());
    cmd.append(" --gbash_color=always");
    cmd.append(" --random_tmpdir=" + (randomTmpDir ? "1" : "0"));
    cmd.append(" --rule=" + executable.getFilename());
    cmd.append(" --output=" + sarOutput.getExecPathString());
    cmd.append(" --minloglevel=1");
    cmd.append(" --compress=gzip");
    // If using an embedded bash, specify where to find it.
    if (!bashVersion.equals(ShRuleClasses.SYSTEM_BASH_VERSION)) {
     cmd.append(" --bash=google3/" + bashPath);
    }

    // Generate the creating action.
    new SpawnAction.Builder(ruleContext)
       .addInputs(getDependencyRunfiles(ruleContext, "deps", Mode.TARGET))
       .addInputs(getDependencyRunfiles(ruleContext, "data", Mode.DATA))
       .addInput(packageTool)
       .setShellCommand(cmd.toString())
        // We need this rule's runfiles tree as an input, as it by definition defines
        // the runtime dependencies that need to be packaged. This is not circular because
        // the .sar is not itself part of the runfiles.
       .addInput(runfiles.getRunfilesMiddleman())
       .addInput(runfiles.getExecutable())
       .addInputManifest(
           runfiles.getRunfilesManifest(), executable.getExecPathString() + ".runfiles/")
       .addOutput(sarOutput)
       .setProgressMessage("Packaging script " + executableScript.getFilename()
           + " as sar archive")
       .setMnemonic("ShPack")
       .build();
  }

  /**
   * Returns the runfiles of the dependency targets listed under the given attribute.
   */
  private static Iterable<Artifact> getDependencyRunfiles(
      RuleContext ruleContext, String attr, Mode mode) {
    ImmutableList.Builder<Artifact> runfiles = ImmutableList.builder();
    for (RunfilesProvider r : ruleContext.getPrerequisites(attr, mode, RunfilesProvider.class)) {
      runfiles.addAll(r.getTransitiveRunfiles(RunfilesCollector.State.DEFAULT).getAllArtifacts());
    }
    return runfiles.build();
  }
}
