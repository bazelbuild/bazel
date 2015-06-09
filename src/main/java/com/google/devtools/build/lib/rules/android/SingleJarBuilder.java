// Copyright 2015 Google Inc. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.Jvm;

import java.util.List;

/**
 * Utility for configuring a SingleJar action. 
 */
public class SingleJarBuilder {
  /**
   * Memory consumption of SingleJar is about 250 bytes per entry in the output file. Unfortunately,
   * the JVM tends to kill the process with an OOM long before we're at the limit. In the most
   * recent example, 400 MB of memory was enough for about 500,000 entries.
   */
  private static final String SINGLEJAR_MAX_MEMORY = "-Xmx1600m";

  /**
   * Estimated resource consumption for a SingleJar action. These values are based on those
   * determined to be appropriate for an action creating deploy jars.
   */
  private static final ResourceSet RESOURCE_SET =
      ResourceSet.createWithRamCpuIo(/*memoryMb = */200.0, /*cpuUsage = */.2, /*ioUsage=*/.2);

  private final RuleContext ruleContext;
  private Artifact outputJar;
  private ImmutableList.Builder<Artifact> inputJars;

  /** Creates a builder using the configuration of the rule as the action configuration. */
  public SingleJarBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.inputJars = ImmutableList.builder();
  }

  /** Sets the artifact to create with the action. */
  public SingleJarBuilder setOutputJar(Artifact outputJar) {
    this.outputJar = outputJar;
    return this;
  }

  /** Adds a jar to the list of jars to be merged. */
  public SingleJarBuilder addInputJar(Artifact inputJar) {
    this.inputJars.add(inputJar);
    return this;
  }

  /** Adds a list of jars to the list of jars to be merged. */
  public SingleJarBuilder addInputJars(Iterable<Artifact> inputJars) {
    this.inputJars.addAll(inputJars);
    return this;
  }

  /** Builds the action as configured. */
  public void build() {
    ImmutableList<Artifact> inputs = inputJars.build();
    List<String> jvmArgs = ImmutableList.of("-client", SINGLEJAR_MAX_MEMORY);
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();
    checkNotNull(outputJar);
    commandLine.addExecPath("--output", outputJar);
    commandLine.add("--dont_change_compression");
    commandLine.add("--exclude_build_data");
    commandLine.addExecPaths("--sources", inputs);
    ruleContext.registerAction(new SpawnAction.Builder()
        .addInputs(inputs)
        .addTransitiveInputs(JavaCompilationHelper.getHostJavabaseInputs(ruleContext))
        .addOutput(outputJar)
        .setResources(RESOURCE_SET)
        .setJarExecutable(
            ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable(),
            ruleContext.getPrerequisiteArtifact("$singlejar", Mode.HOST),
            jvmArgs)
        .setCommandLine(commandLine.build())
        .useParameterFile(ParameterFileType.SHELL_QUOTED)
        .setProgressMessage("Building classes.jar for .aar")
        .setMnemonic("AarClassesJar")
        .build(ruleContext));
  }
}
