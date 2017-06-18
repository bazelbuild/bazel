// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;

/**
 * Helper class to create singlejar actions - singlejar can merge multiple zip files without
 * uncompressing all the entries, making it much faster than uncompressing and then recompressing
 * the files.
 */
@Immutable
public final class SingleJarActionBuilder {

  private static final ImmutableList<String> SOURCE_JAR_COMMAND_LINE_ARGS = ImmutableList.of(
      "--compression",
      "--normalize",
      "--exclude_build_data",
      "--warn_duplicate_resources");

  /**
   * Creates an Action that packages files into a Jar file.
   *
   * @param resources the resources to put into the Jar.
   * @param resourceJars the resource jars to merge into the jar
   * @param outputJar the Jar to create
   */
  public static void createSourceJarAction(
      RuleContext ruleContext,
      Map<PathFragment, Artifact> resources,
      Collection<Artifact> resourceJars,
      Artifact outputJar) {
    Artifact singleJar = getSingleJar(ruleContext);
    SpawnAction.Builder builder = new SpawnAction.Builder();
    // If singlejar's name ends with .jar, it is Java application, otherwise it is native.
    // TODO(asmundak): once https://github.com/bazelbuild/bazel/issues/2241 is fixed (that is,
    // the native singlejar is used on windows) remove support for the Java implementation
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext))
          .setJarExecutable(
              ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable(),
              singleJar,
              JavaToolchainProvider.fromRuleContext(ruleContext).getJvmOptions())
          .setExecutionInfo(ExecutionRequirements.WORKER_MODE_ENABLED);
    } else {
      builder.setExecutable(singleJar);
    }
    builder
        .addOutput(outputJar)
        .addInputs(resources.values())
        .addInputs(resourceJars)
        .setCommandLine(sourceJarCommandLine(outputJar, resources, resourceJars))
        .alwaysUseParameterFile(ParameterFileType.SHELL_QUOTED)
        .setProgressMessage("Building source jar " + outputJar.prettyPrint())
        .setMnemonic("JavaSourceJar");
    ruleContext.registerAction(builder.build(ruleContext));
  }

  /** Returns the SingleJar deploy jar Artifact. */
  private static Artifact getSingleJar(RuleContext ruleContext) {
    Artifact singleJar = JavaToolchainProvider.fromRuleContext(ruleContext).getSingleJar();
    if (singleJar != null) {
      return singleJar;
    }
    return ruleContext.getPrerequisiteArtifact("$singlejar", Mode.HOST);
  }

  private static CommandLine sourceJarCommandLine(Artifact outputJar,
      Map<PathFragment, Artifact> resources, Iterable<Artifact> resourceJars) {
    CustomCommandLine.Builder args = CustomCommandLine.builder();
    args.addExecPath("--output", outputJar);
    args.add(SOURCE_JAR_COMMAND_LINE_ARGS);
    args.addExecPaths("--sources", resourceJars);
    if (!resources.isEmpty()) {
      args.add("--resources");
      for (Map.Entry<PathFragment, Artifact> resource : resources.entrySet()) {
        args.addPaths("%s:%s", resource.getValue().getExecPath(), resource.getKey());
      }
    }
    return args.build();
  }
}
