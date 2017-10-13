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

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParamFileInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nullable;

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

  /** Constructs the base spawn for a singlejar action. */
  private static SpawnAction.Builder singleJarActionBuilder(RuleContext ruleContext) {
    Artifact singleJar = getSingleJar(ruleContext);
    SpawnAction.Builder builder = new SpawnAction.Builder();
    // If singlejar's name ends with .jar, it is Java application, otherwise it is native.
    // TODO(asmundak): once https://github.com/bazelbuild/bazel/issues/2241 is fixed (that is,
    // the native singlejar is used on windows) remove support for the Java implementation
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext))
          .setJarExecutable(
              JavaCommon.getHostJavaExecutable(ruleContext),
              singleJar,
              JavaToolchainProvider.fromRuleContext(ruleContext).getJvmOptions())
          .setExecutionInfo(ExecutionRequirements.WORKER_MODE_ENABLED);
    } else {
      builder.setExecutable(singleJar);
    }
    return builder;
  }

  /**
   * Creates an Action that packages files into a Jar file.
   *
   * @param semantics the current Java semantics, which must be non-{@code null} if {@code
   *     resources} is non-empty
   * @param resources the resources to put into the Jar
   * @param resourceJars the resource jars to merge into the jar
   * @param outputJar the Jar to create
   */
  public static void createSourceJarAction(
      RuleContext ruleContext,
      @Nullable JavaSemantics semantics,
      ImmutableCollection<Artifact> resources,
      NestedSet<Artifact> resourceJars,
      Artifact outputJar) {
    requireNonNull(ruleContext);
    requireNonNull(resourceJars);
    requireNonNull(outputJar);
    if (!resources.isEmpty()) {
      requireNonNull(semantics);
    }
    SpawnAction.Builder builder =
        singleJarActionBuilder(ruleContext)
            .addOutput(outputJar)
            .addInputs(resources)
            .addTransitiveInputs(resourceJars)
            .addCommandLine(
                sourceJarCommandLine(outputJar, semantics, resources, resourceJars),
                ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
            .setProgressMessage("Building source jar %s", outputJar.prettyPrint())
            .setMnemonic("JavaSourceJar");
    ruleContext.registerAction(builder.build(ruleContext));
  }

  /**
   * Creates an Action that merges jars into a single archive.
   *
   * @param jars the jars to merge.
   * @param output the output jar to create
   */
  public static void createSingleJarAction(
      RuleContext ruleContext, NestedSet<Artifact> jars, Artifact output) {
     requireNonNull(ruleContext);
    requireNonNull(jars);
    requireNonNull(output);
    SpawnAction.Builder builder =
        singleJarActionBuilder(ruleContext)
            .addOutput(output)
            .addInputs(jars)
            .addCommandLine(
                sourceJarCommandLine(
                    output, /* semantics= */ null, /* resources= */ ImmutableList.of(), jars),
                ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
            .setProgressMessage("Building singlejar jar %s", output.prettyPrint())
            .setMnemonic("JavaSingleJar");
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

  private static CommandLine sourceJarCommandLine(
      Artifact outputJar,
      JavaSemantics semantics,
      ImmutableCollection<Artifact> resources,
      NestedSet<Artifact> resourceJars) {
    CustomCommandLine.Builder args = CustomCommandLine.builder();
    args.addExecPath("--output", outputJar);
    args.addAll(SOURCE_JAR_COMMAND_LINE_ARGS);
    args.addExecPaths("--sources", resourceJars);
    if (!resources.isEmpty()) {
      args.add("--resources");
      args.addAll(VectorArg.of(resources).mapped(resource -> getResourceArg(semantics, resource)));
    }
    return args.build();
  }

  private static String getResourceArg(JavaSemantics semantics, Artifact resource) {
    return String.format(
        "%s:%s",
        resource.getExecPathString(),
        semantics.getDefaultJavaResourcePath(resource.getRootRelativePath()));
  }
}
