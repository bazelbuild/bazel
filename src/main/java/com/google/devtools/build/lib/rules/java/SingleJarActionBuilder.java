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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.function.Consumer;

/**
 * Helper class to create singlejar actions - singlejar can merge multiple zip files without
 * uncompressing all the entries, making it much faster than uncompressing and then recompressing
 * the files.
 */
@Immutable
public final class SingleJarActionBuilder {

  private static final ImmutableList<String> SOURCE_JAR_COMMAND_LINE_ARGS =
      ImmutableList.of(
          "--compression", "--normalize", "--exclude_build_data", "--warn_duplicate_resources");

  /** Constructs the base spawn for a singlejar action. */
  private static SpawnAction.Builder singleJarActionBuilder(JavaToolchainProvider provider) {
    Artifact singleJar = provider.getSingleJar();
    SpawnAction.Builder builder = new SpawnAction.Builder();
    // If singlejar's name ends with .jar, it is Java application, otherwise it is native.
    // TODO(asmundak): once https://github.com/bazelbuild/bazel/issues/2241 is fixed (that is,
    // the native singlejar is used on windows) remove support for the Java implementation
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .addTransitiveInputs(provider.getJavaRuntime().javaBaseInputsMiddleman())
          .setJarExecutable(
              provider.getJavaRuntime().javaBinaryExecPathFragment(),
              singleJar,
              provider.getJvmOptions())
          .setExecutionInfo(ExecutionRequirements.WORKER_MODE_ENABLED);
    } else {
      builder.setExecutable(singleJar);
    }
    return builder;
  }

  /**
   * Creates an Action that packages files into a Jar file.
   *
   * @param resources the resources to put into the Jar.
   * @param resourceJars the resource jars to merge into the jar
   * @param outputJar the Jar to create
   */
  public static void createSourceJarAction(
      RuleContext ruleContext,
      JavaSemantics semantics,
      NestedSet<Artifact> resources,
      NestedSet<Artifact> resourceJars,
      Artifact outputJar) {
    createSourceJarAction(
        ruleContext,
        ruleContext,
        semantics,
        resources,
        resourceJars,
        outputJar,
        JavaToolchainProvider.from(ruleContext));
  }

  /**
   * Creates an Action that packages files into a Jar file.
   *
   * @param actionRegistry serves for registering action,,
   * @param actionConstructionContext bundles items commonly needed to construct action instances,
   * @param resources the resources to put into the Jar.
   * @param resourceJars the resource jars to merge into the jar
   * @param outputJar the Jar to create
   * @param toolchainProvider is used to retrieve jvm options
   */
  public static void createSourceJarAction(
      ActionRegistry actionRegistry,
      ActionConstructionContext actionConstructionContext,
      JavaSemantics semantics,
      NestedSet<Artifact> resources,
      NestedSet<Artifact> resourceJars,
      Artifact outputJar,
      JavaToolchainProvider toolchainProvider) {
    requireNonNull(resourceJars);
    requireNonNull(outputJar);
    if (!resources.isEmpty()) {
      requireNonNull(semantics);
    }
    SpawnAction.Builder builder =
        singleJarActionBuilder(toolchainProvider)
            .addOutput(outputJar)
            .addTransitiveInputs(resources)
            .addTransitiveInputs(resourceJars)
            .addCommandLine(
                sourceJarCommandLine(outputJar, semantics, resources, resourceJars),
                ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
            .setProgressMessage("Building source jar %s", outputJar.prettyPrint())
            .setMnemonic("JavaSourceJar");

    actionRegistry.registerAction(builder.build(actionConstructionContext));
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
        singleJarActionBuilder(JavaToolchainProvider.from(ruleContext))
            .addOutput(output)
            .addTransitiveInputs(jars)
            .addCommandLine(
                sourceJarCommandLine(
                    output,
                    /* semantics= */ null,
                    /* resources= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
                    jars),
                ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
            .setProgressMessage("Building singlejar jar %s", output.prettyPrint())
            .setMnemonic("JavaSingleJar");
    ruleContext.registerAction(builder.build(ruleContext));
  }

  private static CommandLine sourceJarCommandLine(
      Artifact outputJar,
      JavaSemantics semantics,
      NestedSet<Artifact> resources,
      NestedSet<Artifact> resourceJars) {
    CustomCommandLine.Builder args = CustomCommandLine.builder();
    args.addExecPath("--output", outputJar);
    args.addAll(SOURCE_JAR_COMMAND_LINE_ARGS);
    args.addExecPaths("--sources", resourceJars);
    if (!resources.isEmpty()) {
      args.add("--resources");
      args.addAll(VectorArg.of(resources).mapped(new ResourceArgMapFn(semantics)));
    }
    return args.build();
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class ResourceArgMapFn extends CommandLineItem.ParametrizedMapFn<Artifact> {
    private final JavaSemantics semantics;

    ResourceArgMapFn(JavaSemantics semantics) {
      this.semantics = Preconditions.checkNotNull(semantics);
    }

    @Override
    public void expandToCommandLine(Artifact resource, Consumer<String> args) {
      String execPath = resource.getExecPathString();
      String resourcePath =
          semantics.getDefaultJavaResourcePath(resource.getRootRelativePath()).getPathString();
      StringBuilder sb = new StringBuilder(execPath.length() + resourcePath.length() + 1);
      sb.append(execPath).append(":").append(resourcePath);
      args.accept(sb.toString());
    }

    @Override
    public int maxInstancesAllowed() {
      // Expect only one semantics object.
      return 1;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof ResourceArgMapFn)) {
        return false;
      }
      ResourceArgMapFn that = (ResourceArgMapFn) o;
      return semantics.equals(that.semantics);
    }

    @Override
    public int hashCode() {
      return semantics.hashCode();
    }
  }
}
