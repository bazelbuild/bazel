// Copyright 2020 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

/** An executable tool that is part of {@code java_toolchain}. */
@AutoValue
@AutoCodec
public abstract class JavaToolchainTool {

  /** The executable, possibly a {@code _deploy.jar}. */
  public abstract FilesToRunProvider tool();

  /** Additional inputs required by the tool, e.g. a Class Data Sharing archive. */
  public abstract NestedSet<Artifact> data();

  /**
   * JVM flags to invoke the tool with, or empty if it is not a {@code _deploy.jar}. Location
   * expansion is performed on these flags using the inputs in {@link #data}.
   */
  public abstract ImmutableList<String> jvmOpts();

  @Nullable
  static JavaToolchainTool fromRuleContext(
      RuleContext ruleContext,
      String toolAttribute,
      String dataAttribute,
      String jvmOptsAttribute) {
    FilesToRunProvider tool = ruleContext.getExecutablePrerequisite(toolAttribute);
    if (tool == null) {
      return null;
    }
    NestedSetBuilder<Artifact> dataArtifacts = NestedSetBuilder.stableOrder();
    ImmutableMap.Builder<Label, ImmutableCollection<Artifact>> locations = ImmutableMap.builder();
    for (TransitiveInfoCollection data : ruleContext.getPrerequisites(dataAttribute)) {
      NestedSet<Artifact> files = data.getProvider(FileProvider.class).getFilesToBuild();
      dataArtifacts.addTransitive(files);
      locations.put(AliasProvider.getDependencyLabel(data), files.toList());
    }
    ImmutableList<String> jvmOpts =
        ruleContext.getExpander().withExecLocations(locations.build()).list(jvmOptsAttribute);
    return create(tool, dataArtifacts.build(), jvmOpts);
  }

  @Nullable
  static JavaToolchainTool fromFilesToRunProvider(@Nullable FilesToRunProvider executable) {
    if (executable == null) {
      return null;
    }
    return create(executable, NestedSetBuilder.emptySet(STABLE_ORDER), ImmutableList.of());
  }

  @AutoCodec.Instantiator
  static JavaToolchainTool create(
      FilesToRunProvider tool, NestedSet<Artifact> data, ImmutableList<String> jvmOpts) {
    return new AutoValue_JavaToolchainTool(tool, data, jvmOpts);
  }

  /**
   * Builds the executable command line for the tool and adds its inputs to the given input builder.
   *
   * <p>For a Java command, the executable command line will include {@code java -jar deploy.jar} as
   * well as any JVM flags.
   *
   * @param command the executable command line builder for the tool
   * @param toolchain {@code java_toolchain} for the action being constructed
   * @param inputs inputs for the action being constructed
   */
  void buildCommandLine(
      CustomCommandLine.Builder command,
      JavaToolchainProvider toolchain,
      NestedSetBuilder<Artifact> inputs) {
    inputs.addTransitive(data());
    Artifact executable = tool().getExecutable();
    if (!executable.getExtension().equals("jar")) {
      command.addExecPath(executable);
      inputs.addTransitive(tool().getFilesToRun());
    } else {
      inputs.add(executable).addTransitive(toolchain.getJavaRuntime().javaBaseInputsMiddleman());
      command
          .addPath(toolchain.getJavaRuntime().javaBinaryExecPathFragment())
          .addAll(toolchain.getJvmOptions())
          .addAll(jvmOpts())
          .add("-jar")
          .addPath(executable.getExecPath());
    }
  }

  /**
   * Builds the executable command line for the tool and adds its inputs to the given builder, see
   * also {@link #buildCommandLine(CustomCommandLine.Builder, JavaToolchainProvider,
   * NestedSetBuilder)}.
   */
  CommandLine buildCommandLine(JavaToolchainProvider toolchain, NestedSetBuilder<Artifact> inputs) {
    CustomCommandLine.Builder command = CustomCommandLine.builder();
    buildCommandLine(command, toolchain, inputs);
    return command.build();
  }
}
