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


import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** An executable tool that is part of {@code java_toolchain}. */
@AutoValue
public abstract class JavaToolchainTool {

  @Nullable
  static JavaToolchainTool fromStarlark(
      @Nullable StructImpl struct, JavaToolchainProvider toolchain) throws RuleErrorException {
    if (struct == null) {
      return null;
    }
    try {
      return create(
          struct.getValue("tool", FilesToRunProvider.class),
          Depset.noneableCast(struct.getValue("data"), Artifact.class, "data"),
          Depset.noneableCast(struct.getValue("jvm_opts"), String.class, "jvm_opts"),
          toolchain);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  /** The executable, possibly a {@code _deploy.jar}. */
  public abstract FilesToRunProvider tool();

  /** Additional inputs required by the tool, e.g. a Class Data Sharing archive. */
  public abstract NestedSet<Artifact> data();

  /**
   * JVM flags to invoke the tool with. Location expansion is performed on these flags using the
   * inputs in {@link #data}.
   */
  public abstract NestedSet<String> jvmOpts();

  /** The {@code java_toolchain} this tool belongs to. */
  public abstract JavaToolchainProvider toolchain();

  private record CommandLineKey(
      Artifact executable,
      ImmutableList<String> jvmOpts,
      @Nullable PathFragment javaBinary,
      @Nullable ImmutableList<String> toolchainJvmOpts) {
    static CommandLineKey from(
        Artifact executable, NestedSet<String> jvmOpts, JavaToolchainProvider toolchain)
        throws RuleErrorException {
      ImmutableList<String> jvmOptsList = jvmOpts.toList();
      if (!executable.getExtension().equals("jar")) {
        return new CommandLineKey(executable, jvmOptsList, null, null);
      }
      return new CommandLineKey(
          executable,
          jvmOptsList,
          toolchain.getJavaRuntime().javaBinaryExecPathFragment(),
          toolchain.getJvmOptions().toList());
    }
  }

  /**
   * Cache for the {@link CustomCommandLine} for a given tool.
   *
   * <p>Using weak values since the main benefit is to share the command line between different
   * actions, in which case the {@link CustomCommandLine} object remains strongly reachable anyway.
   */
  private static final LoadingCache<CommandLineKey, CustomCommandLine> commandLineCache =
      Caffeine.newBuilder().weakValues().build(JavaToolchainTool::buildCommandLine);

  private static JavaToolchainTool create(
      FilesToRunProvider tool,
      NestedSet<Artifact> data,
      NestedSet<String> jvmOpts,
      JavaToolchainProvider toolchain) {
    return new AutoValue_JavaToolchainTool(tool, data, jvmOpts, toolchain);
  }

  /**
   * Returns the executable command line for the tool.
   *
   * <p>For a Java command, the executable command line will include {@code java -jar deploy.jar} as
   * well as any JVM flags.
   *
   * @param toolchain {@code java_toolchain} for the action being constructed
   */
  CustomCommandLine getCommandLine() throws RuleErrorException {
    return commandLineCache.get(
        CommandLineKey.from(tool().getExecutable(), jvmOpts(), toolchain()));
  }

  private static CustomCommandLine buildCommandLine(CommandLineKey key) {
    CustomCommandLine.Builder command = CustomCommandLine.builder();

    if (key.javaBinary() == null) {
      command = command.addExecPath(key.executable()).addAll(key.jvmOpts());
    } else {
      command
          .addPath(key.javaBinary())
          .addAll(key.toolchainJvmOpts())
          .addAll(key.jvmOpts())
          .add("-jar")
          .addPath(key.executable().getExecPath());
    }

    return command.build();
  }

  /** Adds its inputs for the tool to provided input builder. */
  void addInputs(NestedSetBuilder<Artifact> inputs) throws RuleErrorException {
    inputs.addTransitive(data());
    Artifact executable = tool().getExecutable();
    // The runfiles of the tool are not added. If this is desired, add getFilesToRun() to inputs
    // instead.
    inputs.add(executable);
    if (executable.getExtension().equals("jar")) {
      inputs.addTransitive(toolchain().getJavaRuntime().javaBaseInputs());
    }
  }

  public JavaToolchainTool withAdditionalJvmFlags(NestedSet<String> additionalJvmFlags) {
    if (additionalJvmFlags.isEmpty()) {
      return this;
    }
    return create(
        tool(),
        data(),
        NestedSetBuilder.<String>stableOrder()
            .addTransitive(jvmOpts())
            .addTransitive(additionalJvmFlags)
            .build(),
        toolchain());
  }
}
