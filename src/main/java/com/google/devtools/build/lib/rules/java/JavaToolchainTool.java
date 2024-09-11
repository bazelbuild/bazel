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
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** An executable tool that is part of {@code java_toolchain}. */
@AutoValue
public abstract class JavaToolchainTool {

  // avoid creating new instances every time the same toolchain provider instance is passed from
  // Starlark to native (for example for registering compilation actions in JavaStarlarkCommon).
  // This is important since each instance of this class has its own commandLineCache
  private static final Interner<JavaToolchainTool> interner = BlazeInterners.newWeakInterner();

  @Nullable
  static JavaToolchainTool fromStarlark(@Nullable StructImpl struct) throws RuleErrorException {
    if (struct == null) {
      return null;
    }
    try {
      JavaToolchainTool tool =
          create(
              struct.getValue("tool", FilesToRunProvider.class),
              Depset.noneableCast(struct.getValue("data"), Artifact.class, "data"),
              Depset.noneableCast(struct.getValue("jvm_opts"), String.class, "jvm_opts"));
      return interner.intern(tool);
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

  /**
   * Cache for the {@link CustomCommandLine} for a given tool.
   *
   * <p>Practically, every {@link JavaToolchainTool} is used with only 1 {@link
   * JavaToolchainProvider}, hence the initial capacity of 1.
   *
   * <p>Using soft values since the main benefit is to share the command line between different
   * actions, in which case the {@link CustomCommandLine} object remains strongly reachable anyway.
   */
  private final transient LoadingCache<JavaToolchainProvider, CustomCommandLine> commandLineCache =
      Caffeine.newBuilder().initialCapacity(1).softValues().build(this::extractCommandLine);


  private static JavaToolchainTool create(
      FilesToRunProvider tool, NestedSet<Artifact> data, NestedSet<String> jvmOpts) {
    return new AutoValue_JavaToolchainTool(tool, data, jvmOpts);
  }

  /**
   * Returns the executable command line for the tool.
   *
   * <p>For a Java command, the executable command line will include {@code java -jar deploy.jar} as
   * well as any JVM flags.
   *
   * @param toolchain {@code java_toolchain} for the action being constructed
   */
  CustomCommandLine getCommandLine(JavaToolchainProvider toolchain) {
    return commandLineCache.get(toolchain);
  }

  private CustomCommandLine extractCommandLine(JavaToolchainProvider toolchain)
      throws RuleErrorException {
    CustomCommandLine.Builder command = CustomCommandLine.builder();

    Artifact executable = tool().getExecutable();
    if (!executable.getExtension().equals("jar")) {
      command = command.addExecPath(executable).addAll(jvmOpts());
    } else {
      command
          .addPath(toolchain.getJavaRuntime().javaBinaryExecPathFragment())
          .addAll(toolchain.getJvmOptions())
          .addAll(jvmOpts())
          .add("-jar")
          .addPath(executable.getExecPath());
    }

    return command.build();
  }

  /** Adds its inputs for the tool to provided input builder. */
  void addInputs(JavaToolchainProvider toolchain, NestedSetBuilder<Artifact> inputs)
      throws RuleErrorException {
    inputs.addTransitive(data());
    Artifact executable = tool().getExecutable();
    if (!executable.getExtension().equals("jar")) {
      inputs.addTransitive(tool().getFilesToRun());
    } else {
      inputs.add(executable).addTransitive(toolchain.getJavaRuntime().javaBaseInputs());
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
            .build());
  }
}
