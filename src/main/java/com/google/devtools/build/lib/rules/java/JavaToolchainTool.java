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

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** An executable tool that is part of {@code java_toolchain}. */
@AutoValue
public abstract class JavaToolchainTool {

  /** The executable, possibly a {@code _deploy.jar}. */
  public abstract FilesToRunProvider tool();

  /** Additional inputs required by the tool, e.g. a Class Data Sharing archive. */
  public abstract NestedSet<Artifact> data();

  /**
   * JVM flags to invoke the tool with, or empty if it is not a {@code _deploy.jar}. Location
   * expansion is performed on these flags using the inputs in {@link #data}.
   */
  public abstract NestedSet<String> jvmOpts();

  /**
   * Cache for the {@link CustomCommandLine} for a given tool.
   *
   * <p>Practically, every {@link JavaToolchainTool} is used with only 1 {@link
   * JavaToolchainProvider} hence the initial capacity of 2 (path stripping on/off).
   *
   * <p>Using soft values since the main benefit is to share the command line between different
   * actions, in which case the {@link CustomCommandLine} object remains strongly reachable anyway.
   */
  private final transient LoadingCache<Pair<JavaToolchainProvider, PathFragment>, CustomCommandLine>
      commandLineCache =
          Caffeine.newBuilder()
              .initialCapacity(2)
              .softValues()
              .build(key -> extractCommandLine(key.first, key.second));

  @Nullable
  static JavaToolchainTool fromRuleContext(
      RuleContext ruleContext, String toolAttribute, String dataAttribute, String jvmOptsAttribute)
      throws InterruptedException {
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
    NestedSet<String> jvmOpts =
        NestedSetBuilder.wrap(
            Order.STABLE_ORDER,
            ruleContext
                .getExpander()
                .withExecLocations(locations.buildOrThrow())
                .list(jvmOptsAttribute));
    return create(tool, dataArtifacts.build(), jvmOpts);
  }

  @Nullable
  static JavaToolchainTool fromFilesToRunProvider(@Nullable FilesToRunProvider executable) {
    if (executable == null) {
      return null;
    }
    return create(
        executable,
        NestedSetBuilder.emptySet(STABLE_ORDER),
        NestedSetBuilder.emptySet(STABLE_ORDER));
  }

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
   * @param stripOutputPath output tree root fragment to use for path stripping or null if disabled.
   */
  CustomCommandLine getCommandLine(
      JavaToolchainProvider toolchain, @Nullable PathFragment stripOutputPath) {
    return commandLineCache.get(Pair.of(toolchain, stripOutputPath));
  }

  private CustomCommandLine extractCommandLine(
      JavaToolchainProvider toolchain, @Nullable PathFragment stripOutputPath) {
    CustomCommandLine.Builder command = CustomCommandLine.builder();

    Artifact executable = tool().getExecutable();
    if (!executable.getExtension().equals("jar")) {
      command.addExecPath(executable);
    } else {
      command
          .addPath(toolchain.getJavaRuntime().javaBinaryExecPathFragment())
          .addAll(toolchain.getJvmOptions())
          .addAll(jvmOpts())
          .add("-jar")
          .addPath(executable.getExecPath());
    }

    if (stripOutputPath != null) {
      command.stripOutputPaths(stripOutputPath);
    }

    return command.build();
  }

  /** Adds its inputs for the tool to provided input builder. */
  void addInputs(JavaToolchainProvider toolchain, NestedSetBuilder<Artifact> inputs) {
    inputs.addTransitive(data());
    Artifact executable = tool().getExecutable();
    if (!executable.getExtension().equals("jar")) {
      inputs.addTransitive(tool().getFilesToRun());
    } else {
      inputs.add(executable).addTransitive(toolchain.getJavaRuntime().javaBaseInputs());
    }
  }
}
