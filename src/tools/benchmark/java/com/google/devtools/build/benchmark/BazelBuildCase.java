// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.benchmark;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.benchmark.codegenerator.CodeGenerator;
import com.google.devtools.build.benchmark.codegenerator.CppCodeGenerator;
import com.google.devtools.build.benchmark.codegenerator.JavaCodeGenerator;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.nio.file.Path;

/** Provides all build target information for Bazel. */
final class BazelBuildCase implements BuildCase {

  private static final ImmutableMap<String, String> BUILD_TARGET_NAME_TO_DESCRIPTION =
      ImmutableMap.<String, String>builder()
          .put("java/AFewFiles", "Java Target: A Few Files")
          .put("java/ManyFiles", "Java Target: Many Files")
          .put("java/LongChainedDeps", "Java Target: Long Chained Deps")
          .put("java/ParallelDeps", "Java Target: Parallel Deps")
          .put("cpp/AFewFiles", "Cpp Target: A Few Files")
          .put("cpp/ManyFiles", "Cpp Target: Many Files")
          .put("cpp/LongChainedDeps", "Cpp Target: Long Chained Deps")
          .put("cpp/ParallelDeps", "Cpp Target: Parallel Deps")
          .build();
  private static final ImmutableSet<String> ALL_TARGET_NAMES = ImmutableSet.<String>of(
      "AFewFiles", "ManyFiles", "LongChainedDeps", "ParallelDeps");
  private static final String WORKSPACE_FILE_NAME = "WORKSPACE";
  private static final ImmutableList<BuildTargetConfig> defaultBuildTargetConfigs =
      getDefaultBuildTargetConfigs();

  private static final BuildEnvConfig FULL_CLEAN_BUILD_CONFIG =
      BuildEnvConfig.newBuilder()
          .setDescription("Full clean build")
          .setCleanBeforeBuild(true)
          .setIncremental(false)
          .build();
  private static final BuildEnvConfig INCREMENTAL_BUILD_CONFIG =
      BuildEnvConfig.newBuilder()
          .setDescription("Incremental build")
          .setCleanBeforeBuild(false)
          .setIncremental(true)
          .build();
  private static final ImmutableList<BuildEnvConfig> BUILD_ENV_CONFIGS =
      ImmutableList.of(FULL_CLEAN_BUILD_CONFIG, INCREMENTAL_BUILD_CONFIG);

  private static final FileSystem fileSystem = new JavaIoFileSystem();

  @Override
  public ImmutableList<BuildTargetConfig> getBuildTargetConfigs() {
    return defaultBuildTargetConfigs;
  }

  @Override
  public ImmutableList<String> getCodeVersions(Builder builder, BenchmarkOptions options)
      throws IOException, CommandException {
    if (options.versionFilter != null) {
      return builder.getCodeVersionsBetweenVersions(options.versionFilter);
    }
    if (options.dateFilter != null) {
      return builder.getCodeVersionsBetweenDates(options.dateFilter);
    }
    return ImmutableList.copyOf(options.versions);
  }

  @Override
  public ImmutableList<BuildEnvConfig> getBuildEnvConfigs() {
    return BUILD_ENV_CONFIGS;
  }

  // TODO(yueg): configurable target, we may not want to run benchmark for all kinds of target
  @Override
  public void prepareGeneratedCode(Path copyDir, Path generatedCodePath) throws IOException {
    // Prepare generated code for copy
    if (!copyDir.toFile().exists()) {
      CodeGenerator codeGenerator = new JavaCodeGenerator();
      codeGenerator.generateNewProject(copyDir + codeGenerator.getDirSuffix(), ALL_TARGET_NAMES);

      codeGenerator = new CppCodeGenerator();
      codeGenerator.generateNewProject(copyDir + codeGenerator.getDirSuffix(), ALL_TARGET_NAMES);
    }

    // Clean generated code path
    if (generatedCodePath.toFile().exists()) {
      try {
        FileSystemUtils.deleteTreesBelow(fileSystem.getPath(generatedCodePath.toString()));
      } catch (IOException e) {
        throw new IOException("Failed to clean directory for generated code", e);
      }
    } else {
      generatedCodePath.toFile().mkdirs();
    }

    // Copy
    try {
      FileSystemUtils.copyTreesBelow(
          fileSystem.getPath(copyDir.toString()),
          fileSystem.getPath(generatedCodePath.toString()),
          Symlinks.FOLLOW);
    } catch (IOException e) {
      throw new IOException("Failed to copy generated code", e);
    }
    if (!generatedCodePath.resolve(WORKSPACE_FILE_NAME).toFile().createNewFile()) {
      throw new IOException("Failed to create workspace file");
    }
  }

  private static ImmutableList<BuildTargetConfig> getDefaultBuildTargetConfigs() {
    ImmutableList.Builder<BuildTargetConfig> builder = ImmutableList.builder();
    for (ImmutableMap.Entry<String, String> entry : BUILD_TARGET_NAME_TO_DESCRIPTION.entrySet()) {
      builder.add(
          BuildTargetConfig.newBuilder()
              .setBuildTarget(entry.getKey())
              .setDescription(entry.getValue())
              .build());
    }
    return builder.build();
  }
}
