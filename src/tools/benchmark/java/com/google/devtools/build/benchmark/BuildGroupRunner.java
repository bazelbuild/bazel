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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.benchmark.codegenerator.CodeGenerator;
import com.google.devtools.build.benchmark.codegenerator.CppCodeGenerator;
import com.google.devtools.build.benchmark.codegenerator.JavaCodeGenerator;
import com.google.devtools.build.lib.shell.CommandException;
import java.io.IOException;
import java.nio.file.Path;

/** Class for running a build group with all build targets and getting performance results. */
class BuildGroupRunner {

  private static final String GENERATED_CODE_FOR_COPY_DIR = "GeneratedCodeForCopy";
  private static final String GENERATED_CODE_DIR = "GeneratedCode";
  private static final String BUILDER_DIR = "BuilderBazel";
  private static final int REPEAT_TIMES = 3;

  private final Path workspace;
  private Builder builder = null;

  BuildGroupRunner(Path workspace) {
    this.workspace = workspace;
  }

  BuildGroupResult run(BenchmarkOptions opt)
      throws IOException, CommandException {
    BuildCase buildCase = new BazelBuildCase();
    ImmutableList<BuildTargetConfig> buildTargetConfigs = buildCase.getBuildTargetConfigs();
    ImmutableList<BuildEnvConfig> buildEnvConfigs = buildCase.getBuildEnvConfigs();

    // Prepare builder (Bazel)
    prepareBuilder();
    System.out.println("Done preparing builder.");

    // Get code versions (commit hashtag for Bazel) and datetimes
    ImmutableList<String> codeVersions = buildCase.getCodeVersions(builder, opt);
    ImmutableList<String> datetimes = builder.getDatetimeForCodeVersions(codeVersions);
    System.out.println("Ready to run benchmark for the following versions:");
    for (String version : codeVersions) {
      System.out.println(version);
    }

    BuildGroupResult.Builder buildGroupResultBuilder =
        getBuildGroupResultBuilder(buildTargetConfigs, buildEnvConfigs, codeVersions, datetimes);

    for (int versionIndex = 0; versionIndex < codeVersions.size(); ++versionIndex) {
      String version = codeVersions.get(versionIndex);
      System.out.format("Benchmark for version %s started.\n", version);

      // Get builder binary (build Bazel binary)
      Path buildBinary = builder.getBuildBinary(version);

      // Repeat several times to calculate average result
      for (int t = 0; t < REPEAT_TIMES; ++t) {
        // Prepare generated code for build
        buildCase.prepareGeneratedCode(
            workspace.resolve(GENERATED_CODE_FOR_COPY_DIR),
            workspace.resolve(GENERATED_CODE_DIR));

        // Target config
        for (int targetIndex = 0; targetIndex < buildTargetConfigs.size(); ++targetIndex) {
          System.out.println(
              "Started target: " + buildTargetConfigs.get(targetIndex).getDescription());

          // Environment config
          for (int envIndex = 0; envIndex < buildEnvConfigs.size(); ++envIndex) {
            System.out.println("Started config: " + buildEnvConfigs.get(envIndex).getDescription());

            double elapsedTime = buildSingleTargetAndGetElapsedTime(
                buildTargetConfigs, buildEnvConfigs, buildBinary, targetIndex, envIndex);

            // Store result
            buildGroupResultBuilder
                .getBuildTargetResultsBuilder(targetIndex)
                .getBuildEnvResultsBuilder(envIndex)
                .getResultsBuilder(versionIndex)
                .addResults(elapsedTime);
          }
        }
      }
    }

    return buildGroupResultBuilder.build();
  }

  private double buildSingleTargetAndGetElapsedTime(
      ImmutableList<BuildTargetConfig> buildTargetConfigs,
      ImmutableList<BuildEnvConfig> buildEnvConfigs,
      Path buildBinary, int targetIndex, int envIndex) throws CommandException {

    BuildTargetConfig targetConfig = buildTargetConfigs.get(targetIndex);
    BuildEnvConfig envConfig = buildEnvConfigs.get(envIndex);

    // Clean if should
    if (envConfig.getCleanBeforeBuild()) {
      builder.clean();
    }

    // Modify generated code if should (only this target)
    if (envConfig.getIncremental()) {
      String targetName = targetConfig.getBuildTarget();
      targetName = targetName.substring(targetName.lastIndexOf('/') + 1, targetName.length());

      CodeGenerator codeGenerator = new JavaCodeGenerator();
      codeGenerator.modifyExistingProject(
          workspace.resolve(GENERATED_CODE_DIR) + codeGenerator.getDirSuffix(),
          ImmutableSet.of(targetName));

      codeGenerator = new CppCodeGenerator();
      codeGenerator.modifyExistingProject(
          workspace.resolve(GENERATED_CODE_DIR) + codeGenerator.getDirSuffix(),
          ImmutableSet.of(targetName));
    }

    // Remove the first target since it's slow
    if (targetIndex == 0 && envIndex == 0) {
      buildTargetAndGetElapsedTime(buildBinary, envConfig, targetConfig);
      builder.clean();
    }
    return buildTargetAndGetElapsedTime(buildBinary, envConfig, targetConfig);
  }

  private double buildTargetAndGetElapsedTime(
      Path buildBinary, BuildEnvConfig envConfig, BuildTargetConfig targetConfig)
      throws CommandException {
    // Run build
    double elapsedTime =
        builder.buildAndGetElapsedTime(
            buildBinary, builder.getCommandFromConfig(targetConfig, envConfig));
    System.out.println(elapsedTime);
    return elapsedTime;
  }

  private static BuildGroupResult.Builder getBuildGroupResultBuilder(
      ImmutableList<BuildTargetConfig> buildTargetConfigs,
      ImmutableList<BuildEnvConfig> buildEnvConfigs,
      ImmutableList<String> codeVersions,
      ImmutableList<String> datetimes) {
    // Initialize a BuildGroupResult object to preserve array length
    BuildGroupResult.Builder buildGroupResultBuilder = BuildGroupResult.newBuilder();
    for (BuildTargetConfig targetConfig : buildTargetConfigs) {
      BuildTargetResult.Builder targetBuilder =
          BuildTargetResult.newBuilder().setBuildTargetConfig(targetConfig);
      prepareBuildEnvConfigs(buildEnvConfigs, targetBuilder, codeVersions, datetimes);
      buildGroupResultBuilder.addBuildTargetResults(targetBuilder.build());
    }
    return buildGroupResultBuilder;
  }

  private static void prepareBuildEnvConfigs(
      ImmutableList<BuildEnvConfig> buildEnvConfigs,
      BuildTargetResult.Builder targetBuilder,
      ImmutableList<String> codeVersions,
      ImmutableList<String> datetimes) {
    for (BuildEnvConfig envConfig : buildEnvConfigs) {
      BuildEnvResult.Builder envBuilder = BuildEnvResult.newBuilder().setConfig(envConfig);
      for (int i = 0; i < codeVersions.size(); ++i) {
        envBuilder.addResults(
            SingleBuildResult.newBuilder()
                .setCodeVersion(codeVersions.get(i))
                .setDatetime(datetimes.get(i))
                .build());
      }
      targetBuilder.addBuildEnvResults(envBuilder.build());
    }
  }

  private void prepareBuilder() throws IOException, CommandException {
    builder =
        new BazelBuilder(workspace.resolve(GENERATED_CODE_DIR), workspace.resolve(BUILDER_DIR));
    builder.prepare();
  }
}
