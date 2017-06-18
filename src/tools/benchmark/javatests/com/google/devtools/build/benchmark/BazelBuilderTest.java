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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BazelBuilderTest {

  private static final double EPSILON = 1e-4;

  @Rule public TemporaryFolder folder = new TemporaryFolder();

  @Test
  public void testGetCommandFromConfig() {
    BuildTargetConfig targetConfig = BuildTargetConfig.newBuilder().setBuildTarget("foo").build();
    BuildEnvConfig envConfig =
        BuildEnvConfig.newBuilder().addBuildArgs("apple").addBuildArgs("mango").build();

    ImmutableList<String> command =
        new BazelBuilder(null, null).getCommandFromConfig(targetConfig, envConfig);

    assertThat(command).containsExactly("build", "foo", "apple", "mango");
  }

  @Test
  public void testBuildAndGetElapsedTime() throws IOException, CommandException {
    Path root = folder.newFolder("BuildAndGetElapsedTime").toPath();
    Path generatedCode = root.resolve("GeneratedCode");
    Files.createDirectories(generatedCode);
    // Prepare binary
    Path buildBinary = root.resolve("binary.sh");
    Files.createFile(buildBinary);
    if (!buildBinary.toFile().setExecutable(true)) {
      fail("Failed to set executable");
    }
    double expectedValue = 10.42;
    try (PrintWriter writer = new PrintWriter(buildBinary.toFile())) {
      writer.format(
          "#!/bin/bash\n>&2 echo 'blah blah Elapsed time: %.2f blah blah'", expectedValue);
    }

    double result =
        new BazelBuilder(generatedCode, null)
            .buildAndGetElapsedTime(buildBinary, ImmutableList.<String>of());

    assertThat(result).isWithin(EPSILON).of(expectedValue);
  }

  @Test
  public void testPrepareFromGitRepo() throws IOException, CommandException {
    Path root = folder.newFolder("Prepare").toPath();
    // Create a git repo for clone
    Path repo = root.resolve("SimpleRepo");
    Files.createDirectories(repo);
    Files.createFile(repo.resolve("BUILD"));
    Files.createFile(repo.resolve("WORKSPACE"));

    ImmutableList<String[]> gitCommands = ImmutableList.of(
        new String[]{"git", "init"},
        new String[]{"git", "add", "."},
        new String[]{"git", "config", "user.email", "you@example.com"},
        new String[]{"git", "config", "user.name", "Your Name"},
        new String[]{"git", "commit", "-m", "empty"});
    for (String[] gitCommand : gitCommands) {
      (new Command(gitCommand, null, repo.toFile())).execute();
    }

    BazelBuilder builder = new BazelBuilder(root.resolve("GeneratedCode"), root.resolve("Builder"));
    builder.prepareFromGitRepo(repo.toString());

    ImmutableSet<String> fileList =
        fileArrayToImmutableSet(root.resolve("Builder").toFile().listFiles());
    assertThat(fileList).containsExactly(".git", "BUILD", "WORKSPACE");
  }

  private static ImmutableSet<String> fileArrayToImmutableSet(File[] files) {
    ImmutableSet.Builder<String> fileNames = ImmutableSet.builder();
    for (File file : files) {
      fileNames.add(file.getName());
    }
    return fileNames.build();
  }
}
