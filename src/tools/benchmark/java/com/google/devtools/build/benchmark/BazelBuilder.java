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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Class that provides all needed feature of Bazel for benchmark. */
class BazelBuilder implements Builder {

  private static final Logger logger = Logger.getLogger(BazelBuilder.class.getName());
  private static final FileSystem fileSystem = new JavaIoFileSystem();

  private static final String BAZEL_BINARY_PATH = "bazel-bin/src/bazel";
  private static final Pattern ELAPSED_TIME_PATTERN = Pattern.compile("(?<=Elapsed time: )[0-9.]+");
  private static final String DEFAULT_GIT_REPO = "https://github.com/bazelbuild/bazel.git";

  private final Path generatedCodeDir;
  private final Path builderDir;
  private Path buildBinary = null;
  private String currentCodeVersion = "";

  BazelBuilder(Path generatedCodeDir, Path builderDir) {
    this.generatedCodeDir = generatedCodeDir;
    this.builderDir = builderDir;
  }

  @Override
  public Path getBuildBinary(String codeVersion) throws IOException, CommandException {
    if (buildBinary != null && currentCodeVersion.equals(codeVersion)) {
      return buildBinary;
    }

    // git checkout codeVersion
    String[] checkoutCommand = {"git", "checkout", codeVersion};
    Command cmd = new Command(checkoutCommand, null, builderDir.toFile());
    cmd.execute();

    // bazel build src:bazel
    String[] buildBazelCommand = {"bazel", "build", "src:bazel"};
    cmd = new Command(buildBazelCommand, null, builderDir.toFile());
    CommandResult result = cmd.execute();

    // Get binary path, bazel output is in stderr
    String output = new String(result.getStderr(), UTF_8).trim();
    if (!output.contains(BAZEL_BINARY_PATH)) {
      throw new IOException("Bazel binary " + BAZEL_BINARY_PATH + " is not in output of build.");
    }
    buildBinary = builderDir.resolve(BAZEL_BINARY_PATH);
    currentCodeVersion = codeVersion;
    return buildBinary;
  }

  @Override
  public ImmutableList<String> getCommandFromConfig(
      BuildTargetConfig targetConfig, BuildEnvConfig envConfig) {
    return ImmutableList.<String>builder()
        .add("build")
        .add(targetConfig.getBuildTarget())
        .addAll(envConfig.getBuildArgsList())
        .build();
  }

  @Override
  public double buildAndGetElapsedTime(Path buildBinary, ImmutableList<String> args)
      throws CommandException {
    List<String> cmdList = new ArrayList<>();
    cmdList.add(buildBinary.toString());
    cmdList.addAll(args);
    String[] cmdArr = new String[cmdList.size()];
    cmdArr = cmdList.toArray(cmdArr);

    // Run build command
    Command cmd = new Command(cmdArr, null, generatedCodeDir.toFile());
    CommandResult result = cmd.execute();

    // Get elapsed time from output
    String output = new String(result.getStderr(), UTF_8).trim();
    Matcher m = ELAPSED_TIME_PATTERN.matcher(output);

    if (m.find()) {
      try {
        return (Double.parseDouble(m.group(0)));
      } catch (NumberFormatException e) {
        // Should not be here since we look for [0-9.]+
        logger.log(Level.SEVERE, "Cannot parse " + m.group(0));
      }
    }
    throw new CommandException(cmd, "Command didn't provide parsable output.");
  }

  @Override
  public void clean() throws CommandException {
    String[] cleanCommand = {"bazel", "clean", "--expunge"};
    Command cmd = new Command(cleanCommand, null, generatedCodeDir.toFile());
    cmd.execute();
  }

  @Override
  public void prepare() throws IOException, CommandException {
    prepareFromGitRepo(DEFAULT_GIT_REPO);
  }

  @Override
  public ImmutableList<String> getCodeVersionsBetweenVersions(VersionFilter versionFilter)
      throws CommandException {
    return getListOfOutputFromCommand(
        "git", "log",
        versionFilter.getFrom() + ".." + versionFilter.getTo(), "--pretty=format:%H", "--reverse");
  }

  @Override
  public ImmutableList<String> getCodeVersionsBetweenDates(DateFilter dateFilter)
      throws CommandException {
    return getListOfOutputFromCommand(
        "git", "log",
        "--after", dateFilter.getFromString(),
        "--before", dateFilter.getToString(), "--pretty=format:%H", "--reverse");
  }

  @Override
  public ImmutableList<String> getDatetimeForCodeVersions(ImmutableList<String> codeVersions)
      throws CommandException {
    return getListOfOutputFromCommandWithAdditionalParam(codeVersions,
        "git", "show", "-s",
        "--date=iso", "--pretty=format:%cd", "--date=format:%Y-%m-%d %H:%M:%S");
  }

  void prepareFromGitRepo(String gitRepo) throws IOException, CommandException {
    // Try to pull git repo first, delete directory if failed.
    if (builderDir.toFile().isDirectory()) {
      try {
        pullGitRepo();
      } catch (CommandException e) {
        FileSystemUtils.deleteTree(fileSystem.getPath(builderDir.toString()));
      }
    }

    if (Files.notExists(builderDir)) {
      try {
        Files.createDirectories(builderDir);
      } catch (IOException e) {
        throw new IOException("Failed to create directory for bazel", e);
      }

      String[] gitCloneCommand = {"git", "clone", gitRepo, "."};
      Command cmd = new Command(gitCloneCommand, null, builderDir.toFile());
      cmd.execute();
    }
    // Assume the directory is what we need if not empty
  }

  private void pullGitRepo() throws CommandException {
    String[] gitCloneCommand = {"git", "pull"};
    Command cmd = new Command(gitCloneCommand, null, builderDir.toFile());
    cmd.execute();
  }

  private ImmutableList<String> getListOfOutputFromCommand(String... command)
      throws CommandException{
    Command cmd = new Command(command, null, builderDir.toFile());
    CommandResult result = cmd.execute();
    String output = new String(result.getStdout(), UTF_8).trim();
    return ImmutableList.copyOf(output.split("\n"));
  }

  private ImmutableList<String> getListOfOutputFromCommandWithAdditionalParam(
      ImmutableList<String> additionalParam, String... command) throws CommandException{
    ImmutableList<String> commandList =
        ImmutableList.<String>builder().add(command).addAll(additionalParam).build();
    String[] finalCommand = commandList.toArray(new String[0]);

    return getListOfOutputFromCommand(finalCommand);
  }
}
