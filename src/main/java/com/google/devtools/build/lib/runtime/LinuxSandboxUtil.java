// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Duration;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Utility functions for the {@code linux-sandbox} embedded tool. */
public final class LinuxSandboxUtil {
  private static final String LINUX_SANDBOX = "linux-sandbox" + OsUtils.executableExtension();

  /** Returns whether using the {@code linux-sandbox} is supported in the command environment. */
  public static boolean isSupported(CommandEnvironment cmdEnv) {
    // We can only use the linux-sandbox if the linux-sandbox exists in the embedded tools.
    // This might not always be the case, e.g. while bootstrapping.
    return getLinuxSandbox(cmdEnv) != null;
  }

  /** Returns the path of the {@code linux-sandbox} binary, or null if it doesn't exist. */
  public static Path getLinuxSandbox(CommandEnvironment cmdEnv) {
    PathFragment execPath = cmdEnv.getBlazeWorkspace().getBinTools().getExecPath(LINUX_SANDBOX);
    return execPath != null ? cmdEnv.getExecRoot().getRelative(execPath) : null;
  }

  /** Returns a new command line builder for the {@code linux-sandbox} tool. */
  public static CommandLineBuilder commandLineBuilder(
      String linuxSandboxPath, Collection<String> commandArguments) {
    return new CommandLineBuilder()
        .setLinuxSandboxPath(linuxSandboxPath)
        .setCommandArguments(commandArguments);
  }

  /**
   * A builder class for constructing the full command line to run a command using the {@code
   * linux-sandbox} tool.
   */
  public static class CommandLineBuilder {
    // TODO(b/62588075): Reconsider using Path/PathFragment instead of Strings for file paths.
    private Optional<String> linuxSandboxPath = Optional.empty();
    private Optional<String> workingDirectory = Optional.empty();
    private Optional<Duration> timeout = Optional.empty();
    private Optional<Duration> killDelay = Optional.empty();
    private Optional<String> stdoutPath = Optional.empty();
    private Optional<String> stderrPath = Optional.empty();
    private Optional<Iterable<Path>> writableFilesAndDirectories = Optional.empty();
    private Optional<Iterable<Path>> tmpfsDirectories = Optional.empty();
    private Optional<Map<Path, Path>> bindMounts = Optional.empty();
    private Optional<String> statisticsPath = Optional.empty();
    private boolean useFakeHostname = false;
    private boolean createNetworkNamespace = false;
    private boolean useFakeRoot = false;
    private boolean useFakeUsername = false;
    private boolean useDebugMode = false;
    private Optional<Collection<String>> commandArguments = Optional.empty();

    private CommandLineBuilder() {
      // Prevent external construction via "new".
    }

    /** Sets the path of the {@code linux-sandbox} tool. */
    public CommandLineBuilder setLinuxSandboxPath(String linuxSandboxPath) {
      this.linuxSandboxPath = Optional.of(linuxSandboxPath);
      return this;
    }

    /** Sets the working directory to use, if any. */
    public CommandLineBuilder setWorkingDirectory(String workingDirectory) {
      this.workingDirectory = Optional.of(workingDirectory);
      return this;
    }

    /** Sets the timeout for the command run using the {@code linux-sandbox} tool. */
    public CommandLineBuilder setTimeout(Duration timeout) {
      this.timeout = Optional.of(timeout);
      return this;
    }

    /**
     * Sets the kill delay for commands run using the {@code linux-sandbox} tool that exceed their
     * timeout.
     */
    public CommandLineBuilder setKillDelay(Duration killDelay) {
      this.killDelay = Optional.of(killDelay);
      return this;
    }

    /** Sets the path to use for redirecting stdout, if any. */
    public CommandLineBuilder setStdoutPath(String stdoutPath) {
      this.stdoutPath = Optional.of(stdoutPath);
      return this;
    }

    /** Sets the path to use for redirecting stderr, if any. */
    public CommandLineBuilder setStderrPath(String stderrPath) {
      this.stderrPath = Optional.of(stderrPath);
      return this;
    }

    /** Sets the files or directories to make writable for the sandboxed process, if any. */
    public CommandLineBuilder setWritableFilesAndDirectories(
        Iterable<Path> writableFilesAndDirectories) {
      this.writableFilesAndDirectories = Optional.of(writableFilesAndDirectories);
      return this;
    }

    /** Sets the directories where to mount an empty tmpfs, if any. */
    public CommandLineBuilder setTmpfsDirectories(Iterable<Path> tmpfsDirectories) {
      this.tmpfsDirectories = Optional.of(tmpfsDirectories);
      return this;
    }

    /**
     * Sets the sources and targets of files or directories to explicitly bind-mount in the sandbox,
     * if any.
     */
    public CommandLineBuilder setBindMounts(Map<Path, Path> bindMounts) {
      this.bindMounts = Optional.of(bindMounts);
      return this;
    }

    /** Sets the path for writing execution statistics (e.g. resource usage). */
    public CommandLineBuilder setStatisticsPath(String statisticsPath) {
      this.statisticsPath = Optional.of(statisticsPath);
      return this;
    }

    /** Sets whether to use a fake 'localhost' hostname inside the sandbox. */
    public CommandLineBuilder setUseFakeHostname(boolean useFakeHostname) {
      this.useFakeHostname = useFakeHostname;
      return this;
    }

    /** Sets whether to create a new network namespace. */
    public CommandLineBuilder setCreateNetworkNamespace(boolean createNetworkNamespace) {
      this.createNetworkNamespace = createNetworkNamespace;
      return this;
    }

    /** Sets whether to pretend to be 'root' inside the namespace. */
    public CommandLineBuilder setUseFakeRoot(boolean useFakeRoot) {
      this.useFakeRoot = useFakeRoot;
      return this;
    }

    /** Sets whether to use a fake 'nobody' username inside the sandbox. */
    public CommandLineBuilder setUseFakeUsername(boolean useFakeUsername) {
      this.useFakeUsername = useFakeUsername;
      return this;
    }

    /** Sets whether to enable debug mode (e.g. to print debugging messages). */
    public CommandLineBuilder setUseDebugMode(boolean useDebugMode) {
      this.useDebugMode = useDebugMode;
      return this;
    }

    /** Sets the command (and its arguments) to run using the {@code linux-sandbox} tool. */
    public CommandLineBuilder setCommandArguments(Collection<String> commandArguments) {
      this.commandArguments = Optional.of(commandArguments);
      return this;
    }

    /**
     * Builds the command line to invoke a specific command using the {@code linux-sandbox} tool.
     */
    public List<String> build() {
      Preconditions.checkState(this.linuxSandboxPath.isPresent(), "linuxSandboxPath is required");
      Preconditions.checkState(this.commandArguments.isPresent(), "commandArguments are required");
      Preconditions.checkState(
          !(this.useFakeUsername && this.useFakeRoot),
          "useFakeUsername and useFakeRoot are exclusive");

      ImmutableList.Builder<String> commandLineBuilder = ImmutableList.<String>builder();

      commandLineBuilder.add(linuxSandboxPath.get());
      if (workingDirectory.isPresent()) {
        commandLineBuilder.add("-W", workingDirectory.get());
      }
      if (timeout.isPresent()) {
        commandLineBuilder.add("-T", Long.toString(timeout.get().getSeconds()));
      }
      if (killDelay.isPresent()) {
        commandLineBuilder.add("-t", Long.toString(killDelay.get().getSeconds()));
      }
      if (stdoutPath.isPresent()) {
        commandLineBuilder.add("-l", stdoutPath.get());
      }
      if (stderrPath.isPresent()) {
        commandLineBuilder.add("-L", stderrPath.get());
      }
      if (writableFilesAndDirectories.isPresent()) {
        for (Path writablePath : writableFilesAndDirectories.get()) {
          commandLineBuilder.add("-w", writablePath.getPathString());
        }
      }
      if (tmpfsDirectories.isPresent()) {
        for (Path tmpfsPath : tmpfsDirectories.get()) {
          commandLineBuilder.add("-e", tmpfsPath.getPathString());
        }
      }
      if (bindMounts.isPresent()) {
        for (Path bindMountTarget : bindMounts.get().keySet()) {
          Path bindMountSource = bindMounts.get().get(bindMountTarget);
          commandLineBuilder.add("-M", bindMountSource.getPathString());
          // The file is mounted in a custom location inside the sandbox.
          if (!bindMountSource.equals(bindMountTarget)) {
            commandLineBuilder.add("-m", bindMountTarget.getPathString());
          }
        }
      }
      if (statisticsPath.isPresent()) {
        commandLineBuilder.add("-S", statisticsPath.get());
      }
      if (useFakeHostname) {
        commandLineBuilder.add("-H");
      }
      if (createNetworkNamespace) {
        commandLineBuilder.add("-N");
      }
      if (useFakeRoot) {
        commandLineBuilder.add("-R");
      }
      if (useFakeUsername) {
        commandLineBuilder.add("-U");
      }
      if (useDebugMode) {
        commandLineBuilder.add("-D");
      }
      commandLineBuilder.add("--");
      commandLineBuilder.addAll(commandArguments.get());

      return commandLineBuilder.build();
    }

    /**
     * Builds the command line to invoke a specific command using the {@code linux-sandbox} tool.
     *
     * @return the command line as an array of strings
     */
    public String[] buildAsArray() {
      return build().toArray(new String[0]);
    }
  }
}
