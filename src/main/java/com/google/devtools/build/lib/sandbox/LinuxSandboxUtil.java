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

package com.google.devtools.build.lib.sandbox;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
    return cmdEnv.getBlazeWorkspace().getBinTools().getEmbeddedPath(LINUX_SANDBOX);
  }

  /** Returns a new command line builder for the {@code linux-sandbox} tool. */
  public static CommandLineBuilder commandLineBuilder(
      Path linuxSandboxPath, List<String> commandArguments) {
    return new CommandLineBuilder(linuxSandboxPath, commandArguments);
  }

  /**
   * A builder class for constructing the full command line to run a command using the {@code
   * linux-sandbox} tool.
   */
  public static class CommandLineBuilder {
    private final Path linuxSandboxPath;
    private final List<String> commandArguments;

    private Path workingDirectory;
    private Duration timeout;
    private Duration killDelay;
    private Path stdoutPath;
    private Path stderrPath;
    private Set<Path> writableFilesAndDirectories = ImmutableSet.of();
    private Set<Path> tmpfsDirectories = ImmutableSet.of();
    private Map<Path, Path> bindMounts = ImmutableMap.of();
    private Path statisticsPath;
    private boolean useFakeHostname = false;
    private boolean createNetworkNamespace = false;
    private boolean useFakeRoot = false;
    private boolean useFakeUsername = false;
    private boolean useDebugMode = false;

    private CommandLineBuilder(Path linuxSandboxPath, List<String> commandArguments) {
      this.linuxSandboxPath = linuxSandboxPath;
      this.commandArguments = commandArguments;
    }

    /** Sets the working directory to use, if any. */
    public CommandLineBuilder setWorkingDirectory(Path workingDirectory) {
      this.workingDirectory = workingDirectory;
      return this;
    }

    /** Sets the timeout for the command run using the {@code linux-sandbox} tool. */
    public CommandLineBuilder setTimeout(Duration timeout) {
      this.timeout = timeout;
      return this;
    }

    /**
     * Sets the kill delay for commands run using the {@code linux-sandbox} tool that exceed their
     * timeout.
     */
    public CommandLineBuilder setKillDelay(Duration killDelay) {
      this.killDelay = killDelay;
      return this;
    }

    /** Sets the path to use for redirecting stdout, if any. */
    public CommandLineBuilder setStdoutPath(Path stdoutPath) {
      this.stdoutPath = stdoutPath;
      return this;
    }

    /** Sets the path to use for redirecting stderr, if any. */
    public CommandLineBuilder setStderrPath(Path stderrPath) {
      this.stderrPath = stderrPath;
      return this;
    }

    /** Sets the files or directories to make writable for the sandboxed process, if any. */
    public CommandLineBuilder setWritableFilesAndDirectories(
        Set<Path> writableFilesAndDirectories) {
      this.writableFilesAndDirectories = writableFilesAndDirectories;
      return this;
    }

    /** Sets the directories where to mount an empty tmpfs, if any. */
    public CommandLineBuilder setTmpfsDirectories(Set<Path> tmpfsDirectories) {
      this.tmpfsDirectories = tmpfsDirectories;
      return this;
    }

    /**
     * Sets the sources and targets of files or directories to explicitly bind-mount in the sandbox,
     * if any.
     */
    public CommandLineBuilder setBindMounts(Map<Path, Path> bindMounts) {
      this.bindMounts = bindMounts;
      return this;
    }

    /** Sets the path for writing execution statistics (e.g. resource usage). */
    public CommandLineBuilder setStatisticsPath(Path statisticsPath) {
      this.statisticsPath = statisticsPath;
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

    /**
     * Builds the command line to invoke a specific command using the {@code linux-sandbox} tool.
     */
    public ImmutableList<String> build() {
      Preconditions.checkState(
          !(this.useFakeUsername && this.useFakeRoot),
          "useFakeUsername and useFakeRoot are exclusive");

      ImmutableList.Builder<String> commandLineBuilder = ImmutableList.builder();

      commandLineBuilder.add(linuxSandboxPath.getPathString());
      if (workingDirectory != null) {
        commandLineBuilder.add("-W", workingDirectory.getPathString());
      }
      if (timeout != null) {
        commandLineBuilder.add("-T", Long.toString(timeout.getSeconds()));
      }
      if (killDelay != null) {
        commandLineBuilder.add("-t", Long.toString(killDelay.getSeconds()));
      }
      if (stdoutPath != null) {
        commandLineBuilder.add("-l", stdoutPath.getPathString());
      }
      if (stderrPath != null) {
        commandLineBuilder.add("-L", stderrPath.getPathString());
      }
      for (Path writablePath : writableFilesAndDirectories) {
        commandLineBuilder.add("-w", writablePath.getPathString());
      }
      for (Path tmpfsPath : tmpfsDirectories) {
        commandLineBuilder.add("-e", tmpfsPath.getPathString());
      }
      for (Path bindMountTarget : bindMounts.keySet()) {
        Path bindMountSource = bindMounts.get(bindMountTarget);
        commandLineBuilder.add("-M", bindMountSource.getPathString());
        // The file is mounted in a custom location inside the sandbox.
        if (!bindMountSource.equals(bindMountTarget)) {
          commandLineBuilder.add("-m", bindMountTarget.getPathString());
        }
      }
      if (statisticsPath != null) {
        commandLineBuilder.add("-S", statisticsPath.getPathString());
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
      commandLineBuilder.addAll(commandArguments);

      return commandLineBuilder.build();
    }
  }
}
