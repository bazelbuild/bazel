// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Utility functions for the process wrapper embedded tool, which should work on most platforms and
 * gives at least some isolation between running actions.
 */
public final class ProcessWrapperUtil {
  private static final String PROCESS_WRAPPER = "process-wrapper" + OsUtils.executableExtension();

  /** Returns whether using the process wrapper is supported in the {@link CommandEnvironment}. */
  public static boolean isSupported(CommandEnvironment cmdEnv) {
    // We can only use this runner, if the process-wrapper exists in the embedded tools.
    // This might not always be the case, e.g. while bootstrapping.
    return getProcessWrapper(cmdEnv) != null;
  }

  /** Returns the {@link Path} of the process wrapper binary, or null if it doesn't exist. */
  public static Path getProcessWrapper(CommandEnvironment cmdEnv) {
    PathFragment execPath = cmdEnv.getBlazeWorkspace().getBinTools().getExecPath(PROCESS_WRAPPER);
    return execPath != null ? cmdEnv.getExecRoot().getRelative(execPath) : null;
  }

  /** Returns a new {@link CommandLineBuilder} for the process wrapper tool. */
  public static CommandLineBuilder commandLineBuilder(
      String processWrapperPath, List<String> commandArguments) {
    return new CommandLineBuilder(processWrapperPath, commandArguments);
  }

  /**
   * A builder class for constructing the full command line to run a command using the
   * process-wrapper tool.
   */
  public static class CommandLineBuilder {
    private final String processWrapperPath;
    private final List<String> commandArguments;
    private Optional<String> stdoutPath = Optional.empty();
    private Optional<String> stderrPath = Optional.empty();
    private Optional<Duration> timeout = Optional.empty();
    private Optional<Duration> killDelay = Optional.empty();
    private Optional<String> statisticsPath = Optional.empty();

    private CommandLineBuilder(String processWrapperPath, List<String> commandArguments) {
      this.processWrapperPath = processWrapperPath;
      this.commandArguments = commandArguments;
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

    /** Sets the timeout for the command run using the process-wrapper tool. */
    public CommandLineBuilder setTimeout(Duration timeout) {
      this.timeout = Optional.of(timeout);
      return this;
    }

    /**
     * Sets the kill delay for commands run using the process-wrapper tool that exceed their
     * timeout.
     */
    public CommandLineBuilder setKillDelay(Duration killDelay) {
      this.killDelay = Optional.of(killDelay);
      return this;
    }

    /** Sets the path for writing execution statistics (e.g. resource usage). */
    public CommandLineBuilder setStatisticsPath(String statisticsPath) {
      this.statisticsPath = Optional.of(statisticsPath);
      return this;
    }

    /** Build the command line to invoke a specific command using the process wrapper tool. */
    public List<String> build() {
      List<String> fullCommandLine = new ArrayList<>();
      fullCommandLine.add(processWrapperPath);

      if (timeout.isPresent()) {
        fullCommandLine.add("--timeout=" + timeout.get().getSeconds());
      }
      if (killDelay.isPresent()) {
        fullCommandLine.add("--kill_delay=" + killDelay.get().getSeconds());
      }
      if (stdoutPath.isPresent()) {
        fullCommandLine.add("--stdout=" + stdoutPath.get());
      }
      if (stderrPath.isPresent()) {
        fullCommandLine.add("--stderr=" + stderrPath.get());
      }
      if (statisticsPath.isPresent()) {
        fullCommandLine.add("--stats=" + statisticsPath.get());
      }

      fullCommandLine.addAll(commandArguments);

      return fullCommandLine;
    }
  }
}
