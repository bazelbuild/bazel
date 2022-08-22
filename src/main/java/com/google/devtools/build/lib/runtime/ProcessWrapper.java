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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Tracks process-wrapper configuration and allows building command lines that rely on it. */
public final class ProcessWrapper {

  /** Name of the process-wrapper binary, without any path components. */
  private static final String BIN_BASENAME = "process-wrapper" + OsUtils.executableExtension();

  /** Path to the process-wrapper binary to use. */
  private final Path binPath;

  /** Grace delay between asking a process to stop and forcibly killing it, or null for none. */
  @Nullable private final Duration killDelay;

  /** Whether to pass {@code --graceful_sigterm} or not to the process-wrapper. */
  private final boolean gracefulSigterm;

  /** Creates a new process-wrapper instance from explicit values. */
  @VisibleForTesting
  public ProcessWrapper(Path binPath, @Nullable Duration killDelay, boolean gracefulSigterm) {
    this.binPath = binPath;
    this.killDelay = killDelay;
    this.gracefulSigterm = gracefulSigterm;
  }

  /**
   * Constructs a new process-wrapper instance based on the context of an invocation.
   *
   * @param cmdEnv command environment for this invocation
   * @return a process-wrapper handler, or null if this is not supported in the current system
   */
  @Nullable
  public static ProcessWrapper fromCommandEnvironment(CommandEnvironment cmdEnv) {
    LocalExecutionOptions options = cmdEnv.getOptions().getOptions(LocalExecutionOptions.class);
    Duration killDelay = options == null ? null : options.getLocalSigkillGraceSeconds();

    boolean gracefulSigterm = options != null && options.processWrapperGracefulSigterm;

    Path path = cmdEnv.getBlazeWorkspace().getBinTools().getEmbeddedPath(BIN_BASENAME);
    if (OS.isPosixCompatible() && path != null && path.exists()) {
      return new ProcessWrapper(path, killDelay, gracefulSigterm);
    } else {
      return null;
    }
  }

  /** Returns a new {@link CommandLineBuilder} for the process-wrapper tool. */
  public CommandLineBuilder commandLineBuilder(List<String> commandArguments) {
    return new CommandLineBuilder(
        binPath.getPathString(), commandArguments, killDelay, gracefulSigterm);
  }

  /**
   * A builder class for constructing the full command line to run a command using the
   * process-wrapper tool.
   */
  public static class CommandLineBuilder {
    private final String processWrapperPath;
    private final List<String> commandArguments;
    @Nullable private final Duration killDelay;
    private boolean gracefulSigterm;

    private Path stdoutPath;
    private Path stderrPath;
    private Duration timeout;
    private Path statisticsPath;

    private CommandLineBuilder(
        String processWrapperPath,
        List<String> commandArguments,
        @Nullable Duration killDelay,
        boolean gracefulSigterm) {
      this.processWrapperPath = processWrapperPath;
      this.commandArguments = commandArguments;
      this.killDelay = killDelay;
      this.gracefulSigterm = gracefulSigterm;
    }

    /** Sets the path to use for redirecting stdout, if any. */
    @CanIgnoreReturnValue
    public CommandLineBuilder setStdoutPath(Path stdoutPath) {
      this.stdoutPath = stdoutPath;
      return this;
    }

    /** Sets the path to use for redirecting stderr, if any. */
    @CanIgnoreReturnValue
    public CommandLineBuilder setStderrPath(Path stderrPath) {
      this.stderrPath = stderrPath;
      return this;
    }

    /** Sets the timeout for the command run using the process-wrapper tool. */
    @CanIgnoreReturnValue
    public CommandLineBuilder setTimeout(Duration timeout) {
      this.timeout = timeout;
      return this;
    }

    /** Sets the path for writing execution statistics (e.g. resource usage). */
    @CanIgnoreReturnValue
    public CommandLineBuilder setStatisticsPath(Path statisticsPath) {
      this.statisticsPath = statisticsPath;
      return this;
    }

    /** Incorporates settings from a spawn's execution info. */
    @CanIgnoreReturnValue
    public CommandLineBuilder addExecutionInfo(Map<String, String> executionInfo) {
      if (executionInfo.containsKey(ExecutionRequirements.GRACEFUL_TERMINATION)) {
        gracefulSigterm = true;
      }
      return this;
    }

    /** Build the command line to invoke a specific command using the process wrapper tool. */
    public ImmutableList<String> build() {
      ImmutableList.Builder<String> fullCommandLine = new ImmutableList.Builder<>();
      fullCommandLine.add(processWrapperPath);

      if (timeout != null) {
        fullCommandLine.add("--timeout=" + timeout.getSeconds());
      }
      if (killDelay != null) {
        fullCommandLine.add("--kill_delay=" + killDelay.getSeconds());
      }
      if (stdoutPath != null) {
        fullCommandLine.add("--stdout=" + stdoutPath);
      }
      if (stderrPath != null) {
        fullCommandLine.add("--stderr=" + stderrPath);
      }
      if (statisticsPath != null) {
        fullCommandLine.add("--stats=" + statisticsPath);
      }
      if (gracefulSigterm) {
        fullCommandLine.add("--graceful_sigterm");
      }

      fullCommandLine.addAll(commandArguments);

      return fullCommandLine.build();
    }
  }
}
