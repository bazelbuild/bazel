package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.time.Duration;

/** Interface to support remote execution in repository_ctx.execute(). */
public interface RepositoryRemoteExecutor {

  /** The result of a remotely executed command. */
  final class ExecutionResult {

    private final int exitCode;
    private final byte[] stdout;
    private final byte[] stderr;

    public ExecutionResult(int exitCode, byte[] stdout, byte[] stderr) {
      this.exitCode = exitCode;
      this.stdout = stdout;
      this.stderr = stderr;
    }

    public int exitCode() {
      return exitCode;
    }

    public byte[] stdout() {
      return stdout;
    }

    public byte[] stderr() {
      return stderr;
    }
  }

  /**
   * Execute a command remotely.
   *
   * @param arguments the command arguments.
   * @param executionProperties the remote platform the command should run on.
   * @param environment any environment variables that should be set in the command's environment.
   * @param workingDirectory  the working directory to run the command under. {@code ""} means that
   *                          the remote system should choose.
   * @param timeout execution timeout.
   */
  ExecutionResult execute(ImmutableList<String> arguments,
      ImmutableMap<String, String> executionProperties,
      ImmutableMap<String, String> environment,
      String workingDirectory,
      Duration timeout) throws IOException, InterruptedException;
}
