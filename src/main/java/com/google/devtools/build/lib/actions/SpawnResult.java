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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.shell.TerminationStatus;
import java.time.Duration;
import java.util.Locale;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * The result of a spawn execution.
 *
 * <p>DO NOT IMPLEMENT THIS INTERFACE! Use {@link SpawnResult.Builder} to create instances instead.
 * This is a temporary situation as long as we still have separate internal and external
 * implementations - the plan is to merge the two into a single immutable, final class.
 */
// TODO(ulfjack): Change this from an interface to an immutable, final class.
public interface SpawnResult {
  /** The status of the attempted Spawn execution. */
  public enum Status {
    /**
     * Subprocess executed successfully, but may have returned a non-zero exit code. See
     * {@link #exitCode} for the actual exit code.
     */
    SUCCESS(true),

    /** Subprocess execution timed out. */
    TIMEOUT(true),

    /**
     * The subprocess ran out of memory. On Linux systems, the kernel may kill processes in
     * low-memory situations, and this status is intended to report such a case back to Bazel.
     */
    OUT_OF_MEMORY(true),

    /**
     * Subprocess did not execute for an unknown reason - only use this if none of the more specific
     * status codes apply.
     */
    EXECUTION_FAILED,

    /** The attempted subprocess was disallowed by a user setting. */
    LOCAL_ACTION_NOT_ALLOWED(true),

    /** The Spawn referred to an non-existent absolute or relative path. */
    COMMAND_NOT_FOUND,

    /**
     * One of the Spawn inputs was a directory. For backwards compatibility, some
     * {@link SpawnRunner} implementations may attempt to run the subprocess anyway. Note that this
     * leads to incremental correctness issues, as Bazel does not track dependencies on directories.
     */
    DIRECTORY_AS_INPUT_DISALLOWED(true),

    /**
     * Too many input files - remote execution systems may refuse to execute subprocesses with an
     * excessive number of input files.
     */
    TOO_MANY_INPUT_FILES(true),

    /**
     * Total size of inputs is too large - remote execution systems may refuse to execute
     * subprocesses if the total size of all inputs exceeds a limit.
     */
    INPUTS_TOO_LARGE(true),

    /**
     * One of the input files to the Spawn was modified during the build - some {@link SpawnRunner}
     * implementations cache checksums and may detect such modifications on a best effort basis.
     */
    FILE_MODIFIED_DURING_BUILD(true),

    /**
     * The {@link SpawnRunner} was unable to establish a required network connection.
     */
    CONNECTION_FAILED,

    /**
     * The remote execution system is overloaded and had to refuse execution for this Spawn.
     */
    REMOTE_EXECUTOR_OVERLOADED,

    /**
     * The result of the remotely executed Spawn could not be retrieved due to errors in the remote
     * caching layer.
     */
    REMOTE_CACHE_FAILED,

    /**
     * The remote execution system did not allow the request due to missing authorization or
     * authentication.
     */
    NOT_AUTHORIZED(true),

    /**
     * The Spawn was malformed.
     */
    INVALID_ARGUMENT;

    private final boolean isUserError;

    private Status(boolean isUserError) {
      this.isUserError = isUserError;
    }

    private Status() {
      this(false);
    }

    public boolean isConsideredUserError() {
      return isUserError;
    }
  }

  /**
   * Returns whether the spawn was actually run, regardless of the exit code. I.e., returns if
   * status == SUCCESS || status == TIMEOUT || status == OUT_OF_MEMORY. Returns false if there were
   * errors that prevented the spawn from being run, such as network errors, missing local files,
   * errors setting up sandboxing, etc.
   */
  boolean setupSuccess();

  boolean isCatastrophe();

  /** The status of the attempted Spawn execution. */
  Status status();

  /**
   * The exit code of the subprocess if the subprocess was executed. Check {@link #status} for
   * {@link Status#SUCCESS} before calling this method.
   */
  int exitCode();

  /**
   * The host name of the executor or {@code null}. This information is intended for debugging
   * purposes, especially for remote execution systems. Remote caches usually do not store the
   * original host name, so this is generally {@code null} for cache hits.
   */
  @Nullable String getExecutorHostName();

  /**
   * Returns the wall time taken by the {@link Spawn}'s execution.
   *
   * @return the measurement, or empty in case of execution errors or when the measurement is not
   *     implemented for the current platform
   */
  Optional<Duration> getWallTime();

  /**
   * Returns the user time taken by the {@link Spawn}'s execution.
   *
   * @return the measurement, or empty in case of execution errors or when the measurement is not
   *     implemented for the current platform
   */
  Optional<Duration> getUserTime();

  /**
   * Returns the system time taken by the {@link Spawn}'s execution.
   *
   * @return the measurement, or empty in case of execution errors or when the measurement is not
   *     implemented for the current platform
   */
  Optional<Duration> getSystemTime();

  /** Whether the spawn result was a cache hit. */
  boolean isCacheHit();

  String getDetailMessage(
      String messagePrefix, String message, boolean catastrophe, boolean forciblyRunRemotely);

  /**
   * Basic implementation of {@link SpawnResult}.
   */
  @Immutable @ThreadSafe
  public static final class SimpleSpawnResult implements SpawnResult {
    private final int exitCode;
    private final Status status;
    private final String executorHostName;
    private final Optional<Duration> wallTime;
    private final Optional<Duration> userTime;
    private final Optional<Duration> systemTime;
    private final boolean cacheHit;

    SimpleSpawnResult(Builder builder) {
      this.exitCode = builder.exitCode;
      this.status = Preconditions.checkNotNull(builder.status);
      this.executorHostName = builder.executorHostName;
      this.wallTime = builder.wallTime;
      this.userTime = builder.userTime;
      this.systemTime = builder.systemTime;
      this.cacheHit = builder.cacheHit;
    }

    @Override
    public boolean setupSuccess() {
      return status == Status.SUCCESS || status == Status.TIMEOUT || status == Status.OUT_OF_MEMORY;
    }

    @Override
    public boolean isCatastrophe() {
      return false;
    }

    @Override
    public int exitCode() {
      return exitCode;
    }

    @Override
    public Status status() {
      return status;
    }

    @Override
    public String getExecutorHostName() {
      return executorHostName;
    }

    @Override
    public Optional<Duration> getWallTime() {
      return wallTime;
    }

    @Override
    public Optional<Duration> getUserTime() {
      return userTime;
    }

    @Override
    public Optional<Duration> getSystemTime() {
      return systemTime;
    }

    @Override
    public boolean isCacheHit() {
      return cacheHit;
    }

    @Override
    public String getDetailMessage(
        String messagePrefix, String message, boolean catastrophe, boolean forciblyRunRemotely) {
      TerminationStatus status = new TerminationStatus(
          exitCode(), status() == Status.TIMEOUT);
      String reason = " (" + status.toShortString() + ")"; // e.g " (Exit 1)"
      String explanation = status.exited() ? "" : ": " + message;

      if (!status().isConsideredUserError()) {
        String errorDetail = status().name().toLowerCase(Locale.US)
            .replace('_', ' ');
        explanation += ". Note: Remote connection/protocol failed with: " + errorDetail;
      }
      if (status() == Status.TIMEOUT) {
        Preconditions.checkState(
            getWallTime().isPresent(), "SpawnAction timed out but wall time wasn't set");
        explanation +=
            String.format(
                " (failed due to timeout after %.2f seconds.)",
                getWallTime().get().toMillis() / 1000.0);
      } else if (status() == Status.OUT_OF_MEMORY) {
        explanation += " (Remote action was terminated due to Out of Memory.)";
      }
      if (status() != Status.TIMEOUT && forciblyRunRemotely) {
        explanation += " Action tagged as local was forcibly run remotely and failed - it's "
            + "possible that the action simply doesn't work remotely";
      }
      return messagePrefix + " failed" + reason + explanation;
    }
  }

  /**
   * Builder class for {@link SpawnResult}.
   */
  public static final class Builder {
    private int exitCode;
    private Status status;
    private String executorHostName;
    private Optional<Duration> wallTime = Optional.empty();
    private Optional<Duration> userTime = Optional.empty();
    private Optional<Duration> systemTime = Optional.empty();
    private boolean cacheHit;

    public SpawnResult build() {
      return new SimpleSpawnResult(this);
    }

    public Builder setExitCode(int exitCode) {
      this.exitCode = exitCode;
      return this;
    }

    public Builder setStatus(Status status) {
      this.status = status;
      return this;
    }

    public Builder setExecutorHostname(String executorHostName) {
      this.executorHostName = executorHostName;
      return this;
    }

    public Builder setWallTime(Duration wallTime) {
      this.wallTime = Optional.of(wallTime);
      return this;
    }

    public Builder setUserTime(Duration userTime) {
      this.userTime = Optional.of(userTime);
      return this;
    }

    public Builder setSystemTime(Duration systemTime) {
      this.systemTime = Optional.of(systemTime);
      return this;
    }

    public Builder setCacheHit(boolean cacheHit) {
      this.cacheHit = cacheHit;
      return this;
    }
  }
}
