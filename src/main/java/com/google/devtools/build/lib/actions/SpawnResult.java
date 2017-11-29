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
    /** Subprocess executed successfully, and returned a zero exit code. */
    SUCCESS(true),

    /** Subprocess executed successfully, but returned a non-zero exit code. */
    NON_ZERO_EXIT(true),

    /** Subprocess execution timed out. */
    TIMEOUT(true),

    /**
     * The subprocess ran out of memory. On Linux systems, the kernel may kill processes in
     * low-memory situations, and this status is intended to report such a case back to Bazel.
     */
    OUT_OF_MEMORY(true),

    /**
     * Subprocess did not execute. This error is not catastrophic - Bazel will try to continue the
     * build if keep_going is enabled, possibly attempt to rerun the same spawn, and attempt to run
     * other actions.
     */
    EXECUTION_FAILED,

    /**
     * Subprocess did not execute. Do not retry, and exit immediately even if keep_going is enabled.
     */
    EXECUTION_FAILED_CATASTROPHICALLY,

    /**
     * Subprocess did not execute in a way that indicates that the user can fix it. For example, a
     * remote system may have denied the execution due to too many inputs or too large inputs.
     */
    EXECUTION_DENIED(true),

    /**
     * The remote execution system is overloaded and had to refuse execution for this Spawn.
     */
    REMOTE_EXECUTOR_OVERLOADED,

    /**
     * The result of the remotely executed Spawn could not be retrieved due to errors in the remote
     * caching layer.
     */
    REMOTE_CACHE_FAILED;

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
   * Returns whether the spawn was actually run, regardless of the exit code. I.e., returns
   * {@code true} if {@link #status} is any of {@link Status#SUCCESS}, {@link Status#NON_ZERO_EXIT},
   * {@link Status#TIMEOUT} or {@link Status#OUT_OF_MEMORY}.
   *
   * <p>Returns false if there were errors that prevented the spawn from being run, such as network
   * errors, missing local files, errors setting up sandboxing, etc.
   */
  boolean setupSuccess();

  /** Returns true if the status was {@link Status#EXECUTION_FAILED_CATASTROPHICALLY}. */
  boolean isCatastrophe();

  /** The status of the attempted Spawn execution. */
  Status status();

  /**
   * The exit code of the subprocess if the subprocess was executed. Should only be called if
   * {@link #status} returns {@link Status#SUCCESS} or {@link Status#NON_ZERO_EXIT}.
   *
   * <p>This method must return a non-zero exit code if the status is {@link Status#TIMEOUT} or
   * {@link Status#OUT_OF_MEMORY}. It is recommended to return 128 + 14 when the status is
   * {@link Status#TIMEOUT}, which corresponds to the Unix signal SIGALRM.
   *
   * <p>This method may throw {@link IllegalStateException} if called for any other status.
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
      return status == Status.SUCCESS
          || status == Status.NON_ZERO_EXIT
          || status == Status.TIMEOUT
          || status == Status.OUT_OF_MEMORY;
    }

    @Override
    public boolean isCatastrophe() {
      return status == Status.EXECUTION_FAILED_CATASTROPHICALLY;
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
      if (status == Status.SUCCESS) {
        Preconditions.checkArgument(exitCode == 0);
      }
      if (status == Status.NON_ZERO_EXIT
          || status == Status.TIMEOUT
          || status == Status.OUT_OF_MEMORY) {
        Preconditions.checkArgument(exitCode != 0);
      }
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

    public Builder setWallTime(Optional<Duration> wallTime) {
      this.wallTime = wallTime;
      return this;
    }

    public Builder setUserTime(Optional<Duration> userTime) {
      this.userTime = userTime;
      return this;
    }

    public Builder setSystemTime(Optional<Duration> systemTime) {
      this.systemTime = systemTime;
      return this;
    }

    public Builder setCacheHit(boolean cacheHit) {
      this.cacheHit = cacheHit;
      return this;
    }
  }
}
