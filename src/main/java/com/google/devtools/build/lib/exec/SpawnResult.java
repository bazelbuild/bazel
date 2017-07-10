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
package com.google.devtools.build.lib.exec;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
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
     * The remote execution system did not allow the request due to missing authorization or
     * authentication.
     */
    NOT_AUTHORIZED,

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

  long getWallTimeMillis();

  /** Whether the spawn result was a cache hit. */
  boolean isCacheHit();

  /**
   * Basic implementation of {@link SpawnResult}.
   */
  @Immutable @ThreadSafe
  public static final class SimpleSpawnResult implements SpawnResult {
    private final int exitCode;
    private final Status status;
    private final String executorHostName;
    private final long wallTimeMillis;
    private final boolean cacheHit;

    SimpleSpawnResult(Builder builder) {
      this.exitCode = builder.exitCode;
      this.status = Preconditions.checkNotNull(builder.status);
      this.executorHostName = builder.executorHostName;
      this.wallTimeMillis = builder.wallTimeMillis;
      this.cacheHit = builder.cacheHit;
    }

    @Override
    public boolean setupSuccess() {
      return status == Status.SUCCESS || status == Status.TIMEOUT || status == Status.OUT_OF_MEMORY;
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
    public long getWallTimeMillis() {
      return wallTimeMillis;
    }

    @Override
    public boolean isCacheHit() {
      return cacheHit;
    }
  }

  /**
   * Builder class for {@link SpawnResult}.
   */
  public static final class Builder {
    private int exitCode;
    private Status status;
    private String executorHostName;
    private long wallTimeMillis;
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

    public Builder setWallTimeMillis(long wallTimeMillis) {
      this.wallTimeMillis = wallTimeMillis;
      return this;
    }

    public Builder setCacheHit(boolean cacheHit) {
      this.cacheHit = cacheHit;
      return this;
    }
  }
}