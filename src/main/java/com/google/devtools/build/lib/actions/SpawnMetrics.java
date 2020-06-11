// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/** Timing, size, and memory statistics for a Spawn execution. */
public final class SpawnMetrics {

  /** Indicates whether the metrics correspond to the remote, local or worker execution. */
  public static enum ExecKind {
    REMOTE("Remote"),
    LOCAL("Local"),
    WORKER("Worker"),
    /**
     * Other kinds of execution (or when it's not clear whether something happened locally or
     * remotely).
     */
    OTHER("Other");

    private final String name;

    private ExecKind(String name) {
      this.name = name;
    }

    @Override
    public String toString() {
      return name;
    }
  }

  /** Any non important stats < than 10% will not be shown in the summary. */
  private static final double STATS_SHOW_THRESHOLD = 0.10;

  public static SpawnMetrics forLocalExecution(Duration wallTime) {
    return Builder.forLocalExec().setTotalTime(wallTime).setExecutionWallTime(wallTime).build();
  }

  private final ExecKind execKind;
  private final Duration totalTime;
  private final Duration parseTime;
  private final Duration fetchTime;
  private final Duration queueTime;
  private final Duration uploadTime;
  private final Duration setupTime;
  private final Duration executionWallTime;
  private final Duration retryTime;
  private final Duration processOutputsTime;
  private final Duration networkTime;
  private final long inputBytes;
  private final long inputFiles;
  private final long memoryEstimateBytes;

  private SpawnMetrics(Builder builder) {
    this.execKind = builder.execKind;
    this.totalTime = builder.totalTime;
    this.parseTime = builder.parseTime;
    this.networkTime = builder.networkTime;
    this.fetchTime = builder.fetchTime;
    this.queueTime = builder.queueTime;
    this.setupTime = builder.setupTime;
    this.uploadTime = builder.uploadTime;
    this.executionWallTime = builder.executionWallTime;
    this.retryTime = builder.retryTime;
    this.processOutputsTime = builder.processOutputsTime;
    this.inputBytes = builder.inputBytes;
    this.inputFiles = builder.inputFiles;
    this.memoryEstimateBytes = builder.memoryEstimateBytes;
  }

  /** The kind of execution the metrics refer to (remote/local/worker). */
  public ExecKind execKind() {
    return execKind;
  }

  /** Returns true if {@link #totalTime()} is zero. */
  public boolean isEmpty() {
    return totalTime.isZero();
  }

  /**
   * Total (measured locally) wall time spent running a spawn. This should be at least as large as
   * all the other times summed together.
   */
  public Duration totalTime() {
    return totalTime;
  }

  /**
   * Total time spent getting on network. This includes time getting network-side errors and the
   * time of the round-trip, found by taking the difference of wall time here and the server time
   * reported by the RPC. This is 0 for locally executed spawns.
   */
  public Duration networkTime() {
    return networkTime;
  }

  /** Total time waiting in queues. Includes queue time for any failed attempts. */
  public Duration queueTime() {
    return queueTime;
  }

  /** The time spent transferring files to the backends. This is 0 for locally executed spawns. */
  public Duration uploadTime() {
    return uploadTime;
  }

  /**
   * The time required to setup the environment in which the spawn is run. This may be 0 for locally
   * executed spawns, or may include time to setup a sandbox or other environment. Does not include
   * failed attempts.
   */
  public Duration setupTime() {
    return setupTime;
  }

  /** Time spent running the subprocess. */
  public Duration executionWallTime() {
    return executionWallTime;
  }

  /**
   * The time taken to convert the spawn into a network request, e.g., collecting runfiles, and
   * digests for all input files.
   */
  public Duration parseTime() {
    return parseTime;
  }

  /** Total time spent fetching remote outputs. */
  public Duration fetchTime() {
    return fetchTime;
  }

  /** Time spent in previous failed attempts. Does not include queue time. */
  public Duration retryTime() {
    return retryTime;
  }

  /** Time spend by the execution framework on processing outputs. */
  public Duration processOutputsTime() {
    return processOutputsTime;
  }

  /** Any time that is not measured by a more specific component, out of {@code totalTime()}. */
  public Duration otherTime() {
    return totalTime
        .minus(parseTime)
        .minus(networkTime)
        .minus(queueTime)
        .minus(uploadTime)
        .minus(setupTime)
        .minus(executionWallTime)
        .minus(fetchTime)
        .minus(retryTime)
        .minus(processOutputsTime);
  }

  /** Total size in bytes of inputs or 0 if unavailable. */
  public long inputBytes() {
    return inputBytes;
  }

  /** Total number of input files or 0 if unavailable. */
  public long inputFiles() {
    return inputFiles;
  }

  /** Estimated memory usage or 0 if unavailable. */
  public long memoryEstimate() {
    return memoryEstimateBytes;
  }

  /**
   * Generates a String representation of the stats.
   *
   * @param total total time used to compute the percentages
   * @param summary whether to exclude input file count and sizes, and memory estimates
   */
  public String toString(Duration total, boolean summary) {
    StringBuilder sb = new StringBuilder();
    sb.append("(");
    sb.append(prettyPercentage(totalTime, total));
    sb.append(" of the time): [");
    List<String> stats = new ArrayList<>(8);
    addStatToString(stats, "parse", !summary, parseTime, total);
    addStatToString(stats, "queue", true, queueTime, total);
    addStatToString(stats, "network", !summary, networkTime, total);
    addStatToString(stats, "upload", !summary, uploadTime, total);
    addStatToString(stats, "setup", true, setupTime, total);
    addStatToString(stats, "process", true, executionWallTime, total);
    addStatToString(stats, "fetch", !summary, fetchTime, total);
    addStatToString(stats, "retry", !summary, retryTime, total);
    addStatToString(stats, "processOutputs", !summary, processOutputsTime, total);
    addStatToString(stats, "other", !summary, otherTime(), total);
    if (!summary) {
      stats.add("input files: " + inputFiles);
      stats.add("input bytes: " + inputBytes);
      stats.add("memory bytes: " + memoryEstimateBytes);
    }
    Joiner.on(", ").appendTo(sb, stats);
    sb.append("]");
    return sb.toString();
  }

  /**
   * Add to {@code strings} the string representation of {@code name} component. If {@code
   * forceShow} is set to false it will only show if it is above certain threshold.
   */
  private static void addStatToString(
      List<String> strings, String name, boolean forceShow, Duration time, Duration totalTime) {
    if (forceShow || isAboveThreshold(time, totalTime)) {
      strings.add(name + ": " + prettyPercentage(time, totalTime));
    }
  }

  private static boolean isAboveThreshold(Duration time, Duration totalTime) {
    return totalTime.toMillis() > 0
        && (((float) time.toMillis() / totalTime.toMillis()) >= STATS_SHOW_THRESHOLD);
  }

  /**
   * Converts relative duration to the percentage string.
   *
   * @return formatted percentage string or "N/A" if result is undefined
   */
  private static String prettyPercentage(Duration duration, Duration total) {
    // Duration.toMillis() != 0 does not imply !Duration.isZero() (due to truncation).
    if (total.toMillis() == 0) {
      // Return "not available" string if total is 0 and result is undefined.
      return "N/A";
    }
    return String.format("%.2f%%", duration.toMillis() * 100.0 / total.toMillis());
  }

  /** Builder class for SpawnMetrics. */
  public static class Builder {
    private ExecKind execKind = null;
    private Duration totalTime = Duration.ZERO;
    private Duration parseTime = Duration.ZERO;
    private Duration networkTime = Duration.ZERO;
    private Duration fetchTime = Duration.ZERO;
    private Duration queueTime = Duration.ZERO;
    private Duration setupTime = Duration.ZERO;
    private Duration uploadTime = Duration.ZERO;
    private Duration executionWallTime = Duration.ZERO;
    private Duration retryTime = Duration.ZERO;
    private Duration processOutputsTime = Duration.ZERO;
    private long inputBytes = 0;
    private long inputFiles = 0;
    private long memoryEstimateBytes = 0;

    public static Builder forLocalExec() {
      return forExec(ExecKind.LOCAL);
    }

    public static Builder forRemoteExec() {
      return forExec(ExecKind.REMOTE);
    }

    public static Builder forWorkerExec() {
      return forExec(ExecKind.WORKER);
    }

    public static Builder forOtherExec() {
      return forExec(ExecKind.OTHER);
    }

    public static Builder forExec(ExecKind kind) {
      return new Builder().setExecKind(kind);
    }

    // Make the constructor private to force users to set the ExecKind by using one of the factory
    // methods.
    private Builder() {}

    public SpawnMetrics build() {
      Preconditions.checkNotNull(execKind, "ExecKind must be explicitly set using `setExecKind`");
      // TODO(ulfjack): Add consistency checks here?
      return new SpawnMetrics(this);
    }

    public Builder setExecKind(ExecKind execKind) {
      this.execKind = execKind;
      return this;
    }

    public Builder setTotalTime(Duration totalTime) {
      this.totalTime = totalTime;
      return this;
    }

    public Builder setParseTime(Duration parseTime) {
      this.parseTime = parseTime;
      return this;
    }

    public Builder setNetworkTime(Duration networkTime) {
      this.networkTime = networkTime;
      return this;
    }

    public Builder setFetchTime(Duration fetchTime) {
      this.fetchTime = fetchTime;
      return this;
    }

    public Builder setQueueTime(Duration queueTime) {
      this.queueTime = queueTime;
      return this;
    }

    public Builder setSetupTime(Duration setupTime) {
      this.setupTime = setupTime;
      return this;
    }

    public Builder setUploadTime(Duration uploadTime) {
      this.uploadTime = uploadTime;
      return this;
    }

    public Builder setExecutionWallTime(Duration executionWallTime) {
      this.executionWallTime = executionWallTime;
      return this;
    }

    public Builder setRetryTime(Duration retryTime) {
      this.retryTime = retryTime;
      return this;
    }

    public Builder setProcessOutputsTime(Duration processOutputsTime) {
      this.processOutputsTime = processOutputsTime;
      return this;
    }

    public Builder setInputBytes(long inputBytes) {
      this.inputBytes = inputBytes;
      return this;
    }

    public Builder setInputFiles(long inputFiles) {
      this.inputFiles = inputFiles;
      return this;
    }

    public Builder setMemoryEstimateBytes(long memoryEstimateBytes) {
      this.memoryEstimateBytes = memoryEstimateBytes;
      return this;
    }

    public Builder addDurations(SpawnMetrics metric) {
      totalTime = totalTime.plus(metric.totalTime());
      parseTime = parseTime.plus(metric.parseTime());
      networkTime = networkTime.plus(metric.networkTime());
      fetchTime = fetchTime.plus(metric.fetchTime());
      queueTime = queueTime.plus(metric.queueTime());
      uploadTime = uploadTime.plus(metric.uploadTime());
      setupTime = setupTime.plus(metric.setupTime());
      executionWallTime = executionWallTime.plus(metric.executionWallTime());
      retryTime = retryTime.plus(metric.retryTime());
      processOutputsTime = processOutputsTime.plus(metric.processOutputsTime());
      return this;
    }

    public Builder addNonDurations(SpawnMetrics metric) {
      inputFiles += metric.inputFiles();
      inputBytes += metric.inputBytes();
      memoryEstimateBytes += metric.memoryEstimate();
      return this;
    }

    public Builder maxNonDurations(SpawnMetrics metric) {
      inputFiles = Long.max(inputFiles, metric.inputFiles());
      inputBytes = Long.max(inputBytes, metric.inputBytes());
      memoryEstimateBytes = Long.max(memoryEstimateBytes, metric.memoryEstimate());
      return this;
    }
  }
}
