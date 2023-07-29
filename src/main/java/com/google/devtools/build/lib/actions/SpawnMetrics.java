// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Timing, size, and memory statistics for a Spawn execution. */
@SuppressWarnings("GoodTime") // Use ints instead of Durations to improve build time (cl/505728570)
public class SpawnMetrics {

  SpawnMetrics(Builder builder) {
    this.execKind = builder.execKind;
    this.totalTimeInMs = builder.totalTimeInMs;
    this.parseTimeInMs = builder.parseTimeInMs;
    this.networkTimeInMs = builder.networkTimeInMs;
    this.fetchTimeInMs = builder.fetchTimeInMs;
    this.queueTimeInMs = builder.queueTimeInMs;
    this.setupTimeInMs = builder.setupTimeInMs;
    this.uploadTimeInMs = builder.uploadTimeInMs;
    this.executionWallTimeInMs = builder.executionWallTimeInMs;
    this.retryTimeInMs = ImmutableMap.copyOf(builder.retryTimeInMs);
    this.processOutputsTimeInMs = builder.processOutputsTimeInMs;
    this.inputBytes = builder.inputBytes;
    this.inputFiles = builder.inputFiles;
    this.memoryEstimateBytes = builder.memoryEstimateBytes;
  }

  /** Indicates whether the metrics correspond to the remote, local or worker execution. */
  public enum ExecKind {
    REMOTE("Remote"),
    LOCAL("Local"),
    WORKER("Worker"),
    /**
     * Other kinds of execution (or when it's not clear whether something happened locally or
     * remotely).
     */
    OTHER("Other");

    private final String name;

    ExecKind(String name) {
      this.name = name;
    }

    @Override
    public String toString() {
      return name;
    }
  }

  ExecKind execKind;
  int totalTimeInMs;
  int parseTimeInMs;
  int fetchTimeInMs;
  int queueTimeInMs;
  int uploadTimeInMs;
  int setupTimeInMs;
  int executionWallTimeInMs;
  int processOutputsTimeInMs;
  int networkTimeInMs;
  // error code to duration in ms
  ImmutableMap<Integer, Integer> retryTimeInMs;
  long inputBytes;
  long inputFiles;
  long memoryEstimateBytes;

  /** Any non-important stats < than 10% will not be shown in the summary. */
  static final double STATS_SHOW_THRESHOLD = 0.10;

  public static SpawnMetrics forLocalExecution(int wallTimeInMs) {
    return SpawnMetrics.Builder.forLocalExec()
        .setTotalTimeInMs(wallTimeInMs)
        .setExecutionWallTimeInMs(wallTimeInMs)
        .build();
  }

  /**
   * Generates a String representation of the stats.
   *
   * @param total total time in milliseconds used to compute the percentages
   * @param summary whether to exclude input file count and sizes, and memory estimates
   */
  public String toString(int total, boolean summary) {
    StringBuilder sb = new StringBuilder();
    sb.append("(");
    sb.append(prettyPercentage(totalTimeInMs, total));
    sb.append(" of the time): [");
    List<String> stats = new ArrayList<>(8);
    addStatToString(stats, "parse", !summary, parseTimeInMs, total);
    addStatToString(stats, "queue", true, queueTimeInMs, total);
    addStatToString(stats, "network", !summary, networkTimeInMs, total);
    addStatToString(stats, "upload", !summary, uploadTimeInMs, total);
    addStatToString(stats, "setup", true, setupTimeInMs, total);
    addStatToString(stats, "process", true, executionWallTimeInMs, total);
    addStatToString(stats, "fetch", !summary, fetchTimeInMs, total);
    addStatToString(stats, "retry", !summary, retryTimeInMs(), total);
    addStatToString(stats, "processOutputs", !summary, processOutputsTimeInMs, total);
    addStatToString(stats, "other", !summary, otherTimeInMs(), total);
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
      List<String> strings, String name, boolean forceShow, int time, int totalTime) {
    if (forceShow || isAboveThreshold(time, totalTime)) {
      strings.add(name + ": " + prettyPercentage(time, totalTime));
    }
  }

  private static boolean isAboveThreshold(int time, int totalTime) {
    return totalTime > 0 && (((float) time / totalTime) >= STATS_SHOW_THRESHOLD);
  }

  /**
   * Converts relative duration to the percentage string.
   *
   * @return formatted percentage string or "N/A" if result is undefined
   */
  private static String prettyPercentage(int duration, int total) {
    // Duration.toMillis() != 0 does not imply !Duration.isZero() (due to truncation).
    if (total == 0) {
      // Return "not available" string if total is 0 and result is undefined.
      return "N/A";
    }
    return String.format(Locale.US, "%.2f%%", duration * 100.0 / total);
  }

  /** The kind of execution the metrics refer to (remote/local/worker). */
  public ExecKind execKind() {
    return execKind;
  }

  /** Returns true if {@link #totalTimeInMs()} is zero. */
  public boolean isEmpty() {
    return totalTimeInMs == 0;
  }

  /**
   * Total (measured locally) wall time in milliseconds spent running a spawn. This should be at
   * least as large as all the other times summed together.
   */
  public int totalTimeInMs() {
    return totalTimeInMs;
  }

  /**
   * Total time in milliseconds spent getting on network. This includes time getting network-side
   * errors and the time of the round-trip, found by taking the difference of wall time here and the
   * server time reported by the RPC. This is 0 for locally executed spawns.
   */
  public int networkTimeInMs() {
    return networkTimeInMs;
  }

  /** Total time in milliseconds waiting in queues. Includes queue time for any failed attempts. */
  public int queueTimeInMs() {
    return queueTimeInMs;
  }

  /**
   * The time in milliseconds spent transferring files to the backends. This is 0 for locally
   * executed spawns.
   */
  public int uploadTimeInMs() {
    return uploadTimeInMs;
  }

  /**
   * The time in milliseconds required to setup the environment in which the spawn is run. This may
   * be 0 for locally executed spawns, or may include time to setup a sandbox or other environment.
   * Does not include failed attempts.
   */
  public int setupTimeInMs() {
    return setupTimeInMs;
  }

  /** Time spent running the subprocess. */
  public int executionWallTimeInMs() {
    return executionWallTimeInMs;
  }

  /**
   * The time in milliseconds taken to convert the spawn into a network request, e.g., collecting
   * runfiles, and digests for all input files.
   */
  public int parseTimeInMs() {
    return parseTimeInMs;
  }

  /** Total time in milliseconds spent fetching remote outputs. */
  public int fetchTimeInMs() {
    return fetchTimeInMs;
  }

  /** Time spent in previous failed attempts. Does not include queue time. */
  public int retryTimeInMs() {
    return retryTimeInMs.values().stream().reduce(0, Integer::sum);
  }

  /** Time spent in previous failed attempts, keyed by error code. Does not include queue time. */
  public Map<Integer, Integer> retryTimeByError() {
    return retryTimeInMs;
  }

  /** Time spend by the execution framework on processing outputs. */
  public int processOutputsTimeInMs() {
    return processOutputsTimeInMs;
  }

  /**
   * Any time in milliseconds that is not measured by a more specific component, out of {@code
   * totalTime()}.
   */
  public int otherTimeInMs() {
    return totalTimeInMs
        - parseTimeInMs
        - networkTimeInMs
        - queueTimeInMs
        - uploadTimeInMs
        - setupTimeInMs
        - executionWallTimeInMs
        - fetchTimeInMs
        - retryTimeInMs()
        - processOutputsTimeInMs;
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

  /** Limit of total size in bytes of inputs or 0 if unavailable. */
  public long inputBytesLimit() {
    return 0;
  }

  /** Limit of total number of input files or 0 if unavailable. */
  public long inputFilesLimit() {
    return 0;
  }

  /** Limit of total size in bytes of outputs or 0 if unavailable. */
  public long outputBytesLimit() {
    return 0;
  }

  /** Limit of total number of output files or 0 if unavailable. */
  public long outputFilesLimit() {
    return 0;
  }

  /** Memory limit or 0 if unavailable. */
  public long memoryLimit() {
    return 0;
  }

  /** Time limit in milliseconds or 0 if unavailable. */
  public int timeLimitInMs() {
    return 0;
  }

  /** Builder class for SpawnMetrics. */
  public static class Builder {
    private ExecKind execKind = null;
    private int totalTimeInMs = 0;
    private int parseTimeInMs = 0;
    private int networkTimeInMs = 0;
    private int fetchTimeInMs = 0;
    private int queueTimeInMs = 0;
    private int setupTimeInMs = 0;
    private int uploadTimeInMs = 0;
    private int executionWallTimeInMs = 0;
    private int processOutputsTimeInMs = 0;
    private Map<Integer, Integer> retryTimeInMs = new HashMap<>();
    private long inputBytes = 0;
    private long inputFiles = 0;
    private long memoryEstimateBytes = 0;
    long inputBytesLimit = 0;
    long inputFilesLimit = 0;
    long outputBytesLimit = 0;
    long outputFilesLimit = 0;
    long memoryBytesLimit = 0;
    int timeLimitInMs = 0;

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
      if (inputBytesLimit == 0
          && inputFilesLimit == 0
          && outputBytesLimit == 0
          && outputFilesLimit == 0
          && memoryBytesLimit == 0
          && timeLimitInMs == 0) {
        return new SpawnMetrics(this);
      }
      return new FullSpawnMetrics(this);
    }

    @CanIgnoreReturnValue
    public Builder setExecKind(ExecKind execKind) {
      this.execKind = execKind;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setTotalTimeInMs(int totalTimeInMs) {
      this.totalTimeInMs = totalTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setParseTimeInMs(int parseTimeInMs) {
      this.parseTimeInMs = parseTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setNetworkTimeInMs(int networkTimeInMs) {
      this.networkTimeInMs = networkTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setFetchTimeInMs(int fetchTimeInMs) {
      this.fetchTimeInMs = fetchTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setQueueTimeInMs(int queueTimeInMs) {
      this.queueTimeInMs = queueTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setSetupTimeInMs(int setupTimeInMs) {
      this.setupTimeInMs = setupTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addSetupTimeInMs(int setupTimeInMs) {
      this.setupTimeInMs = this.setupTimeInMs + setupTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setUploadTimeInMs(int uploadTimeInMs) {
      this.uploadTimeInMs = uploadTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExecutionWallTimeInMs(int executionWallTimeInMs) {
      this.executionWallTimeInMs = executionWallTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addRetryTimeInMs(int errorCode, int retryTimeInMs) {
      Integer t = this.retryTimeInMs.getOrDefault(errorCode, 0);
      this.retryTimeInMs.put(errorCode, t + retryTimeInMs);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setRetryTimeInMs(ImmutableMap<Integer, Integer> retryTimeInMs) {
      this.retryTimeInMs = retryTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setProcessOutputsTimeInMs(int processOutputsTimeInMs) {
      this.processOutputsTimeInMs = processOutputsTimeInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setInputBytes(long inputBytes) {
      this.inputBytes = inputBytes;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setInputFiles(long inputFiles) {
      this.inputFiles = inputFiles;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setMemoryEstimateBytes(long memoryEstimateBytes) {
      this.memoryEstimateBytes = memoryEstimateBytes;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setInputBytesLimit(long inputBytesLimit) {
      this.inputBytesLimit = inputBytesLimit;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setInputFilesLimit(long inputFilesLimit) {
      this.inputFilesLimit = inputFilesLimit;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setOutputBytesLimit(long outputBytesLimit) {
      this.outputBytesLimit = outputBytesLimit;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setOutputFilesLimit(long outputFilesLimit) {
      this.outputFilesLimit = outputFilesLimit;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setMemoryBytesLimit(long memoryBytesLimit) {
      this.memoryBytesLimit = memoryBytesLimit;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setTimeLimitInMs(int timeLimitInMs) {
      this.timeLimitInMs = timeLimitInMs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addDurations(SpawnMetrics metric) {
      totalTimeInMs += metric.totalTimeInMs();
      parseTimeInMs += metric.parseTimeInMs();
      networkTimeInMs += metric.networkTimeInMs();
      fetchTimeInMs += metric.fetchTimeInMs();
      queueTimeInMs += metric.queueTimeInMs();
      uploadTimeInMs += metric.uploadTimeInMs();
      setupTimeInMs += metric.setupTimeInMs();
      executionWallTimeInMs += metric.executionWallTimeInMs();
      for (Map.Entry<Integer, Integer> entry : metric.retryTimeInMs.entrySet()) {
        addRetryTimeInMs(entry.getKey(), entry.getValue());
      }
      processOutputsTimeInMs += metric.processOutputsTimeInMs();
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addNonDurations(SpawnMetrics metric) {
      inputFiles += metric.inputFiles();
      inputBytes += metric.inputBytes();
      memoryEstimateBytes += metric.memoryEstimate();
      inputFilesLimit += metric.inputFilesLimit();
      inputBytesLimit += metric.inputBytesLimit();
      outputFilesLimit += metric.outputFilesLimit();
      outputBytesLimit += metric.outputBytesLimit();
      memoryBytesLimit += metric.memoryLimit();
      timeLimitInMs += metric.timeLimitInMs();
      return this;
    }

    @CanIgnoreReturnValue
    public Builder maxNonDurations(SpawnMetrics metric) {
      inputFiles = Long.max(inputFiles, metric.inputFiles());
      inputBytes = Long.max(inputBytes, metric.inputBytes());
      memoryEstimateBytes = Long.max(memoryEstimateBytes, metric.memoryEstimate());
      inputFilesLimit = Long.max(inputFilesLimit, metric.inputFilesLimit());
      inputBytesLimit = Long.max(inputBytesLimit, metric.inputBytesLimit());
      outputFilesLimit = Long.max(outputFilesLimit, metric.outputFilesLimit());
      outputBytesLimit = Long.max(outputBytesLimit, metric.outputBytesLimit());
      memoryBytesLimit = Long.max(memoryBytesLimit, metric.memoryLimit());
      timeLimitInMs = Integer.max(timeLimitInMs, metric.timeLimitInMs());
      return this;
    }
  }
}
