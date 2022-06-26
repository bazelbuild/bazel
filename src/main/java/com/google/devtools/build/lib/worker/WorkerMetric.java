// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Contains data about worker statistics during execution. This class contains data for {@link
 * com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics}
 */
public final class WorkerMetric {
  private final int workerId;
  private final long processId;
  private final String mnemonic;
  private final boolean isMultiplex;
  private final boolean isSandboxed;
  private boolean isMeasurable;

  private List<WorkerStat> workerStats;

  /** Worker measurement of used memory. */
  public static class WorkerStat {
    private final int usedMemoryInKB;
    private final Instant timestamp;

    public WorkerStat(int usedMemoryInKB, Instant timestamp) {
      this.usedMemoryInKB = usedMemoryInKB;
      this.timestamp = timestamp;
    }

    public int getUsedMemoryInKB() {
      return usedMemoryInKB;
    }

    public Instant getTimestamp() {
      return timestamp;
    }
  }

  public WorkerMetric(
      int workerId, long processId, String mnemonic, boolean isMultiplex, boolean isSandboxed) {
    this.workerId = workerId;
    this.processId = processId;
    this.mnemonic = mnemonic;
    this.isMultiplex = isMultiplex;
    this.isSandboxed = isSandboxed;
    this.workerStats = new ArrayList<>();
    this.isMeasurable = true;
  }

  public void addWorkerStat(WorkerStat workerStat) {
    this.workerStats.add(workerStat);
  }

  public void setIsMeasurable(boolean isMeasurable) {
    this.isMeasurable = isMeasurable;
  }

  public void clear() {
    this.workerStats.clear();
  }

  public int getWorkerId() {
    return workerId;
  }

  public long getProcessId() {
    return processId;
  }

  public String getMnemonic() {
    return mnemonic;
  }

  public boolean getIsMultiplex() {
    return isMultiplex;
  }

  public boolean getIsSandboxed() {
    return isSandboxed;
  }

  public List<WorkerStat> getWorkerStats() {
    return workerStats;
  }

  public boolean getIsMeasurable() {
    return isMeasurable;
  }

  public WorkerMetrics toProto() {
    WorkerMetrics.Builder builder = WorkerMetrics.newBuilder();
    builder.setWorkerId(workerId);
    builder.setProcessId((int) processId);
    builder.setIsMeasurable(isMeasurable);
    builder.setMnemonic(mnemonic);
    builder.setIsSandbox(isSandboxed);
    builder.setIsMultiplex(isMultiplex);

    WorkerMetrics.WorkerStats.Builder statsBuilder = WorkerMetrics.WorkerStats.newBuilder();
    for (WorkerStat workerStat : workerStats) {
      statsBuilder.setCollectTimeInMs(workerStat.timestamp.toEpochMilli());
      statsBuilder.setWorkerMemoryInKb(workerStat.usedMemoryInKB);
      builder.addWorkerStats(statsBuilder.build());
    }

    return builder.build();
  }
}
