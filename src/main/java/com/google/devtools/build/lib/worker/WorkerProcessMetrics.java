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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics.WorkerStats;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Contains data about worker statistics during execution. This class contains data for {@link
 * com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics}
 */
public class WorkerProcessMetrics {

  private final List<Integer> workerIds;

  private final long processId;

  private final String mnemonic;

  private final boolean isMultiplex;

  private final boolean isSandbox;

  private boolean isMeasurable = false;

  private final int workerKeyHash;

  private int memoryInKb = 0;

  // Memory usage prior to this invocation, useful to calculate memory deltas as a result of a
  // particular invocations.
  private int priorMemoryInKb = 0;

  private Optional<Instant> lastCallTime = Optional.empty();

  private Optional<Instant> lastCollectedTime = Optional.empty();

  private final WorkerProcessStatus status;

  private boolean newlyCreated = true;
  private final AtomicInteger actionsExecuted = new AtomicInteger(0);

  private int priorActionsExecuted = 0;

  public WorkerProcessMetrics(
      List<Integer> workerIds,
      long processId,
      WorkerProcessStatus status,
      String mnemonic,
      boolean isMultiplex,
      boolean isSandbox,
      int workerKeyHash) {
    this.workerIds = workerIds;
    this.processId = processId;
    this.status = status;
    this.mnemonic = mnemonic;
    this.isMultiplex = isMultiplex;
    this.isSandbox = isSandbox;
    this.workerKeyHash = workerKeyHash;
  }

  public WorkerProcessMetrics(
      int workerId,
      long processId,
      WorkerProcessStatus status,
      String mnemonic,
      boolean isMultiplex,
      boolean isSandbox,
      int workerKeyHash) {
    this(
        new ArrayList<>(Arrays.asList(workerId)),
        processId,
        status,
        mnemonic,
        isMultiplex,
        isSandbox,
        workerKeyHash);
  }

  public void maybeAddWorkerId(int workerId, WorkerProcessStatus status) {
    // Multiplex workers have multiple worker ids, make sure not to include duplicate worker ids.
    if (workerIds.contains(workerId)) {
      return;
    }
    workerIds.add(workerId);
  }

  public void addCollectedMetrics(int memoryInKb, Instant collectionTime) {
    this.memoryInKb = memoryInKb;
    this.isMeasurable = true;
    this.lastCollectedTime = Optional.of(collectionTime);
  }

  /** Reset relevant internal states before each command. */
  public void onBeforeCommand() {
    newlyCreated = false;
    priorActionsExecuted = actionsExecuted.get();
    priorMemoryInKb = memoryInKb;
  }

  /** Whether the worker process was created during the current invocation. */
  public boolean isNewlyCreated() {
    return newlyCreated;
  }

  public void incrementActionsExecuted() {
    actionsExecuted.incrementAndGet();
  }

  public int getActionsExecuted() {
    return actionsExecuted.get();
  }

  public Optional<Instant> getLastCallTime() {
    return lastCallTime;
  }

  public Optional<Instant> getLastCollectedTime() {
    return lastCollectedTime;
  }

  public void setLastCallTime(Instant lastCallTime) {
    this.lastCallTime = Optional.of(lastCallTime);
  }

  public boolean isMeasurable() {
    return isMeasurable;
  }

  public ImmutableList<Integer> getWorkerIds() {
    return ImmutableList.copyOf(workerIds);
  }

  public long getProcessId() {
    return processId;
  }

  public String getMnemonic() {
    return mnemonic;
  }

  public boolean isMultiplex() {
    return isMultiplex;
  }

  public boolean isSandboxed() {
    return isSandbox;
  }

  public int getWorkerKeyHash() {
    return workerKeyHash;
  }

  public int getUsedMemoryInKb() {
    return memoryInKb;
  }

  public WorkerProcessStatus getStatus() {
    return status;
  }

  public WorkerMetrics toProto() {
    WorkerStats.Builder statsBuilder =
        WorkerStats.newBuilder()
            .setWorkerMemoryInKb(memoryInKb)
            .setPriorWorkerMemoryInKb(priorMemoryInKb);
    if (lastCollectedTime.isPresent()) {
      statsBuilder.setCollectTimeInMs(lastCollectedTime.get().toEpochMilli());
    }
    if (lastCallTime.isPresent()) {
      statsBuilder.setLastActionStartTimeInMs(lastCallTime.get().toEpochMilli());
    }

    WorkerMetrics.Builder builder =
        WorkerMetrics.newBuilder()
            .addAllWorkerIds(workerIds)
            .setProcessId((int) processId)
            .setMnemonic(mnemonic)
            .setIsSandbox(isSandbox)
            .setIsMultiplex(isMultiplex)
            .setIsMeasurable(isMeasurable)
            .setWorkerKeyHash(workerKeyHash)
            .setWorkerStatus(status.toWorkerStatus())
            .setActionsExecuted(actionsExecuted.get())
            .setPriorActionsExecuted(priorActionsExecuted)
            .addWorkerStats(statsBuilder.build());

    if (status.getWorkerCode().isPresent()) {
      builder.setCode(status.getWorkerCode().get());
    }

    return builder.build();
  }
}
