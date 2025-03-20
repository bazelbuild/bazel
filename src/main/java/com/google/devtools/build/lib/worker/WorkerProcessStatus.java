// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.WorkerMetrics.WorkerStatus;
import com.google.devtools.build.lib.server.FailureDetails.Worker.Code;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Optional;

/**
 * This is a state machine instance that encapsulates the status of the worker process, and provides
 * a mechanism to signal to Bazel when to kill a worker.
 */
public class WorkerProcessStatus {

  /** The status of the worker process. */
  public enum Status {
    /**
     * Used as a starting value, before the worker process has been created by Bazel.
     *
     * <p>This state is not logged in the BEP as {@code WorkerSpawnRunner#initializeMetrics} is only
     * called after {@code #prepareExecution} (which sets the status to ALIVE).
     */
    NOT_STARTED(/* priority= */ 0, WorkerStatus.NOT_STARTED),

    /**
     * Worker process has been created by Bazel (the process might not be immediately measurable).
     */
    ALIVE(/* priority= */ 1, WorkerStatus.ALIVE),

    /** Intermediate states: Bazel has marked this worker process to be killed. */
    PENDING_KILL_DUE_TO_UNKNOWN(2, WorkerStatus.ALIVE, "KILLED_UNKNOWN"),
    PENDING_KILL_DUE_TO_INTERRUPTED_EXCEPTION(
        /* priority= */ 3, WorkerStatus.ALIVE, "KILLED_DUE_TO_INTERRUPTED_EXCEPTION"),
    PENDING_KILL_DUE_TO_USER_EXEC_EXCEPTION(
        /* priority= */ 3, WorkerStatus.ALIVE, "KILLED_DUE_TO_USER_EXEC_EXCEPTION"),
    PENDING_KILL_DUE_TO_IO_EXCEPTION(
        /* priority= */ 3, WorkerStatus.ALIVE, "KILLED_DUE_TO_IO_EXCEPTION"),
    PENDING_KILL_DUE_TO_MEMORY_PRESSURE(
        /* priority= */ 4, WorkerStatus.ALIVE, "KILLED_DUE_TO_MEMORY_PRESSURE"),

    /**
     * Semi-terminal status: Bazel has determined that worker process has already been killed due to
     * some unknown reason; if a more specific reason (below) comes along, we can transition to
     * those statuses.
     */
    KILLED_UNKNOWN(/* priority= */ 5, WorkerStatus.KILLED_UNKNOWN),

    /** Terminal statuses: The worker process has been killed and Bazel is aware of the reason. */

    // Bazel killed the worker due to an InterruptedException, can happen when the remote branch
    // interrupts the local branch after winning the race in dynamic execution.
    KILLED_DUE_TO_INTERRUPTED_EXCEPTION(
        /* priority= */ 6, WorkerStatus.KILLED_DUE_TO_INTERRUPTED_EXCEPTION),
    // Bazel killed the worker due to a UserExecException.
    KILLED_DUE_TO_USER_EXEC_EXCEPTION(
        /* priority= */ 6, WorkerStatus.KILLED_DUE_TO_USER_EXEC_EXCEPTION),
    // Bazel killed the worker due to an IOException.
    KILLED_DUE_TO_IO_EXCEPTION(/* priority= */ 6, WorkerStatus.KILLED_DUE_TO_IO_EXCEPTION),

    // This can be a result of Bazel forcibly killing the worker process, which might result in
    // other exceptions caught in the execution threads (as enumerated by the other killed statuses
    // above). Thus, this has the highest priority so that we can override that and set the correct
    // reason why the worker was killed.
    KILLED_DUE_TO_MEMORY_PRESSURE(/* priority= */ 7, WorkerStatus.KILLED_DUE_TO_MEMORY_PRESSURE);

    // The priority of a status determines whether another status can be used to override and
    // update it.
    private final int priority;
    private final Optional<String> killedStatus;
    private final WorkerStatus workerStatus;

    Status(final int priority, WorkerStatus workerStatus, String killedStatus) {
      this.priority = priority;
      this.workerStatus = workerStatus;
      this.killedStatus = Optional.of(killedStatus);
    }

    Status(final int priority, WorkerStatus workerStatus) {
      this.priority = priority;
      this.workerStatus = workerStatus;
      this.killedStatus = Optional.empty();
    }

    Status killedStatus() {
      if (killedStatus.isEmpty()) {
        return this;
      }
      return valueOf(killedStatus.get());
    }

    WorkerStatus toWorkerStatus() {
      return workerStatus;
    }
  }

  private Status status;

  private Optional<Code> workerCode = Optional.empty();

  public WorkerProcessStatus() {
    this.status = Status.NOT_STARTED;
  }

  public Status get() {
    return status;
  }

  // A status is invalid if it is killed or marked to be killed later.
  public synchronized boolean isValid() {
    return status == Status.NOT_STARTED || status == Status.ALIVE;
  }

  public boolean isPendingEviction() {
    return status == Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE;
  }

  public Optional<Code> getWorkerCode() {
    return workerCode;
  }

  private static final ImmutableSet<Status> PENDING_KILL_STATUSES =
      ImmutableSet.of(
          Status.PENDING_KILL_DUE_TO_UNKNOWN,
          Status.PENDING_KILL_DUE_TO_IO_EXCEPTION,
          Status.PENDING_KILL_DUE_TO_INTERRUPTED_EXCEPTION,
          Status.PENDING_KILL_DUE_TO_USER_EXEC_EXCEPTION,
          Status.PENDING_KILL_DUE_TO_MEMORY_PRESSURE);

  private boolean isPendingKill() {
    return PENDING_KILL_STATUSES.contains(status);
  }

  private static final ImmutableSet<Status> KILLED_STATUSES =
      ImmutableSet.of(
          Status.KILLED_UNKNOWN,
          Status.KILLED_DUE_TO_IO_EXCEPTION,
          Status.KILLED_DUE_TO_INTERRUPTED_EXCEPTION,
          Status.KILLED_DUE_TO_USER_EXEC_EXCEPTION,
          Status.KILLED_DUE_TO_MEMORY_PRESSURE);

  public boolean isKilled() {
    return KILLED_STATUSES.contains(status);
  }

  /**
   * Attempts to update the status to its corresponding kill status. Should be called **after** the
   * process is destroyed by Bazel.
   */
  @CanIgnoreReturnValue
  public synchronized boolean setKilled() {
    if (isPendingKill()) {
      status = status.killedStatus();
      return true;
    }
    return false;
  }

  /**
   * Returns whether the WorkerStatus was successfully updated after attempting to update it to a
   * given Status.
   */
  @CanIgnoreReturnValue
  public synchronized boolean maybeUpdateStatus(Status toStatus) {
    if (canTransitionTo(toStatus)) {
      this.status = toStatus;
      return true;
    }
    return false;
  }

  @CanIgnoreReturnValue
  public synchronized boolean maybeUpdateStatus(Status toStatus, Code workerCode) {
    this.workerCode = Optional.of(workerCode);
    return maybeUpdateStatus(toStatus);
  }

  /**
   * Returns whether a state transition can occur.
   *
   * @param toStatus the next state attempted to transition to.
   */
  private boolean canTransitionTo(Status toStatus) {
    return status.priority < toStatus.priority;
  }

  public WorkerStatus toWorkerStatus() {
    return status.toWorkerStatus();
  }
}
