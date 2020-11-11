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

package com.google.devtools.build.lib.worker;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Worker.Code;
import com.google.devtools.build.lib.vfs.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Semaphore;

/**
 * A manager to instantiate and destroy multiplexers. There should only be one {@code
 * WorkerMultiplexer} corresponding to workers with the same {@code WorkerKey}. If the {@code
 * WorkerMultiplexer} has been constructed, other workers should point to the same one.
 */
public class WorkerMultiplexerManager {
  /**
   * A map from the hash of {@code WorkerKey} objects to the corresponding information about the
   * multiplexer instance.
   */
  private static final Map<WorkerKey, InstanceInfo> multiplexerInstance = new HashMap<>();

  /** A semaphore to protect {@code multiplexerInstance} and {@code multiplexerRefCount} objects. */
  private static final Semaphore semMultiplexer = new Semaphore(1);

  private WorkerMultiplexerManager() {}

  /**
   * Returns a {@code WorkerMultiplexer} instance to {@code WorkerProxy}. {@code WorkerProxy}
   * objects with the same {@code WorkerKey} talk to the same {@code WorkerMultiplexer}. Also,
   * record how many {@code WorkerProxy} objects are talking to this {@code WorkerMultiplexer}.
   */
  public static WorkerMultiplexer getInstance(WorkerKey key, Path logFile, Reporter reporter)
      throws InterruptedException {
    semMultiplexer.acquire();
    multiplexerInstance.putIfAbsent(key, new InstanceInfo(logFile));
    multiplexerInstance.get(key).increaseRefCount();
    WorkerMultiplexer workerMultiplexer = multiplexerInstance.get(key).getWorkerMultiplexer();
    workerMultiplexer.setReporter(reporter);
    semMultiplexer.release();
    return workerMultiplexer;
  }

  /**
   * Removes the {@code WorkerMultiplexer} instance and reference count since it is no longer in
   * use.
   */
  public static void removeInstance(WorkerKey key) throws InterruptedException, UserExecException {
    semMultiplexer.acquire();
    try {
      multiplexerInstance.get(key).decreaseRefCount();
      if (multiplexerInstance.get(key).getRefCount() == 0) {
        multiplexerInstance.get(key).getWorkerMultiplexer().interrupt();
        multiplexerInstance.get(key).getWorkerMultiplexer().destroyMultiplexer();
        multiplexerInstance.remove(key);
      }
    } catch (Exception e) {
      String message = "NullPointerException while accessing non-existent multiplexer instance.";
      throw createUserExecException(e, message, Code.MULTIPLEXER_INSTANCE_REMOVAL_FAILURE);
    } finally {
      semMultiplexer.release();
    }
  }

  /** Is called when a build is done, to do per-build cleanup. */
  static void afterCommandCleanup() {
    try {
      semMultiplexer.acquire();
      for (InstanceInfo i : multiplexerInstance.values()) {
        i.getWorkerMultiplexer().setReporter(null);
      }
      semMultiplexer.release();
    } catch (InterruptedException e) {
      // Interrupted during cleanup, not much we can do.
    }
  }

  @VisibleForTesting
  static WorkerMultiplexer getMultiplexer(WorkerKey key) throws UserExecException {
    try {
      return multiplexerInstance.get(key).getWorkerMultiplexer();
    } catch (NullPointerException e) {
      String message = "NullPointerException while accessing non-existent multiplexer instance.";
      throw createUserExecException(e, message, Code.MULTIPLEXER_DOES_NOT_EXIST);
    }
  }

  @VisibleForTesting
  static Integer getRefCount(WorkerKey key) throws UserExecException {
    try {
      return multiplexerInstance.get(key).getRefCount();
    } catch (NullPointerException e) {
      String message = "NullPointerException while accessing non-existent multiplexer instance.";
      throw createUserExecException(e, message, Code.MULTIPLEXER_DOES_NOT_EXIST);
    }
  }

  @VisibleForTesting
  static Integer getInstanceCount() {
    return multiplexerInstance.keySet().size();
  }

  private static UserExecException createUserExecException(
      Exception e, String message, Code detailedCode) {
    return new UserExecException(
        FailureDetail.newBuilder()
            .setMessage(ErrorMessage.builder().message(message).exception(e).build().toString())
            .setWorker(FailureDetails.Worker.newBuilder().setCode(detailedCode))
            .build());
  }

  /** Contains the WorkerMultiplexer instance and reference count. */
  static class InstanceInfo {
    private WorkerMultiplexer workerMultiplexer;
    private Integer refCount;

    public InstanceInfo(Path logFile) {
      this.workerMultiplexer = new WorkerMultiplexer(logFile);
      this.refCount = 0;
    }

    public void increaseRefCount() {
      refCount = refCount + 1;
    }

    public void decreaseRefCount() {
      refCount = refCount - 1;
    }

    public WorkerMultiplexer getWorkerMultiplexer() {
      return workerMultiplexer;
    }

    public Integer getRefCount() {
      return refCount;
    }
  }

  /** Resets the instances. For testing only. */
  @VisibleForTesting
  static void reset() {
    multiplexerInstance.clear();
  }
}
