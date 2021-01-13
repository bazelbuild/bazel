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
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Worker.Code;
import com.google.devtools.build.lib.vfs.Path;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

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

  private WorkerMultiplexerManager() {}

  /**
   * Returns a {@code WorkerMultiplexer} instance to {@code WorkerProxy}. {@code WorkerProxy}
   * objects with the same {@code WorkerKey} talk to the same {@code WorkerMultiplexer}. Also,
   * record how many {@code WorkerProxy} objects are talking to this {@code WorkerMultiplexer}.
   */
  public static synchronized WorkerMultiplexer getInstance(WorkerKey key, Path logFile) {
    InstanceInfo instanceInfo =
        multiplexerInstance.computeIfAbsent(key, k -> new InstanceInfo(logFile, k));
    instanceInfo.increaseRefCount();
    return instanceInfo.getWorkerMultiplexer();
  }

  static void beforeCommand(CommandEnvironment env) {
    setReporter(env.getReporter());
  }

  static void afterCommand() {
    setReporter(null);
  }

  /**
   * Sets the reporter for all existing multiplexer instances. This allows reporting problems
   * encountered while fetching an instance, e.g. during WorkerProxy validation.
   */
  private static synchronized void setReporter(@Nullable EventHandler reporter) {
    for (InstanceInfo m : multiplexerInstance.values()) {
      m.workerMultiplexer.setReporter(reporter);
    }
  }

  /** Removes a {@code WorkerProxy} instance and reference count since it is no longer in use. */
  public static synchronized void removeInstance(WorkerKey key) throws UserExecException {
    InstanceInfo instanceInfo = multiplexerInstance.get(key);
    if (instanceInfo == null) {
      throw createUserExecException(
          "Attempting to remove non-existent multiplexer instance.",
          Code.MULTIPLEXER_INSTANCE_REMOVAL_FAILURE);
    }
    instanceInfo.decreaseRefCount();
    if (instanceInfo.getRefCount() == 0) {
      instanceInfo.getWorkerMultiplexer().interrupt();
      instanceInfo.getWorkerMultiplexer().destroyMultiplexer();
      multiplexerInstance.remove(key);
    }
  }

  @VisibleForTesting
  static WorkerMultiplexer getMultiplexer(WorkerKey key) throws UserExecException {
    InstanceInfo instanceInfo = multiplexerInstance.get(key);
    if (instanceInfo == null) {
      throw createUserExecException(
          "Accessing non-existent multiplexer instance.", Code.MULTIPLEXER_DOES_NOT_EXIST);
    }
    return instanceInfo.getWorkerMultiplexer();
  }

  @VisibleForTesting
  static Integer getRefCount(WorkerKey key) throws UserExecException {
    InstanceInfo instanceInfo = multiplexerInstance.get(key);
    if (instanceInfo == null) {
      throw createUserExecException(
          "Accessing non-existent multiplexer instance.", Code.MULTIPLEXER_DOES_NOT_EXIST);
    }
    return instanceInfo.getRefCount();
  }

  @VisibleForTesting
  static Integer getInstanceCount() {
    return multiplexerInstance.keySet().size();
  }

  private static UserExecException createUserExecException(String message, Code detailedCode) {
    return new UserExecException(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setWorker(FailureDetails.Worker.newBuilder().setCode(detailedCode))
            .build());
  }

  /** Contains the WorkerMultiplexer instance and reference count. */
  static class InstanceInfo {
    private final WorkerMultiplexer workerMultiplexer;
    private Integer refCount;

    public InstanceInfo(Path logFile, WorkerKey workerKey) {
      this.workerMultiplexer = new WorkerMultiplexer(logFile, workerKey);
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
