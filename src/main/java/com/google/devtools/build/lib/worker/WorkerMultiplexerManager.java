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

import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Worker.Code;
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
  private static Map<Integer, InstanceInfo> multiplexerInstance;

  /** A semaphore to protect {@code multiplexerInstance} and {@code multiplexerRefCount} objects. */
  private static Semaphore semMultiplexer;

  static {
    multiplexerInstance = new HashMap<>();
    semMultiplexer = new Semaphore(1);
  }

  private WorkerMultiplexerManager() {}

  /**
   * Returns a {@code WorkerMultiplexer} instance to {@code WorkerProxy}. {@code WorkerProxy}
   * objects with the same workerHash talk to the same {@code WorkerMultiplexer}. Also, record how
   * many {@code WorkerProxy} objects are talking to this {@code WorkerMultiplexer}.
   */
  public static WorkerMultiplexer getInstance(Integer workerHash) throws InterruptedException {
    semMultiplexer.acquire();
    if (!multiplexerInstance.containsKey(workerHash)) {
      multiplexerInstance.put(workerHash, new InstanceInfo());
    }
    multiplexerInstance.get(workerHash).increaseRefCount();
    WorkerMultiplexer workerMultiplexer =
        multiplexerInstance.get(workerHash).getWorkerMultiplexer();
    semMultiplexer.release();
    return workerMultiplexer;
  }

  /**
   * Removes the {@code WorkerMultiplexer} instance and reference count since it is no longer in
   * use.
   */
  public static void removeInstance(Integer workerHash)
      throws InterruptedException, UserExecException {
    semMultiplexer.acquire();
    try {
      multiplexerInstance.get(workerHash).decreaseRefCount();
      if (multiplexerInstance.get(workerHash).getRefCount() == 0) {
        multiplexerInstance.get(workerHash).getWorkerMultiplexer().interrupt();
        multiplexerInstance.get(workerHash).getWorkerMultiplexer().destroyMultiplexer();
        multiplexerInstance.remove(workerHash);
      }
    } catch (Exception e) {
      String message = "NullPointerException while accessing non-existent multiplexer instance.";
      throw createUserExecException(e, message, Code.MULTIPLEXER_INSTANCE_REMOVAL_FAILURE);
    } finally {
      semMultiplexer.release();
    }
  }

  public static WorkerMultiplexer getMultiplexer(Integer workerHash) throws UserExecException {
    try {
      return multiplexerInstance.get(workerHash).getWorkerMultiplexer();
    } catch (NullPointerException e) {
      String message = "NullPointerException while accessing non-existent multiplexer instance.";
      throw createUserExecException(e, message, Code.MULTIPLEXER_DOES_NOT_EXIST);
    }
  }

  public static Integer getRefCount(Integer workerHash) throws UserExecException {
    try {
      return multiplexerInstance.get(workerHash).getRefCount();
    } catch (NullPointerException e) {
      String message = "NullPointerException while accessing non-existent multiplexer instance.";
      throw createUserExecException(e, message, Code.MULTIPLEXER_DOES_NOT_EXIST);
    }
  }

  public static Integer getInstanceCount() {
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

    public InstanceInfo() {
      this.workerMultiplexer = new WorkerMultiplexer();
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
}
