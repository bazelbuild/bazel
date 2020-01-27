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
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Semaphore;

/** A manager to instantiate and distroy multiplexers. */
public class WorkerMultiplexerManager {
  /**
   * There should only be one WorkerMultiplexer corresponding to workers with the same mnemonic. If
   * the WorkerMultiplexer has been constructed, other workers should point to the same one. The
   * hash of WorkerKey is used as key.
   */
  private static Map<Integer, InstanceInfo> multiplexerInstance;

  /** A semaphore to protect multiplexerInstance and multiplexerRefCount objects. */
  private static Semaphore semMultiplexer;

  static {
    multiplexerInstance = new HashMap<>();
    semMultiplexer = new Semaphore(1);
  }

  private WorkerMultiplexerManager() {}

  /**
   * Returns a WorkerMultiplexer instance to WorkerProxy. WorkerProxies with the same workerHash
   * talk to the same WorkerMultiplexer. Also, record how many WorkerProxies are talking to this
   * WorkerMultiplexer.
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

  /** Remove the WorkerMultiplexer instance and reference count since it is no longer in use. */
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
      throw new UserExecException(
          ErrorMessage.builder()
              .message("NullPointerException while accessing non-existent multiplexer instance.")
              .exception(e)
              .build()
              .toString());
    } finally {
      semMultiplexer.release();
    }
  }

  public static WorkerMultiplexer getMultiplexer(Integer workerHash) throws UserExecException {
    try {
      return multiplexerInstance.get(workerHash).getWorkerMultiplexer();
    } catch (NullPointerException e) {
      throw new UserExecException(
          ErrorMessage.builder()
              .message("NullPointerException while accessing non-existent multiplexer instance.")
              .exception(e)
              .build()
              .toString());
    }
  }

  public static Integer getRefCount(Integer workerHash) throws UserExecException {
    try {
      return multiplexerInstance.get(workerHash).getRefCount();
    } catch (NullPointerException e) {
      throw new UserExecException(
          ErrorMessage.builder()
              .message("NullPointerException while accessing non-existent multiplexer instance.")
              .exception(e)
              .build()
              .toString());
    }
  }

  public static Integer getInstanceCount() {
    return multiplexerInstance.keySet().size();
  }

  /** Contains the WorkerMultiplexer instance and reference count */
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
