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

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Semaphore;

/**
 * An intermediate worker that sends request and receives response from the
 * worker processes.
 */
public class WorkerMultiplexerManager {
  /**
   * There should only be one WorkerMultiplexer corresponding to workers with
   * the same mnemonic. If the WorkerMultiplexer has been constructed, other
   * workers should point to the same one. The hash of WorkerKey is used as
   * key.
   */
  private static Map<Integer, WorkerMultiplexer> multiplexerInstance = new HashMap<>();
  /** An accumulator of how many WorkerProxies are referencing a particular WorkerMultiplexer. */
  private static Map<Integer, Integer> multiplexerRefCount = new HashMap<>();
  /** A semaphore to protect multiplexerInstance and multiplexerRefCount objects. */
  private static Semaphore semMultiplexer = new Semaphore(1);

  /**
   * Returns a WorkerMultiplexer instance to WorkerProxy. WorkerProxies with the
   * same workerHash talk to the same WorkerMultiplexer. Also, record how many
   * WorkerProxies are talking to this WorkerMultiplexer.
   */
  public static WorkerMultiplexer getInstance(Integer workerHash) throws InterruptedException {
    semMultiplexer.acquire();
    if (!multiplexerInstance.containsKey(workerHash)) {
      multiplexerInstance.put(workerHash, new WorkerMultiplexer());
      multiplexerRefCount.put(workerHash, 0);
    }
    multiplexerRefCount.put(workerHash, multiplexerRefCount.get(workerHash) + 1);
    WorkerMultiplexer workerMultiplexer = multiplexerInstance.get(workerHash);
    semMultiplexer.release();
    return workerMultiplexer;
  }

  /**
   * Remove the WorkerMultiplexer instance and reference count since it is no
   * longer in use.
   */
  public static void removeInstance(Integer workerHash) throws InterruptedException {
    semMultiplexer.acquire();
    multiplexerRefCount.put(workerHash, multiplexerRefCount.get(workerHash) - 1);
    if (multiplexerRefCount.get(workerHash) == 0) {
      multiplexerInstance.get(workerHash).interrupt();
      multiplexerInstance.get(workerHash).destroyMultiplexer();
      multiplexerInstance.remove(workerHash);
      multiplexerRefCount.remove(workerHash);
    }
    semMultiplexer.release();
  }
}
