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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Future;
import java.util.concurrent.SynchronousQueue;
import javax.annotation.Nullable;

/**
 * Captures state that persists across different invocations of {@link
 * com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction}, specifically {@link
 * StarlarkRepositoryFunction}.
 *
 * <p>This class is used to hold on to a worker thread (in reality just a {@link Future} object)
 * when fetching repos using a worker thread is enabled. The worker thread uses a {@link
 * SkyFunction.Environment} object acquired from the host thread, and can signal the host thread to
 * restart to get a fresh environment object.
 */
class RepoFetchingSkyKeyComputeState implements SkyKeyComputeState {

  /** A signal that the worker thread can send to the host Skyframe thread. */
  enum Signal {
    /**
     * Indicates that the host thread should return {@code null}, causing a Skyframe restart. After
     * sending this signal, the client will immediately block on {@code delegateEnvQueue}, waiting
     * for the host thread to send a fresh {@link SkyFunction.Environment} over.
     */
    RESTART,
    /**
     * Indicates that the worker thread has finished running, either yielding a result or an
     * exception.
     */
    DONE
  }

  /** The channel for the worker thread to send a signal to the host Skyframe thread. */
  final BlockingQueue<Signal> signalQueue = new SynchronousQueue<>();

  /**
   * The channel for the host Skyframe thread to send fresh {@link SkyFunction.Environment} objects
   * back to the worker thread.
   */
  final BlockingQueue<SkyFunction.Environment> delegateEnvQueue = new SynchronousQueue<>();

  /**
   * This future holds on to the worker thread in order to cancel it when necessary; it also serves
   * to tell whether a worker thread is already running.
   */
  // This is volatile since we set it to null to indicate the worker thread isn't running, and this
  // could happen on multiple threads. Canceling a future multiple times is safe, though, so we
  // only need to worry about nullness. Using a mutex/synchronization is an alternative but it means
  // we might block in `close()`, which is potentially bad (see its javadoc).
  @Nullable volatile Future<RepositoryDirectoryValue.Builder> workerFuture = null;

  /**
   * This is where the {@code markerData} for the whole invocation is collected.
   *
   * <p>{@link com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction} creates a
   * new map on each restart, so we can't simply plumb that in.
   */
  final Map<String, String> markerData = new TreeMap<>();

  SkyFunction.Environment signalForFreshEnv() throws InterruptedException {
    signalQueue.put(Signal.RESTART);
    return delegateEnvQueue.take();
  }

  @Override
  public void close() {
    var myWorkerFuture = workerFuture;
    workerFuture = null;
    if (myWorkerFuture != null) {
      myWorkerFuture.cancel(true);
    }
  }
}
