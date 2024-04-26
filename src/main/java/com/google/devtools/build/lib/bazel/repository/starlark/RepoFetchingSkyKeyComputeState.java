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

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
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
  sealed interface Signal {
    /**
     * Indicates that the host thread should return {@code null}, causing a Skyframe restart. After
     * sending this signal, the client will immediately block on {@code delegateEnvQueue}, waiting
     * for the host thread to send a fresh {@link SkyFunction.Environment} over.
     */
    record Restart() implements Signal {}

    /** Indicates that the worker thread has finished running successfully. */
    record Success(RepositoryDirectoryValue.Builder result) implements Signal {}

    /** Indicates that the worker thread has finished running with a failure. */
    record Failure(Throwable e) implements Signal {}
  }

  /** The channel for the worker thread to send a signal to the host Skyframe thread. */
  final BlockingQueue<Signal> signalQueue = new SynchronousQueue<>();

  /**
   * The channel for the host Skyframe thread to send fresh {@link SkyFunction.Environment} objects
   * back to the worker thread.
   */
  final BlockingQueue<SkyFunction.Environment> delegateEnvQueue = new SynchronousQueue<>();

  /** The working thread that actually performs the fetching logic. */
  // This is volatile since we set it to null to indicate the worker thread isn't running, and this
  // could happen on multiple threads. Interrupting and joining a thread multiple times is safe,
  // though, so we only need to worry about nullness. Using a mutex/synchronization is an
  // alternative but it means we might block in `close()`, which is potentially bad (see its
  // javadoc).
  @Nullable volatile Thread workerThread = null;

  /**
   * This is where the recorded inputs & values for the whole invocation is collected.
   *
   * <p>{@link com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction} creates a
   * new map on each restart, so we can't simply plumb that in.
   */
  final Map<RepoRecordedInput, String> recordedInputValues = new TreeMap<>();

  SkyFunction.Environment signalForFreshEnv() throws InterruptedException {
    signalQueue.put(new Signal.Restart());
    return delegateEnvQueue.take();
  }

  public void join() {
    Uninterruptibles.joinUninterruptibly(workerThread);
    workerThread = null;
  }

  @Override
  public void close() {
    if (workerThread == null) {
      return;
    }
    workerThread.interrupt();

    // Wait until the worker thread actually gets interrupted. Be resilient to cases where despite
    // the interrupt above, the worker thread was already trying to post a restart. I'm not sure if
    // that can happen; theoretically, it looks like it shouldn't be but I'm not intimately familiar
    // with the exact semantics of thread interruption and it's cheap to be resilient. The important
    // part is that in case an interrupt happens, a Success or Failure is eventually posted by the
    // worker thread.
    while (true) {
      Signal s = Uninterruptibles.takeUninterruptibly(signalQueue);
      if (s instanceof Signal.Success || s instanceof Signal.Failure) {
        break;
      }
    }

    Uninterruptibles.joinUninterruptibly(workerThread);
    workerThread = null;
  }
}
