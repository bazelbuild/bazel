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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Throwables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;

/**
 * An equivalent of <code>ListenableFuture&lt;SpawnResult&gt;</code>.
 *
 * <p>This is a temporary wrapper for ListenableFuture to be used during the migration of all {@link
 * com.google.devtools.build.lib.exec.SpawnRunner} implementations to async execution. The reason
 * for moving to async execution is that it avoids blocking Skyframe threads for expensive
 * operations (waiting for subprocesses, possibly running remotely). Especially for remote
 * execution, this significantly improves scalability while reducing local thread contention due to
 * the high number of threads currently used to drive it.
 *
 * <p>We cannot use ListenableFuture as is as long as not all implementations have been migrated to
 * async execution. The reason for this is that we need to control which code runs in which thread
 * pool, and Skyframe currently does not expose the underlying thread pool; intentionally so - all
 * Skyframe operations have to run with a Skyframe environment so that Skyframe can track and cache
 * it.
 *
 * <p>As long as some implementations still block on evaluating the future, this blocking must not
 * happen in the implementations' thread pool. Especially for remote execution, the thread pool is
 * also responsible for network operations, and blocking threads would negate the desired benefits.
 *
 * <p>Once all implementations are async, we can use Futures.transform - stealing a little bit of
 * CPU from another thread pool is ok as long as it's non-blocking.
 */
public class FutureSpawn {
  public static FutureSpawn immediate(SpawnResult f) {
    return new FutureSpawn(Futures.immediateFuture(f));
  }

  private final ListenableFuture<? extends SpawnResult> future;
  private final Wrapper wrapper;

  public FutureSpawn(ListenableFuture<? extends SpawnResult> future) {
    this(future, (c) -> c.get());
  }

  private FutureSpawn(ListenableFuture<? extends SpawnResult> future, Wrapper wrapper) {
    this.future = future;
    this.wrapper = wrapper;
  }

  /**
   * Returns the underlying future. This is only intended to be used for getting notified about
   * completion, and should not be used to access the {@link SpawnResult} directly, which should be
   * obtained from {@link #get} instead.
   */
  public ListenableFuture<? extends SpawnResult> getFuture() {
    return future;
  }

  /**
   * Blocks the current thread until completion of the underlying future, and calls the wrappers set
   * on this future in the order in which they were set.
   */
  public SpawnResult get() throws ExecException, InterruptedException {
    return wrapper.apply(
        () -> {
          try {
            return future.get();
          } catch (ExecutionException e) {
            Throwables.propagateIfPossible(
                e.getCause(), ExecException.class, InterruptedException.class);
            throw new RuntimeException(e);
          } catch (CancellationException e) {
            throw new InterruptedException(e.getMessage());
          } catch (InterruptedException e) {
            future.cancel(/*mayInterruptIfRunning*/ true);
            throw e;
          }
        });
  }

  /**
   * Wraps the evaluation within this future with the given wrapper. This is similar to {@link
   * com.google.common.util.concurrent.Futures#lazyTransform} in that the wrapper is executed every
   * time get() is called. However, it ensures that the wrapper code isexecuted in the Skyframe
   * thread pool.
   */
  public FutureSpawn wrap(Wrapper wrapper) {
    Wrapper previousWrapper = this.wrapper;
    // Closure chaining magic: we create a new FutureSpawn with the same ListenableFuture, but with
    // a wrapper that first calls the previous wrapper and then the new wrapper.
    return new FutureSpawn(future, (c) -> wrapper.apply(() -> previousWrapper.apply(c)));
  }

  /**
   * A {@link java.util.concurrent.Callable} equivalent that declares certain exceptions we need for
   * spawn runners.
   */
  @FunctionalInterface
  public interface Callable<T> {
    T get() throws ExecException, InterruptedException;
  }

  /**
   * A {@link java.util.function.Function} equivalent that declares certain exceptions we need for
   * spawn runners.
   */
  @FunctionalInterface
  public interface Wrapper {
    /**
     * This is passed the future or a wrapped future. An implementation is expected to do any
     * desired preprocessing, then call the future, then perform any desired post-processing. Note
     * that this scheme allows catching exceptions from lower layers.
     */
    SpawnResult apply(Callable<SpawnResult> future) throws ExecException, InterruptedException;
  }
}
