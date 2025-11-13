// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import static java.util.concurrent.TimeUnit.NANOSECONDS;

import com.google.common.base.Preconditions;
import java.io.Serial;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Phaser;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.AbstractQueuedSynchronizer;

/**
 * A synchronization aid that allows one or more threads to wait until a set of operations being
 * performed in other threads completes.
 *
 * <p>This class is functionally identical to {@link CountDownLatch}, but additionally supports
 * incrementing the count, using method {@link #increment(int)}. This allows more tasks to be
 * executed before a thread waiting in {@link #await()} is released.
 *
 * <p>For example, here is a way to wait for the completion of a collection of tasks when the number
 * of tasks is not known in advance:
 *
 * {@snippet :
 * final IncrementableCountDownLatch latch
 *     = new IncrementableCountDownLatch(1);  // "lock"
 * for (...) {
 *   latch.increment();
 *   executor.submit(new Runnable() {
 *     public void run() {
 *       // do work
 *       latch.countDown();
 *     }
 *   });
 * }
 * latch.countDown();  // "unlock"
 * latch.await();
 * // all of the work is done
 * }
 *
 * <p>Consider instead using {@link Phaser} or {@link java.util.concurrent.CountedCompleter
 * CountedCompleter}. These are standard higher level synchronizers that also provide an
 * incrementable counter whose countdown can be awaited, while providing additional capabilities.
 * CountedCompleter, available since Java 8, may be useful when controlling tasks running in a fork
 * join pool.
 *
 * @author Doug Lea
 * @author Martin Buchholz
 * @author Shay Raz
 */
public final class IncrementableCountDownLatch {

  /** Synchronization control for IncrementableCountDownLatch. Uses AQS state to represent count. */
  private static final class Sync extends AbstractQueuedSynchronizer {

    Sync(int count) {
      setState(count);
    }

    int getCount() {
      return getState();
    }

    @Override
    public int tryAcquireShared(int acquires) {
      return (getState() == 0) ? 1 : -1;
    }

    @Override
    public boolean tryReleaseShared(int releases) {
      // Decrement count; signal when transition to zero
      while (true) {
        int c = getState();
        if (c == 0) {
          return false;
        }
        if (releases > c) {
          return false;
        }
        int nextc = c - releases;
        if (compareAndSetState(c, nextc)) {
          return nextc == 0;
        }
      }
    }

    void increaseCount(int delta) {
      while (true) {
        int current = getState();
        int next = current + delta;
        if (current == 0) {
          throw new IllegalStateException("already counted down to zero");
        }
        if (next < current) { // overflow
          throw new IllegalArgumentException("count overflow");
        }
        if (compareAndSetState(current, next)) {
          return;
        }
      }
    }

    @Serial private static final long serialVersionUID = 0L;
  }

  private final Sync sync;

  /**
   * Constructs a {@code IncrementableCountDownLatch} initialized with the given count.
   *
   * @param count the number of times {@link #countDown} must be invoked before threads can pass
   *     through {@link #await()}
   * @throws IllegalArgumentException if {@code count} is negative
   */
  public IncrementableCountDownLatch(int count) {
    Preconditions.checkArgument(count >= 0, "count (%s) must be >= 0", count);
    this.sync = new Sync(count);
  }

  /**
   * Causes the current thread to wait until the latch has counted down to zero, unless the thread
   * is {@linkplain Thread#interrupt interrupted}.
   *
   * <p>If the current count is zero then this method returns immediately.
   *
   * <p>If the current count is greater than zero then the current thread becomes disabled for
   * thread scheduling purposes and lies dormant until one of two things happen:
   *
   * <ul>
   *   <li>The count reaches zero due to invocations of the {@link #countDown} method; or
   *   <li>Some other thread {@linkplain Thread#interrupt interrupts} the current thread.
   * </ul>
   *
   * <p>If the current thread:
   *
   * <ul>
   *   <li>has its interrupted status set on entry to this method; or
   *   <li>is {@linkplain Thread#interrupt interrupted} while waiting,
   * </ul>
   *
   * <p>then {@link InterruptedException} is thrown and the current thread's interrupted status is
   * cleared.
   *
   * @throws InterruptedException if the current thread is interrupted while waiting
   */
  public void await() throws InterruptedException {
    sync.acquireSharedInterruptibly(1);
  }

  /**
   * Causes the current thread to wait until the latch has counted down to zero, unless the thread
   * is {@linkplain Thread#interrupt interrupted}, or the specified waiting time elapses.
   *
   * <p>If the current count is zero then this method returns immediately with the value {@code
   * true}.
   *
   * <p>If the current count is greater than zero then the current thread becomes disabled for
   * thread scheduling purposes and lies dormant until one of three things happen:
   *
   * <ul>
   *   <li>The count reaches zero due to invocations of the {@link #countDown} method; or
   *   <li>Some other thread {@linkplain Thread#interrupt interrupts} the current thread; or
   *   <li>The specified waiting time elapses.
   * </ul>
   *
   * <p>If the count reaches zero, then the method returns with the value {@code true}.
   *
   * <p>If the current thread:
   *
   * <ul>
   *   <li>has its interrupted status set on entry to this method; or
   *   <li>is {@linkplain Thread#interrupt interrupted} while waiting,
   * </ul>
   *
   * <p>then {@link InterruptedException} is thrown and the current thread's interrupted status is
   * cleared.
   *
   * <p>If the specified waiting time elapses then the value {@code false} is returned. If the time
   * is less than or equal to zero, the method will not wait at all.
   *
   * @param timeout the maximum time to wait
   * @return {@code true} if the count reached zero and {@code false} if the waiting time elapsed
   *     before the count reached zero
   * @throws InterruptedException if the current thread is interrupted while waiting
   */
  public boolean await(Duration timeout) throws InterruptedException {
    return await(timeout.toNanos(), NANOSECONDS);
  }

  /**
   * Causes the current thread to wait until the latch has counted down to zero, unless the thread
   * is {@linkplain Thread#interrupt interrupted}, or the specified waiting time elapses.
   *
   * <p>If the current count is zero then this method returns immediately with the value {@code
   * true}.
   *
   * <p>If the current count is greater than zero then the current thread becomes disabled for
   * thread scheduling purposes and lies dormant until one of three things happen:
   *
   * <ul>
   *   <li>The count reaches zero due to invocations of the {@link #countDown} method; or
   *   <li>Some other thread {@linkplain Thread#interrupt interrupts} the current thread; or
   *   <li>The specified waiting time elapses.
   * </ul>
   *
   * <p>If the count reaches zero, then the method returns with the value {@code true}.
   *
   * <p>If the current thread:
   *
   * <ul>
   *   <li>has its interrupted status set on entry to this method; or
   *   <li>is {@linkplain Thread#interrupt interrupted} while waiting,
   * </ul>
   *
   * <p>then {@link InterruptedException} is thrown and the current thread's interrupted status is
   * cleared.
   *
   * <p>If the specified waiting time elapses then the value {@code false} is returned. If the time
   * is less than or equal to zero, the method will not wait at all.
   *
   * @param timeout the maximum time to wait
   * @param unit the time unit of the {@code timeout} argument
   * @return {@code true} if the count reached zero and {@code false} if the waiting time elapsed
   *     before the count reached zero
   * @throws InterruptedException if the current thread is interrupted while waiting
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public boolean await(long timeout, TimeUnit unit) throws InterruptedException {
    return sync.tryAcquireSharedNanos(1, unit.toNanos(timeout));
  }

  /**
   * Decrements the count of the latch, releasing all waiting threads if the count reaches zero.
   *
   * <p>If the current count is greater than zero then it is decremented. If the new count is zero
   * then all waiting threads are re-enabled for thread scheduling purposes.
   *
   * <p>If the current count equals zero, then nothing happens.
   */
  public void countDown() {
    sync.releaseShared(1);
  }

  /**
   * Increases the count of the latch by 1.
   *
   * <p>The caller must ensure that the count of the latch is greater than zero.
   *
   * <p>This is a convenience method, equivalent to {@link #increment(int) increment(1)}.
   *
   * @throws IllegalStateException if the latch has already counted down to 0
   */
  public void increment() {
    sync.increaseCount(1);
  }

  /**
   * Increases the count of the latch by {@code delta}.
   *
   * <p>The caller must ensure that the count of the latch is greater than zero.
   *
   * @param delta number of additional calls to {@link #countDown} required till waiting threads are
   *     released
   * @throws IllegalArgumentException if {@code delta} is negative
   * @throws IllegalStateException if the latch has already counted down to 0
   */
  public void increment(int delta) {
    Preconditions.checkArgument(delta >= 0, "delta (%s) must be >= 0", delta);
    sync.increaseCount(delta);
  }

  /**
   * Returns the current count.
   *
   * <p>This method is typically used for debugging and testing purposes.
   *
   * @return the current count
   */
  public long getCount() {
    return sync.getCount();
  }

  /**
   * Returns a string identifying this latch, as well as its state. The state, in brackets, includes
   * the String {@code "Count ="} followed by the current count.
   *
   * @return a string identifying this latch, as well as its state
   */
  @Override
  public String toString() {
    return super.toString() + "[Count = " + sync.getCount() + "]";
  }

  /** Invokes {@link #await()} uninterruptibly. */
  public void awaitUninterruptibly() {
    boolean interrupted = false;
    try {
      while (true) {
        try {
          await();
          return;
        } catch (InterruptedException e) {
          interrupted = true;
        }
      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }
}
