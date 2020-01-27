// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/** A keyed store of locks. */
@ThreadSafe
public interface KeyedLocker<K> {
  /**
   * Used to yield access to the implicit locks granted by {@link #writeLock} or {@link #readLock}.
   */
  @ThreadSafe
  interface AutoUnlocker extends AutoCloseable {
    /** Exception used to indicate illegal use of {@link AutoUnlocker#close}. */
    class IllegalUnlockException extends RuntimeException {
      public IllegalUnlockException(String msg) {
        super(msg);
      }
    }

    /**
     * Closes the {@link AutoUnlocker} instance. If this instance was the last unclosed one
     * returned by {@link #writeLock} with argument {@code k} owned by the current
     * thread, then exclusive access to {@code k} is yielded. If this instance was the last unclosed
     * one returned by {@link #readLock} with argument {@code k}, then a thread can request
     * exclusive write access using {@link #writeLock} with argument {@code k}.
     *
     * <p>This method may only be called at most once per {@link AutoUnlocker} instance and must
     * be called by the same thread that acquired the {@link AutoUnlocker} via {@link #writeLock}
     * or {@link #readLock}. Otherwise, an {@link IllegalUnlockException} is thrown.
     */
    @Override
    void close();
  }

  /**
   * Blocks the current thread until it has exclusive access to do things with {@code k} and
   * returns a {@link AutoUnlocker} instance for yielding the implicit lock.
   *
   * <p>Notably, this means that a thread is allowed to call {@code writeLock(k)} again before
   * calling {@link AutoUnlocker#close} for the first call to {@code writeLock(k)}. Each call to
   * {@code #writeLock} will return a different {@link AutoUnlocker} instance.
   *
   * <p>The intended usage is:
   *
   * <pre>
   * {@code
   * try (AutoUnlocker unlocker = locker.writeLock(k)) {
   *   // Your code here.
   * }
   * }
   * </pre>
   *
   * <p>Note that the usual caveats about mutexes apply here, e.g. the following may deadlock:
   *
   * <pre>
   * {@code
   * // Thread A
   * try (AutoUnlocker unlocker = locker.writeLock(k1)) {
   *   // This will deadlock if Thread B already acquired a writeLock for k2.
   *   try (AutoUnlocker unlocker = locker.writeLock(k2)) {
   *   }
   * }
   * // end Thread A
   *
   * // Thread B
   * try (AutoUnlocker unlocker = locker.writeLock(k2)) {
   *   // This will deadlock if Thread A already acquired a writeLock for k1.
   *   try (AutoUnlocker unlocker = locker.writeLock(k1)) {
   *   }
   * }
   * // end Thread B
   * }
   * </pre>
   */
  AutoUnlocker writeLock(K key);

  /**
   * Blocks the current thread until it has access to read things that have to do with {@code k}.
   * Multiple threads may acquire simultaneous read locks, so long as there is no thread with a
   * write lock.
   *
   * <p>As with {@link #writeLock}, the same thread can call {@code readLock(k)} multiple times for
   * the same k before closing the lock.
   */
  AutoUnlocker readLock(K key);
}
