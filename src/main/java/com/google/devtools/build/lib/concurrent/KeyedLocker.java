// Copyright 2015 Google Inc. All rights reserved.
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

import javax.annotation.concurrent.ThreadSafe;

/** A keyed store of locks. */
@ThreadSafe
public interface KeyedLocker<K> {
  /** Used to yield access to the implicit lock granted by {@link #lock}. */
  @ThreadSafe
  interface AutoUnlocker extends AutoCloseable {
    /**
     * If this was returned by {@code lock(k)}, yields exclusive access to {@code k}.
     *
     * <p>This method should be called at most once, and may only be called by the same thread that
     * acquired the {@link AutoUnlocker} via {@link #lock}. Implementations are free to do anything
     * if this is violated.
     */
    @Override
    void close();
  }

  /**
   * Blocks the current thread until it has exclusive access to do things with {@code k} and
   * returns a {@link AutoUnlocker} instance for yielding the implicit lock. The intended usage
   * is:
   *
   * <pre>
   * {@code
   * try (AutoUnlocker unlocker = locker.lock(k)) {
   *   // Your code here.
   * }
   * }
   * </pre>
   */
  AutoUnlocker lock(K key);
}