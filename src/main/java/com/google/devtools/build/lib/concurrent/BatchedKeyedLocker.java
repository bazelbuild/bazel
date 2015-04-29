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

import java.util.Comparator;

/** A {@link KeyedLocker} that additionally offers batched locking functionality. */
public interface BatchedKeyedLocker<K> extends KeyedLocker<K> {
  /** Factory for {@link BatchedKeyedLocker} instances. */
  interface Factory<K> {
    /**
     * Returns a fresh {@link BatchedKeyedLocker} instance.
     *
     * <p>The given {@link Comparator} instance is used to get consistent ordering for
     * {@link BatchedKeyedLocker#lockBatch}.
     */
    BatchedKeyedLocker<K> create(Comparator<K> comparator);
  }

  /**
   * Similar to {@link #lock}, blocks the current thread until it has exclusive access to do
   * things with all the keys in {@code keys} and returns a single {@link AutoUnlocker} instance
   * for yielding the implicit locks on all the given keys.
   *
   * <p>If a thread has an unclosed {@link AutoUnlocker} instance returned by a call to
   * {@code lockBatch(keys)}, this is equivalent to having separate, unclosed {@link AutoUnlocker}
   * instances for each {@code k} in {@code keys}.
   *
   * <p>Note that use of this method avoids the concerns described in {@link #lock}.
   */
  AutoUnlocker lockBatch(Iterable<K> keys);
}
