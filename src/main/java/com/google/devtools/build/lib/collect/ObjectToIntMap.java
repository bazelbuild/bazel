// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import java.util.function.IntSupplier;

/**
 * A compact, thread-safe hash map from keys to non-negative ints, using about 8 bytes per slot.
 *
 * <p>The map is split into {@link #SHARDS} shards, each an open-addressing (linear probing) table
 * that grows by {@link #GROWTH} once it is {@link #LOAD_FACTOR} full. Shard capacities are not
 * restricted to powers of two (slots are picked by multiplying the hash with the capacity), so
 * growing never doubles the size of a shard and only one shard is copied at a time.
 *
 * <p>Every operation takes the lock of its shard, except for {@link #getUnsynchronized}.
 *
 * <p>There are many small shards because insertions into a shard wait while it grows: the total
 * waiting time shrinks quadratically with the number of shards (each growth copies fewer slots and
 * fewer insertions arrive at the shard meanwhile). With 64 shards, concurrently inserting ~1.8M
 * keys from 30 threads spent ~9 s waiting for locks in total; with 1024 shards, ~0.7 s. An empty
 * map costs {@code SHARDS * INITIAL_CAPACITY} slots (~128 KB).
 *
 * <p>This is used instead of fastutil's {@code Object2IntOpenHashMap} because the capacity of the
 * latter is always a power of two, so it can only grow by doubling: right after growing, it is at
 * most half full, and while growing, the old and the new arrays (3x the old size) are alive at the
 * same time. Growing by {@link #GROWTH} wastes much less memory on maps with millions of keys. It
 * is also not thread-safe, so it would need a single lock for all threads.
 */
public final class ObjectToIntMap<K> {
  @VisibleForTesting static final int SHARDS = 1024;
  @VisibleForTesting static final double LOAD_FACTOR = 0.75;
  @VisibleForTesting static final double GROWTH = 1.25;
  private static final int INITIAL_CAPACITY = 16;

  /** A shard: an open-addressing table guarded by its own monitor. */
  private static final class Shard<K> {
    K[] keys;
    int[] values;
    int limit;
    int size;

    Shard(int capacity) {
      keys = newKeyArray(capacity);
      values = new int[capacity];
      limit = Math.min(capacity - 1, (int) (capacity * LOAD_FACTOR));
    }

    // Safe because the array only ever holds Ks and never escapes this class, so its runtime type
    // (Object[]) is never observed as K[].
    @SuppressWarnings("unchecked")
    static <K> K[] newKeyArray(int capacity) {
      return (K[]) new Object[capacity];
    }
  }

  private final Shard<K>[] shards;

  // Safe because the array is private, only ever holds Shard<K>s and generic arrays can't be
  // created directly.
  @SuppressWarnings("unchecked")
  public ObjectToIntMap() {
    shards = (Shard<K>[]) new Shard<?>[SHARDS];
    for (int s = 0; s < SHARDS; s++) {
      shards[s] = new Shard<>(INITIAL_CAPACITY);
    }
  }

  /**
   * Mixes {@code key.hashCode()} into a 64-bit hash using Fibonacci hashing (multiplying by 2^64 /
   * golden ratio, rounded to odd) so that the upper bits are well mixed even if {@code
   * key.hashCode()} is not.
   *
   * <pre>
   *  63       54 53                             22 21                 0
   *  +----------+---------------------------------+--------------------+
   *  |  shard   |          slot (32-bit)          |       unused       |
   *  | 10 bits  |             32 bits             |      22 bits       |
   *  +----------+---------------------------------+--------------------+
   * </pre>
   */
  private static <K> long hash(K key) {
    return key.hashCode() * 0x9E3779B97F4A7C15L;
  }

  /** Extracts bits 63..54 of {@code h} to select one of the {@link #SHARDS} shards. */
  private static int shard(long h) {
    return (int) (h >>> 54);
  }

  /**
   * Maps bits 53..22 of {@code h} (an unsigned 32-bit integer {@code u}) to {@code [0, capacity)}
   * by taking the upper 32 bits of the 64-bit product {@code u * capacity}:
   *
   * <pre>
   *  63                             32 31                              0
   *  +--------------------------------+--------------------------------+
   *  |      slot in [0, capacity)     |       fractional remainder       |
   *  +--------------------------------+--------------------------------+
   * </pre>
   */
  private static int slot(long h, int capacity) {
    return (int) ((((h >>> 22) & 0xFFFFFFFFL) * capacity) >>> 32);
  }

  /**
   * Returns the slot of {@code key} in {@code shard} if present, otherwise the bitwise complement
   * (a negative number) of the free slot where it would be inserted.
   */
  // The reference comparison is only a fast path before equals().
  @SuppressWarnings("ReferenceEquality")
  private static <K> int probe(Shard<K> shard, K key, long h) {
    K[] ks = shard.keys;
    int capacity = ks.length;
    for (int i = slot(h, capacity); ; ) {
      K k = ks[i];
      if (k == null) {
        return ~i;
      }
      if (k == key || k.equals(key)) {
        return i;
      }
      if (++i == capacity) {
        i = 0;
      }
    }
  }

  /** Returns the value for {@code key}, or -1 if absent. */
  public int get(K key) {
    long h = hash(key);
    Shard<K> shard = shards[shard(h)];
    synchronized (shard) {
      int i = probe(shard, key, h);
      return i >= 0 ? shard.values[i] : -1;
    }
  }

  /**
   * Like {@link #get}, but without taking the lock of the shard, which avoids contention when many
   * threads look keys up. Only safe once the map is not modified anymore, and only if all
   * modifications happen-before the call.
   */
  public int getUnsynchronized(K key) {
    long h = hash(key);
    Shard<K> shard = shards[shard(h)];
    int i = probe(shard, key, h);
    return i >= 0 ? shard.values[i] : -1;
  }

  /**
   * Returns the value of {@code key} if present. Otherwise maps {@code key} to the value returned
   * by {@code newValue}, which must be non-negative and is called at most once per key, under a
   * lock, and returns its bitwise complement (a negative number), so that the caller can tell
   * whether the key was added by this call.
   *
   * <p>This method is not re-entrant; {@code newValue} shouldn't touch the map.
   */
  public int getOrPut(K key, IntSupplier newValue) {
    long h = hash(key);
    Shard<K> shard = shards[shard(h)];
    synchronized (shard) {
      int i = probe(shard, key, h);
      if (i >= 0) {
        return shard.values[i];
      }
      int sizeBefore = shard.size;
      int value = newValue.getAsInt();
      checkArgument(value >= 0, value);
      // The lock is re-entrant, so this is the only line of defense against newValue modifying the
      // shard, which would invalidate the slot returned by probe().
      checkState(shard.size == sizeBefore, "newValue modified the map");
      insertAt(shard, ~i, key, h, value);
      return ~value;
    }
  }

  /** Maps {@code key} to {@code value}, which must be non-negative. */
  public void put(K key, int value) {
    checkArgument(value >= 0, value);
    long h = hash(key);
    Shard<K> shard = shards[shard(h)];
    synchronized (shard) {
      int i = probe(shard, key, h);
      if (i >= 0) {
        shard.values[i] = value;
      } else {
        insertAt(shard, ~i, key, h, value);
      }
    }
  }

  /**
   * Inserts {@code key}, which is absent, at {@code slot}, the free slot returned by {@link #probe}
   * for {@code shard}, unless the shard has to grow first. Holds the lock.
   */
  private static <K> void insertAt(Shard<K> shard, int slot, K key, long h, int value) {
    if (shard.size >= shard.limit) {
      grow(shard);
      slot = ~probe(shard, key, h);
    }
    shard.values[slot] = value;
    shard.keys[slot] = key;
    shard.size++;
  }

  /** Replaces {@code shard}'s arrays with larger ones and rehashes its entries. Holds the lock. */
  private static <K> void grow(Shard<K> shard) {
    K[] oldKeys = shard.keys;
    int[] oldValues = shard.values;
    int capacity = oldKeys.length;
    int newCapacity = Math.max(capacity + 1, (int) (capacity * GROWTH));
    shard.keys = Shard.newKeyArray(newCapacity);
    shard.values = new int[newCapacity];
    shard.limit = Math.min(newCapacity - 1, (int) (newCapacity * LOAD_FACTOR));
    for (int i = 0; i < capacity; i++) {
      K k = oldKeys[i];
      if (k != null) {
        // Keys are unique, so probing only finds a free slot.
        int slot = ~probe(shard, k, hash(k));
        shard.values[slot] = oldValues[i];
        shard.keys[slot] = k;
      }
    }
  }

  /** Returns the number of keys. */
  public int size() {
    int size = 0;
    for (Shard<K> shard : shards) {
      synchronized (shard) {
        size += shard.size;
      }
    }
    return size;
  }

  /** The total number of slots, each costing a key reference and an int. */
  @VisibleForTesting
  long slots() {
    long slots = 0;
    for (Shard<K> shard : shards) {
      synchronized (shard) {
        slots += shard.keys.length;
      }
    }
    return slots;
  }
}
