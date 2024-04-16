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
package com.google.devtools.build.lib.concurrent;

import com.google.common.collect.Interner;
import com.google.common.collect.MapMaker;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.ForOverride;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An extension of the weak interner which uses a global pool in addition to the weak interner's
 * {@code ConcurrentHashMap} to store instances.
 *
 * <p>The reason of implementing {@link PooledInterner} is that the same object can be stored in
 * both weak interner and some other container ({@code InMemoryGraphImple#nodeMap}) in blaze with
 * two equal references, causing some memory overhead.
 *
 * <p>{@link PooledInterner} enables the client to manage where a single object is stored,
 * addressing the memory overhead issue. In more detail,
 *
 * <ul>
 *   <li>If the object is already canonicalized in the global pool, it should not be stored in
 *       {@link #weakInterner} again, thus removing the storage overhead of using a traditional weak
 *       interner;
 *   <li>User can also remove the object from {@link #weakInterner}'s underlying {@link
 *       #internerAsMap} when the object appears in the global pool.
 * </ul>
 *
 * <p>Subclasses are only responsible for providing the appropriate {@link Pool} by overriding
 * {@link #getPool()} method.
 */
public abstract class PooledInterner<T> implements Interner<T> {
  private Interner<T> weakInterner = BlazeInterners.newWeakInterner();
  private Map<?, ?> internerAsMap = getMapReflectively(weakInterner);

  /**
   * Holds all PooledInterner instances registered in the lifetime of this Blaze server as a weak
   * collection to prevent memory leaks. This is also thread-safe to handle multiple PooledInterners
   * from being instantiated concurrently, for instance during class initializations.
   */
  private static final Set<PooledInterner<?>> instances =
      Collections.newSetFromMap(new MapMaker().weakKeys().makeMap());

  protected PooledInterner() {
    instances.add(this);
  }

  /**
   * Interns {@code sample} directly into {@link #weakInterner} without checking the global pool and
   * returns the canonical instance of {@code sample}.
   */
  @CanIgnoreReturnValue
  public final T weakIntern(T sample) {
    return weakInterner.intern(sample);
  }

  /**
   * Removes sample from the weak interner. Client can call this method when the sample is already
   * stored in the global pool in order to reduce the memory overhead.
   */
  public final void removeWeak(Object sample) {
    internerAsMap.remove(sample);
  }

  /**
   * Returns the canonical instance of {@code sample} from either global pool or {@link
   * #weakInterner}.
   */
  @Override
  public final T intern(T sample) {
    Pool<T> pool = getPool();
    return pool != null ? pool.getOrWeakIntern(sample) : weakInterner.intern(sample);
  }

  /**
   * Provides the global pool instance for {@link #intern(Object)} method.
   */
  @ForOverride
  @Nullable
  protected abstract Pool<T> getPool();

  public final int size() {
    return internerAsMap.size();
  }

  /**
   * Shrinks all interner instances' backing map to reclaim memory.
   *
   * <p>WARNING: This must not be called concurrently with any interning operations, because it
   * provides unsynchronized access to multiple mutable static interners.
   */
  @ThreadHostile
  public static final void shrinkAll() {
    System.gc(); // Force gc to maximize shrinkage.
    instances.forEach(PooledInterner::shrink);
  }

  /** Shrinks the weak interner and obtain a new reference to the newly shrunk map. */
  private void shrink() {
    this.weakInterner = shrinkAsNewWeakInterner(weakInterner);
    this.internerAsMap = getMapReflectively(weakInterner);
  }

  /**
   * Shrink an interner by rebuilding a new weak interner and backing map/array. Use this if you
   * expect a GC to clear references into an interner's backing map.
   *
   * <p>If there are references to the backing map, use {@code getMapReflectively} to update them.
   *
   * <p>This is created because backing maps do not automatically resize after removing entries.
   */
  private static <T> Interner<T> shrinkAsNewWeakInterner(Interner<T> fromInterner) {
    Interner<T> toInterner = BlazeInterners.newWeakInterner();
    Map<T, ?> map = getMapReflectively(fromInterner);
    map.keySet().parallelStream()
        .forEach(
            k -> {
              T unused = toInterner.intern(k);
            });
    return toInterner;
  }

  // Returns the backing map of an interner.
  //
  // There was a Guava API review to include the feature of removing from an interner, and the
  // outcome was that we should just get and manipulate the map reflectively.
  //
  // See the description for cl/623798951 for additional context.
  @SuppressWarnings("unchecked")
  private static <T> Map<T, ?> getMapReflectively(Interner<?> interner) {
    try {
      Field field = interner.getClass().getDeclaredField("map");
      field.setAccessible(true);
      return (Map<T, ?>) field.get(interner);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * An alternative container to the weak interner for storing type T instance.
   *
   * <p>A pool is a storage space that already exists during normal program execution and provides
   * lookup functionality for interning, thus eliminating storage overhead from using a classic weak
   * interner.
   */
  public interface Pool<T> {
    /**
     * Returns the canonical instance for the given key in the pool if it is present, otherwise
     * interns the key using its {@linkplain #weakIntern weak interner}.
     *
     * <p>To ensure a single canonical instance, if the key is not present in the pool, it should be
     * weakly interned using synchronization so that it is not concurrently {@linkplain #removeWeak
     * removed from the weak interner}.
     */
    T getOrWeakIntern(T sample);
  }
}
