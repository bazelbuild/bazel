// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Maps;
import com.google.common.collect.UnmodifiableIterator;
import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * A bimap with the following features and restrictions:
 *
 * <ul>
 *   <li>it (lazily) tracks the order in which keys were inserted;
 *   <li>... but only for entries whose values satisfy a predicate;
 *   <li>it's append-only, i.e. it supports addition of new key-value pairs, or replacement of the
 *       value of an existing key, but not deletion of key-value pairs;
 *   <li>... with the restriction that replacement is not allowed to make a previously tracked entry
 *       become untracked.
 * </ul>
 *
 * <p>Tracking the insertion order and prohibiting key deletion allows this bimap to provide a
 * lightweight snapshot view for iterating (in key insertion order) over entries which existed at a
 * given point in time.
 *
 * <p>Intended to be used by {@code native.existing_rules} in Starlark, which needs to be able to
 * iterate, at some later point in time, over the rules which existed in a {@link Package.Builder}
 * at the time of the {@code existing_rules} call. We do not want to track insertion orders of
 * numerous non-rule targets (e.g. files) - hence, filtering by a predicate. And in the common case
 * where {@code existing_rules} is never called in a package, we want to avoid the overhead of
 * keeping track of insertion orders - hence, laziness.
 *
 * <p>In packages with a large number of targets, the use of lightweight snapshots instead of
 * copying results in a noticeable improvement in loading times, e.g. 2.2 times faster loading for a
 * package with 4400 targets and 300 {@code native.existing_rules} calls.
 */
final class SnapshottableBiMap<K, V> implements BiMap<K, V> {
  private final BiMap<K, V> contents = HashBiMap.create();
  private final Predicate<V> track;

  // trackedKeys and trackedKeyOrders are initialized lazily by ensureOrderTracking(). In the case
  // where the order-tracking map represents a package builder's targets, ensureOrderTracking() is
  // intended to be triggered only by a call to {@code native.existing_rules} in Starlark.
  //
  // Holds all keys being tracked, in their relative insertion order.
  private ArrayList<K> trackedKeys;
  // Maps all keys being tracked to their index in trackedKeys.
  private Map<K, Integer> trackedKeyOrders;

  public SnapshottableBiMap(Predicate<V> track) {
    this.track = track;
  }

  /**
   * Returns the underlying contents bimap.
   *
   * <p>Mutating the underlying bimap will violate the guarantees of this class and possibly cause
   * inconsistent behavior in snapshot views. Therefore, the recommended usage pattern is to replace
   * any references to the {@code SnapshottableBiMap} with the underlying map, and ensure that any
   * snapshots of the map are no longer in use at that point.
   *
   * <p>An optimization hack intended only for use from {@link Package.Builder#beforeBuild}.
   */
  BiMap<K, V> getUnderlyingBiMap() {
    return contents;
  }

  @Override
  public int size() {
    return contents.size();
  }

  private int sizeTracked() {
    ensureOrderTracking();
    return trackedKeyOrders.size();
  }

  @Override
  public boolean isEmpty() {
    return contents.isEmpty();
  }

  @Override
  public boolean containsKey(Object key) {
    return contents.containsKey(key);
  }

  @Override
  public boolean containsValue(Object value) {
    return contents.containsValue(value);
  }

  @Override
  @Nullable
  public V get(Object key) {
    return contents.get(key);
  }

  /**
   * Returns the insertion order of the specified key (relative to other tracked keys), or -1 if the
   * key was never inserted into the map or corresponds to a key-value pair whose insertion order we
   * do not track. Replacing a key's value does not change this order if tracking has already begun.
   */
  private int getTrackedKeyOrder(Object key) {
    ensureOrderTracking();
    Integer order = trackedKeyOrders.get(key);
    return order == null ? -1 : order;
  }

  /**
   * Returns the tracked key with the specified insertion order (as determined by {@link
   * #getTrackedKeyOrder}).
   *
   * @throws IndexOutOfBoundsException if the specified insertion order is out of bounds
   */
  private K getTrackedKey(int order) {
    ensureOrderTracking();
    return trackedKeys.get(order);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Note that once key insertion order tracking has started, overriding a key with a different
   * value will not change the key's insertion order.
   *
   * @throws IllegalArgumentException if attempting to replace a key-value pair whose insertion
   *     order was tracked with a key-value pair whose insertion order is not tracked, or if the
   *     given value is already bound to a different key in this map.
   */
  @Override
  @Nullable
  public V put(K key, V value) {
    if (startedOrderTracking()) {
      boolean oldWasTracked = getTrackedKeyOrder(key) >= 0;
      boolean newIsTracked = track.test(value);
      if (oldWasTracked) {
        Preconditions.checkArgument(
            newIsTracked,
            "Cannot replace a key-value pair which is tracked with a key-value pair which is"
                + " not tracked");
      } else {
        if (newIsTracked) {
          recordKeyOrder(key);
        }
      }
    }
    return contents.put(key, value);
  }

  /**
   * @deprecated Not supported, since it's morally equivalent to preceding a {@link #put} call with
   *     a silent {@code this.values().remove(value)}.
   * @throws UnsupportedOperationException always.
   */
  @Deprecated
  @Override
  @Nullable
  public V forcePut(K key, V value) {
    throw new UnsupportedOperationException("Append-only data structure");
  }

  /**
   * @deprecated Not supported.
   * @throws UnsupportedOperationException always.
   */
  @Deprecated
  @Override
  @Nullable
  public V remove(Object key) {
    throw new UnsupportedOperationException("Append-only data structure");
  }

  @Override
  public void putAll(Map<? extends K, ? extends V> map) {
    for (Map.Entry<? extends K, ? extends V> entry : map.entrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }

  /**
   * @deprecated Not supported.
   * @throws UnsupportedOperationException always.
   */
  @Deprecated
  @Override
  public void clear() {
    throw new UnsupportedOperationException("Append-only data structure");
  }

  /**
   * {@inheritDoc}
   *
   * <p>Removing a key from the set does not change the key's order if it was tracked prior to
   * removal. Removal is supported only for consistency with {@link values}.
   */
  @Override
  public Set<K> keySet() {
    return Collections.unmodifiableSet(contents.keySet());
  }

  /**
   * {@inheritDoc}
   *
   * <p>Removing a value from the set does not change the key's order if it was tracked prior to
   * removal. Ideally, we would not want to support removal, but it is required for {@link
   * PackageFunction#handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions}.
   */
  @Override
  public Set<V> values() {
    return Collections.unmodifiableSet(contents.values());
  }

  /**
   * {@inheritDoc}
   *
   * <p>Removing an entry from the set does not change the key's order if it was tracked prior to
   * removal. Removal is supported only for consistency with {@link values}.
   */
  @Override
  public Set<Map.Entry<K, V>> entrySet() {
    return Collections.unmodifiableSet(contents.entrySet());
  }

  /**
   * {@inheritDoc}
   *
   * <p>The returned map is unmodifiable (all modifications will throw an {@link
   * UnsupportedOperationException}.
   */
  @Override
  public BiMap<V, K> inverse() {
    return Maps.unmodifiableBiMap(contents.inverse());
  }

  private boolean startedOrderTracking() {
    Preconditions.checkState((trackedKeys == null) == (trackedKeyOrders == null));
    return trackedKeys != null;
  }

  private void ensureOrderTracking() {
    if (!startedOrderTracking()) {
      trackedKeys = new ArrayList<>();
      trackedKeyOrders = new HashMap<>();

      contents.forEach(
          (key, value) -> {
            if (track.test(value)) {
              recordKeyOrder(key);
            }
          });
    }
  }

  private void recordKeyOrder(K key) {
    int order = trackedKeys.size();
    trackedKeys.add(key);
    trackedKeyOrders.put(key, order);
  }

  /**
   * Returns a lightweight snapshot view of the tracked entries existing in the bimap at the time
   * this method is called.
   *
   * <p>Most method calls on the view returned by this method will start insertion order tracking if
   * it has not been started already. In particular, that implies that after this method had been
   * called, a value whose insertion order was tracked may no longer be replaceable with a value
   * whose insertion order is not tracked. See {@link #put} for details.
   */
  public Map<K, V> getTrackedSnapshot() {
    return new TrackedSnapshot<>(this);
  }

  /**
   * A view of a {@link SnapshottableBiMap}'s contents existing at a certain point in time.
   *
   * <p>Iterators over the view's {@link #keySet}, {@link #entrySet}, or {@link #values} iterate in
   * key insertion order.
   */
  static final class TrackedSnapshot<K, V> extends AbstractMap<K, V> {
    private final SnapshottableBiMap<K, V> underlying;
    // The number of initial elements from `underlying`'s `trackedKeys` list that should be
    // considered to be present in this view. Note that we don't snapshot values, so we'll use
    // whatever the most recent value in `underlying` is even if it changed after this snapshot
    // was created.
    private final int sizeTracked;

    private TrackedSnapshot(SnapshottableBiMap<K, V> underlying) {
      this.underlying = underlying;
      this.sizeTracked = underlying.sizeTracked();
    }

    @Override
    public boolean containsKey(Object key) {
      int order = underlying.getTrackedKeyOrder(key);
      return order >= 0 && order < sizeTracked;
    }

    @Override
    public boolean containsValue(Object value) {
      Object key = underlying.inverse().get(value);
      if (key != null) {
        int order = underlying.getTrackedKeyOrder(key);
        return order >= 0 && order < sizeTracked;
      } else {
        return false;
      }
    }

    @Override
    @Nullable
    public V get(Object key) {
      if (containsKey(key)) {
        return underlying.get(key);
      } else {
        return null;
      }
    }

    /**
     * @deprecated Unsupported operation.
     * @throws UnsupportedOperationException always.
     */
    @Deprecated
    @Override
    @Nullable
    public V put(K key, V value) {
      throw new UnsupportedOperationException("Read-only snapshot");
    }

    /**
     * @deprecated Unsupported operation.
     * @throws UnsupportedOperationException always.
     */
    @Deprecated
    @Override
    @Nullable
    public V remove(Object key) {
      throw new UnsupportedOperationException("Read-only snapshot");
    }

    /**
     * @deprecated Unsupported operation.
     * @throws UnsupportedOperationException always.
     */
    @Deprecated
    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
      throw new UnsupportedOperationException("Read-only snapshot");
    }

    /**
     * @deprecated Unsupported operation.
     * @throws UnsupportedOperationException always.
     */
    @Deprecated
    @Override
    public void clear() {
      throw new UnsupportedOperationException("Read-only snapshot");
    }

    @Override
    public Set<Map.Entry<K, V>> entrySet() {
      return new UnmodifiableSet<Map.Entry<K, V>>() {
        @Override
        public int size() {
          return sizeTracked;
        }

        @Override
        public boolean isEmpty() {
          return sizeTracked == 0;
        }

        @Override
        public boolean contains(Object object) {
          if (!(object instanceof Map.Entry<?, ?>)) {
            return false;
          }
          Map.Entry<?, ?> entry = (Map.Entry<?, ?>) object;
          return TrackedSnapshot.this.containsKey(entry.getKey())
              && TrackedSnapshot.this.containsValue(entry.getValue());
        }

        @Override
        public Iterator<Map.Entry<K, V>> iterator() {
          return new UnmodifiableIterator<Map.Entry<K, V>>() {
            private int nextOrder = 0;

            @Override
            public boolean hasNext() {
              return nextOrder < TrackedSnapshot.this.sizeTracked;
            }

            @Override
            public Map.Entry<K, V> next() {
              if (!hasNext()) {
                throw new NoSuchElementException();
              }
              K key = TrackedSnapshot.this.underlying.getTrackedKey(nextOrder);
              V value = TrackedSnapshot.this.underlying.get(key);
              nextOrder++;
              return new AbstractMap.SimpleEntry<>(key, value);
            }
          };
        }
      };
    }

    private abstract static class UnmodifiableSet<E> extends AbstractSet<E> {
      /**
       * @deprecated Unsupported operation.
       * @throws UnsupportedOperationException always.
       */
      @Deprecated
      @Override
      public boolean add(E entry) {
        throw new UnsupportedOperationException();
      }

      /**
       * @deprecated Unsupported operation.
       * @throws UnsupportedOperationException always.
       */
      @Deprecated
      @Override
      public boolean remove(Object o) {
        throw new UnsupportedOperationException();
      }

      /**
       * @deprecated Not implemented due to lack of need.
       * @throws UnsupportedOperationException always.
       */
      @Deprecated
      @Override
      public boolean containsAll(Collection<?> c) {
        throw new UnsupportedOperationException();
      }

      /**
       * @deprecated Unsupported operation.
       * @throws UnsupportedOperationException always.
       */
      @Deprecated
      @Override
      public boolean addAll(Collection<? extends E> c) {
        throw new UnsupportedOperationException();
      }

      /**
       * @deprecated Unsupported operation.
       * @throws UnsupportedOperationException always.
       */
      @Deprecated
      @Override
      public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException();
      }

      /**
       * @deprecated Unsupported operation.
       * @throws UnsupportedOperationException always.
       */
      @Deprecated
      @Override
      public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException();
      }

      /**
       * @deprecated Unsupported operation.
       * @throws UnsupportedOperationException always.
       */
      @Deprecated
      @Override
      public void clear() {
        throw new UnsupportedOperationException();
      }
    }
  }
}
