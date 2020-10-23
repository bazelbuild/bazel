// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import java.util.AbstractCollection;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * A immutable map implementation for maps with comparable keys. It uses a sorted array
 * and binary search to return the correct values. Its only purpose is to save memory - for n
 * entries, it consumes 8n + 64 bytes, much less than a normal HashMap (43n + 128) or an
 * ImmutableMap (35n + 81).
 *
 * <p>Only a few methods are efficiently implemented: {@link #isEmpty} is O(1), {@link #get} and
 * {@link #containsKey} are O(log(n)), using binary search; {@link #keySet} and {@link #values}
 * refer to the parent instance. All other methods can take O(n) or even make a copy of the
 * contents.
 *
 * <p>This implementation supports neither {@code null} keys nor {@code null} values.
 *
 * @param <K> the type of keys maintained by this map; keys must be comparable
 * @param <V> the type of mapped values
 */
public final class ImmutableSortedKeyMap<K extends Comparable<K>, V> implements Map<K, V> {

  @SuppressWarnings({"rawtypes", "unchecked"})
  private static final ImmutableSortedKeyMap EMPTY_MAP =
      new ImmutableSortedKeyMap(new Comparable<?>[0], new Object[0]);

  /** Returns the empty multimap. */
  @SuppressWarnings("unchecked")
  public static <K extends Comparable<K>, V> ImmutableSortedKeyMap<K, V> of() {
    // Safe because the multimap will never hold any elements.
    return EMPTY_MAP;
  }

  public static <K extends Comparable<K>, V> ImmutableSortedKeyMap<K, V> of(K key0, V value0) {
    return ImmutableSortedKeyMap.<K, V>builder()
        .put(key0, value0)
        .build();
  }

  public static <K extends Comparable<K>, V> ImmutableSortedKeyMap<K, V> of(
      K key0, V value0, K key1, V value1) {
    return ImmutableSortedKeyMap.<K, V>builder()
        .put(key0, value0)
        .put(key1, value1)
        .build();
  }

  @SuppressWarnings("unchecked")
  public static <K extends Comparable<K>, V> ImmutableSortedKeyMap<K, V> copyOf(Map<K, V> data) {
    if (data.isEmpty()) {
      return EMPTY_MAP;
    }
    if (data instanceof ImmutableSortedKeyMap) {
      return (ImmutableSortedKeyMap<K, V>) data;
    }
    Set<K> keySet = data.keySet();
    int size = keySet.size();
    K[] sortedKeys = (K[]) new Comparable<?>[size];
    int index = 0;
    for (K key : keySet) {
      sortedKeys[index] = Preconditions.checkNotNull(key);
      index++;
    }
    Arrays.sort(sortedKeys);
    V[] values = (V[]) new Object[size];
    for (int i = 0; i < size; i++) {
      values[i] = data.get(sortedKeys[i]);
    }
    return new ImmutableSortedKeyMap<>(sortedKeys, values);
  }

  public static <K extends Comparable<K>, V> Builder<K, V> builder() {
    return new Builder<>();
  }

  /**
   * A builder class for ImmutableSortedKeyListMultimap<K, V> instances.
   */
  public static final class Builder<K extends Comparable<K>, V> {
    private final Map<K, V> builderMap = new HashMap<>();

    Builder() {
      // Not public so you must call builder() instead.
    }

    public ImmutableSortedKeyMap<K, V> build() {
      return ImmutableSortedKeyMap.copyOf(builderMap);
    }

    public Builder<K, V> put(K key, V value) {
      builderMap.put(Preconditions.checkNotNull(key), Preconditions.checkNotNull(value));
      return this;
    }

    public Builder<K, V> putAll(Map<? extends K, ? extends V> map) {
      map.forEach((key, value) -> put(key, value));
      return this;
    }
  }

  private class ValuesCollection extends AbstractCollection<V> {

    ValuesCollection() {
    }

    @Override
    public int size() {
      return ImmutableSortedKeyMap.this.size();
    }

    @Override
    public boolean isEmpty() {
      return sortedKeys.length == 0;
    }

    @Override
    public boolean contains(Object o) {
      return ImmutableSortedKeyMap.this.containsValue(o);
    }

    @Override
    public Iterator<V> iterator() {
      if (isEmpty()) {
        return Collections.emptyIterator();
      }
      return new Iterator<V>() {
        private int currentIndex = 0;

        @Override
        public boolean hasNext() {
          return currentIndex < values.length;
        }

        @Override
        public V next() {
          if (currentIndex >= values.length) {
            throw new NoSuchElementException();
          }
          return values[currentIndex++];
        }
      };
    }

    @Override
    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean removeAll(Collection<?> c) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean retainAll(Collection<?> c) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
      throw new UnsupportedOperationException();
    }
  }

  private final K[] sortedKeys;
  private final V[] values;

  private ImmutableSortedKeyMap(K[] sortedKeys, V[] values) {
    this.sortedKeys = sortedKeys;
    this.values = values;
  }

  @Override
  public int size() {
    return sortedKeys.length;
  }

  @Override
  public boolean isEmpty() {
    return sortedKeys.length == 0;
  }

  @Override
  public boolean containsKey(@Nullable Object key) {
    if (key == null) {
      return false;
    }
    int index = Arrays.binarySearch(sortedKeys, key);
    return index >= 0;
  }

  @Override
  public boolean containsValue(@Nullable Object value) {
    return value != null && Arrays.stream(values).anyMatch(v -> v.equals(value));
  }

  @Override
  public V put(K key, V value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public V remove(Object key) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void putAll(Map<? extends K, ? extends V> map) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clear() {
    throw new UnsupportedOperationException();
  }

  @Override
  public V get(@Nullable Object key) {
    if (key == null) {
      return null;
    }
    int index = Arrays.binarySearch(sortedKeys, key);
    return index >= 0 ? values[index] : null;
  }

  @Override
  public Set<K> keySet() {
    return ImmutableSet.copyOf(sortedKeys);
  }

  @Override
  public Collection<V> values() {
    return new ValuesCollection();
  }

  @Override
  public Set<Entry<K, V>> entrySet() {
    return entryStream().collect(toImmutableSet());
  }

  @Override
  public String toString() {
    return Streams.zip(Arrays.stream(sortedKeys), Arrays.stream(values), (k, v) -> k + "=" + v)
        .collect(joining(", ", "{", "}"));
  }

  @Override
  public int hashCode() {
    return entryStream().mapToInt(Entry::hashCode).sum();
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (this == object) {
      return true;
    }
    if (object instanceof Map) {
      throw new UnsupportedOperationException();
    }
    return false;
  }

  private Stream<Entry<K, V>> entryStream() {
    return Streams.zip(Arrays.stream(sortedKeys), Arrays.stream(values), SimpleImmutableEntry::new);
  }
}
