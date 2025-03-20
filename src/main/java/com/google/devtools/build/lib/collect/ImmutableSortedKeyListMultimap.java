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

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;
import com.google.common.primitives.Ints;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.AbstractCollection;
import java.util.AbstractMap;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A immutable multimap implementation for multimaps with comparable keys. It uses a sorted array
 * and binary search to return the correct values. It's only purpose is to save memory - it consumes
 * only about half the memory of the equivalent ImmutableListMultimap. Only a few methods are
 * efficiently implemented: {@link #isEmpty} is O(1), {@link #get} and {@link #containsKey} are
 * O(log(n)), and {@link #asMap} and {@link #values} refer to the parent instance. All other methods
 * can take O(n) or even make a copy of the contents.
 *
 * <p>This implementation supports neither {@code null} keys nor {@code null} values.
 */
public final class ImmutableSortedKeyListMultimap<K extends Comparable<K>, V>
    implements ListMultimap<K, V> {

  @SuppressWarnings({"rawtypes", "unchecked"})
  private static final ImmutableSortedKeyListMultimap EMPTY_MULTIMAP =
      new ImmutableSortedKeyListMultimap(new Comparable<?>[0], new List<?>[0]);

  /** Returns the empty multimap. */
  @SuppressWarnings("unchecked")
  public static <K extends Comparable<K>, V> ImmutableSortedKeyListMultimap<K, V> of() {
    // Safe because the multimap will never hold any elements.
    return EMPTY_MULTIMAP;
  }

  @SuppressWarnings("unchecked")
  public static <K extends Comparable<K>, V> ImmutableSortedKeyListMultimap<K, V> copyOf(
      Multimap<K, V> data) {
    if (data.isEmpty()) {
      return EMPTY_MULTIMAP;
    }
    if (data instanceof ImmutableSortedKeyListMultimap) {
      return (ImmutableSortedKeyListMultimap<K, V>) data;
    }
    Set<K> keySet = data.keySet();
    int size = keySet.size();
    K[] sortedKeys = (K[]) new Comparable<?>[size];
    int index = 0;
    for (K key : keySet) {
      sortedKeys[index++] = Preconditions.checkNotNull(key);
    }
    Arrays.sort(sortedKeys);
    List<V>[] values = (List<V>[]) new List<?>[size];
    for (int i = 0; i < size; i++) {
      values[i] = ImmutableList.copyOf(data.get(sortedKeys[i]));
    }
    return new ImmutableSortedKeyListMultimap<>(sortedKeys, values);
  }

  public static <K extends Comparable<K>, V> Builder<K, V> builder() {
    return new Builder<>();
  }

  /**
   * A builder class for ImmutableSortedKeyListMultimap<K, V> instances.
   */
  public static final class Builder<K extends Comparable<K>, V> {
    private final Multimap<K, V> builderMultimap = ArrayListMultimap.create();

    Builder() {
      // Not public so you must call builder() instead.
    }

    public ImmutableSortedKeyListMultimap<K, V> build() {
      return ImmutableSortedKeyListMultimap.copyOf(builderMultimap);
    }

    @CanIgnoreReturnValue
    public Builder<K, V> put(K key, V value) {
      builderMultimap.put(Preconditions.checkNotNull(key), Preconditions.checkNotNull(value));
      return this;
    }

    @CanIgnoreReturnValue
    public Builder<K, V> putAll(K key, Collection<? extends V> values) {
      Collection<V> valueList = builderMultimap.get(Preconditions.checkNotNull(key));
      for (V value : values) {
        valueList.add(Preconditions.checkNotNull(value));
      }
      return this;
    }

    public Builder<K, V> putAll(K key, V... values) {
      return putAll(Preconditions.checkNotNull(key), Arrays.asList(values));
    }

    @CanIgnoreReturnValue
    public Builder<K, V> putAll(Multimap<? extends K, ? extends V> multimap) {
      multimap.asMap().forEach((key, collectionValue) -> putAll(key, collectionValue));
      return this;
    }
  }

  /**
   * An implementation for the Multimap.asMap method. Note that AbstractMap already provides
   * implementations for all methods except {@link #entrySet}, but we override a few here because we
   * can do it much faster than the existing entrySet-based implementations. Also note that it
   * inherits the type parameters K and V from the parent class.
   */
  private class AsMap extends AbstractMap<K, Collection<V>> {

    AsMap() {
    }

    @Override
    public int size() {
      return sortedKeys.length;
    }

    @Override
    public boolean containsKey(Object key) {
      return ImmutableSortedKeyListMultimap.this.containsKey(key);
    }

    @Nullable
    @Override
    public Collection<V> get(Object key) {
      int index = Arrays.binarySearch(sortedKeys, key);
      // Note the different semantic between Map and Multimap.
      return index >= 0 ? values[index] : null;
    }

    @Nullable
    @Override
    public Collection<V> remove(Object key) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Set<Map.Entry<K, Collection<V>>> entrySet() {
      ImmutableSet.Builder<Map.Entry<K, Collection<V>>> builder = ImmutableSet.builder();
      for (int i = 0; i < sortedKeys.length; i++) {
        builder.add(new SimpleImmutableEntry<>(sortedKeys[i], values[i]));
      }
      return builder.build();
    }
  }

  private class ValuesCollection extends AbstractCollection<V> {

    ValuesCollection() {
    }

    @Override
    public int size() {
      return ImmutableSortedKeyListMultimap.this.size();
    }

    @Override
    public boolean isEmpty() {
      return sortedKeys.length == 0;
    }

    @Override
    public boolean contains(Object o) {
      return ImmutableSortedKeyListMultimap.this.containsValue(o);
    }

    @Override
    public Iterator<V> iterator() {
      if (isEmpty()) {
        return Collections.emptyIterator();
      }
      return new AbstractIterator<V>() {
        private int currentList = 0;
        private int currentIndex = 0;

        @Override
        protected V computeNext() {
          if (currentList >= values.length) {
            return endOfData();
          }
          V result = values[currentList].get(currentIndex);
          // Find the next list/index pair.
          currentIndex++;
          if (currentIndex >= values[currentList].size()) {
            currentIndex = 0;
            currentList++;
          }
          return result;
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
  private final List<V>[] values;

  private ImmutableSortedKeyListMultimap(K[] sortedKeys, List<V>[] values) {
    this.sortedKeys = sortedKeys;
    this.values = values;
  }

  @Override
  public int size() {
    return Ints.saturatedCast(Arrays.stream(values).mapToLong(List::size).sum());
  }

  @Override
  public boolean isEmpty() {
    return sortedKeys.length == 0;
  }

  @Override
  public boolean containsKey(Object key) {
    int index = Arrays.binarySearch(sortedKeys, key);
    return index >= 0;
  }

  @Override
  public boolean containsValue(Object value) {
    return Arrays.stream(values).anyMatch(list -> list.contains(value));
  }

  @Override
  public boolean containsEntry(Object key, Object value) {
    int index = Arrays.binarySearch(sortedKeys, key);
    return index >= 0 && values[index].contains(value);
  }

  @Override
  public boolean put(K key, V value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean remove(Object key, Object value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean putAll(K key, Iterable<? extends V> values) {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean putAll(Multimap<? extends K, ? extends V> multimap) {
    throw new UnsupportedOperationException();
  }

  @Override
  public List<V> replaceValues(K key, Iterable<? extends V> values) {
    throw new UnsupportedOperationException();
  }

  @Override
  public List<V> removeAll(Object key) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clear() {
    throw new UnsupportedOperationException();
  }

  @Override
  public List<V> get(K key) {
    int index = Arrays.binarySearch(sortedKeys, key);
    return index >= 0 ? values[index] : ImmutableList.of();
  }

  @Override
  public Set<K> keySet() {
    return ImmutableSet.copyOf(sortedKeys);
  }

  @Override
  public Multiset<K> keys() {
    return ImmutableMultiset.copyOf(sortedKeys);
  }

  @Override
  public Collection<V> values() {
    return new ValuesCollection();
  }

  @Override
  public Collection<Map.Entry<K, V>> entries() {
    ImmutableList.Builder<Map.Entry<K, V>> builder = ImmutableList.builder();
    for (int i = 0; i < sortedKeys.length; i++) {
      for (V value : values[i]) {
        builder.add(new SimpleImmutableEntry<>(sortedKeys[i], value));
      }
    }
    return builder.build();
  }

  /**
   * {@inheritDoc}
   *
   * <p>Note that only {@code get} and {@code containsKey} are implemented efficiently on the
   * returned map.
   */
  @Override
  public Map<K, Collection<V>> asMap() {
    return new AsMap();
  }

  @Override
  public String toString() {
    return asMap().toString();
  }

  @Override
  public int hashCode() {
    return asMap().hashCode();
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (this == object) {
      return true;
    }
    if (object instanceof Multimap<?, ?> that) {
      return asMap().equals(that.asMap());
    }
    return false;
  }
}
