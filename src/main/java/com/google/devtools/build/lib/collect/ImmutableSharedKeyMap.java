// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import javax.annotation.concurrent.Immutable;

/**
 * Provides a memory-efficient map when the key sets are likely to be shared between multiple
 * instances of this class.
 *
 * <p>This class is appropriate where it is expected that a lot of the key sets will be the same.
 * These key sets are shared and an offset table of indices is computed. Each map instance thus
 * contains only a reference to the shared offset table, and a plain array of instances.
 *
 * <p>The map is sensitive to insertion order. Two maps with different insertion orders are *not*
 * considered equal, and will not share keys.
 *
 * <p>This class explicitly does *not* implement the Map interface, as use of that would lead to a
 * lot of GC churn.
 */
@Immutable
public class ImmutableSharedKeyMap<K, V> extends CompactImmutableMap<K, V> {
  private static final Interner<OffsetTable> offsetTables = BlazeInterners.newWeakInterner();

  private final OffsetTable<K> offsetTable;
  private final Object[] values;

  private static final class OffsetTable<K> {
    private final Object[] keys;
    // Keep a map around to speed up get lookups for larger maps.
    // We make this value lazy to avoid computing for values that end up being thrown away
    // during interning anyway (the majority).
    private volatile ImmutableMap<K, Integer> indexMap;

    private OffsetTable(Object[] keys) {
      this.keys = keys;
    }

    void initIndexMap() {
      if (indexMap == null) {
        synchronized (this) {
          if (indexMap == null) {
            ImmutableMap.Builder<K, Integer> builder = ImmutableMap.builder();
            for (int i = 0; i < keys.length; ++i) {
              @SuppressWarnings("unchecked")
              K key = (K) keys[i];
              builder.put(key, i);
            }
            this.indexMap = builder.build();
          }
        }
      }
    }

    private ImmutableMap<K, Integer> getIndexMap() {
      return indexMap;
    }

    int offsetForKey(K key) {
      return getIndexMap().getOrDefault(key, -1);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof OffsetTable)) {
        return false;
      }
      OffsetTable that = (OffsetTable) o;
      return Arrays.equals(this.keys, that.keys);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(keys);
    }
  }

  protected ImmutableSharedKeyMap(Object[] keys, Object[] values) {
    Preconditions.checkArgument(keys.length == values.length);
    this.values = values;
    this.offsetTable = createOffsetTable(keys);
  }

  protected ImmutableSharedKeyMap(Map<K, V> map) {
    int count = map.size();
    Object[] keys = new Object[count];
    Object[] values = new Object[count];
    int i = 0;
    for (Map.Entry<K, V> entry : map.entrySet()) {
      keys[i] = entry.getKey();
      values[i] = entry.getValue();
      ++i;
    }
    Preconditions.checkArgument(keys.length == values.length);
    this.values = values;
    this.offsetTable = createOffsetTable(keys);
  }

  @SuppressWarnings("unchecked")
  private static <K> OffsetTable<K> createOffsetTable(Object[] keys) {
    OffsetTable<K> offsetTable = new OffsetTable<>(keys);
    OffsetTable<K> internedTable = (OffsetTable<K>) offsetTables.intern(offsetTable);
    internedTable.initIndexMap();
    return internedTable;
  }

  @SuppressWarnings("unchecked")
  @Override
  public V get(K key) {
    int offset = offsetTable.offsetForKey(key);
    return offset != -1 ? (V) values[offset] : null;
  }

  @Override
  public int size() {
    return values.length;
  }

  @SuppressWarnings("unchecked")
  @Override
  public K keyAt(int index) {
    return (K) offsetTable.keys[index];
  }

  @SuppressWarnings("unchecked")
  @Override
  public V valueAt(int index) {
    return (V) values[index];
  }

  @Override
  @SuppressWarnings("ReferenceEquality")
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    ImmutableSharedKeyMap<?, ?> that = (ImmutableSharedKeyMap<?, ?>) o;
    // We can use object identity for the offset table due to
    // it being interned
    return offsetTable == that.offsetTable && Arrays.equals(values, that.values);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(offsetTable, Arrays.hashCode(values));
  }

  public static <K, V> Builder<K, V> builder() {
    return new Builder<>();
  }

  /** Builder for {@link ImmutableSharedKeyMap}. */
  public static class Builder<K, V> {
    private final List<Object> entries = new ArrayList<>();

    private Builder() {}

    public Builder<K, V> put(K key, V value) {
      entries.add(key);
      entries.add(value);
      return this;
    }

    public ImmutableSharedKeyMap<K, V> build() {
      int count = entries.size() / 2;
      Object[] keys = new Object[count];
      Object[] values = new Object[count];
      int entryIndex = 0;
      for (int i = 0; i < count; ++i) {
        keys[i] = entries.get(entryIndex++);
        values[i] = entries.get(entryIndex++);
      }
      return new ImmutableSharedKeyMap<>(keys, values);
    }
  }
}
