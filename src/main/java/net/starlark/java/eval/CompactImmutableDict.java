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

package net.starlark.java.eval;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;
import com.google.common.collect.Maps;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.function.BiConsumer;

/**
 * A deeply immutable {@link Dict} with a custom memory-efficient implementation.
 *
 * <p>Construct an instance by calling {@link #copyOf(Map)}. Iteration order of the given map is
 * preserved.
 *
 * <p>Size cutoffs for the various specialized implementations were chosen using the frequency
 * distribution of dict instances from an example large build in b/507408768#comment3. Compared to
 * {@link ImmutableMap}, additional memory savings come from:
 *
 * <ol>
 *   <li>All sizes: no caching of collection views in {@link #keySet}, {@link #values}, and {@link
 *       #entrySet}.
 *   <li>Size 2: dedicated {@link DoubletonImmutableDict}.
 *   <li>Sizes 3+: {@link ArrayImmutableDict} shares backing key and value arrays with {@link
 *       RegularImmutableStarlarkList}.
 *   <li>Sizes 3-8: {@link LinearImmutableDict} uses linear search with no hash table.
 *   <li>Sizes 9+: {@link HashImmutableDict} uses open hashing instead of entry wrappers.
 * </ol>
 *
 * <p>{@link #equals} and {@link #hashCode} are order-independent and compatible with arbitrary
 * {@link Map} instances. All {@link #equals} implementations catch {@link ClassCastException} and
 * {@link NullPointerException} when calling {@link Map#get} on the given map. This is required by
 * the {@link Map#equals} contract to safely handle comparisons with arbitrary or type-restricted
 * maps where our keys might be incompatible.
 */
abstract sealed class CompactImmutableDict<K, V> extends Dict<K, V> {

  @SuppressWarnings("unchecked")
  public static <K, V> CompactImmutableDict<K, V> empty() {
    return (CompactImmutableDict<K, V>) EmptyImmutableDict.INSTANCE;
  }

  /**
   * Creates an immutable, compact version of the given map.
   *
   * <p>Callers are responsible for ensuring that all keys are {@linkplain Starlark#checkHashable
   * hashable} and all values are {@linkplain Starlark#checkValid valid} starlark objects, which
   * implies that they are non-null.
   */
  @SuppressWarnings("unchecked")
  static <K, V> CompactImmutableDict<K, V> copyOf(Map<? extends K, ? extends V> m) {
    if (m instanceof CompactImmutableDict<?, ?> dict) {
      return (CompactImmutableDict<K, V>) dict;
    }
    int size = m.size();
    return switch (size) {
      case 0 -> empty();
      case 1 -> {
        var e = m.entrySet().iterator().next();
        yield new SingletonImmutableDict<>(e.getKey(), e.getValue());
      }
      case 2 -> {
        var it = m.entrySet().iterator();
        var e1 = it.next();
        var e2 = it.next();
        yield new DoubletonImmutableDict<>(e1.getKey(), e1.getValue(), e2.getKey(), e2.getValue());
      }
      default -> {
        K[] ks = (K[]) new Object[size];
        V[] vs = (V[]) new Object[size];
        int i = 0;
        for (var e : m.entrySet()) {
          ks[i] = e.getKey();
          vs[i] = e.getValue();
          i++;
        }
        yield size <= 8 ? new LinearImmutableDict<>(ks, vs) : new HashImmutableDict<>(ks, vs);
      }
    };
  }

  @Override
  public final Mutability mutability() {
    return Mutability.IMMUTABLE;
  }

  @Override
  public final boolean updateIteratorCount(int delta) {
    return false;
  }

  @Override
  public final void putEntry(K key, V value) throws EvalException {
    throw immutable();
  }

  @Override
  public final <K2 extends K, V2 extends V> void putEntries(Map<K2, V2> map) throws EvalException {
    throw immutable();
  }

  @Override
  public final void clearEntries() throws EvalException {
    throw immutable();
  }

  @Override
  public final Object pop(Object key, Object defaultValue, StarlarkThread thread)
      throws EvalException {
    throw immutable();
  }

  @Override
  public final Tuple popitem() throws EvalException {
    if (isEmpty()) {
      throw Starlark.errorf("popitem: empty dictionary");
    }
    throw immutable();
  }

  @Override
  public final V setdefault(K key, V defaultValue) throws EvalException {
    throw immutable();
  }

  private EvalException immutable() throws EvalException {
    Starlark.checkMutable(this);
    throw new IllegalStateException();
  }

  /** Specialized singleton implementation for an empty dict. */
  private static final class EmptyImmutableDict<K, V> extends CompactImmutableDict<K, V> {
    static final EmptyImmutableDict<?, ?> INSTANCE = new EmptyImmutableDict<>();

    @Override
    public StarlarkList<?> values0(StarlarkThread thread) {
      return StarlarkList.newList(thread.mutability());
    }

    @Override
    public StarlarkList<?> items(StarlarkThread thread) {
      return StarlarkList.newList(thread.mutability());
    }

    @Override
    public StarlarkList<?> keys(StarlarkThread thread) {
      return StarlarkList.newList(thread.mutability());
    }

    @Override
    public Iterator<K> iterator() {
      return Collections.emptyIterator();
    }

    @Override
    public int size() {
      return 0;
    }

    @Override
    public boolean containsKey(Object key) {
      return false;
    }

    @Override
    public boolean containsValue(Object value) {
      return false;
    }

    @Override
    public V get(Object key) {
      return null;
    }

    @Override
    public ImmutableSet<K> keySet() {
      return ImmutableSet.of();
    }

    @Override
    public ImmutableList<V> values() {
      return ImmutableList.of();
    }

    @Override
    public ImmutableSet<Entry<K, V>> entrySet() {
      return ImmutableSet.of();
    }

    @Override
    public void forEach(BiConsumer<? super K, ? super V> action) {
      checkNotNull(action);
    }

    @Override
    public int hashCode() {
      return 0;
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      return o instanceof Map<?, ?> m && m.isEmpty();
    }
  }

  /** Specialized implementation for a dict of size 1. */
  private static final class SingletonImmutableDict<K, V> extends CompactImmutableDict<K, V> {
    private final K k;
    private final V v;

    SingletonImmutableDict(K k, V v) {
      this.k = k;
      this.v = v;
    }

    @Override
    public StarlarkList<?> values0(StarlarkThread thread) {
      return StarlarkList.wrap(thread.mutability(), new Object[] {v});
    }

    @Override
    public StarlarkList<?> items(StarlarkThread thread) {
      return StarlarkList.wrap(thread.mutability(), new Object[] {Tuple.pair(k, v)});
    }

    @Override
    public StarlarkList<?> keys(StarlarkThread thread) {
      return StarlarkList.wrap(thread.mutability(), new Object[] {k});
    }

    @Override
    public Iterator<K> iterator() {
      return Iterators.singletonIterator(k);
    }

    @Override
    public int size() {
      return 1;
    }

    @Override
    public boolean containsKey(Object key) {
      return k.equals(key);
    }

    @Override
    public boolean containsValue(Object value) {
      return v.equals(value);
    }

    @Override
    public V get(Object key) {
      return k.equals(key) ? v : null;
    }

    @Override
    public ImmutableSet<K> keySet() {
      return ImmutableSet.of(k);
    }

    @Override
    public ImmutableList<V> values() {
      return ImmutableList.of(v);
    }

    @Override
    public ImmutableSet<Entry<K, V>> entrySet() {
      return ImmutableSet.of(Maps.immutableEntry(k, v));
    }

    @Override
    public void forEach(BiConsumer<? super K, ? super V> action) {
      action.accept(k, v);
    }

    @Override
    public int hashCode() {
      return k.hashCode() ^ v.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof Map<?, ?> m)) {
        return false;
      }
      if (m.size() != 1) {
        return false;
      }
      try {
        return v.equals(m.get(k));
      } catch (ClassCastException | NullPointerException unused) {
        return false;
      }
    }
  }

  /** Specialized implementation for a dict of size 2. */
  private static final class DoubletonImmutableDict<K, V> extends CompactImmutableDict<K, V> {
    private final K k1;
    private final V v1;
    private final K k2;
    private final V v2;

    DoubletonImmutableDict(K k1, V v1, K k2, V v2) {
      this.k1 = k1;
      this.v1 = v1;
      this.k2 = k2;
      this.v2 = v2;
    }

    @Override
    public StarlarkList<?> values0(StarlarkThread thread) {
      return StarlarkList.wrap(thread.mutability(), new Object[] {v1, v2});
    }

    @Override
    public StarlarkList<?> items(StarlarkThread thread) {
      return StarlarkList.wrap(
          thread.mutability(), new Object[] {Tuple.pair(k1, v1), Tuple.pair(k2, v2)});
    }

    @Override
    public StarlarkList<?> keys(StarlarkThread thread) {
      return StarlarkList.wrap(thread.mutability(), new Object[] {k1, k2});
    }

    @Override
    public Iterator<K> iterator() {
      return Iterators.forArray(k1, k2);
    }

    @Override
    public int size() {
      return 2;
    }

    @Override
    public boolean containsKey(Object key) {
      return k1.equals(key) || k2.equals(key);
    }

    @Override
    public boolean containsValue(Object value) {
      return v1.equals(value) || v2.equals(value);
    }

    @Override
    public V get(Object key) {
      if (k1.equals(key)) {
        return v1;
      }
      if (k2.equals(key)) {
        return v2;
      }
      return null;
    }

    @Override
    public ImmutableSet<K> keySet() {
      return ImmutableSet.of(k1, k2);
    }

    @Override
    public ImmutableList<V> values() {
      return ImmutableList.of(v1, v2);
    }

    @Override
    public ImmutableSet<Entry<K, V>> entrySet() {
      return ImmutableSet.of(Maps.immutableEntry(k1, v1), Maps.immutableEntry(k2, v2));
    }

    @Override
    public void forEach(BiConsumer<? super K, ? super V> action) {
      action.accept(k1, v1);
      action.accept(k2, v2);
    }

    @Override
    public int hashCode() {
      return (k1.hashCode() ^ v1.hashCode()) + (k2.hashCode() ^ v2.hashCode());
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof Map<?, ?> m)) {
        return false;
      }
      if (m.size() != 2) {
        return false;
      }
      try {
        return v1.equals(m.get(k1)) && v2.equals(m.get(k2));
      } catch (ClassCastException | NullPointerException unused) {
        return false;
      }
    }
  }

  /** Partial implementation based on parallel key-value arrays. */
  private abstract static sealed class ArrayImmutableDict<K, V> extends CompactImmutableDict<K, V> {
    final K[] ks;
    final V[] vs;

    ArrayImmutableDict(K[] ks, V[] vs) {
      this.ks = ks;
      this.vs = vs;
    }

    @Override
    public final StarlarkList<?> values0(StarlarkThread thread) {
      return StarlarkList.wrap(thread.mutability(), vs.clone());
    }

    @Override
    public final StarlarkList<?> items(StarlarkThread thread) {
      Object[] items = new Object[ks.length];
      for (int i = 0; i < ks.length; i++) {
        items[i] = Tuple.pair(ks[i], vs[i]);
      }
      return StarlarkList.wrap(thread.mutability(), items);
    }

    @Override
    public final StarlarkList<?> keys(StarlarkThread thread) {
      return StarlarkList.wrap(thread.mutability(), ks.clone());
    }

    @Override
    public final Iterator<K> iterator() {
      return Iterators.forArray(ks);
    }

    @Override
    public final int size() {
      return ks.length;
    }

    @Override
    public final boolean containsValue(Object value) {
      if (value == null) {
        return false;
      }
      for (V v : vs) {
        if (v.equals(value)) {
          return true;
        }
      }
      return false;
    }

    @Override
    public final Set<K> keySet() {
      return new AbstractSet<>() {
        @Override
        public Iterator<K> iterator() {
          return Iterators.forArray(ks);
        }

        @Override
        public int size() {
          return ks.length;
        }

        @Override
        public boolean contains(Object o) {
          return containsKey(o);
        }
      };
    }

    @Override
    public final StarlarkList<V> values() {
      return new RegularImmutableStarlarkList<>(vs);
    }

    @Override
    public final Set<Entry<K, V>> entrySet() {
      return new AbstractSet<>() {
        @Override
        public Iterator<Entry<K, V>> iterator() {
          return new Iterator<>() {
            private int i = 0;

            @Override
            public boolean hasNext() {
              return i < ks.length;
            }

            @Override
            public Entry<K, V> next() {
              if (!hasNext()) {
                throw new NoSuchElementException();
              }
              var e = Maps.immutableEntry(ks[i], vs[i]);
              i++;
              return e;
            }
          };
        }

        @Override
        public int size() {
          return ks.length;
        }

        @Override
        public boolean contains(Object o) {
          if (!(o instanceof Map.Entry<?, ?> e) || e.getValue() == null) {
            return false;
          }
          return e.getValue().equals(get(e.getKey()));
        }
      };
    }

    @Override
    public final void forEach(BiConsumer<? super K, ? super V> action) {
      for (int i = 0; i < ks.length; i++) {
        action.accept(ks[i], vs[i]);
      }
    }

    @Override
    public final int hashCode() {
      int h = 0;
      for (int i = 0; i < ks.length; i++) {
        h += (ks[i].hashCode() ^ vs[i].hashCode());
      }
      return h;
    }

    @Override
    public final boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof Map<?, ?> m)) {
        return false;
      }
      if (m.size() != ks.length) {
        return false;
      }
      try {
        for (int i = 0; i < ks.length; i++) {
          if (!vs[i].equals(m.get(ks[i]))) {
            return false;
          }
        }
        return true;
      } catch (ClassCastException | NullPointerException unused) {
        return false;
      }
    }
  }

  /**
   * Implementation for small dicts where linear search is expected to perform just as well as a
   * hash table.
   */
  private static final class LinearImmutableDict<K, V> extends ArrayImmutableDict<K, V> {

    LinearImmutableDict(K[] ks, V[] vs) {
      super(ks, vs);
    }

    @Override
    public boolean containsKey(Object key) {
      if (key == null) {
        return false;
      }
      for (K k : ks) {
        if (key.equals(k)) {
          return true;
        }
      }
      return false;
    }

    @Override
    public V get(Object key) {
      if (key == null) {
        return null;
      }
      for (int i = 0; i < ks.length; i++) {
        if (key.equals(ks[i])) {
          return vs[i];
        }
      }
      return null;
    }
  }

  /** Open hash table implementation. */
  private static final class HashImmutableDict<K, V> extends ArrayImmutableDict<K, V> {
    // Values are the index of the corresponding element in ks and vs, or -1 for empty.
    private final int[] table;

    HashImmutableDict(K[] ks, V[] vs) {
      super(ks, vs);

      int n = ks.length;
      int tableSize = n * 2; // 0.5 load factor.
      int[] table = new int[tableSize];
      Arrays.fill(table, -1);

      for (int i = 0; i < n; i++) {
        int idx = getTableIndex(ks[i], tableSize);
        while (table[idx] != -1) {
          if (++idx == tableSize) {
            idx = 0;
          }
        }
        table[idx] = i;
      }
      this.table = table;
    }

    private static int getTableIndex(Object k, int tableSize) {
      int hash = k.hashCode();
      hash = hash ^ (hash >>> 16);
      return (hash & 0x7fffffff) % tableSize;
    }

    private int getTableIndex(Object k) {
      return getTableIndex(k, table.length);
    }

    @Override
    public boolean containsKey(Object key) {
      return get(key) != null;
    }

    @Override
    public V get(Object key) {
      if (key == null) {
        return null;
      }
      int tableIdx = getTableIndex(key);
      int kvIdx;
      while ((kvIdx = table[tableIdx]) != -1) {
        if (key.equals(ks[kvIdx])) {
          return vs[kvIdx];
        }
        if (++tableIdx == table.length) {
          tableIdx = 0;
        }
      }
      return null;
    }
  }
}
