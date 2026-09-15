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
/*
 * Copyright (C) 2012 The Guava Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.lib.collect.compacthashset;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.lang.reflect.Array;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * CompactHashSet is an implementation of a Set. All optional operations (adding and removing) are
 * supported. The elements can be any objects.
 *
 * <p>{@code contains(x)}, {@code add(x)} and {@code remove(x)}, are all (expected and amortized)
 * constant time operations. Expected in the hashtable sense (depends on the hash function doing a
 * good job of distributing the elements to the buckets to a distribution not far from uniform), and
 * amortized since some operations can trigger a hash table resize.
 *
 * <p>Unlike {@code java.util.HashSet}, iteration is only proportional to the actual {@code size()},
 * which is optimal, and <i>not</i> the size of the internal hashtable, which could be much larger
 * than {@code size()}. Furthermore, this structure only depends on a fixed number of arrays; {@code
 * add(x)} operations <i>do not</i> create objects for the garbage collector to deal with, and for
 * every element added, the garbage collector will have to traverse {@code 1.5} references on
 * average, in the marking phase, not {@code 5.0} as in {@code java.util.HashSet}.
 *
 * <p>If there are no removals, then {@link #iterator iteration} order is the same as insertion
 * order. Any removal invalidates any ordering guarantees.
 *
 * <p>NOTE: This is an older version of Guava's {@code
 * com.google.java.common.collect.CompactHashSet}, but it outperforms the newer version on large
 * builds significantly, as it uses only 50% of cpu time in comparison.
 */
public class CompactHashSet<E> extends AbstractSet<E> {
  // TODO(bazel-team): cache all field accesses in local vars

  // A partial copy of com.google.common.collect.Hashing.
  private static final int C1 = 0xcc9e2d51;
  private static final int C2 = 0x1b873593;

  /*
   * This method was rewritten in Java from an intermediate step of the Murmur hash function in
   * http://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp, which contained the
   * following header:
   *
   * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
   * hereby disclaims copyright to this source code.
   */
  private static int smear(int hashCode) {
    return C2 * Integer.rotateLeft(hashCode * C1, 15);
  }

  private static int smearedHash(@Nullable Object o) {
    return smear((o == null) ? 0 : o.hashCode());
  }

  private static final int MAX_TABLE_SIZE = Ints.MAX_POWER_OF_TWO;

  private static int closedTableSize(int expectedEntries) {
    // Get the recommended table size.
    // Round down to the nearest power of 2.
    expectedEntries = Math.max(expectedEntries, 2);
    int tableSize = Integer.highestOneBit(expectedEntries);
    // Check to make sure that we will not exceed the maximum load factor.
    if (expectedEntries > (int) (LOAD_FACTOR * tableSize)) {
      tableSize <<= 1;
      return (tableSize > 0) ? tableSize : MAX_TABLE_SIZE;
    }
    return tableSize;
  }

  /** Creates an empty {@code CompactHashSet} instance. */
  public static <E> CompactHashSet<E> create() {
    return new CompactHashSet<>(DEFAULT_SIZE);
  }

  /**
   * Creates a <i>mutable</i> {@code CompactHashSet} instance containing the elements
   * of the given collection in unspecified order.
   *
   * @param collection the elements that the set should contain
   * @return a new {@code CompactHashSet} containing those elements (minus duplicates)
   */
  public static <E> CompactHashSet<E> create(Collection<? extends E> collection) {
    CompactHashSet<E> set = createWithExpectedSize(collection.size());
    set.addAll(collection);
    return set;
  }

  /**
   * Creates a <i>mutable</i> {@code CompactHashSet} instance containing the given
   * elements in unspecified order.
   *
   * @param elements the elements that the set should contain
   * @return a new {@code CompactHashSet} containing those elements (minus duplicates)
   */
  @SafeVarargs
  public static <E> CompactHashSet<E> create(E... elements) {
    CompactHashSet<E> set = createWithExpectedSize(elements.length);
    Collections.addAll(set, elements);
    return set;
  }

  /**
   * Creates a {@code CompactHashSet} instance, with a high enough "initial capacity"
   * that it <i>should</i> hold {@code expectedSize} elements without growth.
   *
   * @param expectedSize the number of elements you expect to add to the returned set
   * @return a new, empty {@code CompactHashSet} with enough capacity to hold {@code
   *         expectedSize} elements without resizing
   * @throws IllegalArgumentException if {@code expectedSize} is negative
   */
  public static <E> CompactHashSet<E> createWithExpectedSize(int expectedSize) {
    return new CompactHashSet<>(expectedSize);
  }

  private static final int MAXIMUM_CAPACITY = 1 << 30;

  private static final float LOAD_FACTOR = 1.0f;

  /**
   * Bitmask that selects the low 32 bits.
   */
  private static final long NEXT_MASK  = (1L << 32) - 1;

  /**
   * Bitmask that selects the high 32 bits.
   */
  private static final long HASH_MASK = ~NEXT_MASK;

  // TODO(bazel-team): decide default size
  private static final int DEFAULT_SIZE = 3;

  private static final int UNSET = -1;

  /**
   * The hashtable. Its values are indexes to both the elements and entries arrays.
   *
   * Currently, the UNSET value means "null pointer", and any non negative value x is
   * the actual index.
   *
   * Its size must be a power of two.
   */
  private transient int[] table;

  /**
   * Contains the logical entries, in the range of [0, size()). The high 32 bits of each
   * long is the smeared hash of the element, whereas the low 32 bits is the "next" pointer
   * (pointing to the next entry in the bucket chain). The pointers in [size(), entries.length)
   * are all "null" (UNSET).
   */
  private transient long[] entries;

  /** The elements contained in the set, in the range of [0, size()). */
  private transient Object[] elements;

  /**
   * Keeps track of modifications of this set, to make it possible to throw
   * ConcurrentModificationException in the iterator. Note that we choose not to make this volatile,
   * so we do less of a "best effort" to track such errors, for better performance.
   */
  private transient int modCount;

  /**
   * When we have this many elements, resize the hashtable.
   */
  private transient int threshold;

  /**
   * The number of elements contained in the set.
   */
  private transient int size;

  /**
   * Constructs a new instance of {@code CompactHashSet} with the specified capacity.
   *
   * @param expectedSize the initial capacity of this {@code CompactHashSet}.
   */
  private CompactHashSet(int expectedSize) {
    Preconditions.checkArgument(expectedSize >= 0, "Initial capacity must be non-negative");
    int buckets = closedTableSize(expectedSize);
    this.table = newTable(buckets);
    this.elements = new Object[expectedSize];
    this.entries = newEntries(expectedSize);
    this.threshold = Math.max(1, (int) (buckets * LOAD_FACTOR));
  }

  private static int[] newTable(int size) {
    int[] array = new int[size];
    Arrays.fill(array, UNSET);
    return array;
  }

  private static long[] newEntries(int size) {
    long[] array = new long[size];
    Arrays.fill(array, UNSET);
    return array;
  }

  private static int getHash(long entry) {
    return (int) (entry >>> 32);
  }

  /**
   * Returns the index, or UNSET if the pointer is "null"
   */
  private static int getNext(long entry) {
    return (int) entry;
  }

  /**
   * Returns a new entry value by changing the "next" index of an existing entry
   */
  private static long swapNext(long entry, int newNext) {
    return (HASH_MASK & entry) | (NEXT_MASK & newNext);
  }

  private int hashTableMask() {
    return table.length - 1;
  }

  @Override
  public boolean add(@Nullable E object) {
    long[] entries = this.entries;
    Object[] elements = this.elements;
    int hash = smearedHash(object);
    int tableIndex = hash & hashTableMask();
    int newEntryIndex = this.size; // current size, and pointer to the entry to be appended
    int next = table[tableIndex];
    if (next == UNSET) { // uninitialized bucket
      table[tableIndex] = newEntryIndex;
    } else {
      int last;
      long entry;
      do {
        last = next;
        entry = entries[next];
        if (getHash(entry) == hash && Objects.equals(object, elements[next])) {
          return false;
        }
        next = getNext(entry);
      } while (next != UNSET);
      entries[last] = swapNext(entry, newEntryIndex);
    }
    if (newEntryIndex == Integer.MAX_VALUE) {
      throw new IllegalStateException("Cannot contain more than Integer.MAX_VALUE elements!");
    }
    int newSize = newEntryIndex + 1;
    resizeMeMaybe(newSize);
    insertEntry(newEntryIndex, object, hash);
    this.size = newSize;
    if (newEntryIndex >= threshold) {
      resizeTable(2 * table.length);
    }
    modCount++;
    return true;
  }

  /**
   * Creates a fresh entry with the specified object at the specified position in the entry arrays.
   */
  private void insertEntry(int entryIndex, E object, int hash) {
    this.entries[entryIndex] = ((long) hash << 32) | (NEXT_MASK & UNSET);
    this.elements[entryIndex] = object;
  }

  /**
   * Returns currentSize + 1, after resizing the entries storage if necessary.
   */
  private void resizeMeMaybe(int newSize) {
    int entriesSize = entries.length;
    if (newSize > entriesSize) {
      int newCapacity = entriesSize + Math.max(1, entriesSize >>> 1);
      if (newCapacity < 0) {
        newCapacity = Integer.MAX_VALUE;
      }
      if (newCapacity != entriesSize) {
        resizeEntries(newCapacity);
      }
    }
  }

  /**
   * Resizes the internal entries array to the specified capacity, which may be greater or less than
   * the current capacity.
   */
  private void resizeEntries(int newCapacity) {
    this.elements = Arrays.copyOf(elements, newCapacity);
    long[] entries = this.entries;
    int oldSize = entries.length;
    entries = Arrays.copyOf(entries, newCapacity);
    if (newCapacity > oldSize) {
      Arrays.fill(entries, oldSize, newCapacity, UNSET);
    }
    this.entries = entries;
  }

  private void resizeTable(int newCapacity) { // newCapacity always a power of two
    int[] oldTable = table;
    int oldCapacity = oldTable.length;
    if (oldCapacity >= MAXIMUM_CAPACITY) {
      threshold = Integer.MAX_VALUE;
      return;
    }
    int newThreshold = 1 + (int) (newCapacity * LOAD_FACTOR);
    int[] newTable = newTable(newCapacity);
    long[] entries = this.entries;

    int mask = newTable.length - 1;
    for (int i = 0; i < size; i++) {
      long oldEntry = entries[i];
      int hash = getHash(oldEntry);
      int tableIndex = hash & mask;
      int next = newTable[tableIndex];
      newTable[tableIndex] = i;
      entries[i] = ((long) hash << 32) | (NEXT_MASK & next);
    }

    this.threshold = newThreshold;
    this.table = newTable;
  }

  @Override
  public boolean contains(@Nullable Object object) {
    int hash = smearedHash(object);
    int next = table[hash & hashTableMask()];
    while (next != UNSET) {
      long entry = entries[next];
      if (getHash(entry) == hash && Objects.equals(object, elements[next])) {
        return true;
      }
      next = getNext(entry);
    }
    return false;
  }

  @Override
  public boolean remove(@Nullable Object object) {
    return remove(object, smearedHash(object));
  }

  private boolean remove(Object object, int hash) {
    int tableIndex = hash & hashTableMask();
    int next = table[tableIndex];
    if (next == UNSET) {
      return false;
    }
    int last = UNSET;
    do {
      if (getHash(entries[next]) == hash && Objects.equals(object, elements[next])) {
        if (last == UNSET) {
          // we need to update the root link from table[]
          table[tableIndex] = getNext(entries[next]);
        } else {
          // we need to update the link from the chain
          entries[last] = swapNext(entries[last], getNext(entries[next]));
        }

        moveEntry(next);
        size--;
        modCount++;
        return true;
      }
      last = next;
      next = getNext(entries[next]);
    } while (next != UNSET);
    return false;
  }

  /**
   * Moves the last entry in the entry array into {@code dstIndex}, and nulls out its old position.
   */
  private void moveEntry(int dstIndex) {
    int srcIndex = size() - 1;
    if (dstIndex < srcIndex) {
      // move last entry to deleted spot
      elements[dstIndex] = elements[srcIndex];
      elements[srcIndex] = null;

      // move the last entry to the removed spot, just like we moved the element
      long lastEntry = entries[srcIndex];
      entries[dstIndex] = lastEntry;
      entries[srcIndex] = UNSET;

      // also need to update whoever's "next" pointer was pointing to the last entry place
      // reusing "tableIndex" and "next"; these variables were no longer needed
      int tableIndex = getHash(lastEntry) & hashTableMask();
      int lastNext = table[tableIndex];
      if (lastNext == srcIndex) {
        // we need to update the root pointer
        table[tableIndex] = dstIndex;
      } else {
        // we need to update a pointer in an entry
        int previous;
        long entry;
        do {
          previous = lastNext;
          lastNext = getNext(entry = entries[lastNext]);
        } while (lastNext != srcIndex);
        // here, entries[previous] points to the old entry location; update it
        entries[previous] = swapNext(entry, dstIndex);
      }
    } else {
      elements[dstIndex] = null;
      entries[dstIndex] = UNSET;
    }
  }

  @Override
  public Iterator<E> iterator() {
    return new Iterator<E>() {
      int expectedModCount = modCount;
      boolean nextCalled = false;
      int index = 0;

      @Override
      public boolean hasNext() {
        return index < size;
      }

      @Override
      @SuppressWarnings("unchecked")
      public E next() {
        checkForConcurrentModification();
        if (!hasNext()) {
          throw new NoSuchElementException();
        }
        nextCalled = true;
        return (E) elements[index++];
      }

      @Override
      public void remove() {
        checkForConcurrentModification();
        Preconditions.checkState(nextCalled, "no calls to next() since the last call to remove()");
        expectedModCount++;
        index--;
        CompactHashSet.this.remove(elements[index], getHash(entries[index]));
        nextCalled = false;
      }

      private void checkForConcurrentModification() {
        if (modCount != expectedModCount) {
          throw new ConcurrentModificationException();
        }
      }
    };
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public boolean isEmpty() {
    return size == 0;
  }

  @Override
  public Object[] toArray() {
    return Arrays.copyOf(elements, size);
  }

  @SuppressWarnings("unchecked")
  @Override
  public <T> T[] toArray(T[] a) {
    if (a.length < size) {
      a = (T[]) Array.newInstance(a.getClass().getComponentType(), size);
    }
    System.arraycopy(elements, 0, a, 0, size);
    return a;
  }

  /**
   * Ensures that this {@code CompactHashSet} has the smallest representation in memory,
   * given its current size.
   */
  public void trimToSize() {
    int size = this.size;
    if (size < entries.length) {
      resizeEntries(size);
    }
    // size / loadFactor gives the table size of the appropriate load factor,
    // but that may not be a power of two. We floor it to a power of two by
    // keeping its highest bit. But the smaller table may have a load factor
    // larger than what we want; then we want to go to the next power of 2 if we can
    int minimumTableSize = Math.max(1, Integer.highestOneBit((int) (size / LOAD_FACTOR)));
    if (minimumTableSize < MAXIMUM_CAPACITY) {
      double load = (double) size / minimumTableSize;
      if (load > LOAD_FACTOR) {
        minimumTableSize <<= 1; // increase to next power if possible
      }
    }

    if (minimumTableSize < table.length) {
      resizeTable(minimumTableSize);
    }
  }

  @Override
  public void clear() {
    modCount++;
    Arrays.fill(elements, 0, size, null);
    Arrays.fill(table, UNSET);
    Arrays.fill(entries, UNSET);
    this.size = 0;
  }
}
