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
package com.google.devtools.build.lib.graph;

import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Wraps collection implementation. Automatically switches from one to another implementation
 * depending on the number of storing elements.
 *
 * <p>Effective collection implementation depends on the size of collection.
 * <ul>
 *   <li> For 1 - singleton immutable List.
 *   <li> For [2..6] - ArrayList.
 *   <li> For [7...) - CompactHasSet.
 * </ul>
 *
 * @param <T>
 */
final class ConcurrentCollectionWrapper<T> {

  private static final int ARRAYLIST_THRESHOLD = 6;
  private static final int INITIAL_HASHSET_CAPACITY = 12;
  // The succs and preds set representation changes depending on its size.
  // It is implemented using the following collections:
  // - null for size = 0.
  // - Collections$SingletonList for size = 1.
  // - ArrayList(6) for size = [2..6].
  // - CompactHashSet(12) for size > 6.
  // These numbers were chosen based on profiling.
  // TODO(dbabkin): according to VCS history this profiling info was obtained for
  // ArrayList/HashSet. Then HashSet had been replaced by CompactHashSet. Optimal threshold for
  // ArrayList/CompactHashSet may differ from 6.

  private volatile Collection<T> collection = null;

  /**
   * Returns {@code Collections.unmodifiableCollection} wrapper around collection. Iteration over
   * returned collection at the same time with concurrent modification will cause {@code
   * java.util.ConcurrentModificationException}
   */
  public Collection<T> get() {
    Collection<T> collection = this.collection;
    return collection == null
        ? Collections.emptyList()
        : Collections.unmodifiableCollection(collection);
  }

  synchronized Collection<T> clear() {
    Collection<T> old = collection;
    collection = null;
    return old != null ? old : Collections.emptyList();
  }

  public int size() {
    Collection<T> collection = this.collection;
    return collection == null ? 0 : collection.size();
  }

  /**
   * Adds 'value' to wrapped collection. Replacing this collection instance for CompactHashSet from
   * ArrayList.
   *
   * @return {@code true} if the collection was modified; {@code false} if the collection was not
   *     modified
   */
  public synchronized boolean add(T value) {
    Collection<T> collection = this.collection;

    if (collection == null) {
      // null -> SingletonList
      this.collection = Collections.singletonList(value);
      return true;
    }
    if (collection.contains(value)) {
      // already exists in this collection
      return false;
    }
    int previousSize = collection.size();

    if (previousSize == 1) {
      // SingletonList -> ArrayList
      Collection<T> newList = new ArrayList<>(ARRAYLIST_THRESHOLD);
      newList.addAll(collection);
      newList.add(value);
      this.collection = newList;
    } else if (previousSize < ARRAYLIST_THRESHOLD) {
      // ArrayList
      collection.add(value);
    } else if (previousSize == ARRAYLIST_THRESHOLD) {
      // ArrayList -> CompactHashSet
      Collection<T> newSet = CompactHashSet.createWithExpectedSize(INITIAL_HASHSET_CAPACITY);
      newSet.addAll(collection);
      newSet.add(value);
      this.collection = newSet;
    } else {
      // HashSet
      collection.add(value);
    }
    return true;
  }

  /**
   * Removes 'value' from wrapped collection. Replacing this collection instance for ArrayList from
   * CompactHashSet.
   *
   * @return {@code true} if the collection was modified; {@code false} if the set collection not
   *     modified
   */
  public synchronized boolean remove(T value) {

    Collection<T> collection = this.collection;
    if (collection == null) {
      // null
      return false;
    }

    int previousSize = collection.size();
    if (previousSize == 1) {
      if (collection.contains(value)) {
        // -> null
        this.collection = null;
        return true;
      } else {
        return false;
      }
    }
    // now remove the value
    if (collection.remove(value)) {
      // may need to change representation
      if (previousSize == 2) {
        // -> SingletonList
        List<T> list = Collections.singletonList(collection.iterator().next());
        this.collection = list;
        return true;
      } else if (previousSize == 1 + ARRAYLIST_THRESHOLD) {
        // -> ArrayList
        Collection<T> newArrayList = new ArrayList<>(ARRAYLIST_THRESHOLD);
        newArrayList.addAll(collection);
        this.collection = newArrayList;
        return true;
      }
      return true;
    }
    return false;
  }
}
