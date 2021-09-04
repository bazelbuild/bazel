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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Utilities for collection classes.
 */
public final class CollectionUtils {

  private CollectionUtils() {}

  /**
   * Given a collection of elements and an equivalence relation, returns a new unordered collection
   * of the disjoint subsets of those elements which are equivalent under the specified relation.
   *
   * <p>Note: the Comparator needs only to implement the less-strict contract of EquivalenceRelation
   * (q.v.). (Hopefully this will one day be a superinterface of Comparator.)
   *
   * @param elements the collection of elements to be partitioned. May contain duplicates.
   * @param equivalenceRelation an equivalence relation over the elements.
   * @return a collection of sets of elements that are equivalent under the specified relation.
   */
  private static <T> Collection<Set<T>> partitionWithComparator(
      Collection<T> elements, Comparator<T> equivalenceRelation) {
    //  TODO(bazel-team): (2009) O(n*m) where n=|elements| and m=|eqClasses|; i.e.,
    //  quadratic.  Use Tarjan's algorithm instead.
    List<Set<T>> eqClasses = new ArrayList<>();
    for (T element : elements) {
      boolean found = false;
      for (Set<T> eqClass : eqClasses) {
        if (equivalenceRelation.compare(eqClass.iterator().next(),
                                        element) == 0) {
          eqClass.add(element);
          found = true;
          break;
        }
      }
      if (!found) {
        Set<T> eqClass = new HashSet<>();
        eqClass.add(element);
        eqClasses.add(eqClass);
      }
    }
    return eqClasses;
  }

  /**
   * See partition(Collection, Comparator).
   */
  public static <T> Collection<Set<T>> partition(Collection<T> elements,
      final EquivalenceRelation<T> equivalenceRelation) {
    return partitionWithComparator(elements, equivalenceRelation::compare);
  }

  /** Returns the set of all elements in the given list that appear more than once. */
  public static <T> Set<T> duplicatedElementsOf(List<T> input) {
    int count = input.size();
    if (count < 2) {
      return ImmutableSet.of();
    }
    Set<T> duplicates = null;
    Set<T> elementSet = CompactHashSet.createWithExpectedSize(count);
    for (T el : input) {
      if (!elementSet.add(el)) {
        if (duplicates == null) {
          duplicates = new HashSet<>();
        }
        duplicates.add(el);
      }
    }
    return duplicates == null ? ImmutableSet.of() : duplicates;
  }

  /**
   * Returns an immutable set of all non-null parameters in the order in which they are specified.
   */
  public static <T> ImmutableSet<T> asSetWithoutNulls(T... elements) {
    return Arrays.stream(elements).filter(Objects::nonNull).collect(toImmutableSet());
  }

  /**
   * Returns true if the given iterable can be verified to be immutable.
   *
   * <p>Note that if this method returns false, that does not mean that the iterable is mutable.
   */
  public static boolean isImmutable(Iterable<?> iterable) {
    return iterable instanceof ImmutableCollection || iterable instanceof IterablesChain;
  }

  /** Throws a runtime exception if the given iterable can not be verified to be immutable. */
  public static void checkImmutable(Iterable<?> iterable) {
    Preconditions.checkState(isImmutable(iterable), iterable.getClass());
  }

  /**
   * Given an iterable, returns an immutable iterable with the same contents.
   */
  public static <T> Iterable<T> makeImmutable(Iterable<T> iterable) {
    return isImmutable(iterable) ? iterable : ImmutableList.copyOf(iterable);
  }

  /**
   * Returns a copy of the Map of Maps parameter.
   */
  public static <KEY_1, KEY_2, VALUE> Map<KEY_1, Map<KEY_2, VALUE>> copyOf(
      Map<KEY_1, ? extends Map<KEY_2, VALUE>> map) {
    return new HashMap<>(Maps.transformValues(map, HashMap::new));
  }

  /**
   * Checks whether the given collection is either {@code null} or {@linkplain Collection#isEmpty
   * empty}.
   */
  public static boolean isNullOrEmpty(@Nullable Collection<?> collection) {
    return collection == null || collection.isEmpty();
  }
}
