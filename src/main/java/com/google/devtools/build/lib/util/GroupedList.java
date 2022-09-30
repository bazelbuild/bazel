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
package com.google.devtools.build.lib.util;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import java.lang.annotation.ElementType;
import java.lang.annotation.Target;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.checkerframework.framework.qual.DefaultQualifierInHierarchy;
import org.checkerframework.framework.qual.LiteralKind;
import org.checkerframework.framework.qual.QualifierForLiterals;
import org.checkerframework.framework.qual.SubtypeOf;

/**
 * Encapsulates a list of groups.
 *
 * <p>Despite the "list" name, it is an error for the same element to appear multiple times in the
 * list. Users are responsible for not trying to add the same element to a GroupedList twice.
 *
 * <p>Groups are implemented as lists to minimize memory use. However, {@link #equals} is defined to
 * treat groups as unordered.
 *
 * <p>The generic type {@code T} <em>must not</em> be a {@link List}.
 */
public final class GroupedList<T> implements Iterable<List<T>> {

  /**
   * Indicates that the annotated element is a compressed {@link GroupedList}, so that it can be
   * safely passed to {@link #create} and friends.
   */
  @SubtypeOf(DefaultObject.class)
  @Target({ElementType.TYPE_USE, ElementType.TYPE_PARAMETER})
  @QualifierForLiterals(LiteralKind.NULL)
  public @interface Compressed {}

  /** Default annotation for type-safety checks of {@link Compressed}. */
  @DefaultQualifierInHierarchy
  @SubtypeOf({})
  @Target({ElementType.TYPE_USE, ElementType.TYPE_PARAMETER})
  private @interface DefaultObject {}

  // Total number of items in the list. At least elements.size(), but might be larger if there are
  // any nested lists.
  private int size = 0;
  // Items in this GroupedList. Each element is either of type T or List<T>.
  private final ArrayList<Object> elements;

  private final CollectionView collectionView = new CollectionView();

  public GroupedList() {
    // We optimize for small lists.
    this.elements = new ArrayList<>(1);
  }

  private GroupedList(int size, Object[] elements) {
    this.size = size;
    this.elements = Lists.newArrayList(elements);
  }

  /**
   * Increases the capacity of the backing list to accommodate the given number of additional
   * groups.
   */
  public void ensureCapacityForAdditionalGroups(int additionalGroups) {
    elements.ensureCapacity(elements.size() + additionalGroups);
  }

  /**
   * Adds a new group with a single element.
   *
   * <p>The caller must ensure that the given element is not already present.
   */
  public void appendSingleton(T t) {
    checkArgument(!(t instanceof List), "Cannot created GroupedList of lists: %s", t);
    elements.add(t);
    size++;
  }

  /**
   * Adds a new group.
   *
   * <p>The caller must ensure that the new group is duplicate-free and does not contain any
   * elements which are already present.
   */
  public void appendGroup(ImmutableList<T> group) {
    addGroup(group, elements);
    size += group.size();
  }

  /**
   * Removes the elements in toRemove from this list. Takes time proportional to the size of the
   * list, so should not be called often.
   */
  public void remove(Set<T> toRemove) {
    if (!toRemove.isEmpty()) {
      size = removeAndGetNewSize(elements, toRemove);
    }
  }

  /**
   * Removes everything in {@code toRemove} from the list of lists, {@code elements}. Returns the
   * new number of elements.
   */
  private static int removeAndGetNewSize(List<Object> elements, Set<?> toRemove) {
    int removedCount = 0;
    int newSize = 0;
    // elements.size is an upper bound of the needed size. Since normally removal happens just
    // before the list is finished and compressed, optimizing this size isn't a concern.
    List<Object> newElements = new ArrayList<>(elements.size());
    for (Object obj : elements) {
      if (obj instanceof List) {
        ImmutableList.Builder<Object> newGroup = new ImmutableList.Builder<>();
        List<?> oldGroup = (List<?>) obj;
        for (Object elt : oldGroup) {
          if (toRemove.contains(elt)) {
            removedCount++;
          } else {
            newGroup.add(elt);
            newSize++;
          }
        }
        addGroup(newGroup.build(), newElements);
      } else if (toRemove.contains(obj)) {
        removedCount++;
      } else {
        newElements.add(obj);
        newSize++;
      }
    }
    // removedCount can be larger if elements had duplicates and the duplicate was also in toRemove.
    checkState(
        removedCount >= toRemove.size(),
        "Requested removal of absent element(s) (toRemove=%s, elements=%s)",
        toRemove,
        elements);
    elements.clear();
    elements.addAll(newElements);
    return newSize;
  }

  /** Returns the group at position {@code i}. {@code i} must be less than {@link #listSize()}. */
  @SuppressWarnings("unchecked") // Cast of Object to List<T> or T.
  public ImmutableList<T> get(int i) {
    Object obj = elements.get(i);
    if (obj instanceof List) {
      return (ImmutableList<T>) obj;
    }
    return ImmutableList.of((T) obj);
  }

  /** Returns the number of groups in this list. */
  public int listSize() {
    return elements.size();
  }

  /**
   * Returns the number of individual elements of type {@code T} in this list, as opposed to the
   * number of groups -- equivalent to adding up the sizes of each group in this list.
   */
  public int numElements() {
    return size;
  }

  public static int numElements(@Compressed Object compressed) {
    if (compressed == EMPTY_LIST) {
      return 0;
    }
    if (compressed.getClass().isArray()) {
      int size = 0;
      for (Object item : (Object[]) compressed) {
        size += sizeOf(item);
      }
      return size;
    }
    // Just a single element.
    return 1;
  }

  /** Returns the number of groups in a compressed {@code GroupedList}. */
  public static int numGroups(@Compressed Object compressed) {
    if (compressed == EMPTY_LIST) {
      return 0;
    }
    if (compressed.getClass().isArray()) {
      return ((Object[]) compressed).length;
    }
    return 1;
  }

  /**
   * Expands a compressed {@code GroupedList} into an {@link Iterable}. Equivalent to {@link
   * #getAllElementsAsIterable()} but potentially more efficient.
   */
  @SuppressWarnings("unchecked")
  public static <T> Iterable<T> compressedToIterable(@Compressed Object compressed) {
    if (compressed == EMPTY_LIST) {
      return ImmutableList.of();
    }
    if (compressed.getClass().isArray()) {
      return GroupedList.<T>create(compressed).getAllElementsAsIterable();
    }
    checkState(!(compressed instanceof List), compressed);
    return ImmutableList.of((T) compressed);
  }

  /**
   * Casts an {@code Object} which is known to be {@link Compressed}.
   *
   * <p>This method should only be used when it is not possible to enforce the type via annotations.
   */
  public static @Compressed Object castAsCompressed(Object obj) {
    checkArgument(!(obj instanceof GroupedList), obj);
    return (@Compressed Object) obj;
  }

  /** Returns true if this list contains no elements. */
  public boolean isEmpty() {
    return elements.isEmpty();
  }

  /**
   * Returns true if this list contains {@code needle}. Takes time proportional to list size. Call
   * {@link #toSet} instead and use the result if doing multiple contains checks.
   */
  public boolean expensiveContains(T needle) {
    return contains(elements, needle);
  }

  private static boolean contains(List<Object> elements, Object needle) {
    for (Object obj : elements) {
      if (obj instanceof List) {
        if (((List<?>) obj).contains(needle)) {
          return true;
        }
      } else if (obj.equals(needle)) {
        return true;
      }
    }
    return false;
  }

  @SerializationConstant @AutoCodec.VisibleForSerialization
  static final @Compressed Object EMPTY_LIST = new Object();

  public @Compressed Object compress() {
    switch (numElements()) {
      case 0:
        return EMPTY_LIST;
      case 1:
        return Iterables.getOnlyElement(elements);
      default:
        return elements.toArray();
    }
  }

  @SuppressWarnings("unchecked")
  public ImmutableSet<T> toSet() {
    ImmutableSet.Builder<T> builder = ImmutableSet.builderWithExpectedSize(size);
    for (Object obj : elements) {
      if (obj instanceof List) {
        builder.addAll((List<T>) obj);
      } else {
        builder.add((T) obj);
      }
    }
    return builder.build();
  }

  private static int sizeOf(Object obj) {
    return obj instanceof List ? ((List<?>) obj).size() : 1;
  }

  public static <E> GroupedList<E> create(@Compressed Object compressed) {
    if (compressed == EMPTY_LIST) {
      return new GroupedList<>();
    }
    if (compressed.getClass().isArray()) {
      int size = 0;
      Object[] compressedArray = ((Object[]) compressed);
      for (Object item : compressedArray) {
        size += sizeOf(item);
      }
      return new GroupedList<>(size, compressedArray);
    }
    // Just a single element.
    return new GroupedList<>(1, new Object[] {compressed});
  }

  @Override
  public int hashCode() {
    // Hashing requires getting an order-independent hash for each element of this.elements. That
    // is too expensive for a hash code.
    throw new UnsupportedOperationException("Should not need to get hash for " + this);
  }

  /**
   * Checks that two lists, neither of which may contain duplicates, have the same elements,
   * regardless of order.
   */
  private static boolean checkUnorderedEqualityWithoutDuplicates(List<?> first, List<?> second) {
    if (first.size() != second.size()) {
      return false;
    }
    // The order-sensitive comparison usually returns true. When it does, the CompactHashSet
    // doesn't need to be constructed.
    return first.equals(second) || CompactHashSet.create(first).containsAll(second);
  }

  /**
   * A grouping-unaware view of a {@code GroupedList} which does not support modifications.
   *
   * <p>This is implemented as a {@code Collection} so that calling {@link Iterables#size} on the
   * return value of {@link #getAllElementsAsIterable} will take constant time.
   */
  private final class CollectionView extends AbstractCollection<T> {

    @Override
    public Iterator<T> iterator() {
      return new UngroupedIterator<>(elements);
    }

    @Override
    public int size() {
      return size;
    }
  }

  /** An iterator that loops through every element in each group. */
  private static final class UngroupedIterator<T> implements Iterator<T> {
    private final List<Object> elements;
    private int outerIndex = 0;
    @Nullable private List<T> currentGroup;
    private int innerIndex = 0;

    private UngroupedIterator(List<Object> elements) {
      this.elements = elements;
    }

    @Override
    public boolean hasNext() {
      return outerIndex < elements.size();
    }

    @SuppressWarnings("unchecked") // Cast of Object to List<T> or T.
    @Override
    public T next() {
      if (currentGroup != null) {
        return nextFromCurrentGroup();
      }
      Object next = elements.get(outerIndex);
      if (next instanceof List) {
        currentGroup = (List<T>) next;
        innerIndex = 0;
        return nextFromCurrentGroup();
      }
      return (T) elements.get(outerIndex++);
    }

    private T nextFromCurrentGroup() {
      T next = currentGroup.get(innerIndex++);
      if (innerIndex == currentGroup.size()) {
        outerIndex++;
        currentGroup = null;
      }
      return next;
    }
  }

  @ThreadHostile
  public Collection<T> getAllElementsAsIterable() {
    return collectionView;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof GroupedList)) {
      return false;
    }
    GroupedList<?> that = (GroupedList<?>) other;
    // We must check the deps, ignoring the ordering of deps in the same group.
    if (this.size != that.size || this.elements.size() != that.elements.size()) {
      return false;
    }
    for (int i = 0; i < this.elements.size(); i++) {
      Object thisElt = this.elements.get(i);
      Object thatElt = that.elements.get(i);
      if (thisElt == thatElt) {
        continue;
      }
      if (thisElt instanceof List) {
        // Recall that each inner item is either a List or a singleton element.
        if (!(thatElt instanceof List)) {
          return false;
        }
        if (!checkUnorderedEqualityWithoutDuplicates((List<?>) thisElt, (List<?>) thatElt)) {
          return false;
        }
      } else if (!thisElt.equals(thatElt)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("elements", elements).add("size", size).toString();
  }

  /**
   * Iterator that returns the next group in elements for each call to {@link #next}. A custom
   * iterator is needed here because, to optimize memory, we store single-element lists as elements
   * internally, and so they must be wrapped before they're returned.
   */
  private class GroupedIterator implements Iterator<List<T>> {
    private final Iterator<Object> iter = elements.iterator();

    @Override
    public boolean hasNext() {
      return iter.hasNext();
    }

    @SuppressWarnings("unchecked") // Cast of Object to List<T> or T.
    @Override
    public List<T> next() {
      Object obj = iter.next();
      if (obj instanceof List) {
        return (List<T>) obj;
      }
      return ImmutableList.of((T) obj);
    }
  }

  @Override
  public Iterator<List<T>> iterator() {
    return new GroupedIterator();
  }

  /**
   * If {@code group} is empty, this function does nothing.
   *
   * <p>If it contains a single element, then that element is added to {@code elements}.
   *
   * <p>If it contains more than one element, then it is added as the next element of {@code
   * elements} (this means {@code elements} may contain both raw objects and {@link
   * ImmutableList}s).
   *
   * <p>Use with caution as there are no checks in place for the integrity of the resulting object
   * (no de-duping or verifying there are no nested lists).
   */
  private static void addGroup(ImmutableList<?> group, List<Object> elements) {
    switch (group.size()) {
      case 0:
        return;
      case 1:
        elements.add(group.get(0));
        return;
      default:
        elements.add(group);
    }
  }
}
