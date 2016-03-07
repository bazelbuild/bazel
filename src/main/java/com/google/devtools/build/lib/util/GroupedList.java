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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.CompactHashSet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * Encapsulates a list of groups. Is intended to be used in "batch" mode -- to set the value of a
 * GroupedList, users should first construct a {@link GroupedListHelper}, add elements to it, and
 * then {@link #append} the helper to a new GroupedList instance. The generic type T <i>must not</i>
 * be a {@link List}.
 *
 * <p>Despite the "list" name, it is an error for the same element to appear multiple times in the
 * list. Users are responsible for not trying to add the same element to a GroupedList twice.
 *
 * <p>Groups are implemented as lists to minimize memory use. However, {@link #equals} is defined
 * to treat groups as unordered.
 */
public class GroupedList<T> implements Iterable<Collection<T>> {
  // Total number of items in the list. At least elements.size(), but might be larger if there are
  // any nested lists.
  private int size = 0;
  // Items in this GroupedList. Each element is either of type T or List<T>.
  // Non-final only for #remove.
  private List<Object> elements;

  public GroupedList() {
    // We optimize for small lists.
    this.elements = new ArrayList<>(1);
  }

  // Only for use when uncompressing a GroupedList.
  private GroupedList(int size, List<Object> elements) {
    this.size = size;
    this.elements = new ArrayList<>(elements);
  }

  /** Appends the list constructed in helper to this list. */
  public void append(GroupedListHelper<T> helper) {
    Preconditions.checkState(helper.currentGroup == null, "%s %s", this, helper);
    // Do a check to make sure we don't have lists here. Note that if helper.elements is empty,
    // Iterables.getFirst will return null, and null is not instanceof List.
    Preconditions.checkState(!(Iterables.getFirst(helper.elements, null) instanceof List),
        "Cannot make grouped list of lists: %s", helper);
    elements.addAll(helper.groupedList);
    size += helper.size();
  }

  public void appendGroup(Collection<T> group) {
    // Do a check to make sure we don't have lists here. Note that if group is empty,
    // Iterables.getFirst will return null, and null is not instanceof List.
    Preconditions.checkState(!(Iterables.getFirst(group, null) instanceof List),
        "Cannot make grouped list of lists: %s", group);
    switch (group.size()) {
      case 0:
        return;
      case 1:
        elements.add(Iterables.getOnlyElement(group));
        break;
      default:
        elements.add(group);
        break;
    }
    size += group.size();
  }

  /**
   * Removes the elements in toRemove from this list. Takes time proportional to the size of the
   * list, so should not be called often.
   */
  public void remove(Set<T> toRemove) {
    elements = remove(elements, toRemove);
    size -= toRemove.size();
  }

  /** Returns the number of elements in this list. */
  public int size() {
    return size;
  }

  /** Returns true if this list contains no elements. */
  public boolean isEmpty() {
    return elements.isEmpty();
  }

  private static final Object EMPTY_LIST = new Object();

  public Object compress() {
    switch (size()) {
      case 0:
        return EMPTY_LIST;
      case 1:
        return Iterables.getOnlyElement(elements);
      default:
        return elements.toArray();
    }
  }

  @SuppressWarnings("unchecked")
  public Set<T> toSet() {
    ImmutableSet.Builder<T> builder = ImmutableSet.builder();
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

  public static <E> GroupedList<E> create(Object compressed) {
    if (compressed == EMPTY_LIST) {
      return new GroupedList<>();
    }
    if (compressed.getClass().isArray()) {
      List<Object> elements = new ArrayList<>();
      int size = 0;
      for (Object item : (Object[]) compressed) {
        size += sizeOf(item);
        elements.add(item);
      }
      return new GroupedList<>(size, elements);
    }
    // Just a single element.
    return new GroupedList<>(1, ImmutableList.of(compressed));
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

  @Override
  public boolean equals(Object other) {
    if (other == null) {
      return false;
    }
    if (this.getClass() != other.getClass()) {
      return false;
    }
    GroupedList<?> that = (GroupedList<?>) other;
    // We must check the deps, ignoring the ordering of deps in the same group.
    if (this.elements.size() != that.elements.size()) {
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
    return MoreObjects.toStringHelper(this)
        .add("elements", elements)
        .add("size", size).toString();

  }

  /**
   * Iterator that returns the next group in elements for each call to {@link #next}. A custom
   * iterator is needed here because, to optimize memory, we store single-element lists as elements
   * internally, and so they must be wrapped before they're returned.
   */
  private class GroupedIterator implements Iterator<Collection<T>> {
    private final Iterator<Object> iter = elements.iterator();

    @Override
    public boolean hasNext() {
      return iter.hasNext();
    }

    @SuppressWarnings("unchecked") // Cast of Object to List<T> or T.
    @Override
    public Collection<T> next() {
      Object obj = iter.next();
      if (obj instanceof List) {
        return (List<T>) obj;
      }
      return ImmutableList.of((T) obj);
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public Iterator<Collection<T>> iterator() {
    return new GroupedIterator();
  }

  /**
   * Removes everything in toRemove from the list of lists, elements. Called both by GroupedList and
   * GroupedListHelper.
   */
  private static <E> List<Object> remove(List<Object> elements, Set<E> toRemove) {
    int removedCount = 0;
    // elements.size is an upper bound of the needed size. Since normally removal happens just
    // before the list is finished and compressed, optimizing this size isn't a concern.
    List<Object> newElements = new ArrayList<>(elements.size());
    for (Object obj : elements) {
      if (obj instanceof List) {
        ImmutableList.Builder<E> newGroup = new ImmutableList.Builder<>();
        @SuppressWarnings("unchecked")
        List<E> oldGroup = (List<E>) obj;
        for (E elt : oldGroup) {
          if (toRemove.contains(elt)) {
            removedCount++;
          } else {
            newGroup.add(elt);
          }
        }
        ImmutableList<E> group = newGroup.build();
        addItem(group, newElements);
      } else {
        if (toRemove.contains(obj)) {
          removedCount++;
        } else {
          newElements.add(obj);
        }
      }
    }
    Preconditions.checkState(
        removedCount == toRemove.size(), "%s %s %s", elements, toRemove, newElements);
    return newElements;
  }

  /**
   * If {@param item} is empty, this function does nothing.
   *
   * <p>If it contains a single element, then that element must not be {@code null}, and that
   * element is added to {@param elements}.
   *
   * <p>If it contains more than one element, then an {@link ImmutableList} copy of {@param item}
   * is added as the next element of {@param elements}. (This means {@param elements} may contain
   * both raw objects and {@link ImmutableList}s.)
   */
  private static void addItem(Collection<?> item, List<Object> elements) {
    switch (item.size()) {
      case 0:
        return;
      case 1:
        elements.add(Preconditions.checkNotNull(Iterables.getOnlyElement(item), elements));
        return;
      default:
        elements.add(ImmutableList.copyOf(item));
    }
  }

  /**
   * Builder-like object for GroupedLists. An already-existing grouped list is appended to by
   * constructing a helper, mutating it, and then appending that helper to the grouped list.
   */
  public static class GroupedListHelper<E> implements Iterable<E> {
    // Non-final only for removal.
    private List<Object> groupedList;
    private List<E> currentGroup = null;
    private final CompactHashSet<E> elements;

    public GroupedListHelper() {
      // Optimize for short lists.
      groupedList = new ArrayList<>(1);
      elements = CompactHashSet.create();
    }

    /** Create with a copy of the contents of {@param elements} as the initial group. */
    private GroupedListHelper(Collection<E> elements) {
      // Optimize for short lists.
      groupedList = new ArrayList<>(1);
      addItem(elements, groupedList);
      this.elements = CompactHashSet.create(elements);
    }

    /**
     * Add an element to this list. If in a group, will be added to the current group. Otherwise,
     * goes in a group of its own.
     */
    public void add(E elt) {
      Preconditions.checkState(elements.add(Preconditions.checkNotNull(elt)), "%s %s", elt, this);
      if (currentGroup == null) {
        groupedList.add(elt);
      } else {
        currentGroup.add(elt);
      }
    }

    /**
     * Remove all elements of toRemove from this list. It is a fatal error if any elements of
     * toRemove are not present. Takes time proportional to the size of the list, so should not be
     * called often.
     */
    public void remove(Set<E> toRemove) {
      groupedList = GroupedList.remove(groupedList, toRemove);
      int oldSize = size();
      elements.removeAll(toRemove);
      Preconditions.checkState(oldSize == size() + toRemove.size(), "%s %s", toRemove, this);
    }

    /**
     * Starts a group. All elements added until {@link #endGroup} will be in the same group. Each
     * call of startGroup must be paired with a following {@link #endGroup} call. Any duplicate
     * elements added to this group will be silently deduplicated.
     */
    public void startGroup() {
      Preconditions.checkState(currentGroup == null, this);
      currentGroup = new ArrayList<>();
    }

    /** Ends a group started with {@link #startGroup}. */
    public void endGroup() {
      Preconditions.checkNotNull(currentGroup);
      addItem(currentGroup, groupedList);
      currentGroup = null;
    }

    /** Returns true if elt is present in the list. */
    public boolean contains(E elt) {
      return elements.contains(elt);
    }

    private int size() {
      return elements.size();
    }

    /** Returns true if list is empty. */
    public boolean isEmpty() {
      return elements.isEmpty();
    }

    @Override
    public Iterator<E> iterator() {
      return elements.iterator();
    }

    /** Create a GroupedListHelper from a collection of elements, all put in the same group.*/
    public static <F> GroupedListHelper<F> create(Collection<F> elements) {
      GroupedListHelper<F> helper = new GroupedListHelper<>(elements);
      Preconditions.checkState(helper.elements.size() == elements.size(),
          "%s %s", helper, elements);
      return helper;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("groupedList", groupedList)
          .add("elements", elements)
          .add("currentGroup", currentGroup).toString();
    }
  }
}
