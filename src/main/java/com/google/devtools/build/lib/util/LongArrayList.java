// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import java.util.Arrays;

/**
 * A list of primitive long values.
 *
 * <p>Grows its backing array internally when necessary and such that constant amortized addition of
 * elements is guaranteed.
 *
 * <p>Does not shrink its array except by explicit calls to {@link #trim}.
 */
public class LongArrayList {

  private static final int DEFAULT_CAPACITY = 12;

  private long[] array;
  private int size;

  /**
   * Initialize a new LongArrayList with default capacity.
   */
  public LongArrayList() {
    this.array = new long[DEFAULT_CAPACITY];
  }

  /**
   * Initialize a new LongArrayList with space for elements equal to the given capacity.
   * @throws IndexOutOfBoundsException if the capacity is negative
   */
  public LongArrayList(int capacity) {
    Preconditions.checkArgument(capacity >= 0, "Initial capacity must not be negative.");
    this.array = new long[capacity];
  }

  /**
   * Create a new LongArrayList backed by the given array. No copy is made.
   */
  public LongArrayList(long[] array) {
    Preconditions.checkNotNull(array);
    this.array = array;
    this.size = array.length;
  }

  /**
   * Add a value at a specific position to this list. All elements at larger indices will
   * shift to the right by one.
   * @param position may be any index within the array or equal to the size, to append at the end
   * @throws IndexOutOfBoundsException if the index is outside the interval [0, {@link #size()})
   */
  public void add(int position, long value) {
    Preconditions.checkPositionIndex(position, size);
    copyBackAndGrow(position, 1);
    set(position, value);
  }

  /**
   * Add a value to the end of this list.
   */
  public void add(long value) {
    add(size, value);
  }

  /**
   * Add all elements from another LongArrayList at the end of this one.
   * @see #addAll(LongArrayList, int)
   */
  public boolean addAll(LongArrayList other) {
    return addAll(other.array, 0, other.size, size);
  }

  /**
   * Add all elements from another LongArrayList at a certain position within or at the end of
   * this one.
   * @param other
   * @param position at which position to add these elements, adds at the end if equal to the size
   * @return whether this list changed
   * @throws IndexOutOfBoundsException if the index is outside the interval [0, {@link #size()}]
   */
  public boolean addAll(LongArrayList other, int position) {
    return addAll(other.array, 0, other.size, position);
  }

  /**
   * Add all elements from the given array to the end of this array.
   * @see #addAll(long[], int, int, int)
   */
  public boolean addAll(long[] array) {
    return addAll(array, 0, array.length, size);
  }

  /**
   * Add certain elements from the given array to the end of this array.
   * @see #addAll(long[], int, int, int)
   */
  public boolean addAll(long[] array, int fromIndex, int length) {
    return addAll(array, fromIndex, length, size);
  }

  /**
   * Add certain elements from the given array at a certain position in this list.
   * @param array the array from which to take the elements
   * @param fromIndex the position of the first element to add
   * @param length how many elements to add
   * @param position at which position to add these elements, adds at the end if equal to the size
   * @return whether this list has changed
   * @throws IndexOutOfBoundsException if fromIndex and length violate the boundaries of the given
   *    array or atIndex is not a valid index in this array or equal to the size
   */
  public boolean addAll(long[] array, int fromIndex, int length, int position) {
    Preconditions.checkNotNull(array);
    Preconditions.checkPositionIndex(fromIndex + length, array.length);
    if (length == 0) {
      return false;
    }
    // check other positions later to allow "adding" empty arrays anywhere within this array
    Preconditions.checkElementIndex(fromIndex, array.length);
    Preconditions.checkPositionIndex(position, size);
    copyBackAndGrow(position, length);
    System.arraycopy(array, fromIndex, this.array, position, length);
    return true;
  }

  /**
   * Resize the backing array to fit at least this many elements if necessary.
   */
  public void ensureCapacity(int capacity) {
    if (capacity > array.length) {
      long[] newArray = new long[growCapacity(capacity)];
      System.arraycopy(array, 0, newArray, 0, size);
      array = newArray;
    }
  }

  /**
   * @return the element at the specified index
   * @throws IndexOutOfBoundsException if the index is outside the interval [0, {@link #size()})
   */
  public long get(int index) {
    Preconditions.checkElementIndex(index, size);
    return array[index];
  }

  /**
   * Search for the first index at which the given value is found.
   * @return -1 if the value is not found, the index at which it was found otherwise
   */
  public int indexOf(long value) {
    for (int index = 0; index < size; index++) {
      if (array[index] == value) {
        return index;
      }
    }
    return -1;
  }

  /**
   * Remove the element at the specified index and shift all elements at higher indices down by
   * one.
   * @return the removed element
   * @throws IndexOutOfBoundsException if the index is outside the interval [0, {@link #size()})
   */
  public long remove(int index) {
    Preconditions.checkElementIndex(index, size);
    long previous = array[index];
    System.arraycopy(array, index + 1, array, index, size - index - 1);
    size--;
    return previous;
  }

  /**
   * Remove the first occurrence of a value and shift all elements at higher indices down by one.
   * @return true, if the list changed and thus contained the value, false otherwise
   */
  public boolean remove(long value) {
    int index = indexOf(value);
    if (index == -1) {
      return false;
    }
    remove(index);
    return true;
  }

  /**
   * Overwrites the element at a certain index with the given value and returns the previous
   * element.
   * @throws IndexOutOfBoundsException if the index is outside the interval [0, {@link #size()})
   */
  public long set(int index, long value) {
    Preconditions.checkElementIndex(index, size);
    long previous = array[index];
    array[index] = value;
    return previous;
  }

  /**
   * @return the amount of elements in this list
   */
  public int size() {
    return size;
  }

  /**
   * Sort the list in ascending order.
   */
  public void sort() {
    Arrays.sort(array, 0, size);
  }

  /**
   * Sort the sub list from the first index (inclusive) to the second index (exclusive) in
   * ascending order.
   * @see Arrays#sort(long[], int, int)
   * @throws IndexOutOfBoundsException if fromIndex is outside the interval [0, {@link #size()})
   * or toIndex is outside [0, {@link #size()}]
   */
  public void sort(int fromIndex, int toIndex) {
    Arrays.sort(array, fromIndex, toIndex);
  }

  /**
   * Build a String of the form [0, 1, 2]
   */
  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("[");
    String separator = "";
    for (int index = 0; index < size; index++) {
      sb.append(separator);
      sb.append(array[index]);
      separator = ", ";
    }
    sb.append("]");
    return sb.toString();
  }

  /**
   * Remove any excess capacity to save space.
   */
  public void trim() {
    if (size < array.length) {
      long[] newArray = new long[size];
      System.arraycopy(array, 0, newArray, 0, size);
      array = newArray;
    }
  }

  /**
   * Copy the end of the array from a certain index back to make room for length many items and
   * adds length to the size.
   *
   * @param fromIndex may be equal to the current size, then no element needs to be copied, but the
   *    array may grow
   */
  private void copyBackAndGrow(int fromIndex, int length) {
    int newSize = size + length;
    ensureCapacity(newSize);
    System.arraycopy(array, fromIndex, array, fromIndex + length, size - fromIndex);
    size = newSize;
  }

  /**
   * The new capacity when growing the array to contain at least newSize many elements.
   * Uses a growth factor of 1.5.
   */
  private int growCapacity(int newSize) {
    return newSize + (newSize >> 1);
  }
}

