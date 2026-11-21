// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.coverageoutputgenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * A store for MC/DC coverage data.
 *
 * <p>Implemented as an open-addressed hash map to avoid excessive object creation.
 *
 * <p>Similar to {@link BranchCoverage}, this class stores MC/DC data as arrays of primitives for
 * memory efficiency. {@link McdcCoverage} objects are only created when requested, typically after
 * all data has been read and merged.
 */
final class McdcCoverageData implements Iterable<McdcCoverage> {

  // Key data
  // A slot is filled if the corresponding expression value is non-null
  private int[] lineNumberKeyData;
  private int[] groupSizeKeyData;
  private char[] senseKeyData;
  private int[] indexKeyData;
  private String[] expressionKeyData;

  // Value data
  private long[] takenData;

  private int capacity;
  private int size;

  private static final int INITIAL_CAPACITY = 16;

  private McdcCoverageData(
      int[] lineNumberKeyData,
      int[] groupSizeKeyData,
      char[] senseKeyData,
      int[] indexKeyData,
      String[] expressionKeyData,
      long[] takenData,
      int size) {
    int capacity = lineNumberKeyData.length;
    if (groupSizeKeyData.length != capacity
        || senseKeyData.length != capacity
        || indexKeyData.length != capacity
        || expressionKeyData.length != capacity
        || takenData.length != capacity) {
      throw new IllegalArgumentException(
          String.format(
              "Input arrays must have the same length. lineNumberKeyData: %d, groupSizeKeyData:"
                  + " %d, senseKeyData: %d, indexKeyData: %d, expressionKeyData: %d, takenData:"
                  + " %d",
              lineNumberKeyData.length,
              groupSizeKeyData.length,
              senseKeyData.length,
              indexKeyData.length,
              expressionKeyData.length,
              takenData.length));
    }
    this.lineNumberKeyData = lineNumberKeyData;
    this.groupSizeKeyData = groupSizeKeyData;
    this.senseKeyData = senseKeyData;
    this.indexKeyData = indexKeyData;
    this.expressionKeyData = expressionKeyData;
    this.takenData = takenData;
    this.capacity = capacity;
    this.size = size;
  }

  public static McdcCoverageData copy(McdcCoverageData other) {
    return new McdcCoverageData(
        Arrays.copyOf(other.lineNumberKeyData, other.lineNumberKeyData.length),
        Arrays.copyOf(other.groupSizeKeyData, other.groupSizeKeyData.length),
        Arrays.copyOf(other.senseKeyData, other.senseKeyData.length),
        Arrays.copyOf(other.indexKeyData, other.indexKeyData.length),
        Arrays.copyOf(other.expressionKeyData, other.expressionKeyData.length),
        Arrays.copyOf(other.takenData, other.takenData.length),
        other.size);
  }

  public static McdcCoverageData create() {
    return new McdcCoverageData(
        new int[INITIAL_CAPACITY],
        new int[INITIAL_CAPACITY],
        new char[INITIAL_CAPACITY],
        new int[INITIAL_CAPACITY],
        new String[INITIAL_CAPACITY],
        new long[INITIAL_CAPACITY],
        /* size= */ 0);
  }

  /** Adds all data in the given MC/DC coverage to this one. */
  public void add(McdcCoverageData other) {
    for (int i = 0; i < other.capacity; i++) {
      if (!other.indexIsFilled(i)) {
        continue;
      }
      addMcdc(
          other.lineNumberKeyData[i],
          other.groupSizeKeyData[i],
          other.senseKeyData[i],
          other.takenData[i],
          other.indexKeyData[i],
          other.expressionKeyData[i]);
    }
  }

  /** Combines the data in the given MC/DC coverage instances into a new one. */
  public static McdcCoverageData merge(McdcCoverageData other1, McdcCoverageData other2) {
    McdcCoverageData merged = McdcCoverageData.copy(other1);
    merged.add(other2);
    return merged;
  }

  /**
   * Adds an MC/DC record to this collection.
   *
   * <p>If the record already exists (same key), the taken count is added to the existing count.
   *
   * @param lineNumber The line number where the condition appears
   * @param groupSize The total number of conditions in this MC/DC group
   * @param sense The sense character ('t' or 'f') for this condition
   * @param taken Number of times this condition was sensitized
   * @param index Unique index within the MC/DC group
   * @param expression Textual representation of the Boolean expression
   */
  public void addMcdc(
      int lineNumber, int groupSize, char sense, long taken, int index, String expression) {
    ensureCapacity();
    int slotIndex = findIndex(lineNumber, groupSize, sense, index, expression);
    if (!indexIsFilled(slotIndex)) {
      lineNumberKeyData[slotIndex] = lineNumber;
      groupSizeKeyData[slotIndex] = groupSize;
      senseKeyData[slotIndex] = sense;
      indexKeyData[slotIndex] = index;
      expressionKeyData[slotIndex] = expression;
      takenData[slotIndex] = taken;
      size++;
    } else {
      takenData[slotIndex] += taken;
    }
  }

  /**
   * Adds an MC/DC record to this collection.
   *
   * <p>If the record already exists (same key), the taken count is added to the existing count.
   *
   * @param record The MC/DC record to add.
   */
  public void addMcdc(McdcCoverage record) {
    addMcdc(
        record.lineNumber(),
        record.groupSize(),
        record.sense(),
        record.taken(),
        record.index(),
        record.expression());
  }

  public int size() {
    return size;
  }

  public int nrMcdcHit() {
    int count = 0;
    for (int i = 0; i < takenData.length; i++) {
      if (takenData[i] > 0) {
        count++;
      }
    }
    return count;
  }

  /**
   * Returns all MC/DC records as a list.
   *
   * <p>This creates McdcCoverage objects for all stored records.
   */
  public List<McdcCoverage> getAllRecords() {
    List<McdcCoverage> records = new ArrayList<>(size);
    for (int i = 0; i < capacity; i++) {
      if (indexIsFilled(i)) {
        records.add(
            McdcCoverage.create(
                lineNumberKeyData[i],
                groupSizeKeyData[i],
                senseKeyData[i],
                takenData[i],
                indexKeyData[i],
                expressionKeyData[i]));
      }
    }
    return records;
  }

  /**
   * Find the index into the arrays that correspond to the given key data. If the key is not
   * currently in the map, this will point to the slot that should be used for it. The index is
   * found via linear probing.
   *
   * @param lineNumber The line number where the condition appears
   * @param groupSize The total number of conditions in this MC/DC group
   * @param sense The sense character ('t' or 'f')
   * @param index Unique index within the MC/DC group
   * @param expression Textual representation of the Boolean expression
   * @return The index into the arrays that correspond to the given key data.
   */
  private int findIndex(int lineNumber, int groupSize, char sense, int index, String expression) {
    if (expression == null) {
      throw new IllegalArgumentException("Expression must not be null");
    }
    int hashValue = hash(lineNumber, groupSize, sense, index, expression);
    hashValue &= Integer.MAX_VALUE;
    int slotIndex = hashValue % capacity;
    int originalIndex = slotIndex;
    while (true) {
      if (!indexIsFilled(slotIndex)) {
        return slotIndex;
      }
      if (doesIndexContain(slotIndex, lineNumber, groupSize, sense, index, expression)) {
        return slotIndex;
      }
      slotIndex = (slotIndex + 1) % capacity;
      if (slotIndex == originalIndex) {
        throw new IllegalStateException("MC/DC data store is overloaded.");
      }
    }
  }

  private boolean doesIndexContain(
      int slotIndex, int lineNumber, int groupSize, char sense, int index, String expression) {
    return lineNumber == lineNumberKeyData[slotIndex]
        && groupSize == groupSizeKeyData[slotIndex]
        && sense == senseKeyData[slotIndex]
        && index == indexKeyData[slotIndex]
        && expression.equals(expressionKeyData[slotIndex]);
  }

  private void ensureCapacity() {
    // Maximum load should be ~75%
    if (size < 3 * (capacity / 4)) {
      return;
    }

    // 0x40000000 is the smallest int that overflows when doubled.
    if (capacity >= 0x40000000) {
      throw new IllegalStateException("MC/DC data store cannot be expanded further.");
    }
    int newCapacity = capacity * 2;
    int[] oldLineNumberKeyData = lineNumberKeyData;
    int[] oldGroupSizeKeyData = groupSizeKeyData;
    char[] oldSenseKeyData = senseKeyData;
    int[] oldIndexKeyData = indexKeyData;
    String[] oldExpressionKeyData = expressionKeyData;
    long[] oldTakenData = takenData;

    lineNumberKeyData = new int[newCapacity];
    groupSizeKeyData = new int[newCapacity];
    senseKeyData = new char[newCapacity];
    indexKeyData = new int[newCapacity];
    expressionKeyData = new String[newCapacity];
    takenData = new long[newCapacity];
    capacity = newCapacity;

    for (int i = 0; i < oldLineNumberKeyData.length; i++) {
      if (oldExpressionKeyData[i] != null) {
        int lineNumber = oldLineNumberKeyData[i];
        int groupSize = oldGroupSizeKeyData[i];
        char sense = oldSenseKeyData[i];
        int index = oldIndexKeyData[i];
        String expression = oldExpressionKeyData[i];
        int slotIndex = findIndex(lineNumber, groupSize, sense, index, expression);
        lineNumberKeyData[slotIndex] = lineNumber;
        groupSizeKeyData[slotIndex] = groupSize;
        senseKeyData[slotIndex] = sense;
        indexKeyData[slotIndex] = index;
        expressionKeyData[slotIndex] = expression;
        takenData[slotIndex] = oldTakenData[i];
      }
    }
  }

  private int hash(int lineNumber, int groupSize, char sense, int index, String expression) {
    return Objects.hash(lineNumber, groupSize, sense, index, expression);
  }

  private boolean indexIsFilled(int index) {
    return expressionKeyData[index] != null;
  }

  @Override
  public Iterator<McdcCoverage> iterator() {
    return new McdcCoverageIterator();
  }

  private class McdcCoverageIterator implements Iterator<McdcCoverage> {

    int idx;

    McdcCoverageIterator() {
      idx = -1;
      advanceToNextPopulatedSlot();
    }

    private void advanceToNextPopulatedSlot() {
      do {
        idx++;
      } while (idx < capacity && !indexIsFilled(idx));
    }

    @Override
    public McdcCoverage next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      McdcCoverage result =
          McdcCoverage.create(
              lineNumberKeyData[idx],
              groupSizeKeyData[idx],
              senseKeyData[idx],
              takenData[idx],
              indexKeyData[idx],
              expressionKeyData[idx]);
      advanceToNextPopulatedSlot();
      return result;
    }

    @Override
    public boolean hasNext() {
      return idx < capacity;
    }
  }
}
