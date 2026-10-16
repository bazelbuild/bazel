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

import com.google.common.collect.ImmutableList;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A store for branch coverage data.
 *
 * <p>Implemented as an open-addressed hash map to avoid excessive object creation.
 *
 * <p>Potentially many instances of this class will be created and then later merged. To avoid
 * excessive GC and memory pressure we store the data as arrays of primitives for as long as we can,
 * only creating the {@link BranchCoverageItem} object when it's requested, which should only happen
 * after all branch data has been read and merged. This may make the interface to this class
 * atypical and a little cumbersome, but wrapping the data in an object when adding to this map
 * would be very expensive relative to the necessary work being done.
 */
final class BranchCoverage implements Iterable<Entry<BranchCoverageKey, BranchCoverageItem>> {

  // Key data
  // A slot is filled if the corresponding branchId value is non-null
  private int[] lineNumberKeyData;
  private String[] branchIdKeyData;
  private String[] blockIdKeyData;

  // Value data
  private long[] executionCountData;

  // This is simply keyed by line number
  private final BitSet evaluatedData;

  private int capacity;
  private int size;

  private static final int INITIAL_CAPACITY = 32;

  private BranchCoverage(
      int[] lineNumberKeyData,
      String[] branchIdKeydData,
      String[] blockIdKeyData,
      long[] executionCountData,
      BitSet evaluatedData,
      int size) {
    int capacity = lineNumberKeyData.length;
    if (branchIdKeydData.length != capacity
        || blockIdKeyData.length != capacity
        || executionCountData.length != capacity) {
      throw new IllegalArgumentException(
          String.format(
              "Input arrays must have the same length. lineNumberKeyData: %d, branchIdKeydData: %d,"
                  + " blockIdKeyData: %d, executionCountData: %d",
              lineNumberKeyData.length,
              branchIdKeydData.length,
              blockIdKeyData.length,
              executionCountData.length));
    }
    this.lineNumberKeyData = lineNumberKeyData;
    this.branchIdKeyData = branchIdKeydData;
    this.blockIdKeyData = blockIdKeyData;
    this.executionCountData = executionCountData;
    this.evaluatedData = evaluatedData;
    this.capacity = capacity;
    this.size = size;
  }

  public static BranchCoverage copy(BranchCoverage other) {
    return new BranchCoverage(
        Arrays.copyOf(other.lineNumberKeyData, other.lineNumberKeyData.length),
        Arrays.copyOf(other.branchIdKeyData, other.branchIdKeyData.length),
        Arrays.copyOf(other.blockIdKeyData, other.blockIdKeyData.length),
        Arrays.copyOf(other.executionCountData, other.executionCountData.length),
        (BitSet) other.evaluatedData.clone(),
        other.size);
  }

  public static BranchCoverage create() {
    return new BranchCoverage(
        new int[INITIAL_CAPACITY],
        new String[INITIAL_CAPACITY],
        new String[INITIAL_CAPACITY],
        new long[INITIAL_CAPACITY],
        new BitSet(INITIAL_CAPACITY),
        /* size= */ 0);
  }

  /** Adds all data in the given branch coverage to this one. */
  public void add(BranchCoverage other) {
    for (int i = 0; i < other.capacity; i++) {
      if (!other.indexIsFilled(i)) {
        continue;
      }
      addBranch(
          other.lineNumberKeyData[i],
          other.blockIdKeyData[i],
          other.branchIdKeyData[i],
          other.evaluatedData.get(other.lineNumberKeyData[i]),
          other.executionCountData[i]);
    }
  }

  /** Combines the data in the given branch coverage instances into a new one. */
  public static BranchCoverage merge(BranchCoverage other1, BranchCoverage other2) {
    BranchCoverage merged = BranchCoverage.copy(other1);
    merged.add(other2);
    return merged;
  }

  /**
   * Adds a branch to this collection.
   *
   * <p>If the branch already exists, the execution count and evaluated status are combined.
   *
   * @param lineNumber The line number of the branch.
   * @param blockId The block ID of the branch.
   * @param branchId The branch ID of the branch.
   * @param evaluated Whether the branch was evaluated.
   * @param executionCount The number of times the branch was executed.
   */
  public void addBranch(
      int lineNumber, String blockId, String branchId, boolean evaluated, long executionCount) {
    ensureCapacity();
    if (evaluated) {
      evaluatedData.set(lineNumber);
    }
    int index = findIndex(lineNumber, blockId, branchId);
    if (!indexIsFilled(index)) {
      lineNumberKeyData[index] = lineNumber;
      blockIdKeyData[index] = blockId;
      branchIdKeyData[index] = branchId;
      executionCountData[index] = executionCount;
      size++;
    } else {
      executionCountData[index] += executionCount;
    }
  }

  /**
   * Adds a branch to this collection.
   *
   * <p>If the branch already exists, the execution count and evaluated status are combined.
   *
   * @param branch The branch to add.
   */
  public void addBranch(BranchCoverageItem branch) {
    addBranch(
        branch.lineNumber(),
        branch.blockNumber(),
        branch.branchNumber(),
        branch.evaluated(),
        branch.nrOfExecutions());
  }

  public int size() {
    return size;
  }

  public ImmutableList<BranchCoverageKey> getKeys() {
    ImmutableList.Builder<BranchCoverageKey> builder = ImmutableList.builder();
    for (int i = 0; i < lineNumberKeyData.length; i++) {
      if (indexIsFilled(i)) {
        builder.add(
            BranchCoverageKey.create(lineNumberKeyData[i], blockIdKeyData[i], branchIdKeyData[i]));
      }
    }
    return builder.build();
  }

  public BranchCoverageItem get(BranchCoverageKey key) {
    return get(key.lineNumber(), key.blockNumber(), key.branchNumber());
  }

  /**
   * Returns the branch coverage data for the given line number, block ID, and branch ID.
   *
   * <p>Returns null if no data has been added for the given key.
   *
   * @param lineNumber The line number of the branch.
   * @param blockId The block ID of the branch.
   * @param branchId The branch ID of the branch.
   * @return The branch coverage data for the given line number, block ID, and branch ID.
   */
  @Nullable
  public BranchCoverageItem get(int lineNumber, String blockId, String branchId) {
    int index = findIndex(lineNumber, blockId, branchId);
    if (!indexIsFilled(index)) {
      return null;
    }
    return BranchCoverageItem.create(
        lineNumberKeyData[index],
        blockIdKeyData[index],
        branchIdKeyData[index],
        evaluatedData.get(lineNumber),
        executionCountData[index]);
  }

  public boolean containsKey(BranchCoverageKey key) {
    return containsKey(key.lineNumber(), key.blockNumber(), key.branchNumber());
  }

  public boolean containsKey(int lineNumber, String blockId, String branchId) {
    int index = findIndex(lineNumber, blockId, branchId);
    return indexIsFilled(index);
  }

  public int executedBranchesCount() {
    int count = 0;
    for (int i = 0; i < executionCountData.length; i++) {
      if (executionCountData[i] > 0) {
        count++;
      }
    }
    return count;
  }

  /**
   * Find the index into the arrays that correspond to the given key data. If the key is not
   * currently in the map, this will point to the slot that should be used for it. The index is
   * found simply via linear probing.
   *
   * @param lineNumber The line number of the branch.
   * @param blockId The block ID of the branch.
   * @param branchId The branch ID of the branch.
   * @return The index into the arrays that correspond to the given key data.
   */
  private int findIndex(int lineNumber, String blockId, String branchId) {
    if (branchId == null) {
      throw new IllegalArgumentException("Branch ID must not be null");
    }
    int hashValue = hash(lineNumber, blockId, branchId);
    hashValue &= Integer.MAX_VALUE;
    int index = hashValue % capacity;
    int originalIndex = index;
    while (true) {
      if (!indexIsFilled(index)) {
        return index;
      }
      if (doesIndexContain(index, lineNumber, blockId, branchId)) {
        return index;
      }
      index = (index + 1) % capacity;
      if (index == originalIndex) {
        throw new IllegalStateException("Branch data store is overloaded.");
      }
    }
  }

  private boolean doesIndexContain(int index, int lineNumber, String blockId, String branchId) {
    return lineNumber == lineNumberKeyData[index]
        && blockId.equals(blockIdKeyData[index])
        && branchId.equals(branchIdKeyData[index]);
  }

  private void ensureCapacity() {
    // Maximum load should be ~75%
    if (size < 3 * (capacity / 4)) {
      return;
    }

    // 0x40000000 is the smallest int that overflows when doubled.
    if (capacity >= 0x40000000) {
      throw new IllegalStateException("Branch data store cannot be expanded further.");
    }
    int newCapacity = capacity * 2;
    int[] oldLineNumberKeyData = lineNumberKeyData;
    String[] oldBranchIdKeyData = branchIdKeyData;
    String[] oldBlockIdKeyData = blockIdKeyData;
    long[] oldExecutionCountData = executionCountData;

    lineNumberKeyData = new int[newCapacity];
    branchIdKeyData = new String[newCapacity];
    blockIdKeyData = new String[newCapacity];
    executionCountData = new long[newCapacity];
    capacity = newCapacity;
    for (int i = 0; i < oldLineNumberKeyData.length; i++) {
      if (oldBranchIdKeyData[i] != null) {
        int lineNumber = oldLineNumberKeyData[i];
        String blockId = oldBlockIdKeyData[i];
        String branchId = oldBranchIdKeyData[i];
        int index = findIndex(lineNumber, blockId, branchId);
        lineNumberKeyData[index] = lineNumber;
        blockIdKeyData[index] = blockId;
        branchIdKeyData[index] = branchId;
        executionCountData[index] = oldExecutionCountData[i];
      }
    }
  }

  private int hash(int lineNumber, String blockId, String branchId) {
    // blockId is almost always "0" and branchId is typically a small single digit number
    // clustering seems to be a little less bad putting blockId at the end instead of branchId
    return Objects.hash(lineNumber, branchId, blockId);
  }

  private boolean indexIsFilled(int index) {
    return branchIdKeyData[index] != null;
  }

  @Override
  public Iterator<Entry<BranchCoverageKey, BranchCoverageItem>> iterator() {
    return new BranchCoverageIterator();
  }

  private class BranchCoverageIterator
      implements Iterator<Entry<BranchCoverageKey, BranchCoverageItem>> {

    int idx;

    BranchCoverageIterator() {
      idx = -1;
      advanceToNextPopulatedSlot();
    }

    private void advanceToNextPopulatedSlot() {
      do {
        idx++;
      } while (idx < capacity && !indexIsFilled(idx));
    }

    @Override
    public Entry<BranchCoverageKey, BranchCoverageItem> next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Entry<BranchCoverageKey, BranchCoverageItem> result =
          new AbstractMap.SimpleImmutableEntry<>(
              BranchCoverageKey.create(
                  lineNumberKeyData[idx], blockIdKeyData[idx], branchIdKeyData[idx]),
              BranchCoverageItem.create(
                  lineNumberKeyData[idx],
                  blockIdKeyData[idx],
                  branchIdKeyData[idx],
                  evaluatedData.get(lineNumberKeyData[idx]),
                  executionCountData[idx]));
      advanceToNextPopulatedSlot();
      return result;
    }

    @Override
    public boolean hasNext() {
      return idx < capacity;
    }
  }
}
