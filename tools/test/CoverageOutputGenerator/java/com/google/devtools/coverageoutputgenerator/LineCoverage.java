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

import static java.lang.Math.max;

import java.util.AbstractMap;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.NoSuchElementException;

/** Stores line coverage data. */
final class LineCoverage implements Iterable<Entry<Integer, Long>> {

  private long[] lineExecutions;
  private final BitSet instrumentedLines;

  private LineCoverage(long[] lineExecutions, BitSet instrumentedLines) {
    this.lineExecutions = lineExecutions;
    this.instrumentedLines = instrumentedLines;
  }

  /** Creates a new LineCoverage instance. */
  public static LineCoverage create() {
    return new LineCoverage(new long[32], new BitSet(32));
  }

  /** Creates a copy of the given LineCoverage. */
  public static LineCoverage copy(LineCoverage other) {
    long[] lineExecutions = Arrays.copyOf(other.lineExecutions, other.lineExecutions.length);
    BitSet instrumentedLines = new BitSet(other.instrumentedLines.length());
    instrumentedLines.or(other.instrumentedLines);
    return new LineCoverage(lineExecutions, instrumentedLines);
  }

  /** Adds the line data from the given LineCoverage to this one. */
  public void add(LineCoverage other) {
    ensureLineCapacity(other.lineExecutions.length - 1);
    instrumentedLines.or(other.instrumentedLines);
    for (int i = 0; i < other.lineExecutions.length; i++) {
      lineExecutions[i] += other.lineExecutions[i];
    }
  }

  /**
   * Adds data for a single line. If the line already exists, the execution count is added to the
   * existing count.
   *
   * @param lineNumber the line number to add
   * @param executionCount the number of times the line was executed
   * @throws IllegalArgumentException if the line number is negative
   */
  public void addLine(int lineNumber, long executionCount) {
    if (lineNumber < 0) {
      throw new IllegalArgumentException("Line number must be non-negative.");
    }
    ensureLineCapacity(lineNumber);
    lineExecutions[lineNumber] += executionCount;
    instrumentedLines.set(lineNumber);
  }

  public int numberOfInstrumentedLines() {
    return instrumentedLines.cardinality();
  }

  public int numberOfExecutedLines() {
    int count = 0;
    for (int i = 0; i < lineExecutions.length; i++) {
      if (lineExecutions[i] > 0) {
        count++;
      }
    }
    return count;
  }

  private void ensureLineCapacity(int lineNumber) {
    int n = lineNumber + 1;
    if (n > lineExecutions.length) {
      int growthSize = (lineExecutions.length * 3) / 2 + 1;
      int newSize = max(n, growthSize);
      lineExecutions = Arrays.copyOf(lineExecutions, newSize);
    }
  }

  @Override
  public Iterator<Entry<Integer, Long>> iterator() {
    return new LineCoverageIterator();
  }

  private class LineCoverageIterator implements Iterator<Entry<Integer, Long>> {

    int idx = -1;

    LineCoverageIterator() {
      advanceToNextInstrumentedLine();
    }

    private void advanceToNextInstrumentedLine() {
      do {
        idx++;
      } while (idx < lineExecutions.length && !instrumentedLines.get(idx));
    }

    @Override
    public Entry<Integer, Long> next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Entry<Integer, Long> result =
          new AbstractMap.SimpleImmutableEntry<>(idx, lineExecutions[idx]);
      advanceToNextInstrumentedLine();
      return result;
    }

    @Override
    public boolean hasNext() {
      return idx < lineExecutions.length;
    }
  }
}
