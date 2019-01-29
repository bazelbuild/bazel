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

package com.google.devtools.build.lib.syntax;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * A table to keep track of line numbers in source files. The client creates a LineNumberTable for
 * their buffer using {@link #create}. The client can then ask for the line and column given a
 * position using ({@link #getLineAndColumn(int)}).
 */
@AutoCodec
@Immutable
public class LineNumberTable implements Serializable {
  private static final Interner<LineNumberTable> LINE_NUMBER_TABLE_INTERNER =
      BlazeInterners.newWeakInterner();

  /** A mapping from line number (line >= 1) to character offset into the file. */
  private final int[] linestart;

  private final PathFragment path;
  private final int bufferLength;

  public LineNumberTable(char[] buffer, PathFragment path) {
    this(computeLinestart(buffer), path, buffer.length);
  }

  LineNumberTable(int[] linestart, PathFragment path, int bufferLength) {
    this.linestart = linestart;
    this.path = path;
    this.bufferLength = bufferLength;
  }

  @AutoCodec.Instantiator
  static LineNumberTable createForSerialization(
      int[] linestart, PathFragment path, int bufferLength) {
    return LINE_NUMBER_TABLE_INTERNER.intern(new LineNumberTable(linestart, path, bufferLength));
  }

  static LineNumberTable create(char[] buffer, PathFragment path) {
    return new LineNumberTable(buffer, path);
  }

  private int getLineAt(int offset) {
    if (offset < 0) {
      throw new IllegalStateException("Illegal position: " + offset);
    }
    int lowBoundary = 1;
    int highBoundary = linestart.length - 1;
    while (true) {
      if ((highBoundary - lowBoundary) <= 1) {
        if (linestart[highBoundary] > offset) {
          return lowBoundary;
        } else {
          return highBoundary;
        }
      }
      int medium = lowBoundary + ((highBoundary - lowBoundary) >> 1);
      if (linestart[medium] > offset) {
        highBoundary = medium;
      } else {
        lowBoundary = medium;
      }
    }
  }

  LineAndColumn getLineAndColumn(int offset) {
    int line = getLineAt(offset);
    int column = offset - linestart[line] + 1;
    return new LineAndColumn(line, column);
  }

  PathFragment getPath(int offset) {
    return path;
  }

  Pair<Integer, Integer> getOffsetsForLine(int line) {
    if (line <= 0 || line >= linestart.length) {
      throw new IllegalArgumentException("Illegal line: " + line);
    }
    return Pair.of(
        linestart[line], line < linestart.length - 1 ? linestart[line + 1] : bufferLength);
  }

  @Override
  public int hashCode() {
    return Objects.hash(Arrays.hashCode(linestart), path, bufferLength);
  }

  @Override
  public boolean equals(Object other) {
    if (other == null || !other.getClass().equals(getClass())) {
      return false;
    }
    LineNumberTable that = (LineNumberTable) other;
    return this.bufferLength == that.bufferLength
        && Arrays.equals(this.linestart, that.linestart)
        && Objects.equals(this.path, that.path);
  }

  private static int[] computeLinestart(char[] buffer) {
    // Compute the size.
    int size = 2;
    for (int i = 0; i < buffer.length; i++) {
      if (buffer[i] == '\n') {
        size++;
      }
    }
    int[] linestart = new int[size];

    int index = 0;
    linestart[index++] = 0; // The 0th line does not exist - so we fill something in
    // to make sure the start pos for the 1st line ends up at
    // linestart[1]. Using 0 is useful for tables that are
    // completely empty.
    linestart[index++] = 0; // The first line ("line 1") starts at offset 0.

    // Scan the buffer and record the offset of each line start. Doing this
    // once upfront is faster than checking each char as it is pulled from
    // the buffer.
    for (int i = 0; i < buffer.length; i++) {
      if (buffer[i] == '\n') {
        linestart[index++] = i + 1;
      }
    }
    return linestart;
  }
}
