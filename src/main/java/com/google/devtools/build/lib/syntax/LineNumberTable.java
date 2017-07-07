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

import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.Serializable;
import java.nio.CharBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A table to keep track of line numbers in source files. The client creates a LineNumberTable for
 * their buffer using {@link #create}. The client can then ask for the line and column given a
 * position using ({@link #getLineAndColumn(int)}).
 */
public abstract class LineNumberTable implements Serializable {

  /**
   * Returns the (line, column) pair for the specified offset.
   */
  abstract LineAndColumn getLineAndColumn(int offset);

  /**
   * Returns the (start, end) offset pair for a specified line, not including
   * newline chars.
   */
  abstract Pair<Integer, Integer> getOffsetsForLine(int line);

  /**
   * Returns the path corresponding to the given offset.
   */
  abstract PathFragment getPath(int offset);

  static LineNumberTable create(char[] buffer, PathFragment path) {
    // If #line appears within a BUILD file, we assume it has been preprocessed
    // by gconfig2blaze.  We ignore all actual newlines and compute the logical
    // LNT based only on the presence of #line markers.
    return StringUtilities.containsSubarray(buffer, "\n#line ".toCharArray())
        ? new HashLine(buffer, path)
        : new Regular(buffer, path);
  }

  /**
   * Line number table implementation for regular source files.  Records
   * offsets of newlines.
   */
  @Immutable
  public static class Regular extends LineNumberTable {

    /**
     * A mapping from line number (line >= 1) to character offset into the file.
     */
    private final int[] linestart;
    private final PathFragment path;
    private final int bufferLength;

    public Regular(char[] buffer, PathFragment path) {
      // Compute the size.
      int size = 2;
      for (int i = 0; i < buffer.length; i++) {
        if (buffer[i] == '\n') {
          size++;
        }
      }
      linestart = new int[size];

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
      this.bufferLength = buffer.length;
      this.path = path;
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

    @Override
    LineAndColumn getLineAndColumn(int offset) {
      int line = getLineAt(offset);
      int column = offset - linestart[line] + 1;
      return new LineAndColumn(line, column);
    }

    @Override
    PathFragment getPath(int offset) {
      return path;
    }

    @Override
    Pair<Integer, Integer> getOffsetsForLine(int line) {
      if (line <= 0 || line >= linestart.length) {
        throw new IllegalArgumentException("Illegal line: " + line);
      }
      return Pair.of(linestart[line], line < linestart.length - 1
          ? linestart[line + 1]
          : bufferLength);
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
      Regular that = (Regular) other;
      return this.bufferLength == that.bufferLength
          && Arrays.equals(this.linestart, that.linestart)
          && Objects.equals(this.path, that.path);
    }
  }

  /**
   * Line number table implementation for source files that have been
   * preprocessed. Ignores newlines and uses only #line directives.
   */
  @Immutable
  public static class HashLine extends LineNumberTable {

    /**
     * Represents a "#line" directive
     */
    private static class SingleHashLine implements Serializable {
      private final int offset;
      private final int line;
      private final PathFragment path;

      SingleHashLine(int offset, int line, PathFragment path) {
        this.offset = offset;
        this.line = line;
        this.path = path;
      }
    }

    private static final Ordering<SingleHashLine> hashOrdering =
        new Ordering<SingleHashLine>() {

          @Override
          public int compare(SingleHashLine o1, SingleHashLine o2) {
            return Integer.compare(o1.offset, o2.offset);
          }
        };

    private static final Pattern pattern = Pattern.compile("\n#line ([0-9]+) \"([^\"\\n]+)\"");

    private final List<SingleHashLine> table;
    private final PathFragment defaultPath;
    private final int bufferLength;

    public HashLine(char[] buffer, PathFragment defaultPath) {
      CharSequence bufString = CharBuffer.wrap(buffer);
      Matcher m = pattern.matcher(bufString);
      List<SingleHashLine> unorderedTable = new ArrayList<>();
      Map<String, PathFragment> pathCache = new HashMap<>();
      while (m.find()) {
        String pathString = m.group(2);
        PathFragment pathFragment = pathCache.computeIfAbsent(pathString, defaultPath::getRelative);
        unorderedTable.add(new SingleHashLine(
                m.start(0) + 1,  //offset (+1 to skip \n in pattern)
                Integer.parseInt(m.group(1)),  // line number
                pathFragment));  // filename is an absolute path
      }
      this.table = hashOrdering.immutableSortedCopy(unorderedTable);
      this.bufferLength = buffer.length;
      this.defaultPath = Preconditions.checkNotNull(defaultPath);
    }

    private SingleHashLine getHashLine(int offset) {
      if (offset < 0) {
        throw new IllegalStateException("Illegal position: " + offset);
      }
      int binarySearchIndex =
          Collections.binarySearch(table, new SingleHashLine(offset, -1, null), hashOrdering);
      if (binarySearchIndex >= 0) {
        // An exact match in the binary search. Return it.
        return table.get(binarySearchIndex);
      } else if (binarySearchIndex < -1) {
        // See docs at Collections#binarySearch. Our index is -(insertionPoint + 1). To get to the
        // nearest existing value in the original list, we must subtract 2.
        return table.get(-binarySearchIndex - 2);
      } else {
        return null;
      }
    }

    @Override
    LineAndColumn getLineAndColumn(int offset) {
      SingleHashLine hashLine = getHashLine(offset);
      return hashLine == null ? new LineAndColumn(-1, 1) : new LineAndColumn(hashLine.line, 1);
    }

    @Override
    PathFragment getPath(int offset) {
      SingleHashLine hashLine = getHashLine(offset);
      return hashLine == null ? defaultPath : hashLine.path;
    }

    /**
     * Returns 0, 0 for an unknown line
     */
    @Override
    Pair<Integer, Integer> getOffsetsForLine(int line) {
      for (int ii = 0, len = table.size(); ii < len; ii++) {
        if (table.get(ii).line == line) {
          return Pair.of(table.get(ii).offset, ii < len - 1
              ? table.get(ii + 1).offset
              : bufferLength);
        }
      }
      return Pair.of(0, 0);
    }

    @Override
    public int hashCode() {
      return Objects.hash(table, defaultPath, bufferLength);
    }

    @Override
    public boolean equals(Object other) {
      if (other == null || !other.getClass().equals(getClass())) {
        return false;
      }
      HashLine that = (HashLine) other;
      return this.bufferLength == that.bufferLength
          && this.defaultPath.equals(that.defaultPath)
          && this.table.equals(that.table);
    }
  }
}
