// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.Path;

import java.io.Serializable;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A table to keep track of line numbers in source files. The client creates a LineNumberTable for
 * their buffer using {@link #create}. The client can then ask for the line and column given a
 * position using ({@link #getLineAndColumn(int)}).
 */
abstract class LineNumberTable implements Serializable {

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
  abstract Path getPath(int offset);

  static LineNumberTable create(char[] buffer, Path path) {
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
  private static class Regular extends LineNumberTable  {

    /**
     * A mapping from line number (line >= 1) to character offset into the file.
     */
    private final int[] linestart;
    private final Path path;
    private final int bufferLength;

    private Regular(char[] buffer, Path path) {
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

    private int getLineAt(int pos) {
      if (pos < 0) {
        throw new IllegalArgumentException("Illegal position: " + pos);
      }
      int lowBoundary = 1, highBoundary = linestart.length - 1;
      while (true) {
        if ((highBoundary - lowBoundary) <= 1) {
          if (linestart[highBoundary] > pos) {
            return lowBoundary;
          } else {
            return highBoundary;
          }
        }
        int medium = lowBoundary + ((highBoundary - lowBoundary) >> 1);
        if (linestart[medium] > pos) {
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
    Path getPath(int offset) {
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
  }

  /**
   * Line number table implementation for source files that have been
   * preprocessed. Ignores newlines and uses only #line directives.
   */
  // TODO(bazel-team): Use binary search instead of linear search.
  @Immutable
  private static class HashLine extends LineNumberTable {

    /**
     * Represents a "#line" directive
     */
    private static class SingleHashLine implements Serializable {
      final private int offset;
      final private int line;
      final private Path path;

      SingleHashLine(int offset, int line, Path path) {
        this.offset = offset;
        this.line = line;
        this.path = path;
      }
    }

    private static final Pattern pattern = Pattern.compile("\n#line ([0-9]+) \"([^\"\\n]+)\"");

    private final List<SingleHashLine> table;
    private final Path defaultPath;
    private final int bufferLength;

    private HashLine(char[] buffer, Path defaultPath) {
      // Not especially efficient, but that's fine: we just exec'd Python.
      String bufString = new String(buffer);
      Matcher m = pattern.matcher(bufString);
      ImmutableList.Builder<SingleHashLine> tableBuilder = ImmutableList.builder();
      while (m.find()) {
        tableBuilder.add(new SingleHashLine(
            m.start(0) + 1,  //offset (+1 to skip \n in pattern)
            Integer.valueOf(m.group(1)),  // line number
            defaultPath.getRelative(m.group(2))));  // filename is an absolute path
      }
      this.table = tableBuilder.build();
      this.bufferLength = buffer.length;
      this.defaultPath = defaultPath;
    }

    @Override
    LineAndColumn getLineAndColumn(int offset) {
      int line = -1;
      for (int ii = 0, len = table.size(); ii < len; ii++) {
        SingleHashLine hash = table.get(ii);
        if (hash.offset > offset) {
          break;
        }
        line = hash.line;
      }
      return new LineAndColumn(line, 1);
    }

    @Override
    Path getPath(int offset) {
      Path path = this.defaultPath;
      for (int ii = 0, len = table.size(); ii < len; ii++) {
        SingleHashLine hash = table.get(ii);
        if (hash.offset > offset) {
          break;
        }
        path = hash.path;
      }
      return path;
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
  }
}
