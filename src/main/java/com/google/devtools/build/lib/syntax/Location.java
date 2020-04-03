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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.Serializable;
import java.util.Objects;

/**
 * A Location denotes a position within a Starlark file.
 *
 * <p>A location is a triple {@code (file, line, column)}, where {@code file} is the apparent name
 * of the file, {@code line} is the optional 1-based line number, and {@code column} is the optional
 * 1-based column number measured in UTF-16 code units. If the column is zero it is not displayed.
 * If the line number is also zero, it too is not displayed; in this case, the location denotes the
 * file as a whole.
 */
public abstract class Location implements Serializable, Comparable<Location> {

  // TODO(adonovan): merge with Lexer.LexerLocation and make this the only implementation of
  // Location, once the parser no longer eagerly creates Locations but instead
  // records token offsets and the LineNumberTable in the tree.
  @AutoCodec
  @Immutable
  static class FileLineColumn extends Location {
    final String file;
    final LineAndColumn linecol;

    FileLineColumn(String file, LineAndColumn linecol) {
      this.file = Preconditions.checkNotNull(file);
      this.linecol = linecol;
    }

    @Override
    public String file() {
      return file;
    }

    @Override
    protected LineAndColumn getLineAndColumn() {
      return linecol;
    }
  }

  /**
   * Returns a Location for the given file, line and column. If {@code column} is non-zero, {@code
   * line} too must be non-zero.
   */
  public static Location fromFileLineColumn(String file, int line, int column) {
    Preconditions.checkArgument(line != 0 || column == 0, "non-zero column but no line number");
    return new FileLineColumn(file, new LineAndColumn(line, column));
  }

  /** Returns a Location for the file as a whole. */
  public static Location fromFile(String file) {
    return new FileLineColumn(file, ZERO);
  }

  private static final LineAndColumn ZERO = new LineAndColumn(0, 0);

  /** Returns the name of the file containing this location. */
  public abstract String file();

  protected abstract LineAndColumn getLineAndColumn();

  /** Returns the line number of this location. */
  public final int line() {
    return getLineAndColumn().line;
  }

  /** Returns the column number of this location. */
  public final int column() {
    return getLineAndColumn().column;
  }

  /**
   * Formats the location as {@code "file:line:col"}. If the column is zero, it is omitted. If the
   * line is also zero, it too is omitted.
   */
  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append(file());
    LineAndColumn linecol = getLineAndColumn();
    if (linecol.line != 0) {
      buf.append(':').append(linecol.line);
      if (linecol.column != 0) {
        buf.append(':').append(linecol.column);
      }
    }
    return buf.toString();
  }

  /** A value class that describes the line and column of a location. */
  // TODO(adonovan): make private when we combine with LexerLocation.
  @AutoCodec
  @Immutable
  public static final class LineAndColumn {

    public final int line;
    public final int column;

    public LineAndColumn(int line, int column) {
      this.line = line;
      this.column = column;
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof LineAndColumn)) {
        return false;
      }
      LineAndColumn lac = (LineAndColumn) o;
      return lac.line == line && lac.column == column;
    }

    @Override
    public int hashCode() {
      return line * 41 + column;
    }

    @Override
    public String toString() {
      return line + ":" + column;
    }
  }

  /** Returns a three-valued lexicographical comparison of two Locations. */
  @Override
  public final int compareTo(Location that) {
    int cmp = this.file().compareTo(that.file());
    if (cmp != 0) {
      return cmp;
    }
    LineAndColumn x = this.getLineAndColumn();
    LineAndColumn y = that.getLineAndColumn();
    return Long.compare(((long) x.line << 32) | x.column, ((long) y.line << 32) | y.column);
  }

  @Override
  public final int hashCode() {
    return Objects.hash(file(), getLineAndColumn());
  }

  @Override
  public final boolean equals(Object that) {
    return this == that
        || that instanceof Location
            && this.file().equals(((Location) that).file())
            && this.getLineAndColumn().equals(((Location) that).getLineAndColumn());
  }

  /** A location for built-in functions. */
  @AutoCodec public static final Location BUILTIN = fromFile("<builtin>");
}
