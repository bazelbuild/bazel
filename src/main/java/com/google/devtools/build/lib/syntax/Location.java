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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import java.io.Serializable;
import javax.annotation.concurrent.Immutable;

/**
 * A Location denotes a position within a Starlark file.
 *
 * <p>A location is a triple {@code (file, line, column)}, where {@code file} is the apparent name
 * of the file, {@code line} is the optional 1-based line number, and {@code column} is the optional
 * 1-based column number measured in UTF-16 code units. If the column is zero it is not displayed.
 * If the line number is also zero, it too is not displayed; in this case, the location denotes the
 * file as a whole.
 */
@Immutable
public final class Location implements Serializable, Comparable<Location> {

  private final String file;
  private final int line;
  private final int column;

  public Location(String file, int line, int column) {
    this.file = Preconditions.checkNotNull(file);
    this.line = line;
    this.column = column;
  }

  /** Returns the name of the file containing this location. */
  public String file() {
    return file;
  }

  /** Returns the line number of this location. */
  public int line() {
    return line;
  }

  /** Returns the column number of this location. */
  public int column() {
    return column;
  }

  /**
   * Returns a Location for the given file, line and column. If {@code column} is non-zero, {@code
   * line} too must be non-zero.
   */
  public static Location fromFileLineColumn(String file, int line, int column) {
    Preconditions.checkArgument(line != 0 || column == 0, "non-zero column but no line number");
    return new Location(file, line, column);
  }

  /** Returns a Location for the file as a whole. */
  public static Location fromFile(String file) {
    return new Location(file, 0, 0);
  }

  /**
   * Formats the location as {@code "file:line:col"}. If the column is zero, it is omitted. If the
   * line is also zero, it too is omitted.
   */
  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append(file);
    if (line != 0) {
      buf.append(':').append(line);
      if (column != 0) {
        buf.append(':').append(column);
      }
    }
    return buf.toString();
  }

  /** Returns a three-valued lexicographical comparison of two Locations. */
  @Override
  public final int compareTo(Location that) {
    int cmp = this.file().compareTo(that.file());
    if (cmp != 0) {
      return cmp;
    }
    return Long.compare(
        ((long) this.line << 32) | this.column, ((long) that.line << 32) | that.column);
  }

  @Override
  public int hashCode() {
    return 97 * file.hashCode() + 37 * line + column;
  }

  @Override
  public boolean equals(Object that) {
    return this == that
        || (that instanceof Location
            && this.file.equals(((Location) that).file)
            && this.line == ((Location) that).line
            && this.column == ((Location) that).column);
  }

  /** A location for built-in functions. */
  @SerializationConstant public static final Location BUILTIN = fromFile("<builtin>");
}
