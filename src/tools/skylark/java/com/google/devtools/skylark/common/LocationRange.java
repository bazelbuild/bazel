// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.common;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import javax.annotation.Nullable;

/**
 * Location range of a linter warning.
 *
 * <p>It is a closed interval: [start, end], i.e. the end location is the last character in the
 * range.
 */
public class LocationRange {
  public final Location start;
  public final Location end;

  public LocationRange(Location start, Location end) {
    Preconditions.checkArgument(
        Location.compare(start, end) <= 0,
        "end location (%s) should be after start location (%s)",
        end,
        start);
    this.start = start;
    this.end = end;
  }

  public static LocationRange from(com.google.devtools.build.lib.events.Location location) {
    Location start = Location.from(location.getStartLineAndColumn());
    Location end = Location.from(location.getEndLineAndColumn());
    return new LocationRange(start, end);
  }

  public static int compare(LocationRange l1, LocationRange l2) {
    int cmp = Location.compare(l1.start, l2.start);
    if (cmp != 0) {
      return cmp;
    }
    return Location.compare(l1.end, l2.end);
  }

  @Override
  public String toString() {
    return start + "-" + end;
  }

  /** Single location in a file (line + column). */
  public static class Location {
    public final int line;
    public final int column;

    public Location(int line, int column) {
      this.line = line;
      this.column = column;
    }

    public static Location from(@Nullable LineAndColumn lac) {
      // LineAndColumn may be null, e.g. if a BuildFileAST contains no statements:
      if (lac == null) {
        return new Location(1, 1);
      }
      return new Location(lac.getLine(), lac.getColumn());
    }

    public static int compare(Location l1, Location l2) {
      int cmp = l1.line - l2.line;
      if (cmp != 0) {
        return cmp;
      }
      return l1.column - l2.column;
    }

    @Override
    public String toString() {
      return line + ":" + column;
    }
  }
}
