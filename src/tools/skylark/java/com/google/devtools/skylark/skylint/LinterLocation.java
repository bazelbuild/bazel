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

package com.google.devtools.skylark.skylint;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Location.LineAndColumn;

/** Location of a linter warning. */
public class LinterLocation {
  public final int line;
  public final int column;

  public LinterLocation(int line, int column) {
    this.line = line;
    this.column = column;
  }

  public static LinterLocation from(Location location) {
    LineAndColumn lac = location.getStartLineAndColumn();
    // LineAndColumn may be null, e.g. if a BuildFileAST contains no statements:
    if (lac == null) {
      return new LinterLocation(1, 1);
    }
    return new LinterLocation(lac.getLine(), lac.getColumn());
  }

  @Override
  public String toString() {
    return ":" + line + ":" + column;
  }
}
