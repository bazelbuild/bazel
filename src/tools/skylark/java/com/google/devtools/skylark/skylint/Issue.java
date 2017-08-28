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

/** An issue found by the linter. */
public class Issue {
  // TODO(skylark-team): Represent issues more efficiently than just by a string
  public final String message;
  public final Location location;

  public Issue(String message, Location location) {
    this.message = message;
    this.location = location;
  }

  @Override
  public String toString() {
    return location + ": " + message;
  }

  public static int compare(Issue i1, Issue i2) {
    Location l1 = i1.location;
    Location l2 = i2.location;
    int pathComparison = l1.getPath().compareTo(l2.getPath());
    if (pathComparison != 0) {
      return pathComparison;
    }
    return l1.getStartOffset() - l2.getStartOffset();
  }
}
