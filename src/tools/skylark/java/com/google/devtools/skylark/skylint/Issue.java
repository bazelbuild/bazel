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
import com.google.devtools.skylark.common.LocationRange;

/** An issue found by the linter. */
public class Issue {
  public final String category;
  public final String message;
  public final LocationRange location;

  public Issue(String category, String message, LocationRange location) {
    this.category = category;
    this.message = message;
    this.location = location;
  }

  public static Issue create(String category, String message, Location location) {
    return new Issue(category, message, LocationRange.from(location));
  }

  @Override
  public String toString() {
    return location + ": " + message + " [" + category + "]";
  }

  public String prettyPrint(String path) {
    return path + ":" + toString();
  }

  public static int compareLocation(Issue i1, Issue i2) {
    return LocationRange.compare(i1.location, i2.location);
  }
}
