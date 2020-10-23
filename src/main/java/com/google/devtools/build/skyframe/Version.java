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
package com.google.devtools.build.skyframe;

/**
 *  A Version defines a value in a version tree used in persistent data structures.
 *  See http://en.wikipedia.org/wiki/Persistent_data_structure.
 */
public interface Version {
  /**
   * Defines a partial order relation on versions. Returns true if this object is at most
   * {@code other} in that partial order. If x.equals(y), then x.atMost(y).
   *
   * <p>If x.atMost(y) returns false, then there are two possibilities: y < x in the partial order,
   * so y.atMost(x) returns true and !x.equals(y), or x and y are incomparable in this partial
   * order. This may be because x and y are instances of different Version implementations (although
   * it is legal for different Version implementations to be comparable as well).
   * See http://en.wikipedia.org/wiki/Partially_ordered_set.
   */
  boolean atMost(Version other);

  /**
   * Returns whether {@code this < other} in the partial order of versions, similarly to {@link
   * #atMost}.
   *
   * <p>Returns true iff the 2 versions are comparable in the partial order and {@code this} is
   * strictly lower than {@code other}. False result means that either the elements are comparable
   * and {@code this >= other} or the versions are not comparable in the partial order.
   */
  default boolean lowerThan(Version other) {
    return atMost(other) && !equals(other);
  }
}
