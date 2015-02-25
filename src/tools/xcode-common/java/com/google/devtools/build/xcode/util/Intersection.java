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

package com.google.devtools.build.xcode.util;

import com.google.common.collect.Sets;

import java.util.Set;

/**
 * Provides utility methods that make intersections operations type safe.
 *
 * {@link Sets#intersection(Set, Set)} requires no type bound on the second set, which could lead to
 * calls which always return an empty set.
 */
public class Intersection {
  private Intersection() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Returns the intersection of two sets.
   */
  public static <T> Set<T> of(Set<T> set1, Set<T> set2) {
    return Sets.intersection(set1, set2);
  }
}
