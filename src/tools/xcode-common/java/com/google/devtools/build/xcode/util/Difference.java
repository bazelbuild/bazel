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
 * Provides utility methods that make difference operations type safe.
 *
 * {@link Sets#difference(Set, Set)} requires no type bound on the second set, which has led to
 * calls which can never subtract any elements because the set being subtracted cannot contain any
 * elements which may exist in the first set.
 */
public class Difference {
  private Difference() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Returns the elements in set1 which are not in set2. set2 may contain extra elements which will
   * be ignored.
   *
   * @param set1 Set whose elements to return
   * @param set2 Set whose elements are to be subtracted
   */
  public static <T> Set<T> of(Set<T> set1, Set<T> set2) {
    return Sets.difference(set1, set2);
  }
}
