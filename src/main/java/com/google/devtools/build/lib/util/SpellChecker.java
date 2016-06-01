// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

/**
 * Class that provides functions to do spell checking, i.e. detect typos
 * and make suggestions.
 */
public final class SpellChecker {
  /**
   * Computes the edit distance between two strings. The edit distance is
   * the minimum number of insertions, deletions and replacements to
   * transform a string into the other string.
   *
   * maxEditDistance is the maximum distance the function can return. If
   * it would be greater, the function returns -1. It is useful for
   * speeding up the computations.
   */
  public static int editDistance(String s1, String s2, int maxEditDistance) {
    // This is the Levenshtein distance, as described here:
    // http://en.wikipedia.org/wiki/Levenshtein_distance
    //
    // We don't need to keep the full matrix. To update a cell, we only
    // need top-left, top, and left values. Using a single array is
    // sufficient. Top value is still in row[j] from the last iteration.
    // Top-left value is stored in 'previous'. Left value is row[j - 1].

    if (s1.equals(s2)) {
      return 0;
    }
    // Short-circuit based on string length.
    if (Math.abs(s1.length() - s2.length()) > maxEditDistance) {
      return -1;
    }

    int[] row = new int[s2.length() + 1];
    for (int i = 0; i <= s2.length(); i++) {
      row[i] = i;
    }

    for (int i = 1; i <= s1.length(); i++) {
      row[0] = i;
      int bestInTheRow = row[0];
      int previous = i - 1;

      for (int j = 1; j <= s2.length(); j++) {
        int old = row[j];

        row[j] = Math.min(
            previous + (s1.charAt(i - 1) == s2.charAt(j - 1) ? 0 : 1),
            1 + Math.min(row[j - 1], row[j]));
        previous = old;
        bestInTheRow = Math.min(bestInTheRow, row[j]);
      }
      if (bestInTheRow > maxEditDistance) {
        return -1;
      }
    }
    int result = row[s2.length()];
    return result <= maxEditDistance ? result : -1;
  }
}
