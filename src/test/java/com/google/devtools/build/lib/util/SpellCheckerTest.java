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

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SpellChecker}.
 */
@RunWith(JUnit4.class)
public class SpellCheckerTest {

  private void assertDistance(String s1, String s2, int distance) {
    assertThat(SpellChecker.editDistance(s1, s2, 100)).isEqualTo(distance);
    assertThat(SpellChecker.editDistance(s1, s2, distance)).isEqualTo(distance);

    // Symmetry
    assertThat(SpellChecker.editDistance(s2, s1, 100)).isEqualTo(distance);
    assertThat(SpellChecker.editDistance(s2, s1, distance)).isEqualTo(distance);
  }

  @Test
  public void editDistance_1() throws Exception {
    // Deletion
    assertDistance("abcdef", "abdef", 1);
    assertDistance("abcdef", "abcde", 1);
    assertDistance("abcdef", "bcdef", 1);

    // Replacement
    assertDistance("abcdef", "_bcdef", 1);
    assertDistance("abcdef", "abc_ef", 1);
    assertDistance("abcdef", "abcde_", 1);

    // Insertion
    assertDistance("abcdef", "_abcdef", 1);
    assertDistance("abcdef", "abcd_ef", 1);
    assertDistance("abcdef", "abcdef_", 1);
  }

  @Test
  public void editDistance_general() throws Exception {
    assertDistance("", "", 0);
    assertDistance("abcd", "abcd", 0);
    assertDistance("abcde", "", 5);
    assertDistance("abcde", "12345", 5);
    assertDistance("ab", "ba", 2);
    assertDistance("abba", "acca", 2);
    assertDistance("abaa", "aaca", 2);
    assertDistance("kitten", "sitting", 3);
    assertDistance("kitten kitten", "sitting sitting", 6);
    assertDistance("flaw", "lawn", 2);
  }

  @Test
  public void editDistance_maxDistance() throws Exception {
    assertThat(SpellChecker.editDistance("kitten", "sitting", 0)).isEqualTo(-1);
    assertThat(SpellChecker.editDistance("kitten", "sitting", 1)).isEqualTo(-1);
    assertThat(SpellChecker.editDistance("kitten", "sitting", 2)).isEqualTo(-1);
    assertThat(SpellChecker.editDistance("kitten", "sitting", 3)).isEqualTo(3);
    assertThat(SpellChecker.editDistance("kitten", "sitting", 4)).isEqualTo(3);

    assertThat(SpellChecker.editDistance("abcdefg", "s", 2)).isEqualTo(-1);
  }
}
