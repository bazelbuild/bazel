// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.rules.ninja.file.ArrayViewCharSequence;
import com.google.devtools.build.lib.bazel.rules.ninja.file.CharacterArrayIterator;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link NinjaSeparatorPredicate}.
 */
@RunWith(JUnit4.class)
public class NinjaSeparatorPredicateTest {
  @Test
  public void testIsSeparator() {
    doTestIsSeparator("", false);
    doTestIsSeparator("\n ", false);
    doTestIsSeparator("\n", null);
    doTestIsSeparator("\na", true);
    doTestIsSeparator("\n\n", true);
  }

  @Test
  public void testAdjacent() {
    doTestSplitAdjacent("abc", "def", false);
    doTestSplitAdjacent("abc\n", "def", true);
    doTestSplitAdjacent("abc\n", " def", false);
    doTestSplitAdjacent("abc\n", "", false);
    doTestSplitAdjacent("abc\n", " ", false);
    doTestSplitAdjacent("abc", "\n", false);
  }

  private void doTestSplitAdjacent(String left, String right, boolean expected) {
    char[] chars = left.toCharArray();
    ArrayViewCharSequence sequence = new ArrayViewCharSequence(chars, 0, chars.length);
    boolean value = NinjaSeparatorPredicate.INSTANCE
        .splitAdjacent(sequence.iteratorAtEnd(), iterator(right));
    assertThat(value).isEqualTo(expected);
  }

  private void doTestIsSeparator(String s, Boolean expected) {
    Boolean value = NinjaSeparatorPredicate.INSTANCE.isSeparator(iterator(s));
    if (expected == null) {
      assertThat(value).isNull();
    } else {
      assertThat(value).isEqualTo(expected);
    }
  }

  private CharacterArrayIterator iterator(String s) {
    char[] chars = s.toCharArray();
    ArrayViewCharSequence sequence = new ArrayViewCharSequence(chars, 0, chars.length);
    return (CharacterArrayIterator) sequence.iterator();
  }
}
