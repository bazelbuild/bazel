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
import static com.google.common.truth.Truth8.assertThat;

import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ListIterator;
import java.util.Optional;
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

  private static void doTestSplitAdjacent(String left, String right, boolean expected) {
    boolean value = NinjaSeparatorPredicate.INSTANCE
        .splitAdjacent(fragment(left).iteratorAtEnd(), fragment(right).iterator());
    assertThat(value).isEqualTo(expected);
  }

  private static void doTestIsSeparator(String s, Boolean expected) {
    ListIterator<Byte> iterator = fragment(s).iterator();
    if (!iterator.hasNext()) {
      return;
    }
    Optional<Boolean> value = NinjaSeparatorPredicate.INSTANCE.isSeparator(iterator.next(),
        iterator);
    if (expected == null) {
      assertThat(value).isEmpty();
    } else {
      assertThat(value).isPresent();
      assertThat(value.get()).isEqualTo(expected);
    }
  }

  private static ByteBufferFragment fragment(String s) {
    byte[] bytes = s.getBytes(StandardCharsets.ISO_8859_1);
    return new ByteBufferFragment(ByteBuffer.wrap(bytes), 0, bytes.length);
  }
}
