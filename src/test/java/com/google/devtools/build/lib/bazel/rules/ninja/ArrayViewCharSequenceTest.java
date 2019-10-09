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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ArrayViewCharSequence;
import java.util.ListIterator;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ArrayViewCharSequence} and
 * {@link com.google.devtools.build.lib.bazel.rules.ninja.file.CharacterArrayIterator}
 */
@RunWith(JUnit4.class)
public class ArrayViewCharSequenceTest {
  @Test
  public void testCharSequenceMethods() {
    final char[] chars = "0123456789".toCharArray();
    ArrayViewCharSequence sequence = new ArrayViewCharSequence(chars, 1, 9);
    assertThat(sequence.length()).isEqualTo(8);
    assertThat(sequence.toString()).isEqualTo("12345678");
    assertThat(sequence.subSequence(2, 4).toString()).isEqualTo("34");
    assertThat(sequence.subSequence(0, 8).toString()).isEqualTo("12345678");

    for (int i = 0; i < 8; i++) {
      char ch = sequence.charAt(i);
      assertThat(String.valueOf(ch)).isEqualTo(String.valueOf(i + 1));
    }
  }

  @Test
  public void testCharSequenceAssertions() {
    final char[] chars = "0123456789".toCharArray();
    ArrayViewCharSequence sequence = new ArrayViewCharSequence(chars, 1, 9);
    assertThat(sequence.length()).isEqualTo(8);

    assertThrows(IndexOutOfBoundsException.class, () -> sequence.charAt(-1));
    assertThrows(IndexOutOfBoundsException.class, () -> sequence.charAt(8));

    assertThrows(IndexOutOfBoundsException.class, () -> sequence.subSequence(-1, 4));
    assertThrows(IndexOutOfBoundsException.class, () -> sequence.subSequence(1, 9));
  }

  @Test
  public void testStartsWith() {
    final char[] chars = "0123456789".toCharArray();
    ArrayViewCharSequence sequence = new ArrayViewCharSequence(chars, 1, 9);
    assertThat(sequence.startsWith("1")).isTrue();
    assertThat(sequence.startsWith("12")).isTrue();
    assertThat(sequence.startsWith("123")).isTrue();
    assertThat(sequence.startsWith("12345678")).isTrue();

    assertThat(sequence.startsWith("123456789")).isFalse();
    assertThat(sequence.startsWith("01")).isFalse();
    assertThat(sequence.startsWith("a")).isFalse();

    assertThat(sequence.startsWith("")).isTrue();
  }

  @Test
  public void testIterator() {
    final char[] chars = "0123456789".toCharArray();
    ArrayViewCharSequence sequence = new ArrayViewCharSequence(chars, 1, 9);
    ListIterator<Character> iterator = sequence.iterator();

    assertThrows(UnsupportedOperationException.class, () -> iterator.add('c'));
    assertThrows(UnsupportedOperationException.class, () -> iterator.set('c'));
    assertThrows(UnsupportedOperationException.class, iterator::remove);

    goForward(iterator);
    goBackward(iterator);

    ListIterator<Character> iteratorAtEnd = sequence.iteratorAtEnd();
    goBackward(iteratorAtEnd);
    goForward(iteratorAtEnd);
  }

  private void goForward(ListIterator<Character> iterator) {
    assertThat(iterator.hasNext()).isTrue();
    assertThat(iterator.hasPrevious()).isFalse();
    assertThat(iterator.nextIndex()).isEqualTo(0);
    assertThat(iterator.previousIndex()).isEqualTo(-1);
    for (int i = 0; i < 8; i++) {
      assertThat(iterator.hasNext()).isTrue();
      char ch = iterator.next();
      assertThat(String.valueOf(ch)).isEqualTo(String.valueOf(i + 1));
    }
  }

  private void goBackward(ListIterator<Character> iterator) {
    assertThat(iterator.hasNext()).isFalse();
    assertThat(iterator.hasPrevious()).isTrue();
    assertThat(iterator.nextIndex()).isEqualTo(8);
    assertThat(iterator.previousIndex()).isEqualTo(7);
    for (int i = 7; i >= 0; i--) {
      assertThat(iterator.hasPrevious()).isTrue();
      char ch = iterator.previous();
      assertThat(String.valueOf(ch)).isEqualTo(String.valueOf(i + 1));
    }
  }

  @Test
  public void testMerge() {
    final char[] chars = "0123456789".toCharArray();
    ArrayViewCharSequence first = new ArrayViewCharSequence(chars, 1, 9);
    final char[] abcChars = "abcdefg".toCharArray();
    ArrayViewCharSequence second = new ArrayViewCharSequence(abcChars, 1, 4);

    assertThat(ArrayViewCharSequence.merge(ImmutableList.of(first))).isSameInstanceAs(first);
    ArrayViewCharSequence merged = ArrayViewCharSequence.merge(ImmutableList.of(first, second));
    assertThat(merged.length()).isEqualTo(11);
    assertThat(merged.toString()).isEqualTo("12345678bcd");
  }
}
