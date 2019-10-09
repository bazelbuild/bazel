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

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ArrayViewCharSequence;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import com.google.devtools.build.lib.bazel.rules.ninja.file.TokenAssembler;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link TokenAssembler}.
 */
@RunWith(JUnit4.class)
public class TokenAssemblerTest {
  @Test
  public void testAssembleLines() throws GenericParsingException {
    // Glue two parts of the same token together
    doSameBufferTest("0123456789", 1, 3, 3, 5, "1234");
    // The '\n' symbol happened to be the last in the buffer, we should correctly
    // not merge two parts, but create two separate tokens
    doSameBufferTest("01\n3456789", 1, 3, 3, 5, "1\n", "34");
    // The "\n " sequence does not separate a new token, because of the starting space
    doSameBufferTest("01\n 3456789", 1, 4, 4, 6, "1\n 34");

    doTwoBuffersTest("abc", "def", "abcdef");
    doTwoBuffersTest("abc\n", "def", "abc\n", "def");
    doTwoBuffersTest("abc\n", " def", "abc\n def");
  }

  private void doTwoBuffersTest(String s1, String s2, String... expected)
      throws GenericParsingException {
    List<String> list = Lists.newArrayList();
    TokenAssembler assembler = new TokenAssembler(item -> list.add(item.toString()),
        NinjaSeparatorPredicate.INSTANCE);

    final char[] chars1 = s1.toCharArray();
    final char[] chars2 = s2.toCharArray();
    assembler.fragment(0, new ArrayViewCharSequence(chars1, 0, s1.length()));
    assembler.fragment(chars1.length, new ArrayViewCharSequence(chars2, 0, s2.length()));

    assembler.wrapUp();

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }

  private void doSameBufferTest(String s, int start1, int end1, int start2, int end2,
      String... expected) throws GenericParsingException {
    List<String> list = Lists.newArrayList();
    TokenAssembler assembler = new TokenAssembler(item -> list.add(item.toString()),
        NinjaSeparatorPredicate.INSTANCE);

    final char[] chars = s.toCharArray();
    assembler.fragment(0, new ArrayViewCharSequence(chars, start1, end1));
    assembler.fragment(0, new ArrayViewCharSequence(chars, start2, end2));

    assembler.wrapUp();

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }
}
