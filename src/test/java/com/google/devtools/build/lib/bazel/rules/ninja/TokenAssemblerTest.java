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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import com.google.devtools.build.lib.bazel.rules.ninja.file.TokenAssembler;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
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

  private static void doTwoBuffersTest(String s1, String s2, String... expected)
      throws GenericParsingException {
    List<String> list = Lists.newArrayList();
    TokenAssembler assembler = new TokenAssembler(item -> list.add(item.toString()),
        NinjaSeparatorPredicate.INSTANCE);

    final byte[] chars1 = s1.getBytes(StandardCharsets.ISO_8859_1);
    final byte[] chars2 = s2.getBytes(StandardCharsets.ISO_8859_1);

    assembler.wrapUp(
        ImmutableList.of(0, chars1.length),
        ImmutableList.of(
            ImmutableList.of(new ByteBufferFragment(ByteBuffer.wrap(chars1), 0, s1.length())),
            ImmutableList.of(new ByteBufferFragment(ByteBuffer.wrap(chars2), 0, s2.length())))
    );

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }

  private static void doSameBufferTest(String s, int start1, int end1, int start2, int end2,
      String... expected) throws GenericParsingException {
    List<String> list = Lists.newArrayList();
    TokenAssembler assembler = new TokenAssembler(item -> list.add(item.toString()),
        NinjaSeparatorPredicate.INSTANCE);

    final byte[] chars = s.getBytes(StandardCharsets.ISO_8859_1);
    assembler.wrapUp(
        ImmutableList.of(0, 0),
        ImmutableList.of(
            ImmutableList.of(new ByteBufferFragment(ByteBuffer.wrap(chars), start1, end1)),
            ImmutableList.of(new ByteBufferFragment(ByteBuffer.wrap(chars), start2, end2)))
    );

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }
}
