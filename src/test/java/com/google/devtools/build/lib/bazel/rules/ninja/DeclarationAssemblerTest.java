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
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteFragmentAtOffset;
import com.google.devtools.build.lib.bazel.rules.ninja.file.DeclarationAssembler;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorFinder;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DeclarationAssembler}. */
@RunWith(JUnit4.class)
public class DeclarationAssemblerTest {
  @Test
  public void testAssembleLines() throws GenericParsingException, IOException {
    // Glue two parts of the same token together
    doSameBufferTest("0123456789", 1, 3, 3, 5, "1234");
    // The '\n' symbol happened to be the last in the buffer, we should correctly
    // not merge two parts, but create two separate tokens
    doSameBufferTest("01\n3456789", 1, 3, 3, 5, "1\n", "34");
    // The "\n " sequence does not separate a new token, because of the starting space
    doSameBufferTest("01\n 3456789", 1, 4, 4, 6, "1\n 34");

    doTwoBuffersTest("abc", "def", 0, "abcdef");
    doTwoBuffersTest("abc\n", "def", 0, "abc\n", "def");
    doTwoBuffersTest("abc\n", " def", 0, "abc\n def");
    doTwoBuffersTest("abc", "\ndef", 1, "abc\n", "def");

    doTwoBuffersTest("abc$\n", "def", 0, "abc$\ndef");
    doTwoBuffersTest("abc$", "\ndef", 1, "abc$\ndef");

    doTwoBuffersComplexTest("abc", "def", 0, "abcdef");
    doTwoBuffersComplexTest("abc\n", "def", 0, "abc\n", "def");
    doTwoBuffersComplexTest("abc\n", " def", 0, "abc\n def");
    doTwoBuffersComplexTest("abc", "\ndef", 1, "abc\n", "def");

    doTwoBuffersComplexTest("abc$\n", "def", 0, "abc$\ndef");
    doTwoBuffersComplexTest("abc$", "\ndef", 1, "abc$\ndef");
  }

  private static void doTwoBuffersTest(String s1, String s2, int secondOffset, String... expected)
      throws GenericParsingException, IOException {
    List<String> list = Lists.newArrayList();
    final byte[] chars1 = s1.getBytes(StandardCharsets.ISO_8859_1);
    final byte[] chars2 = s2.getBytes(StandardCharsets.ISO_8859_1);

    DeclarationAssembler assembler =
        new DeclarationAssembler(
            (byteFragmentAtOffset) -> {
              list.add(byteFragmentAtOffset.getFragment().toString());
              assertThat(byteFragmentAtOffset.getRealStartOffset()).isAnyOf(0, secondOffset + chars1.length);
            },
            NinjaSeparatorFinder.INSTANCE);

    assembler.wrapUp(
        Lists.newArrayList(
            new ByteFragmentAtOffset(
                0, new ByteBufferFragment(ByteBuffer.wrap(chars1), 0, s1.length())),
            new ByteFragmentAtOffset(
                chars1.length, new ByteBufferFragment(ByteBuffer.wrap(chars2), 0, s2.length()))));

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }

  private static void doTwoBuffersComplexTest(String s1, String s2, int secondOffset, String... expected)
      throws GenericParsingException, IOException {
    List<String> list = Lists.newArrayList();
    // Imitate that there was some other part of 10 bytes in the beginning of first buffer.
    String startIgnore = "0123456789";
    int startLen = startIgnore.length();
    final byte[] chars1 = (startIgnore + s1).getBytes(StandardCharsets.ISO_8859_1);
    final byte[] chars2 = s2.getBytes(StandardCharsets.ISO_8859_1);

    DeclarationAssembler assembler =
        new DeclarationAssembler(
            (byteFragmentAtOffset) -> {
              list.add(byteFragmentAtOffset.getFragment().toString());
              // The resulting offsets of the declarations should be shifted on startLen.
              // (The offset of the second declaration is the length of the first buffer +
              // the number of bytes for the starting separator, i.e. secondOffset)
              assertThat(byteFragmentAtOffset.getRealStartOffset())
                  .isAnyOf(startLen, secondOffset + chars1.length);
            },
            NinjaSeparatorFinder.INSTANCE);

    assembler.wrapUp(
        Lists.newArrayList(
            // Skip first 10 bytes in first buffer.
            new ByteFragmentAtOffset(
                0, new ByteBufferFragment(ByteBuffer.wrap(chars1), startLen, chars1.length)),
            new ByteFragmentAtOffset(
                chars1.length, new ByteBufferFragment(ByteBuffer.wrap(chars2), 0, s2.length()))));

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }

  private static void doSameBufferTest(
      String s, int start1, int end1, int start2, int end2, String... expected)
      throws GenericParsingException, IOException {
    List<String> list = Lists.newArrayList();
    DeclarationAssembler assembler =
        new DeclarationAssembler(
            (byteFragmentAtOffset) -> {
              list.add(byteFragmentAtOffset.getFragment().toString());
              assertThat(byteFragmentAtOffset.getRealStartOffset()).isAnyOf(start1, start2);
            },
            NinjaSeparatorFinder.INSTANCE);

    final byte[] chars = s.getBytes(StandardCharsets.ISO_8859_1);
    assembler.wrapUp(
        Lists.newArrayList(
            new ByteFragmentAtOffset(
                0, new ByteBufferFragment(ByteBuffer.wrap(chars), start1, end1)),
            new ByteFragmentAtOffset(
                0, new ByteBufferFragment(ByteBuffer.wrap(chars), start2, end2))));

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }
}
