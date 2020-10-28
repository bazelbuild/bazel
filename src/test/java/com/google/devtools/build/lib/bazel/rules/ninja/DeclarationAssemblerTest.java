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
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Strings;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.bazel.rules.ninja.file.DeclarationAssembler;
import com.google.devtools.build.lib.bazel.rules.ninja.file.FileFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DeclarationAssembler}. */
@RunWith(JUnit4.class)
public class DeclarationAssemblerTest {
  @Test
  public void testOneCharacterFragment() throws Exception {
    List<ByteBuffer> buffers = new ArrayList<>();
    buffers.add(ByteBuffer.wrap("abcdefghij\na".getBytes(ISO_8859_1)));
    buffers.add(ByteBuffer.wrap("bcdefg\n".getBytes(ISO_8859_1)));

    List<FileFragment> fragments = new ArrayList<>();
    fragments.add(new FileFragment(buffers.get(0), 0, 0, 11)); // abcdefghij\n
    fragments.add(new FileFragment(buffers.get(0), 0, 11, 12)); // a
    fragments.add(new FileFragment(buffers.get(1), 12, 0, 7)); // bcdefg\n

    List<String> result = new ArrayList<>();
    DeclarationAssembler assembler = new DeclarationAssembler(d -> result.add(d.toString()));
    assembler.wrapUp(fragments);
    assertThat(result).containsExactly("abcdefghij\n", "abcdefg\n").inOrder();
  }

  @Test
  public void testAssembleLines() throws Exception {
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
    doTwoBuffersTest("abc", "\ndef", "abc\n", "def");

    doTwoBuffersTest("abc$\n", "def", "abc$\ndef");
    doTwoBuffersTest("abc$", "\ndef", "abc$\ndef");
  }

  @Test
  public void testMergeTwoDifferentBuffers() throws Exception {
    List<Pair<Long, String>> offsetStringPairList = Lists.newArrayList();
    String unrelatedFirstBuffer = Strings.repeat(" ", 100);
    String s1 = "hello";
    String s2 = "goodbye";
    byte[] chars1 = (unrelatedFirstBuffer + s1).getBytes(ISO_8859_1);
    ByteBuffer bytes1 = ByteBuffer.wrap(chars1);
    byte[] chars2 = s2.getBytes(ISO_8859_1);
    ByteBuffer bytes2 = ByteBuffer.wrap(chars2);

    DeclarationAssembler assembler =
        new DeclarationAssembler(
            fragment -> {
              offsetStringPairList.add(
                  new Pair<>(fragment.getFragmentOffset(), fragment.toString()));
            });

    assembler.wrapUp(
        Lists.newArrayList(
            new FileFragment(bytes1, 0, unrelatedFirstBuffer.length(), chars1.length),
            new FileFragment(bytes2, chars1.length, 0, s2.length())));

    assertThat(Iterables.getOnlyElement(offsetStringPairList))
        .isEqualTo(new Pair<>((long) unrelatedFirstBuffer.length(), "hellogoodbye"));
  }

  private static void doTwoBuffersTest(String s1, String s2, String... expected)
      throws GenericParsingException, IOException {
    List<String> list = Lists.newArrayList();
    final byte[] chars1 = s1.getBytes(ISO_8859_1);
    ByteBuffer bytes1 = ByteBuffer.wrap(chars1);
    final byte[] chars2 = s2.getBytes(ISO_8859_1);
    ByteBuffer bytes2 = ByteBuffer.wrap(chars2);

    DeclarationAssembler assembler =
        new DeclarationAssembler(
            fragment -> {
              list.add(fragment.toString());
              assertThat(fragment.getFileOffset()).isAnyOf(0L, (long) chars1.length);
            });

    assembler.wrapUp(
        Lists.newArrayList(
            new FileFragment(bytes1, 0, 0, s1.length()),
            new FileFragment(bytes2, chars1.length, 0, s2.length())));

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }

  private static void doSameBufferTest(
      String s, int start1, int end1, int start2, int end2, String... expected)
      throws GenericParsingException, IOException {
    List<String> list = Lists.newArrayList();
    DeclarationAssembler assembler =
        new DeclarationAssembler(
            fragment -> {
              list.add(fragment.toString());
              assertThat(fragment.getFileOffset()).isEqualTo(0);
            });

    byte[] chars = s.getBytes(ISO_8859_1);
    ByteBuffer bytes = ByteBuffer.wrap(chars);
    assembler.wrapUp(
        Lists.newArrayList(
            new FileFragment(bytes, 0, start1, end1), new FileFragment(bytes, 0, start2, end2)));

    assertThat(list).isEqualTo(Arrays.asList(expected));
  }
}
