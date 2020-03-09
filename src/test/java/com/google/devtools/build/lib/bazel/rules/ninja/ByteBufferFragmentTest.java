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
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment} */
@RunWith(JUnit4.class)
public class ByteBufferFragmentTest {
  @Test
  public void testMethods() {
    final byte[] bytes = "0123456789".getBytes(StandardCharsets.ISO_8859_1);
    ByteBufferFragment fragment = new ByteBufferFragment(ByteBuffer.wrap(bytes), 1, 9);
    assertThat(fragment.length()).isEqualTo(8);
    assertThat(fragment.toString()).isEqualTo("12345678");
    assertThat(fragment.subFragment(2, 4).toString()).isEqualTo("34");
    assertThat(fragment.subFragment(0, 8).toString()).isEqualTo("12345678");
  }

  @Test
  public void testMerge() {
    final byte[] bytes = "0123456789".getBytes(StandardCharsets.ISO_8859_1);
    ByteBufferFragment first = new ByteBufferFragment(ByteBuffer.wrap(bytes), 1, 9);
    final byte[] abcBytes = "abcdefg".getBytes(StandardCharsets.ISO_8859_1);
    ByteBufferFragment second = new ByteBufferFragment(ByteBuffer.wrap(abcBytes), 1, 4);

    assertThat(ByteBufferFragment.merge(ImmutableList.of(first))).isSameInstanceAs(first);
    ByteBufferFragment merged = ByteBufferFragment.merge(ImmutableList.of(first, second));
    assertThat(merged.length()).isEqualTo(11);
    assertThat(merged.toString()).isEqualTo("12345678bcd");
  }

  @Test
  public void testEscapeCharacters() {
    final byte[] bytes = "\0\n\t\r".getBytes(StandardCharsets.ISO_8859_1);
    ByteBufferFragment fragment = new ByteBufferFragment(ByteBuffer.wrap(bytes), 0, 3);
    assertThat(fragment.length()).isEqualTo(3);
    assertThat(fragment.toString()).isEqualTo("\0\n\t");
    assertThat(fragment.subFragment(1, 3).toString()).isEqualTo("\n\t");
  }
}
