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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.bazel.rules.ninja.file.BufferSplitter;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteFragmentAtOffset;
import com.google.devtools.build.lib.bazel.rules.ninja.file.DeclarationConsumer;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorFinder;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BufferSplitter} */
@RunWith(JUnit4.class)
public class BufferSplitterTest {
  @Test
  public void testTokenizeSimple() throws Exception {
    List<String> list = ImmutableList.of("one", "two", "three");

    List<String> result = Lists.newArrayList();
    int offsetValue = 123;
    DeclarationConsumer consumer =
        (byteFragmentAtOffset) -> {
          result.add(byteFragmentAtOffset.getFragment().toString());
          assertThat(byteFragmentAtOffset.getOffset()).isEqualTo(offsetValue);
        };

    byte[] chars = String.join("\n", list).getBytes(StandardCharsets.ISO_8859_1);
    ByteBufferFragment fragment = new ByteBufferFragment(ByteBuffer.wrap(chars), 0, chars.length);
    BufferSplitter tokenizer =
        new BufferSplitter(fragment, consumer, NinjaSeparatorFinder.INSTANCE, offsetValue);
    List<ByteFragmentAtOffset> edges = tokenizer.call();
    assertThat(result).containsExactly("two\n");
    assertThat(
            edges.stream()
                .map(pair -> Preconditions.checkNotNull(pair.getFragment()).toString())
                .collect(Collectors.toList()))
        .containsExactly("one\n", "three")
        .inOrder();
  }

  @Test
  public void testTokenizeWithDetails() throws Exception {
    List<String> list =
        ImmutableList.of("one", " one-detail", "two", "\ttwo-detail", "three", " three-detail");
    byte[] chars = String.join("\n", list).getBytes(StandardCharsets.ISO_8859_1);

    List<String> result = Lists.newArrayList();
    DeclarationConsumer consumer =
        (byteFragmentAtOffset) -> result.add(byteFragmentAtOffset.getFragment().toString());

    ByteBufferFragment fragment = new ByteBufferFragment(ByteBuffer.wrap(chars), 0, chars.length);
    BufferSplitter tokenizer =
        new BufferSplitter(fragment, consumer, NinjaSeparatorFinder.INSTANCE, 0);
    List<ByteFragmentAtOffset> edges = tokenizer.call();
    assertThat(result).containsExactly("two\n\ttwo-detail\n");
    assertThat(
            edges.stream()
                .map(pair -> Preconditions.checkNotNull(pair.getFragment()).toString())
                .collect(Collectors.toList()))
        .containsExactly("one\n one-detail\n", "three\n three-detail")
        .inOrder();
  }
}
