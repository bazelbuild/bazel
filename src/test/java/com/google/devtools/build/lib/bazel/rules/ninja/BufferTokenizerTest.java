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
import com.google.devtools.build.lib.bazel.rules.ninja.file.BufferTokenizer;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import com.google.devtools.build.lib.bazel.rules.ninja.file.TokenAndFragmentsConsumer;
import com.google.monitoring.runtime.instrumentation.common.com.google.common.collect.ImmutableList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link BufferTokenizer}
 */
@RunWith(JUnit4.class)
public class BufferTokenizerTest {
  @Test
  public void testTokenizeSimple() {
    List<String> list = ImmutableList.of("one", "two", "three");
    char[] chars = String.join("\n", list).toCharArray();
    WholeTokensAndFragmentsConsumer consumer = new WholeTokensAndFragmentsConsumer();
    BufferTokenizer tokenizer = new BufferTokenizer(
        chars, consumer, NinjaSeparatorPredicate.INSTANCE, 0, 0, chars.length);
    tokenizer.run();
    assertThat(consumer.getResult()).isEqualTo(list);
    assertThat(consumer.getFragments()).isEqualTo(ImmutableList.of("one\n", "three"));
  }

  @Test
  public void testTokenizeWithDetails() {
    List<String> list = ImmutableList.of("one", " one-detail", "two", "\ttwo-detail",
        "three", " three-detail");
    char[] chars = String.join("\n", list).toCharArray();
    WholeTokensAndFragmentsConsumer consumer = new WholeTokensAndFragmentsConsumer();
    BufferTokenizer tokenizer = new BufferTokenizer(
        chars, consumer, NinjaSeparatorPredicate.INSTANCE, 0, 0, chars.length);
    tokenizer.run();
    assertThat(consumer.getResult()).isEqualTo(ImmutableList.of(
        "one\n one-detail", "two\n\ttwo-detail", "three\n three-detail"
    ));
    assertThat(consumer.getFragments()).isEqualTo(ImmutableList.of(
        "one\n one-detail\n", "three\n three-detail"
    ));
  }

  private static class WholeTokensAndFragmentsConsumer implements TokenAndFragmentsConsumer {
    private final List<String> result = Lists.newArrayList();
    private final List<String> fragments = Lists.newArrayList();

    @Override
    public void token(CharSequence sequence) {
      result.add(sequence.toString().trim());
    }

    @Override
    public void fragment(int offset, ArrayViewCharSequence sequence) {
      fragments.add(sequence.toString());
      result.add(sequence.toString().trim());
    }

    @Override
    public void error(Throwable throwable) {
      throw new RuntimeException(throwable);
    }

    public List<String> getResult() {
      return result;
    }

    public List<String> getFragments() {
      return fragments;
    }
  }
}
