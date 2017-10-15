// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link StringTrie}.
 */
@RunWith(JUnit4.class)
public class StringTrieTest {
  @Test
  public void empty() {
    StringTrie<Integer> cut = new StringTrie<>();
    assertThat(cut.get("")).isNull();
    assertThat(cut.get("a")).isNull();
    assertThat(cut.get("ab")).isNull();
  }

  @Test
  public void simple() {
    StringTrie<Integer> cut = new StringTrie<>();
    cut.put("a", 1);
    cut.put("b", 2);

    assertThat(cut.get("")).isNull();
    assertThat(cut.get("a").intValue()).isEqualTo(1);
    assertThat(cut.get("ab").intValue()).isEqualTo(1);
    assertThat(cut.get("abc").intValue()).isEqualTo(1);

    assertThat(cut.get("b").intValue()).isEqualTo(2);
  }

  @Test
  public void ancestors() {
    StringTrie<Integer> cut = new StringTrie<>();
    cut.put("abc", 3);
    assertThat(cut.get("")).isNull();
    assertThat(cut.get("a")).isNull();
    assertThat(cut.get("ab")).isNull();
    assertThat(cut.get("abc").intValue()).isEqualTo(3);
    assertThat(cut.get("abcd").intValue()).isEqualTo(3);

    cut.put("a", 1);
    assertThat(cut.get("a").intValue()).isEqualTo(1);
    assertThat(cut.get("ab").intValue()).isEqualTo(1);
    assertThat(cut.get("abc").intValue()).isEqualTo(3);

    cut.put("", 0);
    assertThat(cut.get("").intValue()).isEqualTo(0);
    assertThat(cut.get("b").intValue()).isEqualTo(0);
    assertThat(cut.get("a").intValue()).isEqualTo(1);
  }
}
