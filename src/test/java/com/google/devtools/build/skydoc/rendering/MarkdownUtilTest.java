// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.rendering;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for MarkdownUtil. */
@RunWith(JUnit4.class)
public class MarkdownUtilTest {

  MarkdownUtil util = new MarkdownUtil();

  @Test
  public void markdownCodeSpan() {
    assertThat(MarkdownUtil.markdownCodeSpan("")).isEqualTo("``");
    assertThat(MarkdownUtil.markdownCodeSpan("foo bar ")).isEqualTo("`foo bar `");
  }

  @Test
  public void markdownCodeSpan_backticks() {
    assertThat(MarkdownUtil.markdownCodeSpan("foo`bar")).isEqualTo("``foo`bar``");
    assertThat(MarkdownUtil.markdownCodeSpan("foo``bar")).isEqualTo("```foo``bar```");
    assertThat(MarkdownUtil.markdownCodeSpan("foo`bar```baz``quz"))
        .isEqualTo("````foo`bar```baz``quz````");
  }

  @Test
  public void markdownCodeSpan_backticksPadding() {
    assertThat(MarkdownUtil.markdownCodeSpan("`foo")).isEqualTo("`` `foo ``");
    assertThat(MarkdownUtil.markdownCodeSpan("``foo")).isEqualTo("``` ``foo ```");
    assertThat(MarkdownUtil.markdownCodeSpan("foo`")).isEqualTo("`` foo` ``");
    assertThat(MarkdownUtil.markdownCodeSpan("foo``")).isEqualTo("``` foo`` ```");
  }
}
