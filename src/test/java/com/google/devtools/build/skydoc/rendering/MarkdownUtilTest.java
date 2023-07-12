package com.google.devtools.build.skydoc.rendering;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;

public class MarkdownUtilTest {

  MarkdownUtil util = new MarkdownUtil();

  @Test
  public void markdownCodeSpan() {
    assertThat(util.markdownCodeSpan("")).isEqualTo("``");
    assertThat(util.markdownCodeSpan("foo bar ")).isEqualTo("`foo bar `");
  }

  @Test
  public void markdownCodeSpan_backticks() {
    assertThat(util.markdownCodeSpan("foo`bar")).isEqualTo("``foo`bar``");
    assertThat(util.markdownCodeSpan("foo``bar")).isEqualTo("```foo``bar```");
    assertThat(util.markdownCodeSpan("foo`bar```baz``quz")).isEqualTo("````foo`bar```baz``quz````");
  }

  @Test
  public void markdownCodeSpan_backticksPadding() {
    assertThat(util.markdownCodeSpan("`foo")).isEqualTo("`` `foo ``");
    assertThat(util.markdownCodeSpan("``foo")).isEqualTo("``` ``foo ```");
    assertThat(util.markdownCodeSpan("foo`")).isEqualTo("`` foo` ``");
    assertThat(util.markdownCodeSpan("foo``")).isEqualTo("``` foo`` ```");
  }
}
