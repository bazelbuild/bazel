// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.util.WindowsParamFileEscaper.escapeString;

import com.google.common.collect.ImmutableSet;
import java.util.Arrays;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WindowsParamFileEscaper}. */
@RunWith(JUnit4.class)
public class WindowsParamFileEscaperTest {

  @Test
  public void testEscapeString() throws Exception {
    assertThat(escapeString("")).isEqualTo("\"\"");
    assertThat(escapeString("foo")).isEqualTo("foo");
    assertThat(escapeString("'foo'")).isEqualTo("'foo'");
    assertThat(escapeString("\"foo\"")).isEqualTo("\\\"foo\\\"");
    assertThat(escapeString("\\foo")).isEqualTo("\\foo");
    assertThat(escapeString("foo bar")).isEqualTo("\"foo bar\"");
    assertThat(escapeString("foo\tbar")).isEqualTo("\"foo\tbar\"");
    assertThat(escapeString("foo\rbar")).isEqualTo("\"foo\rbar\"");
    assertThat(escapeString("foo\n'foo'\n")).isEqualTo("\"foo\n'foo'\n\"");
    assertThat(escapeString("foo\fbar")).isEqualTo("foo\fbar");
    assertThat(escapeString("foo\u000Bbar")).isEqualTo("foo\u000Bbar");
    assertThat(escapeString("${filename%.c}.o")).isEqualTo("${filename%.c}.o");
  }

  @Test
  public void escapesBackslashesBeforeQuotes() {
    for (int count = 1; count <= 3; count++) {
      String backslashes = "\\".repeat(count);
      String escapedBackslashes = "\\".repeat(2 * count + 1);
      assertThat(escapeString("foo" + backslashes + "\"bar"))
          .isEqualTo("foo" + escapedBackslashes + "\"bar");
      assertThat(escapeString("foo " + backslashes + "\"bar"))
          .isEqualTo("\"foo " + escapedBackslashes + "\"bar\"");
    }
  }

  @Test
  public void escapesTrailingBackslashesInQuotedArguments() {
    for (int count = 1; count <= 3; count++) {
      assertThat(escapeString("C:\\Program Files\\SDK" + "\\".repeat(count)))
          .isEqualTo("\"C:\\Program Files\\SDK" + "\\".repeat(2 * count) + "\"");
    }
  }

  @Test
  public void preservesBackslashesWithoutFollowingQuotes() {
    assertThat(escapeString("C:\\SDK\\")).isEqualTo("C:\\SDK\\");
    assertThat(escapeString("\\\\server\\share\\")).isEqualTo("\\\\server\\share\\");
    assertThat(escapeString("C:\\Program Files\\SDK")).isEqualTo("\"C:\\Program Files\\SDK\"");
  }

  @Test
  public void escapeAllPreservesArgumentBoundaries() {
    assertThat(
            WindowsParamFileEscaper.escapeAll(
                Arrays.asList("", "C:\\Program Files\\SDK\\", "next", "")))
        .containsExactly("\"\"", "\"C:\\Program Files\\SDK\\\\\"", "next", "\"\"")
        .inOrder();
  }

  @Test
  public void testEscapeAll() throws Exception {
    Set<String> escaped =
        ImmutableSet.copyOf(
            WindowsParamFileEscaper.escapeAll(
                Arrays.asList("foo", "'foo'", "foo\n", "\"foo", "foo \tbar")));
    assertThat(escaped).containsExactly("foo", "'foo'", "\"foo\n\"", "\\\"foo", "\"foo \tbar\"");
  }
}
