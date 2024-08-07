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
import static com.google.devtools.build.lib.util.StringUtil.decodeBytestringUtf8;
import static com.google.devtools.build.lib.util.StringUtil.encodeBytestringUtf8;
import static com.google.devtools.build.lib.util.StringUtil.joinEnglishList;
import static com.google.devtools.build.lib.util.StringUtil.joinEnglishListSingleQuoted;

import com.google.common.collect.ImmutableList;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link StringUtil}. */
@RunWith(JUnit4.class)
public class StringUtilTest {

  @Test
  public void testJoinEnglishList() throws Exception {
    assertThat(joinEnglishList(ImmutableList.of())).isEqualTo("nothing");
    assertThat(joinEnglishList(Arrays.asList("one"))).isEqualTo("one");
    assertThat(joinEnglishList(Arrays.asList("one", "two"))).isEqualTo("one or two");
    assertThat(joinEnglishList(Arrays.asList("one", "two"), "and")).isEqualTo("one and two");
    assertThat(joinEnglishList(Arrays.asList("one", "two", "three")))
        .isEqualTo("one, two, or three");
    assertThat(joinEnglishList(Arrays.asList("one", "two", "three"), "and"))
        .isEqualTo("one, two, and three");
    assertThat(joinEnglishList(Arrays.asList("one", "two", "three"), "or even", "\"", true))
        .isEqualTo("\"one\", \"two\", or even \"three\"");
    assertThat(joinEnglishList(Arrays.asList("one", "two", "three"), "then", "'", false))
        .isEqualTo("'one', 'two' then 'three'");
  }

  @Test
  public void testJoinEnglishListSingleQuoted() throws Exception {
    assertThat(joinEnglishListSingleQuoted(ImmutableList.of())).isEqualTo("nothing");
    assertThat(joinEnglishListSingleQuoted(Arrays.asList("one"))).isEqualTo("'one'");
    assertThat(joinEnglishListSingleQuoted(Arrays.asList("one", "two")))
        .isEqualTo("'one' or 'two'");
    assertThat(joinEnglishListSingleQuoted(Arrays.asList("one", "two", "three")))
        .isEqualTo("'one', 'two', or 'three'");
  }

  @Test
  public void listItemsWithLimit() throws Exception {
    assertThat(
            StringUtil.listItemsWithLimit(
                    new StringBuilder("begin/"), 3, ImmutableList.of("a", "b", "c"))
                .append("/end")
                .toString())
        .isEqualTo("begin/a, b, c/end");

    assertThat(
            StringUtil.listItemsWithLimit(
                    new StringBuilder("begin/"), 3, ImmutableList.of("a", "b", "c", "d", "e"))
                .append("/end")
                .toString())
        .isEqualTo("begin/a, b, c ...(omitting 2 more item(s))/end");
  }

  @Test
  @SuppressWarnings("UnicodeEscape")
  public void testDecodeBytestringUtf8() throws Exception {
    // Regular ASCII passes through OK.
    assertThat(decodeBytestringUtf8("ascii")).isEqualTo("ascii");

    // Valid UTF-8 is decoded.
    assertThat(decodeBytestringUtf8("\u00E2\u0081\u0089"))
        .isEqualTo("\u2049"); // U+2049 EXCLAMATION QUESTION MARK
    assertThat(decodeBytestringUtf8("\u00f0\u009f\u008c\u00b1"))
        .isEqualTo("\uD83C\uDF31"); // U+1F331 SEEDLING

    // Strings that contain characters that can't be UTF-8 are returned as-is,
    // to support Windows file paths.
    assertThat(decodeBytestringUtf8("\u2049"))
        .isEqualTo("\u2049"); // U+2049 EXCLAMATION QUESTION MARK
    assertThat(decodeBytestringUtf8("\uD83C\uDF31")).isEqualTo("\uD83C\uDF31"); // U+1F331 SEEDLING

    // Strings that might be either UTF-8 or Unicode are optimistically decoded,
    // and returned as-is if decoding succeeded with no replacement characters.
    assertThat(decodeBytestringUtf8("\u00FC\u006E\u00EF\u0063\u00F6\u0064\u00EB"))
        .isEqualTo("\u00FC\u006E\u00EF\u0063\u00F6\u0064\u00EB"); // "ünïcödë"

    // A string that is both valid ISO-8859-1 and valid UTF-8 will be decoded
    // as UTF-8.
    assertThat(decodeBytestringUtf8("\u00C2\u00A3")) // "Â£"
        .isEqualTo("\u00A3"); // U+00A3 POUND SIGN
  }

  @Test
  public void testEncodeBytestringUtf8() throws Exception {
    assertThat(encodeBytestringUtf8("ascii")).isEqualTo("ascii");
    assertThat(encodeBytestringUtf8("\u2049")) // U+2049 EXCLAMATION QUESTION MARK
        .isEqualTo("\u00E2\u0081\u0089");
    assertThat(encodeBytestringUtf8("\uD83C\uDF31")) // U+1F331 SEEDLING
        .isEqualTo("\u00f0\u009f\u008c\u00b1");
  }
}
