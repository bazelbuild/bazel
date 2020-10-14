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
import static com.google.devtools.build.lib.util.StringUtilities.combineKeys;
import static com.google.devtools.build.lib.util.StringUtilities.joinLines;
import static com.google.devtools.build.lib.util.StringUtilities.layoutTable;
import static com.google.devtools.build.lib.util.StringUtilities.prettyPrintBytes;
import static org.junit.Assert.fail;

import com.google.common.collect.Maps;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link StringUtilities}. */
@RunWith(JUnit4.class)
public class StringUtilitiesTest {

  // Tests of StringUtilities.joinLines()

  @Test
  public void emptyLinesYieldsEmptyString() {
    assertThat(joinLines()).isEmpty();
  }

  @Test
  public void twoLinesGetjoinedNicely() {
    assertThat(joinLines("line 1", "line 2")).isEqualTo("line 1\nline 2");
  }

  @Test
  public void aTrailingNewlineIsAvailableWhenYouNeedIt() {
    assertThat(joinLines("two lines", "with trailing newline", ""))
        .isEqualTo("two lines\nwith trailing newline\n");
  }

  // Tests of StringUtilities.combineKeys()

  /** Simple test of format */
  @Test
  public void combineKeysFormat() {
    assertThat(combineKeys("a", "b!c", "<d>")).isEqualTo("<a><b!!c><!<d!>>");
  }

  /**
   * Test that combining different keys gives different results,
   * i.e. that there are no collisions.
   * We test all combinations of up to 3 keys from the test_keys
   * array (defined below).
   */
  @Test
  public void testCombineKeys() {
    // This map is really just used as a set, but
    // if the test fails, the values in the map may be
    // useful for debugging.
    Map<String,String[]> map = new HashMap<>();
    for (int numKeys = 0; numKeys <= 3; numKeys++) {
      testCombineKeys(map, numKeys, new String[numKeys]);
    }
  }

  private void testCombineKeys(Map<String,String[]> map,
        int n, String[] keys) {
    if (n == 0) {
      String[] keys_copy = keys.clone();
      String combined_key = combineKeys(keys_copy);
      String[] prev_keys = map.put(combined_key, keys_copy);
      if (prev_keys != null) {
        fail("combineKeys collision:\n"
            + "key sequence 1: " + Arrays.toString(prev_keys) + "\n"
            + "key sequence 2: " + Arrays.toString(keys_copy) + "\n"
            + "combined key sequence 1: " + combineKeys(prev_keys) + "\n"
            + "combined key sequence 2: " + combineKeys(keys_copy) + "\n");
      }
    } else {
      for (String key : test_keys) {
        keys[n - 1] = key;
        testCombineKeys(map, n - 1, keys);
      }
    }
  }

  private static final String[] test_keys = {
        // ordinary strings
        "", "a", "word", "//depot/foo/bar",
        // likely delimiter characters
        " ", ",", "\\", "\"", "\'", "\0", "\u00ff",
        // strings starting in special delimiter
        " foo", ",foo", "\\foo", "\"foo", "\'foo", "\0foo", "\u00fffoo",
        // strings ending in special delimiter
        "bar ", "bar,", "bar\\", "bar\"", "bar\'", "bar\0", "bar\u00ff",
        // white-box testing of the delimiters that combineKeys() uses
        "<", ">", "!", "!<", "!>", "!!", "<!", ">!"
  };

  @Test
  public void replaceAllLiteral() throws Exception {
    assertThat(StringUtilities.replaceAllLiteral("bababa", "ba", "ab")).isEqualTo("ababab");
    assertThat(StringUtilities.replaceAllLiteral("bababa", "ba", "")).isEmpty();
    assertThat(StringUtilities.replaceAllLiteral("bababa", "", "ab")).isEqualTo("bababa");
  }

  @Test
  public void testLayoutTable() throws Exception {
    Map<String, String> data = Maps.newTreeMap();
    data.put("foo", "bar");
    data.put("bang", "baz");
    data.put("lengthy key", "lengthy value");

    assertThat(layoutTable(data))
        .isEqualTo(joinLines("bang: baz", "foo: bar", "lengthy key: lengthy value"));
  }

  @Test
  public void testPrettyPrintBytes() {
    String[] expected = {
      "2B",
      "23B",
      "234B",
      "2345B",
      "23KB",
      "234KB",
      "2345KB",
      "23MB",
      "234MB",
      "2345MB",
      "23456MB",
      "234GB",
      "2345GB",
      "23456GB",
    };
    double x = 2.3456;
    for (int ii = 0; ii < expected.length; ++ii) {
      assertThat(prettyPrintBytes((long) x)).isEqualTo(expected[ii]);
      x = x * 10.0;
    }
  }

  @Test
  public void sanitizeControlChars() {
    assertThat(StringUtilities.sanitizeControlChars("\000")).isEqualTo("<?>");
    assertThat(StringUtilities.sanitizeControlChars("\001")).isEqualTo("<?>");
    assertThat(StringUtilities.sanitizeControlChars("\r")).isEqualTo("\\r");
    assertThat(StringUtilities.sanitizeControlChars(" abc123")).isEqualTo(" abc123");
  }

  @Test
  public void containsSubarray() {
    assertThat(StringUtilities.containsSubarray("abcde".toCharArray(), "ab".toCharArray()))
        .isTrue();
    assertThat(StringUtilities.containsSubarray("abcde".toCharArray(), "de".toCharArray()))
        .isTrue();
    assertThat(StringUtilities.containsSubarray("abcde".toCharArray(), "bc".toCharArray()))
        .isTrue();
    assertThat(StringUtilities.containsSubarray("abcde".toCharArray(), "".toCharArray())).isTrue();
  }

  @Test
  public void notContainsSubarray() {
    assertThat(StringUtilities.containsSubarray("abc".toCharArray(), "abcd".toCharArray()))
        .isFalse();
    assertThat(StringUtilities.containsSubarray("abc".toCharArray(), "def".toCharArray()))
        .isFalse();
    assertThat(StringUtilities.containsSubarray("abcde".toCharArray(), "bd".toCharArray()))
        .isFalse();
  }

  @Test
  public void toPythonStyleFunctionName() {
    assertThat(StringUtilities.toPythonStyleFunctionName("a")).isEqualTo("a");
    assertThat(StringUtilities.toPythonStyleFunctionName("aB")).isEqualTo("a_b");
    assertThat(StringUtilities.toPythonStyleFunctionName("aBC")).isEqualTo("a_b_c");
    assertThat(StringUtilities.toPythonStyleFunctionName("aBcD")).isEqualTo("a_bc_d");
  }
}
