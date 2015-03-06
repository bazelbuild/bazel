// Copyright 2014 Google Inc. All rights reserved.
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Maps;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A test for {@link StringUtilities}.
 */
@RunWith(JUnit4.class)
public class StringUtilitiesTest {

  // Tests of StringUtilities.joinLines()

  @Test
  public void emptyLinesYieldsEmptyString() {
    assertThat(joinLines()).isEmpty();
  }

  @Test
  public void twoLinesGetjoinedNicely() {
    assertEquals("line 1\nline 2", joinLines("line 1", "line 2"));
  }

  @Test
  public void aTrailingNewlineIsAvailableWhenYouNeedIt() {
    assertEquals("two lines\nwith trailing newline\n",
        joinLines("two lines", "with trailing newline", ""));
  }

  // Tests of StringUtilities.combineKeys()

  /** Simple sanity test of format */
  @Test
  public void combineKeysFormat() {
    assertEquals("<a><b!!c><!<d!>>", combineKeys("a", "b!c", "<d>"));
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
    assertEquals("ababab",
                 StringUtilities.replaceAllLiteral("bababa", "ba", "ab"));
    assertThat(StringUtilities.replaceAllLiteral("bababa", "ba", "")).isEmpty();
    assertEquals("bababa",
        StringUtilities.replaceAllLiteral("bababa", "", "ab"));
  }

  @Test
  public void testLayoutTable() throws Exception {
    Map<String, String> data = Maps.newTreeMap();
    data.put("foo", "bar");
    data.put("bang", "baz");
    data.put("lengthy key", "lengthy value");

    assertEquals(joinLines("bang: baz",
                           "foo: bar",
                           "lengthy key: lengthy value"), layoutTable(data));
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
      assertEquals(expected[ii], prettyPrintBytes((long) x));
      x = x * 10.0;
    }
  }

  @Test
  public void sanitizeControlChars() {
    assertEquals("<?>", StringUtilities.sanitizeControlChars("\000"));
    assertEquals("<?>", StringUtilities.sanitizeControlChars("\001"));
    assertEquals("\\r", StringUtilities.sanitizeControlChars("\r"));
    assertEquals(" abc123", StringUtilities.sanitizeControlChars(" abc123"));
  }

  @Test
  public void containsSubarray() {
    assertTrue(StringUtilities.containsSubarray("abcde".toCharArray(), "ab".toCharArray()));
    assertTrue(StringUtilities.containsSubarray("abcde".toCharArray(), "de".toCharArray()));
    assertTrue(StringUtilities.containsSubarray("abcde".toCharArray(), "bc".toCharArray()));
    assertTrue(StringUtilities.containsSubarray("abcde".toCharArray(), "".toCharArray()));
  }

  @Test
  public void notContainsSubarray() {
    assertFalse(StringUtilities.containsSubarray("abc".toCharArray(), "abcd".toCharArray()));
    assertFalse(StringUtilities.containsSubarray("abc".toCharArray(), "def".toCharArray()));
    assertFalse(StringUtilities.containsSubarray("abcde".toCharArray(), "bd".toCharArray()));
  }

  @Test
  public void toPythonStyleFunctionName() {
    assertEquals("a", StringUtilities.toPythonStyleFunctionName("a"));
    assertEquals("a_b", StringUtilities.toPythonStyleFunctionName("aB"));
    assertEquals("a_b_c", StringUtilities.toPythonStyleFunctionName("aBC"));
    assertEquals("a_bc_d", StringUtilities.toPythonStyleFunctionName("aBcD"));
  }
}
