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
import static com.google.devtools.build.lib.util.StringUtil.capitalize;
import static com.google.devtools.build.lib.util.StringUtil.indent;
import static com.google.devtools.build.lib.util.StringUtil.joinEnglishList;
import static com.google.devtools.build.lib.util.StringUtil.splitAndInternString;
import static com.google.devtools.build.lib.util.StringUtil.stripSuffix;
import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link StringUtil}. */
@RunWith(JUnit4.class)
public class StringUtilTest {

  @Test
  public void testJoinEnglishList() throws Exception {
    assertEquals("nothing",
        joinEnglishList(Collections.emptyList()));
    assertEquals("one",
        joinEnglishList(Arrays.asList("one")));
    assertEquals("one or two",
        joinEnglishList(Arrays.asList("one", "two")));
    assertEquals("one and two",
        joinEnglishList(Arrays.asList("one", "two"), "and"));
    assertEquals("one, two or three",
        joinEnglishList(Arrays.asList("one", "two", "three")));
    assertEquals("one, two and three",
        joinEnglishList(Arrays.asList("one", "two", "three"), "and"));
    assertEquals("'one', 'two' and 'three'",
        joinEnglishList(Arrays.asList("one", "two", "three"), "and", "'"));
  }

  @Test
  public void splitAndIntern() throws Exception {
    assertThat(splitAndInternString("       ")).isEmpty();
    assertThat(splitAndInternString(null)).isEmpty();
    List<String> list1 = splitAndInternString("    x y    z    z");
    List<String> list2 = splitAndInternString("a z    c z");

    assertThat(list1).containsExactly("x", "y", "z", "z").inOrder();
    assertThat(list2).containsExactly("a", "z", "c", "z").inOrder();
    assertThat(list1.get(3)).isSameAs(list1.get(2));
    assertThat(list2.get(1)).isSameAs(list1.get(2));
    assertThat(list2.get(3)).isSameAs(list2.get(1));
  }

  @Test
  public void listItemsWithLimit() throws Exception {
    assertEquals("begin/a, b, c/end", StringUtil.listItemsWithLimit(
        new StringBuilder("begin/"), 3, ImmutableList.of("a", "b", "c")).append("/end").toString());

    assertEquals("begin/a, b, c ...(omitting 2 more item(s))/end", StringUtil.listItemsWithLimit(
        new StringBuilder("begin/"), 3, ImmutableList.of("a", "b", "c", "d", "e"))
            .append("/end").toString());
  }

  @Test
  public void testIndent() throws Exception {
    assertThat(indent("", 0)).isEmpty();
    assertThat(indent("", 1)).isEmpty();
    assertEquals("a", indent("a", 1));
    assertEquals("\n  a", indent("\na", 2));
    assertEquals("a\n  b", indent("a\nb", 2));
    assertEquals("a\n b\n c\n d", indent("a\nb\nc\nd", 1));
    assertEquals("\n ", indent("\n", 1));
  }

  @Test
  public void testStripSuffix() throws Exception {
    assertThat(stripSuffix("", "")).isEmpty();
    assertThat(stripSuffix("", "a")).isNull();
    assertEquals("a", stripSuffix("a", ""));
    assertEquals("a", stripSuffix("aa", "a"));
    assertThat(stripSuffix("ab", "c")).isNull();
  }

  @Test
  public void testCapitalize() throws Exception {
    assertThat(capitalize("")).isEmpty();
    assertEquals("Joe", capitalize("joe"));
    assertEquals("Joe", capitalize("Joe"));
    assertEquals("O", capitalize("o"));
    assertEquals("O", capitalize("O"));
  }
}
