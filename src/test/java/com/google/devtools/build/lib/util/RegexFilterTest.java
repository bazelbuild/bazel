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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link RegexFilter}. */
@RunWith(JUnit4.class)
public class RegexFilterTest {
  protected RegexFilter filter = null;

  protected RegexFilter createFilter(String filterString) throws OptionsParsingException {
    filter = new RegexFilter.RegexFilterConverter().convert(filterString);
    return filter;
  }

  private static RegexFilter safeCreateFilter(String filterString) {
    try {
      return new RegexFilter.RegexFilterConverter().convert(filterString);
    } catch (OptionsParsingException e) {
      throw new RuntimeException(e);
    }
  }

  protected void assertIncluded(String value) {
    assertThat(filter.isIncluded(value)).isTrue();
  }

  protected void assertExcluded(String value) {
    assertThat(filter.isIncluded(value)).isFalse();
  }

  @Test
  public void emptyFilter() throws Exception {
    createFilter("");
    assertIncluded("a/b/c");
    assertIncluded("d");
  }

  @Test
  public void inclusions() throws Exception {
    createFilter("a/b,+^c,_test$");
    assertThat(filter.toString()).isEqualTo("(?:(?>^c)|(?>_test$)|(?>a/b))");
    assertIncluded("a/b");
    assertIncluded("a/b/c");
    assertIncluded("c");
    assertIncluded("c/d");
    assertIncluded("e/a/b");
    assertIncluded("f/1/2/3/_test");
    assertExcluded("a");
    assertExcluded("a/c");
    assertExcluded("d");
    assertExcluded("e/f/g");
    assertExcluded("f/_test2");
  }

  @Test
  public void exclusions() throws Exception {
    createFilter("-a/b,-^c,-_test$");
    assertThat(filter.toString()).isEqualTo("-(?:(?>^c)|(?>_test$)|(?>a/b))");
    assertExcluded("a/b");
    assertExcluded("a/b/c");
    assertExcluded("c");
    assertExcluded("c/d");
    assertExcluded("f/a/b/d");
    assertExcluded("f/a_test");
    assertIncluded("a");
    assertIncluded("a/c");
    assertIncluded("d");
    assertIncluded("e/f/g");
    assertIncluded("f/a_test_case");
  }

  @Test
  public void inclusionsAndExclusions() throws Exception {
    createFilter("a,-^c,,-,+,d,+a/b/c,-a/b,a/b/d");
    assertThat(filter.toString())
        .isEqualTo("(?:(?>a)|(?>a/b/c)|(?>a/b/d)|(?>d)),-(?:(?>^c)|(?>a/b))");
    assertIncluded("a");
    assertIncluded("a/c");
    assertExcluded("a/b");
    assertExcluded("a/b/c"); // Exclusions take precedence over inclusions. Order is not important.
    assertExcluded("a/b/d"); // Exclusions take precedence over inclusions. Order is not important.
    assertExcluded("a/c/a/b/d");
    assertExcluded("c");
    assertExcluded("c/d");
    assertIncluded("d/e");
    assertExcluded("e");
  }

  @Test
  public void commas() throws Exception {
    createFilter("a\\,b,c\\,d");
    assertThat(filter.toString()).isEqualTo("(?:(?>a\\,b)|(?>c\\,d))");
    assertIncluded("a,b");
    assertIncluded("c,d");
    assertExcluded("a");
    assertExcluded("b,c");
    assertExcluded("d");
  }

  @Test
  public void invalidExpression() throws Exception {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> createFilter("*a"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "Failed to build valid regular expression: Dangling meta character '*' "
                + "near index");
  }

  @Test
  public void equals() throws Exception {
    new EqualsTester()
        .addEqualityGroup(createFilter("a,b,c"), createFilter("a,b,c"))
        .addEqualityGroup(createFilter("a,b,c,d"))
        .addEqualityGroup(createFilter("a,b,-c"), createFilter("a,b,-c"))
        .addEqualityGroup(createFilter("a,b,-c,-d"))
        .addEqualityGroup(createFilter("-a,-b,-c"), createFilter("-a,-b,-c"))
        .addEqualityGroup(createFilter("-a,-b,-c,-d"))
        .addEqualityGroup(createFilter(""), createFilter(""))
        .testEquals();
  }

  @Test
  public void codec() throws Exception {
    new SerializationTester(
            ImmutableList.of(
                    "",
                    "a/b,+^c,_test$",
                    "-a/b,-^c,-_test$",
                    "a,-^c,,-,+,d,+a/b/c,-a/b,a/b/d",
                    "a\\,b,c\\,d")
                .stream()
                .map(RegexFilterTest::safeCreateFilter)
                .collect(toImmutableList()))
        .runTests();
  }

  @Test
  public void initialDoubleDash_error() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> createFilter("--compilation_mode"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "Failed to build filter: value looks like another flag (--compilation_mode). Either"
                + " escape the value with \"\\-\\-\", or pass an explicit value to the flag.");
  }

  @Test
  public void initialDoubleDash_escaped() throws OptionsParsingException {
    createFilter("\\-\\-compilation_mode");
    assertIncluded("--compilation_mode");
  }
}
