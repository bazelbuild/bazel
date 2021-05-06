// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

/**
 * Tests for {@link JUnit4Options}
 */
@RunWith(JUnit4.class)
public class JUnit4OptionsTest {

  private static final Map<String, String> EMPTY_ENV = Collections.emptyMap();

  @Test
  public void testParse_noArgs() throws Exception {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.<String>of());
    assertThat(options.getTestIncludeCategories()).isNull();
    assertThat(options.getTestExcludeCategories()).isNull();
    assertThat(options.getTestIncludeFilter()).isNull();
    assertThat(options.getUnparsedArgs()).isEmpty();
  }

  @Test
  public void testParse_onlyUnparsedArgs() {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--bar", "baz"));
    assertThat(options.getTestIncludeFilter()).isNull();
    assertThat(options.getUnparsedArgs()).isEqualTo(new String[] {"--bar", "baz"});
  }

  @Test
  public void testParse_includeCategories() {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_categories", "java.lang.Object"));
    assertThat(options.getTestIncludeCategories()).isNotNull();
  }

  @Test
  public void testParse_excludeCategories() {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_exclude_categories", "java.lang.Object"));
    assertThat(options.getTestExcludeCategories()).isNotNull();
  }

  @Test
  public void testParse_withTwoArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--test_filter", "foo"));
    assertThat(options.getTestIncludeFilter()).isEqualTo("foo");
    assertThat(options.getUnparsedArgs()).isEmpty();
  }

  @Test
  public void testParse_withOneArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_filter=foo"));
    assertThat(options.getTestIncludeFilter()).isEqualTo("foo");
    assertThat(options.getUnparsedArgs()).isEmpty();
  }

  @Test
  public void testParse_testFilterAndUnparsedArgs() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--bar", "--test_filter=foo", "--baz"));
    assertThat(options.getTestIncludeFilter()).isEqualTo("foo");
    assertThat(options.getUnparsedArgs()).isEqualTo(new String[] {"--bar", "--baz"});
  }

  @Test
  public void testParse_testLastTestFilterWins() throws Exception {
    JUnit4Options options =
        JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_filter=foo", "--test_filter=bar"));
    assertThat(options.getTestIncludeFilter()).isEqualTo("bar");
    assertThat(options.getUnparsedArgs()).isEmpty();
  }

  @Test
  public void testParse_testFilterMissingSecondArg() throws Exception {
    assertThrows(
        RuntimeException.class,
        () -> JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_filter")));
  }

  @Test
  public void testParse_testFilterExcludeWithTwoArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--test_exclude_filter", "foo"));
    assertThat(options.getTestExcludeFilter()).isEqualTo("foo");
    assertThat(options.getUnparsedArgs()).isEmpty();
  }

  @Test
  public void testParse_testFilterExcludewithOneArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--test_exclude_filter=foo"));
    assertThat(options.getTestExcludeFilter()).isEqualTo("foo");
    assertThat(options.getUnparsedArgs()).isEmpty();
  }

  @Test
  public void testParse_unknownOptionName() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--unknown=foo"));
    assertThat(options.getUnparsedArgs()).isEqualTo(new String[] {"--unknown=foo"});
  }

  @Test
  public void testParse_withTestFilterFromEnv() throws Exception {
    Map<String, String> env = new HashMap<>();
    env.put("TESTBRIDGE_TEST_ONLY", "foo");
    JUnit4Options options = JUnit4Options.parse(env, ImmutableList.<String>of());
    assertThat(options.getTestIncludeFilter()).isEqualTo("foo");
    assertThat(options.getUnparsedArgs()).isEmpty();
  }

  @Test
  public void testParse_testFilterArgOverridesEnv() throws Exception {
    Map<String, String> env = new HashMap<>();
    env.put("TESTBRIDGE_TEST_ONLY", "foo");
    JUnit4Options options = JUnit4Options.parse(env, ImmutableList.of("--test_filter=bar"));
    assertThat(options.getTestIncludeFilter()).isEqualTo("bar");
    assertThat(options.getUnparsedArgs()).isEmpty();
  }
}
