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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link JUnit4Options}
 */
@RunWith(JUnit4.class)
public class JUnit4OptionsTest {

  private static final Map<String, String> EMPTY_ENV = Collections.emptyMap();

  @Test
  public void testParse_noArgs() throws Exception {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.<String>of());
    assertNull(options.getTestIncludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }

  @Test
  public void testParse_onlyUnparsedArgs() {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--bar", "baz"));
    assertNull(options.getTestIncludeFilter());
    assertArrayEquals(new String[] {"--bar", "baz"}, options.getUnparsedArgs());
  }

  @Test
  public void testParse_withTwoArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--test_filter", "foo"));
    assertEquals("foo", options.getTestIncludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }

  @Test
  public void testParse_withOneArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_filter=foo"));
    assertEquals("foo", options.getTestIncludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }

  @Test
  public void testParse_testFilterAndUnparsedArgs() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--bar", "--test_filter=foo", "--baz"));
    assertEquals("foo", options.getTestIncludeFilter());
    assertArrayEquals(new String[] {"--bar", "--baz"}, options.getUnparsedArgs());
  }

  @Test
  public void testParse_testLastTestFilterWins() throws Exception {
    JUnit4Options options =
        JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_filter=foo", "--test_filter=bar"));
    assertEquals("bar", options.getTestIncludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }

  @Test
  public void testParse_testFilterMissingSecondArg() throws Exception {
    try {
      JUnit4Options.parse(EMPTY_ENV, ImmutableList.of("--test_filter"));
      fail();
    } catch (RuntimeException e) {
      // expected
    }
  }

  @Test
  public void testParse_testFilterExcludeWithTwoArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--test_exclude_filter", "foo"));
    assertEquals("foo", options.getTestExcludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }

  @Test
  public void testParse_testFilterExcludewithOneArgTestFilter() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--test_exclude_filter=foo"));
    assertEquals("foo", options.getTestExcludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }

  @Test
  public void testParse_unknownOptionName() throws Exception {
    JUnit4Options options = JUnit4Options.parse(
        EMPTY_ENV, ImmutableList.of("--unknown=foo"));
    assertArrayEquals(new String[] {"--unknown=foo"}, options.getUnparsedArgs());
  }

  @Test
  public void testParse_withTestFilterFromEnv() throws Exception {
    Map<String, String> env = new HashMap<>();
    env.put("TESTBRIDGE_TEST_ONLY", "foo");
    JUnit4Options options = JUnit4Options.parse(env, ImmutableList.<String>of());
    assertEquals("foo", options.getTestIncludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }

  @Test
  public void testParse_testFilterArgOverridesEnv() throws Exception {
    Map<String, String> env = new HashMap<>();
    env.put("TESTBRIDGE_TEST_ONLY", "foo");
    JUnit4Options options = JUnit4Options.parse(env, ImmutableList.of("--test_filter=bar"));
    assertEquals("bar", options.getTestIncludeFilter());
    assertEquals(0, options.getUnparsedArgs().length);
  }
}
