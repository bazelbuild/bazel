// Copyright 2009-2015 Google Inc. All Rights Reserved.
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
package com.google.devtools.build.lib.syntax;

import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 * Tests for {@link GlobList}
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class GlobListTest {

  @Test
  public void testParse_glob() throws Exception {
    String expression = "glob(['abc'])";
    assertEquals(expression, GlobList.parse(expression).toExpression());
  }

  @Test
  public void testParse_multipleGlobs() throws Exception {
    String expression = "glob(['abc']) + glob(['def']) + glob(['ghi'])";
    assertEquals(expression, GlobList.parse(expression).toExpression());
  }

  @Test
  public void testParse_multipleLists() throws Exception {
    String expression = "['abc'] + ['def'] + ['ghi']";
    assertEquals(expression, GlobList.parse(expression).toExpression());
  }

  @Test
  public void testParse_complexExpression() throws Exception {
    String expression = "glob(['abc', 'def', 'ghi'], "
      + "exclude=['rst', 'uvw', 'xyz']) "
      + "+ glob(['abc', 'def', 'ghi'], exclude=['rst', 'uvw', 'xyz'])";
    assertEquals(expression, GlobList.parse(expression).toExpression());
  }

  @Test
  public void testConcat_GlobToGlob() throws Exception {
    GlobList<String> glob1 = GlobList.parse(
        "glob(['abc'], exclude=['def']) + glob(['xyz'])");
    GlobList<String> glob2 = GlobList.parse(
        "glob(['xyzzy']) + glob(['foo'], exclude=['bar'])");
    GlobList<String> cat = GlobList.concat(glob1, glob2);
    assertEquals(glob1.toExpression() + " + " + glob2.toExpression(), cat.toExpression());
  }

  @Test
  public void testConcat_GlobToList() throws Exception {
    GlobList<String> glob = GlobList.parse(
        "glob(['abc'], exclude=['def']) + glob(['xyz'])");
    List<String> list = ImmutableList.of("xyzzy", "foo", "bar");
    GlobList<String> cat = GlobList.concat(list, glob);
    assertEquals("['xyzzy', 'foo', 'bar'] + glob(['abc'], exclude=['def']) + glob(['xyz'])",
        cat.toExpression());
  }

  @Test
  public void testConcat_ListToGlob() throws Exception {
    GlobList<String> glob = GlobList.parse(
        "glob(['abc'], exclude=['def']) + glob(['xyz'])");
    List<String> list = ImmutableList.of("xyzzy", "foo", "bar");
    GlobList<String> cat = GlobList.concat(glob, list);
    assertEquals("glob(['abc'], exclude=['def']) + glob(['xyz']) + ['xyzzy', 'foo', 'bar']",
        cat.toExpression());
  }

  @Test
  public void testGetCriteria() throws Exception {
    List<String> include = ImmutableList.of("abc", "def", "ghi");
    List<String> exclude = ImmutableList.of("rst", "uvw", "xyz");
    List<String> matches = ImmutableList.of("xyzzy", "foo", "bar");
    GlobList<String> glob = GlobList.captureResults(include, exclude, matches);
    assertEquals(matches, glob);
    ImmutableList<GlobCriteria> criteria = glob.getCriteria();
    assertEquals(1, criteria.size());
    assertEquals(include, criteria.get(0).getIncludePatterns());
    assertEquals(exclude, criteria.get(0).getExcludePatterns());
  }
}
