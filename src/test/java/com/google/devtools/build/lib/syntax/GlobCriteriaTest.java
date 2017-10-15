// Copyright 2009 The Bazel Authors. All Rights Reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Links for {@link GlobCriteria}
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class GlobCriteriaTest {

  @Test
  public void testParse_EmptyList() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("[]");
    assertThat(gc.isGlob()).isFalse();
    assertThat(gc.getIncludePatterns()).isEmpty();
    assertThat(gc.getExcludePatterns()).isEmpty();
  }

  @Test
  public void testParse_SingleList() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("['abc']");
    assertThat(gc.isGlob()).isFalse();
    assertThat(gc.getIncludePatterns()).containsExactly("abc");
    assertThat(gc.getExcludePatterns()).isEmpty();
  }

  @Test
  public void testParse_MultipleList() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("['abc', 'def', 'ghi']");
    assertThat(gc.isGlob()).isFalse();
    assertThat(gc.getIncludePatterns()).containsExactly("abc", "def", "ghi").inOrder();
    assertThat(gc.getExcludePatterns()).isEmpty();
  }

  @Test
  public void testParse_EmptyGlob() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob([])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).isEmpty();
    assertThat(gc.getExcludePatterns()).isEmpty();
  }

  @Test
  public void testParse_SingleGlob() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob(['abc'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).containsExactly("abc");
    assertThat(gc.getExcludePatterns()).isEmpty();
  }

  @Test
  public void testParse_MultipleGlob() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob(['abc', 'def', 'ghi'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).containsExactly("abc", "def", "ghi").inOrder();
    assertThat(gc.getExcludePatterns()).isEmpty();
  }

  @Test
  public void testParse_EmptyGlobWithExclude() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob([], exclude=['xyz'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).isEmpty();
    assertThat(gc.getExcludePatterns()).containsExactly("xyz");
  }

  @Test
  public void testParse_SingleGlobWithExclude() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob(['abc'], exclude=['xyz'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).containsExactly("abc");
    assertThat(gc.getExcludePatterns()).containsExactly("xyz");
  }

  @Test
  public void testParse_MultipleGlobWithExclude() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob(['abc', 'def', 'ghi'], exclude=['xyz'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).containsExactly("abc", "def", "ghi").inOrder();
    assertThat(gc.getExcludePatterns()).containsExactly("xyz");
  }

  @Test
  public void testParse_MultipleGlobWithMultipleExclude() throws Exception {
    GlobCriteria gc = GlobCriteria.parse(
        "glob(['abc', 'def', 'ghi'], exclude=['rst', 'uvw', 'xyz'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).containsExactly("abc", "def", "ghi").inOrder();
    assertThat(gc.getExcludePatterns()).containsExactly("rst", "uvw", "xyz").inOrder();
  }

  @Test
  public void testParse_GlobWithSlashesAndWildcards() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob(['java/src/net/jsunit/*.java'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).containsExactly("java/src/net/jsunit/*.java");
    assertThat(gc.getExcludePatterns()).isEmpty();
  }

  @Test
  public void testParse_ExcludeWithInvalidLabel() throws Exception {
    GlobCriteria gc = GlobCriteria.parse("glob(['abc', 'def', 'ghi'], exclude=['xyz~'])");
    assertThat(gc.isGlob()).isTrue();
    assertThat(gc.getIncludePatterns()).containsExactly("abc", "def", "ghi").inOrder();
    assertThat(gc.getExcludePatterns()).containsExactly("xyz~");
  }

  @Test
  public void testParse_InvalidFormat_TooManySpacesList() throws Exception {
    try {
      GlobCriteria.parse("glob(['abc,  'def', 'ghi'], exclude=['xyz~'])");
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testParse_InvalidFormat_MissingQuote() throws Exception {
    try {
      GlobCriteria.parse("glob(['abc, 'def', 'ghi'], exclude=['xyz~'])");
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testParse_InvalidFormat_TooManySpacesExclude() throws Exception {
    try {
      GlobCriteria.parse("glob(['abc', 'def', 'ghi'],  exclude=['xyz~'])");
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testParse_InvalidFormat_MissingQuoteExclude() throws Exception {
    try {
      GlobCriteria.parse("glob(['abc, 'def', 'ghi'], exclude=['xyz~])");
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testParse_InvalidFormat_ExcludeWithList() throws Exception {
    try {
      GlobCriteria.parse("['abc, 'def', 'ghi'], exclude=['xyz~']");
      fail();
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testParse_veryLongString() throws Exception {
    StringBuilder builder = new StringBuilder();
    builder.append("['File0.java'");
    for (int i = 1; i < 5000; ++i) {
      builder.append(", 'File").append(i).append(".java'");
    }
    builder.append("]");
    String s = builder.toString();
    GlobCriteria gc = GlobCriteria.parse(s);
    assertThat(gc.toString()).isEqualTo(s);
  }
}
