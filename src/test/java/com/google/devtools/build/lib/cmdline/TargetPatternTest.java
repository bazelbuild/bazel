// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;


import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.cmdline.TargetPattern.Type;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link TargetPattern}.
 */
@RunWith(JUnit4.class)
public class TargetPatternTest {

  @Test
  public void testPassingValidations() throws TargetParsingException {
    parse("foo:bar");
    parse("foo:all");
    parse("foo/...:all");
    parse("foo:*");

    parse("//foo");
    parse("//foo:bar");
    parse("//foo:all");

    parse("//foo/all");
    parse("java/com/google/foo/Bar.java");
    parse("//foo/...:all");

    parse("//...");
    parse("@repo//foo:bar");
    parse("@repo//foo:all");
    parse("@repo//:bar");
  }

  @Test
  public void testInvalidPatterns() throws TargetParsingException {
    try {
      parse("Bar&&&java");
      fail();
    } catch (TargetParsingException expected) {
    }
  }

  @Test
  public void testNormalize() {
    // Good cases.
    assertEquals("empty", TargetPattern.normalize("empty"));
    assertEquals("a/b", TargetPattern.normalize("a/b"));
    assertEquals("a/b/c", TargetPattern.normalize("a/b/c"));
    assertEquals("a/b/c.d", TargetPattern.normalize("a/b/c.d"));
    assertEquals("a/b/c..", TargetPattern.normalize("a/b/c.."));
    assertEquals("a/b/c...", TargetPattern.normalize("a/b/c..."));

    assertEquals("a/b", TargetPattern.normalize("a/b/")); // Remove trailing empty segments
    assertEquals("a/c", TargetPattern.normalize("a//c")); // Remove empty inner segments
    assertEquals("a/d", TargetPattern.normalize("a/./d")); // Remove inner dot segments
    assertEquals("a", TargetPattern.normalize("a/."));     // Remove trailing dot segments
    // Remove .. segment and its predecessor
    assertEquals("a/e", TargetPattern.normalize("a/b/../e"));
    // Remove trailing .. segment and its predecessor
    assertEquals("a/g", TargetPattern.normalize("a/g/b/.."));
    // Remove double .. segments and two predecessors
    assertEquals("a/h", TargetPattern.normalize("a/b/c/../../h"));
    // Don't remove leading .. segments
    assertEquals("../a", TargetPattern.normalize("../a"));
    assertEquals("../../a", TargetPattern.normalize("../../a"));
    assertEquals("../../../a", TargetPattern.normalize("../../../a"));
    assertEquals("../../b", TargetPattern.normalize("a/../../../b"));
  }

  @Test
  public void testTargetsBelowDirectoryContainsNestedPatterns() throws Exception {
    // Given an outer pattern '//foo/...',
    TargetPattern outerPattern = parseAsExpectedType("//foo/...", Type.TARGETS_BELOW_DIRECTORY);
    // And a nested inner pattern '//foo/bar/...',
    TargetPattern innerPattern = parseAsExpectedType("//foo/bar/...", Type.TARGETS_BELOW_DIRECTORY);
    // Then the outer pattern contains the inner pattern,,
    assertTrue(outerPattern.containsDirectoryOfTBDForTBD(innerPattern));
    // And the inner pattern does not contain the outer pattern.
    assertFalse(innerPattern.containsDirectoryOfTBDForTBD(outerPattern));
  }

  @Test
  public void testTargetsBelowDirectoryIsExcludableFromForIndependentPatterns() throws Exception {
    // Given a pattern '//foo/...',
    TargetPattern patternFoo = parseAsExpectedType("//foo/...", Type.TARGETS_BELOW_DIRECTORY);
    // And a pattern '//bar/...',
    TargetPattern patternBar = parseAsExpectedType("//bar/...", Type.TARGETS_BELOW_DIRECTORY);
    // Then neither pattern contains the other.
    assertFalse(patternFoo.containsDirectoryOfTBDForTBD(patternBar));
    assertFalse(patternBar.containsDirectoryOfTBDForTBD(patternFoo));
  }

  @Test
  public void testTargetsBelowDirectoryContainsForOtherPatternTypes() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdFoo of '//foo/...',
    TargetPattern tbdFoo = parseAsExpectedType("//foo/...", Type.TARGETS_BELOW_DIRECTORY);

    // And target patterns of each type other than TargetsBelowDirectory, e.g. 'foo/bar',
    // '//foo:bar', and 'foo:all',
    TargetPattern pathAsTargetPattern = parseAsExpectedType("foo/bar", Type.PATH_AS_TARGET);
    TargetPattern singleTargetPattern = parseAsExpectedType("//foo:bar", Type.SINGLE_TARGET);
    TargetPattern targetsInPackagePattern = parseAsExpectedType("foo:all", Type.TARGETS_IN_PACKAGE);

    // Then the non-TargetsBelowDirectory patterns do not contain tbdFoo.
    assertFalse(pathAsTargetPattern.containsDirectoryOfTBDForTBD(tbdFoo));
    // And are not considered to be a contained directory of the TargetsBelowDirectory pattern.
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(pathAsTargetPattern));

    assertFalse(singleTargetPattern.containsDirectoryOfTBDForTBD(tbdFoo));
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(singleTargetPattern));

    assertFalse(targetsInPackagePattern.containsDirectoryOfTBDForTBD(tbdFoo));
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(targetsInPackagePattern));
  }

  @Test
  public void testTargetsBelowDirectoryDoesNotContainCoincidentPrefixPatterns() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdFoo of '//foo/...',
    TargetPattern tbdFoo = parseAsExpectedType("//foo/...", Type.TARGETS_BELOW_DIRECTORY);

    // And target patterns with prefixes equal to the directory of the TBD pattern, but not below
    // it,
    TargetPattern targetsBelowDirectoryPattern =
        parseAsExpectedType("//food/...", Type.TARGETS_BELOW_DIRECTORY);
    TargetPattern pathAsTargetPattern = parseAsExpectedType("food/bar", Type.PATH_AS_TARGET);
    TargetPattern singleTargetPattern = parseAsExpectedType("//food:bar", Type.SINGLE_TARGET);
    TargetPattern targetsInPackagePattern =
        parseAsExpectedType("food:all", Type.TARGETS_IN_PACKAGE);

    // Then the non-TargetsBelowDirectory patterns are not contained by tbdFoo.
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(targetsBelowDirectoryPattern));
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(pathAsTargetPattern));
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(singleTargetPattern));
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(targetsInPackagePattern));
  }

  @Test
  public void testDepotRootTargetsBelowDirectoryContainsPatterns() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdDepot of '//...',
    TargetPattern tbdDepot = parseAsExpectedType("//...", Type.TARGETS_BELOW_DIRECTORY);

    // And target patterns of each type other than TargetsBelowDirectory, e.g. 'foo/bar',
    // '//foo:bar', and 'foo:all',
    TargetPattern tbdFoo = parseAsExpectedType("//foo/...", Type.TARGETS_BELOW_DIRECTORY);
    TargetPattern pathAsTargetPattern = parseAsExpectedType("foo/bar", Type.PATH_AS_TARGET);
    TargetPattern singleTargetPattern = parseAsExpectedType("//foo:bar", Type.SINGLE_TARGET);
    TargetPattern targetsInPackagePattern = parseAsExpectedType("foo:all", Type.TARGETS_IN_PACKAGE);

    // Then the patterns are contained by tbdDepot, and do not contain tbdDepot.
    assertTrue(tbdDepot.containsDirectoryOfTBDForTBD(tbdFoo));
    assertFalse(tbdFoo.containsDirectoryOfTBDForTBD(tbdDepot));

    assertFalse(tbdDepot.containsDirectoryOfTBDForTBD(pathAsTargetPattern));
    assertFalse(pathAsTargetPattern.containsDirectoryOfTBDForTBD(tbdDepot));

    assertFalse(tbdDepot.containsDirectoryOfTBDForTBD(singleTargetPattern));
    assertFalse(singleTargetPattern.containsDirectoryOfTBDForTBD(tbdDepot));

    assertFalse(tbdDepot.containsDirectoryOfTBDForTBD(targetsInPackagePattern));
    assertFalse(targetsInPackagePattern.containsDirectoryOfTBDForTBD(tbdDepot));
  }

  private static TargetPattern parse(String pattern) throws TargetParsingException {
    return TargetPattern.defaultParser().parse(pattern);
  }

  private static TargetPattern parseAsExpectedType(String pattern, Type expectedType)
      throws TargetParsingException {
    TargetPattern parsedPattern = parse(pattern);
    assertThat(parsedPattern.getType()).isEqualTo(expectedType);
    assertThat(parsedPattern.getOriginalPattern()).isEqualTo(pattern);
    return parsedPattern;
  }
}
