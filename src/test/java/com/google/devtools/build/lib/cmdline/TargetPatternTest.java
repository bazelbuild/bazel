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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.cmdline.TargetPattern.ContainsTBDForTBDResult;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.cmdline.TargetPattern}. */
@RunWith(JUnit4.class)
public class TargetPatternTest {
  private void expectError(String pattern) {
    assertThrows(TargetParsingException.class, () -> parse(pattern));
  }

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
    expectError("Bar\\java");
    expectError("");
    expectError("\\");
  }

  @Test
  public void testNormalize() {
    // Good cases.
    assertThat(TargetPattern.normalize("empty")).isEqualTo("empty");
    assertThat(TargetPattern.normalize("a/b")).isEqualTo("a/b");
    assertThat(TargetPattern.normalize("a/b/c")).isEqualTo("a/b/c");
    assertThat(TargetPattern.normalize("a/b/c.d")).isEqualTo("a/b/c.d");
    assertThat(TargetPattern.normalize("a/b/c..")).isEqualTo("a/b/c..");
    assertThat(TargetPattern.normalize("a/b/c...")).isEqualTo("a/b/c...");

    assertThat(TargetPattern.normalize("a/b/")).isEqualTo("a/b"); // Remove trailing empty segments
    assertThat(TargetPattern.normalize("a//c")).isEqualTo("a/c"); // Remove empty inner segments
    assertThat(TargetPattern.normalize("a/./d")).isEqualTo("a/d"); // Remove inner dot segments
    assertThat(TargetPattern.normalize("a/.")).isEqualTo("a"); // Remove trailing dot segments
    // Remove .. segment and its predecessor
    assertThat(TargetPattern.normalize("a/b/../e")).isEqualTo("a/e");
    // Remove trailing .. segment and its predecessor
    assertThat(TargetPattern.normalize("a/g/b/..")).isEqualTo("a/g");
    // Remove double .. segments and two predecessors
    assertThat(TargetPattern.normalize("a/b/c/../../h")).isEqualTo("a/h");
    // Don't remove leading .. segments
    assertThat(TargetPattern.normalize("../a")).isEqualTo("../a");
    assertThat(TargetPattern.normalize("../../a")).isEqualTo("../../a");
    assertThat(TargetPattern.normalize("../../../a")).isEqualTo("../../../a");
    assertThat(TargetPattern.normalize("a/../../../b")).isEqualTo("../../b");
  }

  @Test
  public void testTargetsBelowDirectoryContainsColonStar() throws Exception {
    // Given an outer pattern '//foo/...', that matches rules only,
    TargetPattern outerPattern =
        parseAsExpectedType("//foo/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // And a nested inner pattern '//foo/bar/...:*', that matches all targets,
    TargetPattern innerPattern =
        parseAsExpectedType("//foo/bar/...:*", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // Then a directory exclusion would exactly describe the subtraction of the inner pattern from
    // the outer pattern,
    assertThat(outerPattern.containsTBDForTBD(innerPattern))
        .isEqualTo(ContainsTBDForTBDResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT);
    // And the inner pattern does not contain the outer pattern.
    assertThat(innerPattern.containsTBDForTBD(outerPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
  }

  @Test
  public void testTargetsBelowDirectoryColonStarContains() throws Exception {
    // Given an outer pattern '//foo/...:*', that matches all targets,
    TargetPattern outerPattern =
        parseAsExpectedType("//foo/...:*", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // And a nested inner pattern '//foo/bar/...', that matches rules only,
    TargetPattern innerPattern =
        parseAsExpectedType("//foo/bar/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // Then a directory exclusion would be too broad,
    assertThat(outerPattern.containsTBDForTBD(innerPattern))
        .isEqualTo(ContainsTBDForTBDResult.DIRECTORY_EXCLUSION_WOULD_BE_TOO_BROAD);
    // And the inner pattern does not contain the outer pattern.
    assertThat(innerPattern.containsTBDForTBD(outerPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
  }

  @Test
  public void testTargetsBelowDirectoryContainsNestedPatterns() throws Exception {
    // Given an outer pattern '//foo/...',
    TargetPattern outerPattern =
        parseAsExpectedType("//foo/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // And a nested inner pattern '//foo/bar/...',
    TargetPattern innerPattern =
        parseAsExpectedType("//foo/bar/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // Then the outer pattern contains the inner pattern,
    assertThat(outerPattern.containsTBDForTBD(innerPattern))
        .isEqualTo(ContainsTBDForTBDResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT);
    // And the inner pattern does not contain the outer pattern.
    assertThat(innerPattern.containsTBDForTBD(outerPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
  }

  @Test
  public void testTargetsBelowDirectoryIsExcludableFromForIndependentPatterns() throws Exception {
    // Given a pattern '//foo/...',
    TargetPattern patternFoo =
        parseAsExpectedType("//foo/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // And a pattern '//bar/...',
    TargetPattern patternBar =
        parseAsExpectedType("//bar/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    // Then neither pattern contains the other.
    assertThat(patternFoo.containsTBDForTBD(patternBar)).isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(patternBar.containsTBDForTBD(patternFoo)).isEqualTo(ContainsTBDForTBDResult.OTHER);
  }

  @Test
  public void testTargetsBelowDirectoryContainsForOtherPatternTypes() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdFoo of '//foo/...',
    TargetPattern tbdFoo =
        parseAsExpectedType("//foo/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);

    // And target patterns of each type other than TargetsBelowDirectory, e.g. 'foo/bar',
    // '//foo:bar', and 'foo:all',
    TargetPattern pathAsTargetPattern =
        parseAsExpectedType("foo/bar", TargetPattern.Type.PATH_AS_TARGET);
    TargetPattern singleTargetPattern =
        parseAsExpectedType("//foo:bar", TargetPattern.Type.SINGLE_TARGET);
    TargetPattern targetsInPackagePattern =
        parseAsExpectedType("foo:all", TargetPattern.Type.TARGETS_IN_PACKAGE);

    // Then the non-TargetsBelowDirectory patterns do not contain tbdFoo.
    assertThat(pathAsTargetPattern.containsTBDForTBD(tbdFoo))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    // And are not considered to be a contained directory of the TargetsBelowDirectory pattern.
    assertThat(tbdFoo.containsTBDForTBD(pathAsTargetPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);

    assertThat(singleTargetPattern.containsTBDForTBD(tbdFoo))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(tbdFoo.containsTBDForTBD(singleTargetPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);

    assertThat(targetsInPackagePattern.containsTBDForTBD(tbdFoo))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(tbdFoo.containsTBDForTBD(targetsInPackagePattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
  }

  @Test
  public void testTargetsBelowDirectoryDoesNotContainCoincidentPrefixPatterns() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdFoo of '//foo/...',
    TargetPattern tbdFoo =
        parseAsExpectedType("//foo/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);

    // And target patterns with prefixes equal to the directory of the TBD pattern, but not below
    // it,
    TargetPattern targetsBelowDirectoryPattern =
        parseAsExpectedType("//food/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    TargetPattern pathAsTargetPattern =
        parseAsExpectedType("food/bar", TargetPattern.Type.PATH_AS_TARGET);
    TargetPattern singleTargetPattern =
        parseAsExpectedType("//food:bar", TargetPattern.Type.SINGLE_TARGET);
    TargetPattern targetsInPackagePattern =
        parseAsExpectedType("food:all", TargetPattern.Type.TARGETS_IN_PACKAGE);

    // Then the non-TargetsBelowDirectory patterns are not contained by tbdFoo.
    assertThat(tbdFoo.containsTBDForTBD(targetsBelowDirectoryPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(tbdFoo.containsTBDForTBD(pathAsTargetPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(tbdFoo.containsTBDForTBD(singleTargetPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(tbdFoo.containsTBDForTBD(targetsInPackagePattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
  }

  @Test
  public void testDepotRootTargetsBelowDirectoryContainsPatterns() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdDepot of '//...',
    TargetPattern tbdDepot =
        parseAsExpectedType("//...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);

    // And target patterns of each type other than TargetsBelowDirectory, e.g. 'foo/bar',
    // '//foo:bar', and 'foo:all',
    TargetPattern tbdFoo =
        parseAsExpectedType("//foo/...", TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    TargetPattern pathAsTargetPattern =
        parseAsExpectedType("foo/bar", TargetPattern.Type.PATH_AS_TARGET);
    TargetPattern singleTargetPattern =
        parseAsExpectedType("//foo:bar", TargetPattern.Type.SINGLE_TARGET);
    TargetPattern targetsInPackagePattern =
        parseAsExpectedType("foo:all", TargetPattern.Type.TARGETS_IN_PACKAGE);

    // Then the patterns are contained by tbdDepot, and do not contain tbdDepot.
    assertThat(tbdDepot.containsTBDForTBD(tbdFoo))
        .isEqualTo(ContainsTBDForTBDResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT);
    assertThat(tbdFoo.containsTBDForTBD(tbdDepot)).isEqualTo(ContainsTBDForTBDResult.OTHER);

    assertThat(tbdDepot.containsTBDForTBD(pathAsTargetPattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(pathAsTargetPattern.containsTBDForTBD(tbdDepot))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);

    assertThat(tbdDepot.containsTBDForTBD(singleTargetPattern)).
        isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(singleTargetPattern.containsTBDForTBD(tbdDepot))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);

    assertThat(tbdDepot.containsTBDForTBD(targetsInPackagePattern))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
    assertThat(targetsInPackagePattern.containsTBDForTBD(tbdDepot))
        .isEqualTo(ContainsTBDForTBDResult.OTHER);
  }

  private static TargetPattern parse(String pattern) throws TargetParsingException {
    return TargetPattern.defaultParser().parse(pattern);
  }

  private static TargetPattern parseAsExpectedType(String pattern, TargetPattern.Type expectedType)
      throws TargetParsingException {
    TargetPattern parsedPattern = parse(pattern);
    assertThat(parsedPattern.getType()).isEqualTo(expectedType);
    assertThat(parsedPattern.getOriginalPattern()).isEqualTo(pattern);
    return parsedPattern;
  }
}
