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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory.ContainsResult;
import java.util.Map;
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
    TargetsBelowDirectory outerPattern = parseAsTBD("//foo/...");
    // And a nested inner pattern '//foo/bar/...:*', that matches all targets,
    TargetsBelowDirectory innerPattern = parseAsTBD("//foo/bar/...:*");
    // Then a directory exclusion would exactly describe the subtraction of the inner pattern from
    // the outer pattern,
    assertThat(outerPattern.contains(innerPattern))
        .isEqualTo(ContainsResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT);
    // And the inner pattern does not contain the outer pattern.
    assertThat(innerPattern.contains(outerPattern)).isEqualTo(ContainsResult.NOT_CONTAINED);
  }

  @Test
  public void testTargetsBelowDirectoryColonStarContains() throws Exception {
    // Given an outer pattern '//foo/...:*', that matches all targets,
    TargetsBelowDirectory outerPattern = parseAsTBD("//foo/...:*");
    // And a nested inner pattern '//foo/bar/...', that matches rules only,
    TargetsBelowDirectory innerPattern = parseAsTBD("//foo/bar/...");
    // Then a directory exclusion would be too broad,
    assertThat(outerPattern.contains(innerPattern))
        .isEqualTo(ContainsResult.DIRECTORY_EXCLUSION_WOULD_BE_TOO_BROAD);
    // And the inner pattern does not contain the outer pattern.
    assertThat(innerPattern.contains(outerPattern)).isEqualTo(ContainsResult.NOT_CONTAINED);
  }

  @Test
  public void testTargetsBelowDirectoryContainsNestedPatterns() throws Exception {
    // Given an outer pattern '//foo/...',
    TargetsBelowDirectory outerPattern = parseAsTBD("//foo/...");
    // And a nested inner pattern '//foo/bar/...',
    TargetsBelowDirectory innerPattern = parseAsTBD("//foo/bar/...");
    // Then the outer pattern contains the inner pattern,
    assertThat(outerPattern.contains(innerPattern))
        .isEqualTo(ContainsResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT);
    // And the inner pattern does not contain the outer pattern.
    assertThat(innerPattern.contains(outerPattern)).isEqualTo(ContainsResult.NOT_CONTAINED);
  }

  @Test
  public void testTargetsBelowDirectoryIsExcludableFromForIndependentPatterns() throws Exception {
    // Given a pattern '//foo/...',
    TargetsBelowDirectory patternFoo = parseAsTBD("//foo/...");
    // And a pattern '//bar/...',
    TargetsBelowDirectory patternBar = parseAsTBD("//bar/...");
    // Then neither pattern contains the other.
    assertThat(patternFoo.contains(patternBar)).isEqualTo(ContainsResult.NOT_CONTAINED);
    assertThat(patternBar.contains(patternFoo)).isEqualTo(ContainsResult.NOT_CONTAINED);
  }

  @Test
  public void testTargetsBelowDirectoryDoesNotContainCoincidentPrefixPatterns() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdFoo of '//foo/...',
    TargetsBelowDirectory tbdFoo = parseAsTBD("//foo/...");

    // And a target pattern with prefix equal to the directory of the TBD pattern, but not below it,
    TargetsBelowDirectory targetsBelowDirectoryPattern = parseAsTBD("//food/...");

    // Then it is not contained in the first pattern.
    assertThat(tbdFoo.contains(targetsBelowDirectoryPattern))
        .isEqualTo(ContainsResult.NOT_CONTAINED);
  }

  @Test
  public void testDepotRootTargetsBelowDirectoryContainsPatterns() throws Exception {
    // Given a TargetsBelowDirectory pattern, tbdDepot of '//...',
    TargetsBelowDirectory tbdDepot = parseAsTBD("//...");

    // And a target pattern for a directory,
    TargetsBelowDirectory tbdFoo = parseAsTBD("//foo/...");

    // Then the pattern is contained by tbdDepot, and does not contain tbdDepot.
    assertThat(tbdDepot.contains(tbdFoo))
        .isEqualTo(ContainsResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT);
    assertThat(tbdFoo.contains(tbdDepot)).isEqualTo(ContainsResult.NOT_CONTAINED);
  }

  @Test
  public void testRenameRepository() throws Exception {
    Map<RepositoryName, RepositoryName> renaming =
        ImmutableMap.of(
            RepositoryName.create("@foo"), RepositoryName.create("@bar"),
            RepositoryName.create("@myworkspace"), RepositoryName.create("@"));

    // Expecting renaming
    assertThat(TargetPattern.renameRepository("@foo//package:target", renaming))
        .isEqualTo("@bar//package:target");
    assertThat(TargetPattern.renameRepository("@myworkspace//package:target", renaming))
        .isEqualTo("@//package:target");
    assertThat(TargetPattern.renameRepository("@foo//foo/...", renaming))
        .isEqualTo("@bar//foo/...");
    assertThat(TargetPattern.renameRepository("@myworkspace//foo/...", renaming))
        .isEqualTo("@//foo/...");

    // No renaming should occur
    assertThat(TargetPattern.renameRepository("@//package:target", renaming))
        .isEqualTo("@//package:target");
    assertThat(TargetPattern.renameRepository("@unrelated//package:target", renaming))
        .isEqualTo("@unrelated//package:target");
    assertThat(TargetPattern.renameRepository("foo/package:target", renaming))
        .isEqualTo("foo/package:target");
    assertThat(TargetPattern.renameRepository("foo/...", renaming)).isEqualTo("foo/...");
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

  private static TargetsBelowDirectory parseAsTBD(String pattern) throws TargetParsingException {
    return (TargetsBelowDirectory)
        parseAsExpectedType(pattern, TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
  }
}
