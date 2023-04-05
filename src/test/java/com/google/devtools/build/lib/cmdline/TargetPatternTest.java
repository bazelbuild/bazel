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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.TargetPattern.InterpretPathAsTarget;
import com.google.devtools.build.lib.cmdline.TargetPattern.SingleTarget;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory.ContainsResult;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsInPackage;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.cmdline.TargetPattern}. */
@RunWith(JUnit4.class)
public class TargetPatternTest {
  private static Label label(String raw) {
    return Label.parseCanonicalUnchecked(raw);
  }

  private static PackageIdentifier pkg(String raw) {
    try {
      return PackageIdentifier.parse(raw);
    } catch (LabelSyntaxException e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void validPatterns_mainRepo_atRepoRoot() throws TargetParsingException {
    TargetPattern.Parser parser =
        new TargetPattern.Parser(
            PathFragment.EMPTY_FRAGMENT,
            RepositoryName.MAIN,
            RepositoryMapping.create(
                ImmutableMap.of("repo", RepositoryName.createUnvalidated("canonical_repo")),
                RepositoryName.MAIN));

    assertThat(parser.parse(":foo")).isEqualTo(new SingleTarget(":foo", label("@@//:foo")));
    assertThat(parser.parse("foo:bar"))
        .isEqualTo(new SingleTarget("foo:bar", label("@@//foo:bar")));
    assertThat(parser.parse("foo:all"))
        .isEqualTo(new TargetsInPackage("foo:all", pkg("@@//foo"), "all", false, true));
    assertThat(parser.parse("foo/...:all"))
        .isEqualTo(new TargetsBelowDirectory("foo/...:all", pkg("@@//foo"), true));
    assertThat(parser.parse("foo:*"))
        .isEqualTo(new TargetsInPackage("foo:*", pkg("@@//foo"), "*", false, false));
    assertThat(parser.parse("foo")).isEqualTo(new InterpretPathAsTarget("foo", "foo"));
    assertThat(parser.parse("...")).isEqualTo(new TargetsBelowDirectory("...", pkg("@@//"), true));
    assertThat(parser.parse("foo/bar")).isEqualTo(new InterpretPathAsTarget("foo/bar", "foo/bar"));

    assertThat(parser.parse("//foo")).isEqualTo(new SingleTarget("//foo", label("@@//foo:foo")));
    assertThat(parser.parse("//foo:bar"))
        .isEqualTo(new SingleTarget("//foo:bar", label("@@//foo:bar")));
    assertThat(parser.parse("//foo:all"))
        .isEqualTo(new TargetsInPackage("//foo:all", pkg("@@//foo"), "all", true, true));

    assertThat(parser.parse("//foo/all"))
        .isEqualTo(new SingleTarget("//foo/all", label("@@//foo/all:all")));
    assertThat(parser.parse("//foo/...:all"))
        .isEqualTo(new TargetsBelowDirectory("//foo/...:all", pkg("@@//foo"), true));
    assertThat(parser.parse("//..."))
        .isEqualTo(new TargetsBelowDirectory("//...", pkg("@@//"), true));

    assertThat(parser.parse("@repo"))
        .isEqualTo(new SingleTarget("@repo", label("@@canonical_repo//:repo")));
    assertThat(parser.parse("@repo//foo:bar"))
        .isEqualTo(new SingleTarget("@repo//foo:bar", label("@@canonical_repo//foo:bar")));
    assertThat(parser.parse("@repo//foo:all"))
        .isEqualTo(
            new TargetsInPackage(
                "@repo//foo:all", pkg("@@canonical_repo//foo"), "all", true, true));
    assertThat(parser.parse("@repo//:bar"))
        .isEqualTo(new SingleTarget("@repo//:bar", label("@@canonical_repo//:bar")));
    assertThat(parser.parse("@repo//..."))
        .isEqualTo(new TargetsBelowDirectory("@repo//...", pkg("@@canonical_repo//"), true));

    assertThat(parser.parse("@@repo"))
        .isEqualTo(new SingleTarget("@@repo", label("@@repo//:repo")));
    assertThat(parser.parse("@@repo//foo:all"))
        .isEqualTo(new TargetsInPackage("@@repo//foo:all", pkg("@@repo//foo"), "all", true, true));
    assertThat(parser.parse("@@repo//:bar"))
        .isEqualTo(new SingleTarget("@@repo//:bar", label("@@repo//:bar")));
  }

  @Test
  public void validPatterns_mainRepo_inSomeRelativeDirectory() throws TargetParsingException {
    TargetPattern.Parser parser =
        new TargetPattern.Parser(
            PathFragment.create("base"),
            RepositoryName.MAIN,
            RepositoryMapping.create(
                ImmutableMap.of("repo", RepositoryName.createUnvalidated("canonical_repo")),
                RepositoryName.MAIN));

    assertThat(parser.parse(":foo")).isEqualTo(new SingleTarget(":foo", label("@@//base:foo")));
    assertThat(parser.parse("foo:bar"))
        .isEqualTo(new SingleTarget("foo:bar", label("@@//base/foo:bar")));
    assertThat(parser.parse("foo:all"))
        .isEqualTo(new TargetsInPackage("foo:all", pkg("@@//base/foo"), "all", false, true));
    assertThat(parser.parse("foo/...:all"))
        .isEqualTo(new TargetsBelowDirectory("foo/...:all", pkg("@@//base/foo"), true));
    assertThat(parser.parse("foo:*"))
        .isEqualTo(new TargetsInPackage("foo:*", pkg("@@//base/foo"), "*", false, false));
    assertThat(parser.parse("foo")).isEqualTo(new InterpretPathAsTarget("foo", "base/foo"));
    assertThat(parser.parse("..."))
        .isEqualTo(new TargetsBelowDirectory("...", pkg("@@//base"), true));
    assertThat(parser.parse("foo/bar"))
        .isEqualTo(new InterpretPathAsTarget("foo/bar", "base/foo/bar"));

    assertThat(parser.parse("//foo")).isEqualTo(new SingleTarget("//foo", label("@@//foo:foo")));
    assertThat(parser.parse("//foo:bar"))
        .isEqualTo(new SingleTarget("//foo:bar", label("@@//foo:bar")));
    assertThat(parser.parse("//foo:all"))
        .isEqualTo(new TargetsInPackage("//foo:all", pkg("@@//foo"), "all", true, true));

    assertThat(parser.parse("//foo/all"))
        .isEqualTo(new SingleTarget("//foo/all", label("@@//foo/all:all")));
    assertThat(parser.parse("//foo/...:all"))
        .isEqualTo(new TargetsBelowDirectory("//foo/...:all", pkg("@@//foo"), true));
    assertThat(parser.parse("//..."))
        .isEqualTo(new TargetsBelowDirectory("//...", pkg("@@//"), true));

    assertThat(parser.parse("@repo"))
        .isEqualTo(new SingleTarget("@repo", label("@@canonical_repo//:repo")));
    assertThat(parser.parse("@repo//foo:bar"))
        .isEqualTo(new SingleTarget("@repo//foo:bar", label("@@canonical_repo//foo:bar")));
    assertThat(parser.parse("@repo//foo:all"))
        .isEqualTo(
            new TargetsInPackage(
                "@repo//foo:all", pkg("@@canonical_repo//foo"), "all", true, true));
    assertThat(parser.parse("@repo//:bar"))
        .isEqualTo(new SingleTarget("@repo//:bar", label("@@canonical_repo//:bar")));
    assertThat(parser.parse("@repo//..."))
        .isEqualTo(new TargetsBelowDirectory("@repo//...", pkg("@@canonical_repo//"), true));

    assertThat(parser.parse("@@repo"))
        .isEqualTo(new SingleTarget("@@repo", label("@@repo//:repo")));
    assertThat(parser.parse("@@repo//foo:all"))
        .isEqualTo(new TargetsInPackage("@@repo//foo:all", pkg("@@repo//foo"), "all", true, true));
    assertThat(parser.parse("@@repo//:bar"))
        .isEqualTo(new SingleTarget("@@repo//:bar", label("@@repo//:bar")));
  }

  @Test
  public void validPatterns_nonMainRepo() throws TargetParsingException {
    TargetPattern.Parser parser =
        new TargetPattern.Parser(
            PathFragment.EMPTY_FRAGMENT,
            RepositoryName.createUnvalidated("my_repo"),
            RepositoryMapping.create(
                ImmutableMap.of("repo", RepositoryName.createUnvalidated("canonical_repo")),
                RepositoryName.createUnvalidated("my_repo")));

    assertThat(parser.parse(":foo")).isEqualTo(new SingleTarget(":foo", label("@@my_repo//:foo")));
    assertThat(parser.parse("foo:bar"))
        .isEqualTo(new SingleTarget("foo:bar", label("@@my_repo//foo:bar")));
    assertThat(parser.parse("foo:all"))
        .isEqualTo(new TargetsInPackage("foo:all", pkg("@@my_repo//foo"), "all", false, true));
    assertThat(parser.parse("foo/...:all"))
        .isEqualTo(new TargetsBelowDirectory("foo/...:all", pkg("@@my_repo//foo"), true));
    assertThat(parser.parse("foo:*"))
        .isEqualTo(new TargetsInPackage("foo:*", pkg("@@my_repo//foo"), "*", false, false));
    assertThat(parser.parse("foo")).isEqualTo(new SingleTarget("foo", label("@@my_repo//:foo")));
    assertThat(parser.parse("..."))
        .isEqualTo(new TargetsBelowDirectory("...", pkg("@@my_repo//"), true));
    assertThat(parser.parse("foo/bar"))
        .isEqualTo(new SingleTarget("foo/bar", label("@@my_repo//:foo/bar")));

    assertThat(parser.parse("//foo"))
        .isEqualTo(new SingleTarget("//foo", label("@@my_repo//foo:foo")));
    assertThat(parser.parse("//foo:bar"))
        .isEqualTo(new SingleTarget("//foo:bar", label("@@my_repo//foo:bar")));
    assertThat(parser.parse("//foo:all"))
        .isEqualTo(new TargetsInPackage("//foo:all", pkg("@@my_repo//foo"), "all", true, true));

    assertThat(parser.parse("//foo/all"))
        .isEqualTo(new SingleTarget("//foo/all", label("@@my_repo//foo/all:all")));
    assertThat(parser.parse("//foo/...:all"))
        .isEqualTo(new TargetsBelowDirectory("//foo/...:all", pkg("@@my_repo//foo"), true));
    assertThat(parser.parse("//..."))
        .isEqualTo(new TargetsBelowDirectory("//...", pkg("@@my_repo//"), true));

    assertThat(parser.parse("@repo"))
        .isEqualTo(new SingleTarget("@repo", label("@@canonical_repo//:repo")));
    assertThat(parser.parse("@repo//foo:bar"))
        .isEqualTo(new SingleTarget("@repo//foo:bar", label("@@canonical_repo//foo:bar")));
    assertThat(parser.parse("@repo//foo:all"))
        .isEqualTo(
            new TargetsInPackage(
                "@repo//foo:all", pkg("@@canonical_repo//foo"), "all", true, true));
    assertThat(parser.parse("@repo//:bar"))
        .isEqualTo(new SingleTarget("@repo//:bar", label("@@canonical_repo//:bar")));
    assertThat(parser.parse("@repo//..."))
        .isEqualTo(new TargetsBelowDirectory("@repo//...", pkg("@@canonical_repo//"), true));

    assertThat(parser.parse("@@repo"))
        .isEqualTo(new SingleTarget("@@repo", label("@@repo//:repo")));
    assertThat(parser.parse("@@repo//foo:all"))
        .isEqualTo(new TargetsInPackage("@@repo//foo:all", pkg("@@repo//foo"), "all", true, true));
    assertThat(parser.parse("@@repo//:bar"))
        .isEqualTo(new SingleTarget("@@repo//:bar", label("@@repo//:bar")));
  }

  @Test
  public void invalidPatterns() throws Exception {
    ImmutableList<String> badPatterns =
        ImmutableList.of("//Bar\\java", "", "/foo", "///foo", "@", "@foo//", "@@");
    ImmutableMap<String, RepositoryName> repoMappingEntries =
        ImmutableMap.of("repo", RepositoryName.createUnvalidated("canonical_repo"));
    for (TargetPattern.Parser parser :
        ImmutableList.of(
            new TargetPattern.Parser(
                PathFragment.EMPTY_FRAGMENT,
                RepositoryName.MAIN,
                RepositoryMapping.create(repoMappingEntries, RepositoryName.MAIN)),
            new TargetPattern.Parser(
                PathFragment.create("base"),
                RepositoryName.MAIN,
                RepositoryMapping.create(repoMappingEntries, RepositoryName.MAIN)),
            new TargetPattern.Parser(
                PathFragment.EMPTY_FRAGMENT,
                RepositoryName.create("my_repo"),
                RepositoryMapping.create(repoMappingEntries, RepositoryName.create("my_repo"))))) {
      for (String pattern : badPatterns) {
        try {
          TargetPattern parsed = parser.parse(pattern);
          fail(
              String.format(
                  "parsing should have failed for pattern \"%s\" with parser in repo %s at"
                      + " relative directory [%s], but succeeded with the result:\n%s",
                  pattern, parser.getCurrentRepo(), parser.getRelativeDirectory(), parsed));
        } catch (TargetParsingException expected) {
        }
      }
    }
  }

  @Test
  public void invalidParser_nonMainRepo_nonEmptyRelativeDirectory() throws Exception {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new TargetPattern.Parser(
                PathFragment.create("base"),
                RepositoryName.create("my_repo"),
                RepositoryMapping.ALWAYS_FALLBACK));
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

  private static TargetsBelowDirectory parseAsTBD(String pattern) throws TargetParsingException {
    TargetPattern parsedPattern = TargetPattern.defaultParser().parse(pattern);
    assertThat(parsedPattern.getType()).isEqualTo(TargetPattern.Type.TARGETS_BELOW_DIRECTORY);
    assertThat(parsedPattern.getOriginalPattern()).isEqualTo(pattern);
    return (TargetsBelowDirectory) parsedPattern;
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
}
