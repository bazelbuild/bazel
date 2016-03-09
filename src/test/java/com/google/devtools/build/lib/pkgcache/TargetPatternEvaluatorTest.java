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
package com.google.devtools.build.lib.pkgcache;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.pkgcache.FilteringPolicies.FILTER_TESTS;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Set;

/** Tests for {@link TargetPatternEvaluator}. */
@RunWith(JUnit4.class)
public class TargetPatternEvaluatorTest extends AbstractTargetPatternEvaluatorTest {
  private PathFragment fooOffset;

  private Set<Label> rulesBeneathFoo;
  private Set<Label> rulesBeneathFooBar;
  private Set<Label> rulesBeneathOtherrules;
  private Set<Label> rulesInFoo;
  private Set<Label> rulesInFooBar;
  private Set<Label> rulesInOtherrules;
  private Set<Label> targetsInFoo;
  private Set<Label> targetsInFooBar;
  private Set<Label> targetsBeneathFoo;
  private Set<Label> targetsInOtherrules;

  @Before
  public final void createFiles() throws Exception {
    // TODO(ulfjack): Also disable the implicit C++ outputs in Google's internal version.
    boolean hasImplicitCcOutputs = ruleClassProvider.getRuleClassMap().get("cc_library")
        .getImplicitOutputsFunction() != ImplicitOutputsFunction.NONE;

    scratch.file("foo/BUILD",
        "cc_library(name = 'foo1', srcs = [ 'foo1.cc' ], hdrs = [ 'foo1.h' ])",
        "exports_files(['baz/bang'])");
    scratch.file("foo/bar/BUILD",
        "cc_library(name = 'bar1', alwayslink = 1)",
        "cc_library(name = 'bar2')",
        "exports_files(['wiz/bang', 'wiz/all', 'baz', 'baz/bang', 'undeclared.h'])");

    // 'filegroup' and 'test_suite' are rules, but 'exports_files' is not.
    scratch.file("otherrules/BUILD",
        "test_suite(name = 'suite1')",
        "filegroup(name='group', srcs=['suite/somefile'])",
        "exports_files(['suite/somefile'])",
        "cc_library(name = 'wiz', linkstatic = 1)");
    scratch.file("nosuchpkg/subdir/empty", "");

    Path foo = scratch.dir("foo");
    fooOffset = foo.relativeTo(rootDirectory);

    rulesBeneathFoo = labels("//foo:foo1", "//foo/bar:bar1", "//foo/bar:bar2");
    rulesBeneathFooBar = labels("//foo/bar:bar1", "//foo/bar:bar2");
    rulesBeneathOtherrules = labels(
        "//otherrules:suite1", "//otherrules:wiz", "//otherrules:group");
    rulesInFoo = labels("//foo:foo1");
    rulesInFooBar = labels("//foo/bar:bar1", "//foo/bar:bar2");
    rulesInOtherrules = rulesBeneathOtherrules;

    targetsInFoo = labels(
        "//foo:foo1",
        "//foo:foo1",
        "//foo:foo1.cc",
        "//foo:foo1.h",
        "//foo:BUILD",
        "//foo:baz/bang");
    if (hasImplicitCcOutputs) {
      targetsInFoo.addAll(labels("//foo:libfoo1.a", "//foo:libfoo1.so"));
    }
    targetsInFooBar = labels(
        "//foo/bar:bar1",
        "//foo/bar:bar2",
        "//foo/bar:BUILD",
        "//foo/bar:wiz/bang",
        "//foo/bar:wiz/all",
        "//foo/bar:baz",
        "//foo/bar:baz/bang",
        "//foo/bar:undeclared.h");
    if (hasImplicitCcOutputs) {
      targetsInFooBar.addAll(labels("//foo/bar:libbar1.lo", "//foo/bar:libbar2.a"));
    }
    targetsBeneathFoo = Sets.newHashSet();
    targetsBeneathFoo.addAll(targetsInFoo);
    targetsBeneathFoo.addAll(targetsInFooBar);

    targetsInOtherrules = labels(
        "//otherrules:group",
        "//otherrules:wiz",
        "//otherrules:suite1",
        "//otherrules:BUILD",
        "//otherrules:suite/somefile",
        "//otherrules:wiz",
        "//otherrules:suite1");
    if (hasImplicitCcOutputs) {
      targetsInOtherrules.addAll(labels("//otherrules:libwiz.a"));
    }
  }

  private void invalidate(String file) throws InterruptedException {
    skyframeExecutor.invalidateFilesUnderPathForTesting(reporter,
        ModifiedFileSet.builder().modify(new PathFragment(file)).build(), rootDirectory);
  }

  private void invalidate(ModifiedFileSet modifiedFileSet) throws InterruptedException {
    skyframeExecutor.invalidateFilesUnderPathForTesting(reporter, modifiedFileSet, rootDirectory);
  }

  private void setDeletedPackages(Set<PackageIdentifier> deletedPackages) {
    skyframeExecutor.setDeletedPackages(deletedPackages);
  }

  private TargetPatternEvaluator shiftOffset() {
    parser.updateOffset(fooOffset);
    return parser;
  }

  private Set<Label> parseList(String... patterns)
      throws TargetParsingException, InterruptedException {
    return targetsToLabels(
        getFailFast(parseTargetPatternList(parser, parsingListener, Arrays.asList(patterns),
            false)));
  }

  private Set<Label> parseListKeepGoingExpectFailure(String... patterns)
      throws TargetParsingException, InterruptedException {
    ResolvedTargets<Target> result =
        parseTargetPatternList(parser, parsingListener, Arrays.asList(patterns), true);
    assertTrue(result.hasError());
    return targetsToLabels(result.getTargets());
  }

  private Set<Label> parseList(
      FilteringPolicy policy, String... patterns)
      throws TargetParsingException, InterruptedException {
    return targetsToLabels(getFailFast(
        parseTargetPatternList(parser, parsingListener, Arrays.asList(patterns), policy, false)));
  }

  private Set<Label> parseListRelative(String... patterns)
      throws TargetParsingException, InterruptedException {
    return targetsToLabels(getFailFast(parseTargetPatternList(
        shiftOffset(), parsingListener, Arrays.asList(patterns), false)));
  }

  private static Set<Target> getFailFast(ResolvedTargets<Target> result) {
    assertFalse(result.hasError());
    return result.getTargets();
  }

  private void expectError(TargetPatternEvaluator parser, String expectedError,
      String target) throws InterruptedException {
    try {
      parser.parseTargetPattern(parsingListener, target, false);
      fail("target='" + target + "', expected error: " + expectedError);
    } catch (TargetParsingException e) {
      assertThat(e.getMessage()).contains(expectedError);
    }
  }

  private void expectError(String expectedError, String target) throws InterruptedException {
    expectError(parser, expectedError, target);
  }

  private void expectErrorRelative(String expectedError, String target)
      throws InterruptedException {
    expectError(shiftOffset(), expectedError, target);
  }

  private Label parseIndividualTarget(String targetLabel) throws Exception {
    return Iterables.getOnlyElement(
        getFailFast(parser.parseTargetPattern(parsingListener, targetLabel, false))).getLabel();
  }

  private Label parseIndividualTargetRelative(String targetLabel) throws Exception {
    return Iterables.getOnlyElement(
        getFailFast(
            shiftOffset().parseTargetPattern(parsingListener, targetLabel, false))).getLabel();
  }

  @Test
  public void testParsingStandardLabel() throws Exception {
    assertEquals("//foo:foo1",
        parseIndividualTarget("//foo:foo1").toString());
  }

  @Test
  public void testAbsolutePatternEndsWithSlashAll() throws Exception {
    scratch.file("foo/all/BUILD", "cc_library(name = 'all')");
    assertEquals("//foo/all:all", parseIndividualTarget("//foo/all").toString());
    assertNoEvents();
  }

  @Test
  public void testWildcardConflict() throws Exception {
    scratch.file("foo/lib/BUILD",
        "cc_library(name = 'lib1')",
        "cc_library(name = 'lib2')",
        "cc_library(name = 'all-targets')",
        "cc_library(name = 'all')");

    assertWildcardConflict("//foo/lib:all", ":all");
    eventCollector.clear();
    assertWildcardConflict("//foo/lib:all-targets", ":all-targets");
  }

  private void assertWildcardConflict(String label, String suffix) throws Exception {
    assertEquals(label, parseIndividualTarget(label).toString());
    assertSame(1, eventCollector.count());
    assertContainsEvent(String.format("The target pattern '%s' is ambiguous: '%s' is both "
        + "a wildcard, and the name of an existing cc_library rule; "
        + "using the latter interpretation", label, suffix));
  }

  @Test
  public void testMissingPackage() throws Exception {
    try {
      parseIndividualTarget("//missing:foo1");
      fail("TargetParsingException expected");
    } catch (TargetParsingException e) {
      assertThat(e.getMessage()).startsWith("no such package");
    }
  }

  @Test
  public void testParsingStandardLabelWithRelativeParser() throws Exception {
    assertEquals("//foo:foo1", parseIndividualTargetRelative("//foo:foo1").toString());
  }

  @Test
  public void testMissingLabel() throws Exception {
    try {
      parseIndividualTarget("//foo:missing");
      fail("TargetParsingException expected");
    } catch (TargetParsingException e) {
      assertThat(e.getMessage()).startsWith("no such target");
    }
  }

  @Test
  public void testParsingStandardLabelShorthand() throws Exception {
    assertEquals("//foo:foo1",
                 parseIndividualTarget("foo:foo1").toString());
  }

  @Test
  public void testParsingStandardLabelShorthandRelative() throws Exception {
    assertEquals("//foo:foo1", parseIndividualTargetRelative(":foo1").toString());
  }

  @Test
  public void testSingleSlashPatternCantBeParsed() throws Exception {
    expectError("not a valid absolute pattern (absolute target patterns must start with exactly "
        + "two slashes): '/single/slash'",
        "/single/slash");
  }

  @Test
  public void testTripleSlashPatternCantBeParsed() throws Exception {
    expectError("not a valid absolute pattern (absolute target patterns must start with exactly "
        + "two slashes): '///triple/slash'",
        "///triple/slash");
  }

  @Test
  public void testSingleSlashPatternCantBeParsedWithRelativeParser() throws Exception {
    expectErrorRelative("not a valid absolute pattern (absolute target patterns must start with "
        + "exactly two slashes): '/single/slash'",
        "/single/slash");
  }

  @Test
  public void testUnsupportedTargets() throws Exception {
    String expectedError = "no such target '//foo:foo': target 'foo' not declared in package 'foo'"
        + " defined by /workspace/foo/BUILD";
    expectError(expectedError, "foo");
    expectError("The package part of 'foo/' should not end in a slash", "foo/");
  }

  @Test
  public void testModifiedBuildFile() throws Exception {
    assertThat(parseList("foo:all")).containsExactlyElementsIn(rulesInFoo);
    assertNoEvents();

    scratch.overwriteFile("foo/BUILD",
        "cc_library(name = 'foo1', srcs = [ 'foo1.cc' ], hdrs = [ 'foo1.h' ])",
        "cc_library(name = 'foo2', srcs = [ 'foo1.cc' ], hdrs = [ 'foo1.h' ])");
    invalidate("foo/BUILD");
    assertThat(parseList("foo:all")).containsExactlyElementsIn(labels("//foo:foo1", "//foo:foo2"));
  }

  @Test
  public void testParserOffsetUpdated() throws Exception {
    scratch.file("nest/BUILD",
        "cc_library(name = 'nested1', srcs = [ ])");
    scratch.file("nest/nest/BUILD",
        "cc_library(name = 'nested2', srcs = [ ])");

    updateOffset(new PathFragment("nest"));
    assertThat(parseList(":all")).containsExactlyElementsIn(labels("//nest:nested1"));
    updateOffset(new PathFragment("nest/nest"));
    assertThat(parseList(":all")).containsExactlyElementsIn(labels("//nest/nest:nested2"));
  }

  protected void updateOffset(PathFragment rel) {
    parser.updateOffset(rel);
  }

  private void runFindTargetsInPackage(String suffix) throws Exception {
    // 'my_package:all'
    assertThat(parseList("foo" + suffix)).containsExactlyElementsIn(rulesInFoo);
    assertThat(parseList("foo/bar" + suffix)).containsExactlyElementsIn(rulesInFooBar);
    assertThat(parseList("otherrules" + suffix)).containsExactlyElementsIn(rulesInOtherrules);
    assertNoEvents();
    String msg1 = "while parsing 'nosuchpkg" + suffix + "': no such package 'nosuchpkg': "
        + "BUILD file not found on package path";
    expectError(msg1, "nosuchpkg" + suffix);

    String msg2 = "while parsing 'nosuchdirectory" + suffix
        + "': no such package 'nosuchdirectory': "
        + "BUILD file not found on package path";
    expectError(msg2, "nosuchdirectory" + suffix);
    assertThat(parsingListener.events).containsExactly(Pair.of("nosuchpkg" + suffix, msg1),
        Pair.of("nosuchdirectory" + suffix, msg2));
  }

  private void runFindTargetsInPackageAbsolute(String suffix) throws Exception {
    // '//my_package:all'
    assertThat(parseList("//foo" + suffix)).containsExactlyElementsIn(rulesInFoo);
    assertThat(parseList("//foo/bar" + suffix)).containsExactlyElementsIn(rulesInFooBar);
    assertThat(parseList("//otherrules" + suffix)).containsExactlyElementsIn(rulesInOtherrules);
    assertNoEvents();
    expectError("while parsing 'nosuchpkg" + suffix + "': no such package 'nosuchpkg': "
            + "BUILD file not found on package path",
        "nosuchpkg" + suffix);
    expectError("while parsing '//nosuchpkg" + suffix + "': no such package 'nosuchpkg': "
            + "BUILD file not found on package path",
        "//nosuchpkg" + suffix);
  }

  @Test
  public void testFindRulesInPackage() throws Exception {
    runFindTargetsInPackage(":all");
    runFindTargetsInPackageAbsolute(":all");
  }

  private void runFindRulesRecursively(String suffix) throws Exception {
    assertThat(parseList("foo" + suffix)).containsExactlyElementsIn(rulesBeneathFoo);
    assertThat(parseList("//foo" + suffix)).containsExactlyElementsIn(rulesBeneathFoo);
    assertThat(parseList("//foo/bar" + suffix)).containsExactlyElementsIn(rulesBeneathFooBar);
    assertThat(parseList("//foo" + suffix)).containsExactlyElementsIn(rulesBeneathFoo);
    assertThat(parseList("otherrules" + suffix)).containsExactlyElementsIn(rulesBeneathOtherrules);
    assertThat(parseList("//foo" + suffix)).containsExactlyElementsIn(rulesBeneathFoo);
    assertNoEvents();
    eventCollector.clear();
  }

  @Test
  public void testNoTargetsFoundRecursiveDirectory() throws Exception {
    try {
      parseList("nosuchpkg/...");
      fail();
    } catch (TargetParsingException e) {
      assertThat(e).hasMessage("no targets found beneath 'nosuchpkg'");
    }
  }

  @Test
  public void testFindRulesRecursively() throws Exception {
    runFindRulesRecursively("/...:all");
    runFindRulesRecursively("/...");
  }

  private void runFindAllRules(String pattern) throws Exception {
    assertThat(parseList(pattern))
        .containsExactlyElementsIn(Sets.union(rulesBeneathFoo, rulesBeneathOtherrules));
    assertNoEvents();
    eventCollector.clear();
  }

  @Test
  public void testFindAllRules() throws Exception {
    runFindAllRules("//...:all");
    runFindAllRules("//...");
    runFindAllRules("...");
  }

  private void runFindAllTargets(String pattern) throws Exception {
    assertThat(parseList(pattern))
        .containsExactlyElementsIn(Sets.union(targetsBeneathFoo, targetsInOtherrules));
    assertNoEvents();
    eventCollector.clear();
  }

  @Test
  public void testFindAllTargets() throws Exception {
    runFindAllTargets("//...:all-targets");
    runFindAllTargets("//...:*");
    runFindAllTargets("...:*");
  }

  @Test
  public void testFindAllRulesRecursivelyWithExperimental() throws Exception {
    scratch.file("experimental/BUILD",
        "cc_library(name = 'experimental', srcs = [ 'experimental.cc' ])");
    assertThat(parseList("//..."))
        .containsExactlyElementsIn(Sets.union(rulesBeneathFoo, rulesBeneathOtherrules));
    assertNoEvents();
  }

  @Test
  public void testFindAllRulesRecursivelyExperimental() throws Exception {
    scratch.file("experimental/BUILD",
        "cc_library(name = 'experimental', srcs = [ 'experimental.cc' ])");
    assertThat(parseList("//experimental/..."))
        .containsExactlyElementsIn(labels("//experimental:experimental"));
    assertNoEvents();
  }

  @Test
  public void testDefaultPackage() throws Exception {
    scratch.file("experimental/BUILD",
                "cc_library(name = 'experimental', srcs = [ 'experimental.cc' ])");
    assertEquals("//experimental:experimental", parseIndividualTarget("//experimental").toString());
    assertEquals("//experimental:experimental", parseIndividualTarget("experimental").toString());
    assertNoEvents();
  }

  /**
   * Test that the relative path label parsing behaves as stated in the target-syntax documentation.
   */
  @Test
  public void testRelativePathLabel() throws Exception {
    scratch.file("sub/BUILD", "exports_files(['dir2/dir2'])");
    scratch.file("sub/dir/BUILD", "exports_files(['dir2'])");
    scratch.file("sub/dir/dir/BUILD", "exports_files(['dir'])");
    // sub/dir/dir is a package
    assertEquals("//sub/dir/dir:dir", parseIndividualTarget("sub/dir/dir").toString());
    // sub/dir is a package but not sub/dir/dir2
    assertEquals("//sub/dir:dir2", parseIndividualTarget("sub/dir/dir2").toString());
    // sub is a package but not sub/dir2
    assertEquals("//sub:dir2/dir2", parseIndividualTarget("sub/dir2/dir2").toString());
  }

  @Test
  public void testFindsLongestPlausiblePackageName() throws Exception {
    assertEquals("//foo/bar:baz",
                 parseIndividualTarget("foo/bar/baz").toString());
    assertEquals("//foo/bar:baz/bang",
                 parseIndividualTarget("foo/bar/baz/bang").toString());
    assertEquals("//foo:baz/bang",
        parseIndividualTarget("foo/baz/bang").toString());
  }

  @Test
  public void testGivesUpIfPackageDoesNotExist() throws Exception {
    expectError("couldn't determine target from filename 'does/not/exist'",
        "does/not/exist");
  }

  @Test
  public void testParsesIterableOfLabels() throws Exception {
    Set<Label> labels = Sets.newHashSet(Label.parseAbsolute("//foo/bar:bar1"),
        Label.parseAbsolute("//foo:foo1"));
    assertEquals(labels, parseList("//foo/bar:bar1", "//foo:foo1"));
    parsingListener.assertEmpty();
  }

  @Test
  public void testParseAbsoluteWithRelativeParser() throws Exception {
    Set<Label> labels = Sets.newHashSet(Label.parseAbsolute("//foo/bar:bar1"),
        Label.parseAbsolute("//foo:foo1"));
    assertEquals(labels, parseListRelative("//foo/bar:bar1", "//foo:foo1"));
    parsingListener.assertEmpty();
  }

  @Test
  public void testMultisegmentLabelsWithNoSlashSlash() throws Exception {
    assertEquals("//foo/bar:wiz/bang",
        parseIndividualTarget("foo/bar:wiz/bang").toString());
    assertEquals("//foo/bar:wiz/all",
        parseIndividualTarget("foo/bar:wiz/all").toString());
  }

  @Test
  public void testMultisegmentLabelsWithNoSlashSlashRelative() throws Exception {
    assertEquals("//foo/bar:wiz/bang",
        parseIndividualTargetRelative("bar:wiz/bang").toString());
    assertEquals("//foo/bar:wiz/all",
        parseIndividualTargetRelative("bar:wiz/all").toString());
  }

  @Test
  public void testAll() throws Exception {
    expectError("couldn't determine target from filename 'all'", "all");
  }

  /** Regression test for a bug. */
  @Test
  public void testDotDotDotDoesntMatchDeletedPackages() throws Exception {
    scratch.file("x/y/BUILD", "cc_library(name='y')");
    scratch.file("x/z/BUILD", "cc_library(name='z')");
    setDeletedPackages(Sets.newHashSet(PackageIdentifier.createInDefaultRepo("x/y")));
    assertEquals(Sets.newHashSet(Label.parseAbsolute("//x/z")),
        parseList("x/..."));
  }

  @Test
  public void testDotDotDotDoesntMatchDeletedPackagesRelative() throws Exception {
    scratch.file("x/y/BUILD", "cc_library(name='y')");
    scratch.file("x/z/BUILD", "cc_library(name='z')");
    setDeletedPackages(Sets.newHashSet(PackageIdentifier.createInDefaultRepo("x/y")));

    parser.updateOffset(new PathFragment("x"));
    assertEquals(Sets.newHashSet(Label.parseAbsolute("//x/z")),
        targetsToLabels(getFailFast(parser.parseTargetPattern(parsingListener, "...", false))));
  }

  @Test
  public void testDeletedPackagesIncrementality() throws Exception {
    scratch.file("x/y/BUILD", "cc_library(name='y')");
    scratch.file("x/z/BUILD", "cc_library(name='z')");

    assertEquals(Sets.newHashSet(Label.parseAbsolute("//x/y"), Label.parseAbsolute("//x/z")),
        parseList("x/..."));

    setDeletedPackages(Sets.newHashSet(PackageIdentifier.createInDefaultRepo("x/y")));
    assertEquals(Sets.newHashSet(Label.parseAbsolute("//x/z")), parseList("x/..."));

    setDeletedPackages(ImmutableSet.<PackageIdentifier>of());
    assertEquals(Sets.newHashSet(Label.parseAbsolute("//x/y"), Label.parseAbsolute("//x/z")),
        parseList("x/..."));
  }

  @Test
  public void testSequenceOfTargetPatterns_Union() throws Exception {
    // No prefix negation operator => union.  Order is not significant.
    assertThat(parseList("foo/...", "foo/bar/...")).containsExactlyElementsIn(rulesBeneathFoo);
    assertThat(parseList("foo/bar/...", "foo/...")).containsExactlyElementsIn(rulesBeneathFoo);
  }

  @Test
  public void testSequenceOfTargetPatterns_UnionRelative() throws Exception {
    // No prefix negation operator => union.  Order is not significant.
    assertThat(parseListRelative("...", "bar/...")).containsExactlyElementsIn(rulesBeneathFoo);
    assertThat(parseListRelative("bar/...", "...")).containsExactlyElementsIn(rulesBeneathFoo);
  }

  @Test
  public void testSequenceOfTargetPatterns_SetDifference() throws Exception {
    // Prefix negation operator => set difference.  Order is significant.
    assertThat(parseList("foo/...", "-foo/bar/...")).containsExactlyElementsIn(rulesInFoo);
    assertThat(parseList("-foo/bar/...", "foo/...")).containsExactlyElementsIn(rulesBeneathFoo);
  }

  @Test
  public void testSequenceOfTargetPatterns_SetDifferenceRelative() throws Exception {
    // Prefix negation operator => set difference.  Order is significant.
    assertThat(parseListRelative("...", "-bar/...")).containsExactlyElementsIn(rulesInFoo);
    assertThat(parseListRelative("-bar/...", "...")).containsExactlyElementsIn(rulesBeneathFoo);
  }

  @Test
  public void testAllTargetsWildcard() throws Exception {
    assertThat(parseList("foo:all-targets")).containsExactlyElementsIn(targetsInFoo);
    assertThat(parseList("foo/bar:all-targets")).containsExactlyElementsIn(targetsInFooBar);
    assertThat(parseList("otherrules:all-targets")).containsExactlyElementsIn(targetsInOtherrules);
    assertThat(parseList("foo/...:all-targets")).containsExactlyElementsIn(targetsBeneathFoo);

    assertThat(parseList("foo:*")).containsExactlyElementsIn(targetsInFoo);
    assertThat(parseList("foo/bar:*")).containsExactlyElementsIn(targetsInFooBar);
    assertThat(parseList("otherrules:*")).containsExactlyElementsIn(targetsInOtherrules);
    assertThat(parseList("foo/...:*")).containsExactlyElementsIn(targetsBeneathFoo);
  }

  @Test
  public void testAllTargetsWildcardRelative() throws Exception {
    assertThat(parseListRelative(":all-targets")).containsExactlyElementsIn(targetsInFoo);
    assertThat(parseListRelative("//foo:all-targets")).containsExactlyElementsIn(targetsInFoo);
    assertThat(parseListRelative("bar:all-targets")).containsExactlyElementsIn(targetsInFooBar);
    assertThat(parseListRelative("//foo/bar:all-targets"))
        .containsExactlyElementsIn(targetsInFooBar);
    assertThat(parseListRelative("...:all-targets")).containsExactlyElementsIn(targetsBeneathFoo);
    assertThat(parseListRelative("//foo/...:all-targets"))
        .containsExactlyElementsIn(targetsBeneathFoo);

    assertThat(parseListRelative(":*")).containsExactlyElementsIn(targetsInFoo);
    assertThat(parseListRelative("//foo:*")).containsExactlyElementsIn(targetsInFoo);
    assertThat(parseListRelative("bar:*")).containsExactlyElementsIn(targetsInFooBar);
    assertThat(parseListRelative("//foo/bar:*")).containsExactlyElementsIn(targetsInFooBar);
    assertThat(parseListRelative("...:*")).containsExactlyElementsIn(targetsBeneathFoo);
    assertThat(parseListRelative("//foo/...:*")).containsExactlyElementsIn(targetsBeneathFoo);
  }

  @Test
  public void testFactoryMethod() throws Exception {
    Path workspace = scratch.dir("/client/workspace");
    Path underWorkspace = scratch.dir("/client/workspace/foo");
    Path notUnderWorkspace = scratch.dir("/client/otherclient");

    updateOffset(workspace, underWorkspace);
    updateOffset(workspace, workspace);

    // The client must be equal to or underneath the workspace.
    try {
      updateOffset(workspace, notUnderWorkspace);
      fail("Should have failed because client was not underneath the workspace");
    } catch (IllegalArgumentException expected) {
    }
  }

  private void updateOffset(Path workspace, Path workingDir) {
    parser.updateOffset(workingDir.relativeTo(workspace));
  }

  private void setupSubDirectoryCircularSymlink() throws Exception {
    Path parent = scratch.file("parent/BUILD", "sh_library(name = 'parent')").getParentDirectory();
    Path child = parent.getRelative("child");
    child.createDirectory();
    Path badBuild = child.getRelative("BUILD");
    badBuild.createSymbolicLink(badBuild);
    reporter.removeHandler(failFastHandler);
  }

  @Test
  public void testSubdirectoryCircularSymlinkKeepGoing() throws Exception {
    setupSubDirectoryCircularSymlink();
    assertThat(parseListKeepGoing("//parent/...").getFirst())
        .containsExactlyElementsIn(labels("//parent:parent"));
  }

  @Test
  public void testSubdirectoryCircularSymlinkNoKeepGoing() throws Exception {
    setupSubDirectoryCircularSymlink();
    try {
      parseList("//parent/...");
      fail();
    } catch (TargetParsingException e) {
      // Expected.
    }
  }

  /** Regression test for bug: "Bogus 'helpful' error message" */
  @Test
  public void testHelpfulMessageForFileOutsideOfAnyPackage() throws Exception {
    scratch.file("goo/wiz/file");
    expectError("couldn't determine target from filename 'goo/wiz/file'",
                "goo/wiz/file");
    expectError("couldn't determine target from filename 'goo/wiz'",
        "goo/wiz");
  }

  /** Regression test for bug: "Bogus 'helpful' error message" */
  @Test
  public void testHelpfulMessageForDirectoryWhichIsASubdirectoryOfAPackage() throws Exception {
    scratch.file("bar/BUILD");
    scratch.file("bar/quux/somefile");
    expectError("no such target '//bar:quux': target 'quux' not declared in package 'bar'; "
            + "however, a source directory of this name exists.  (Perhaps add "
            + "'exports_files([\"quux\"])' to bar/BUILD, or define a filegroup?) defined by "
            + "/workspace/bar/BUILD",
        "bar/quux");
  }

  /** Regression test for bug: "Uplevel references in blaze target patterns cause crash" */
  @Test
  public void testNoCrashWhenUplevelReferencesUsed() throws Exception {
    scratch.file("/other/workspace/project/BUILD");
    expectError(
        "Invalid package name '../other/workspace/project': ",
        "../other/workspace/project/...:all");
    expectError(
        "Invalid package name '../other/workspace/project': ", "../other/workspace/project/...");
    expectError(
        "Invalid package name 'foo/../../other/workspace/project': ",
        "foo/../../other/workspace/project/...");
    expectError(
        "Invalid package name '../other/workspace/project': ", "../other/workspace/project:all");
  }

  @Test
  public void testPassingValidations() {
    expectValidationPass("foo:bar");
    expectValidationPass("foo:all");
    expectValidationPass("foo/...:all");
    expectValidationPass("foo:*");

    expectValidationPass("//foo");
    expectValidationPass("foo");
    expectValidationPass("foo/bar");
    expectValidationPass("//foo:bar");
    expectValidationPass("//foo:all");

    expectValidationPass("//foo/all");
    expectValidationPass("java/com/google/foo/Bar.java");
    expectValidationPass("//foo/...:all");
  }

  @Test
  public void testFailingValidations() {
    expectValidationFail("");
    expectValidationFail("\\");
    expectValidationFail("foo:**");
    expectValidationFail("//foo/*");
  }

  private void expectValidationFail(String target) {
    try {
      TargetPattern.defaultParser().parse(target);
      fail("TargetParsingException expected from parse(" + target + ")");
    } catch (TargetParsingException expected) {
      /* ignore */
    }

    // Ensure that validateTargetPattern's checking is strictly weaker than
    // that of parseTargetPattern.
    try {
      parser.parseTargetPattern(parsingListener, target, false);
      fail("parseTargetPattern(" + target + ") inconsistent with parseTargetPattern!");
    } catch (TargetParsingException expected) {
      /* ignore */
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private void expectValidationPass(String target) {
    try {
      TargetPattern.defaultParser().parse(target);
    } catch (TargetParsingException e) {
      fail("Expected " + target + " to pass; got exception: " + e);
    }
  }

  @Test
  public void testSetOffset() throws Exception {
    assertEquals("//foo:foo1", parseIndividualTarget("foo:foo1").toString());

    parser.updateOffset(new PathFragment("foo"));
    assertEquals("//foo:foo1", parseIndividualTarget(":foo1").toString());
  }

  @Test
  public void testTestTargetParsing() throws Exception {
    scratch.file("test/BUILD",
        "cc_library(name = 'bar1', alwayslink = 1)",
        "cc_library(name = 'bar2')",
        "cc_test(name = 'test1', deps = ['bar1'], tags = ['local'])",
        "cc_test(name = 'test2', deps = ['bar2'], tags = ['local'])",
        "py_test(name = 'manual_test', tags = ['exclusive', 'manual'], srcs=['py_test.py'])",
        "test_suite(name = 'suite1')");

    Set<Label> testRules = labels("//test:test1", "//test:test2");
    Set<Label> allTestRules =
      labels("//test:test1", "//test:test2", "//test:manual_test");
    assertThat(parseList(FILTER_TESTS, "test/...")).containsExactlyElementsIn(testRules);
    assertThat(parseList(FILTER_TESTS, "test:all")).containsExactlyElementsIn(testRules);
    assertThat(parseList(FILTER_TESTS, "test:*")).containsExactlyElementsIn(testRules);
    assertThat(parseList(FILTER_TESTS, "test:test1", "test/test2", "//test:suite1"))
        .containsExactlyElementsIn(testRules);
    assertThat(parseList(FILTER_TESTS, "test:all", "//test:manual_test"))
        .containsExactlyElementsIn(allTestRules);
    assertThat(parseList(FILTER_TESTS, "test:all", "test/manual_test"))
        .containsExactlyElementsIn(allTestRules);
  }

  /** Regression test for bug: "blaze test "no targets found" warning now fatal" */
  @Test
  public void testNoTestsInRecursivePattern() throws Exception {
    assertThat(parseList(FILTER_TESTS, "foo/..."))
        .containsExactlyElementsIn(labels()); // doesn't throw
  }

  @Test
  public void testKeepGoingBadPackage() throws Exception {
    assertKeepGoing(rulesBeneathFoo,
        "Skipping '//missing_pkg': no such package 'missing_pkg': "
            + "BUILD file not found on package path",
        "//missing_pkg", "foo/...");
  }

  @Test
  public void testKeepGoingPartiallyBadPackage() throws Exception {
    scratch.file("x/y/BUILD",
        "filegroup(name = 'a')",
        "BROKEN",
        "filegroup(name = 'b')");

    reporter.removeHandler(failFastHandler);
    Pair<Set<Label>, Boolean> result = parseListKeepGoing("//x/...");

    assertContainsEvent("name 'BROKEN' is not defined");
    assertThat(result.first)
        .containsExactlyElementsIn(
            Sets.newHashSet(Label.parseAbsolute("//x/y:a"), Label.parseAbsolute("//x/y:b")));
    assertFalse(result.second);
  }

  @Test
  public void testKeepGoingMissingRecursiveDirectory() throws Exception {
    assertKeepGoing(rulesBeneathFoo,
        "Skipping 'nosuchpkg/...': no targets found beneath 'nosuchpkg'",
        "nosuchpkg/...", "foo/...");
    eventCollector.clear();
    assertKeepGoing(rulesBeneathFoo,
        "Skipping 'nosuchdirectory/...': no targets found beneath 'nosuchdirectory'",
        "nosuchdirectory/...", "foo/...");
  }

  @Test
  public void testKeepGoingMissingTarget() throws Exception {
    assertKeepGoing(rulesBeneathFoo,
        "Skipping '//otherrules:missing_target': no such target "
            + "'//otherrules:missing_target': target 'missing_target' not declared in "
            + "package 'otherrules'",
        "//otherrules:missing_target", "foo/...");
  }

  @Test
  public void testKeepGoingOnAllRulesBeneath() throws Exception {
    scratch.file("foo/bar/bad/BUILD", "invalid build file");

    reporter.removeHandler(failFastHandler);
    Pair<Set<Label>, Boolean> result = parseListKeepGoing("foo/...");
    assertThat(result.first).containsExactlyElementsIn(rulesBeneathFoo);
    assertContainsEvent("syntax error at 'build'");
    assertContainsEvent("package contains errors");
    reporter.addHandler(failFastHandler);

    // Even though there was a loading error in the package, parsing the target pattern was
    // successful.
    assertFalse(result.second);
  }

  @Test
  public void testKeepGoingBadFilenameTarget() throws Exception {
    assertKeepGoing(rulesBeneathFoo,
        "couldn't determine target from filename 'bad/filename/target'",
        "bad/filename/target", "foo/...");
  }

  @Test
  public void testMoreThanOneBadPatternFailFast() throws Exception {
    try {
      parseTargetPatternList(parser, parsingListener,
          ImmutableList.of("bad/filename/target", "other/bad/filename/target"),
          /*keepGoing=*/false);
      fail();
    } catch (TargetParsingException e) {
      assertThat(e.getMessage()).contains("couldn't determine target from filename");
    }
  }

  @Test
  public void testMentioningBuildFile() throws Exception {
    ResolvedTargets<Target> result = parseTargetPatternList(parser, parsingListener,
        Arrays.asList("//foo/bar/BUILD"), false);

    assertFalse(result.hasError());
    assertThat(result.getTargets()).hasSize(1);

    Label label = Iterables.getOnlyElement(result.getTargets()).getLabel();
    assertEquals("BUILD", label.getName());
    assertEquals("foo/bar", label.getPackageName());

  }

  /**
   * Regression test for bug: '"Target pattern parsing failed. Continuing anyway" appears, even
   * without --keep_going'
   */
  @Test
  public void testLoadingErrorsAreNotParsingErrors() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("loading/BUILD",
        "cc_library(name='y', deps=['a'])",
        "cc_library(name='a', deps=['b'])",
        "cc_library(name='b', deps=['c'])",
        "genrule(name='c', outs=['c.out'])");

    Pair<Set<Label>, Boolean> result = parseListKeepGoing("//loading:y");
    assertEquals(Label.parseAbsolute("//loading:y"), Iterables.getOnlyElement(result.first));
    assertContainsEvent("missing value for mandatory attribute");
    assertFalse(result.second);
  }

  private void assertKeepGoing(Set<Label> expectedLabels, String expectedEvent, String... toParse)
      throws Exception {
    reporter.removeHandler(failFastHandler);
    assertThat(parseListKeepGoingExpectFailure(toParse)).containsExactlyElementsIn(expectedLabels);
    assertContainsEvent(expectedEvent);
    reporter.addHandler(failFastHandler);
  }

  /** Regression test for bug: "IllegalStateException in BuildTool.prepareToBuild()" */
  @Test
  public void testTestingIsSubset() throws Exception {
    scratch.file("test/BUILD",
        "cc_library(name = 'bar1')",
        "cc_test(name = 'test', deps = [':bar1'], tags = ['manual'])");

    assertThat(parseList(FILTER_TESTS, "//test:test", "-//test:all"))
        .containsExactlyElementsIn(labels());
  }

  @Test
  public void testAddedPkg() throws Exception {
    invalidate(ModifiedFileSet.EVERYTHING_MODIFIED);
    scratch.dir("h/i/j/k/BUILD");
    scratch.file("h/BUILD", "sh_library(name='h')");
    assertThat(parseList("//h/...")).containsExactlyElementsIn(labels("//h"));

    scratch.file("h/i/j/BUILD", "sh_library(name='j')");

    // Modifications not yet known.
    assertThat(parseList("//h/...")).containsExactlyElementsIn(labels("//h"));

    ModifiedFileSet modifiedFileSet = ModifiedFileSet.builder()
        .modify(new PathFragment("h/i/j/BUILD")).build();
    invalidate(modifiedFileSet);

    assertThat(parseList("//h/...")).containsExactly(Label.parseAbsolute("//h/i/j:j"),
        Label.parseAbsolute("//h"));
  }

  @Test
  public void testAddedFilesAndDotDotDot() throws Exception {
    invalidate(ModifiedFileSet.EVERYTHING_MODIFIED);
    reporter.removeHandler(failFastHandler);
    scratch.dir("h");
    try {
      parseList("//h/...");
      fail("TargetParsingException expected");
    } catch (TargetParsingException e) {
      // expected
    }

    scratch.file("h/i/j/k/BUILD", "sh_library(name='l')");
    ModifiedFileSet modifiedFileSet = ModifiedFileSet.builder()
        .modify(new PathFragment("h"))
        .modify(new PathFragment("h/i"))
        .modify(new PathFragment("h/i/j"))
        .modify(new PathFragment("h/i/j/k"))
        .modify(new PathFragment("h/i/j/k/BUILD"))
        .build();
    invalidate(modifiedFileSet);
    reporter.addHandler(failFastHandler);
    Set<Label> nonEmptyResult = parseList("//h/...");
    assertThat(nonEmptyResult).containsExactly(Label.parseAbsolute("//h/i/j/k:l"));
  }

  @Test
  public void testBrokenSymlinkRepaired() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path tuv = scratch.dir("t/u/v");
    tuv.getChild("BUILD").createSymbolicLink(new PathFragment("../../BUILD"));

    try {
      parseList("//t/...");
      fail("TargetParsingException expected");
    } catch (TargetParsingException e) {
      // expected
    }

    scratch.file("t/BUILD", "sh_library(name='t')");
    ModifiedFileSet modifiedFileSet = ModifiedFileSet.builder()
        .modify(new PathFragment("t/BUILD"))
        .build();

    invalidate(modifiedFileSet);
    reporter.addHandler(failFastHandler);
    Set<Label> result = parseList("//t/...");

    assertThat(result).containsExactly(Label.parseAbsolute("//t:t"),
        Label.parseAbsolute("//t/u/v:t"));
  }

  @Test
  public void testInfiniteTreeFromSymlinks() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path ab = scratch.dir("a/b");
    ab.getChild("c").createSymbolicLink(new PathFragment("../b"));
    scratch.file("a/b/BUILD", "filegroup(name='g')");
    ResolvedTargets<Target> result = parseTargetPatternList(parser, parsingListener,
        ImmutableList.of("//a/b/..."), true);
    assertThat(targetsToLabels(result.getTargets())).containsExactly(
        Label.parseAbsolute("//a/b:g"));
  }

  @Test
  public void testSymlinkCycle() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path ab = scratch.dir("a/b");
    ab.getChild("c").createSymbolicLink(new PathFragment("c"));
    scratch.file("a/b/BUILD", "filegroup(name='g')");
    ResolvedTargets<Target> result = parseTargetPatternList(parser, parsingListener,
        ImmutableList.of("//a/b/..."), true);
    assertThat(targetsToLabels(result.getTargets())).contains(
        Label.parseAbsolute("//a/b:g"));
  }

  @Test
  public void testPerDirectorySymlinkTraversalOptOut() throws Exception {
    scratch.dir("from-b");
    scratch.file("from-b/BUILD", "filegroup(name = 'from-b')");
    scratch.dir("from-c");
    scratch.file("from-c/BUILD", "filegroup(name = 'from-c')");
    Path ab = scratch.dir("a/b");
    ab.getChild("symlink").createSymbolicLink(new PathFragment("../../from-b"));
    scratch.dir("a/b/not-a-symlink");
    scratch.file("a/b/not-a-symlink/BUILD", "filegroup(name = 'not-a-symlink')");
    scratch.file(
        "a/b/DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN");
    Path ac = scratch.dir("a/c");
    ac.getChild("symlink").createSymbolicLink(new PathFragment("../../from-c"));
    ResolvedTargets<Target> result = parseTargetPatternList(parser, parsingListener,
        ImmutableList.of("//a/..."), true);
    assertThat(targetsToLabels(result.getTargets())).containsExactly(
        Label.parseAbsolute("//a/c/symlink:from-c"),
        Label.parseAbsolute("//a/b/not-a-symlink:not-a-symlink"));
  }

  @Test
  public void testDoesNotRecurseIntoSymlinksToOutputBase() throws Exception {
    Path outputBaseBuildFile = outputBase.getRelative("workspace/test/BUILD");
    scratch.file(outputBaseBuildFile.getPathString(), "filegroup(name='c')");
    PathFragment targetFragment = outputBase.asFragment().getRelative("workspace/test");
    Path d = scratch.dir("d");
    d.getChild("c").createSymbolicLink(targetFragment);
    rootDirectory.getChild("convenience").createSymbolicLink(targetFragment);
    Set<Label> result = parseList("//...");
    assertThat(result).doesNotContain(Label.parseAbsolute("//convenience:c"));
    assertThat(result).doesNotContain(Label.parseAbsolute("//d/c:c"));
  }
}

