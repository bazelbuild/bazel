// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import static com.google.common.base.Splitter.on;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.testutil.AbstractQueryTest.QueryHelper.ResultAndTargets;
import com.google.devtools.build.lib.query2.testutil.QueryTest;
import com.google.devtools.build.lib.query2.testutil.SkyframeQueryHelper;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the {@code SkyQueryEnvironment}, which is currently activated by specifying unordered
 * results.
 */
@RunWith(JUnit4.class)
public class SkyQueryTest extends QueryTest {
  private static final String RULES_ONLY = "//...";
  private static final String UNFILTERED = "//...:*";

  private static final Function<Target, Label> TO_LABEL = Target::getLabel;

  @Before
  public final void disableOrderedResults() throws Exception {
    helper.setOrderedResults(false);
  }

  // SkyQueryEnvironment may put out an additional error relating to package loading, so the error
  // wanted by the test may not be the first one.
  @Override
  @Test
  public void testTestsOperatorReportsMissingTargets() throws Exception {
    writeFile("c/BUILD", "test_suite(name='c', tests=['//d'])");
    writeFile("d/BUILD");
    EvalThrowsResult result = evalThrows("tests(//c)", false);
    assertPackageLoadingCode(result.getFailureDetail(), Code.TARGET_MISSING);
    String expected =
        "couldn't expand 'tests' attribute of test_suite //c:c: no such target '//d:d'";
    if (!result.getMessage().startsWith(expected)) {
      assertContainsEvent(expected);
    }
  }

  @Override
  public void badRuleInDeps() throws Exception {
    runBadRuleInDeps(FailureDetails.Query.Code.SKYQUERY_TRANSITIVE_TARGET_ERROR);
  }

  @Test
  public void testUniverseWithNegativeOperator() throws Exception {
    // Given two packages "a" and "a/b", where "a/b:b" has a dependency on "a:a",
    writeFile(
        "a/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        foo_library(name = 'a', visibility = ['//visibility:public'])
        """);
    writeFile(
        "a/b/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        foo_library(name = 'b', deps = ['//a:a'])
        """);

    // Given a universe scope that contains everything in and under "a" but excludes everything in
    // and under "a/b",
    helper.setUniverseScope("//a/...,-//a/b/...");

    // Then "//a:all" evaluates to contain its one target,
    assertThat(eval("//a:all")).hasSize(1);

    // And then a query that refers to the package "a/b" fails because that package was excluded:
    EvalThrowsResult result = evalThrows("deps(//a/b:b)", false);
    assertPackageLoadingCode(result.getFailureDetail(), Code.PACKAGE_MISSING);
    assertThat(result.getMessage()).containsMatch("no such package 'a/b'");
  }

  @Test
  public void testRuleOnlyUniverseAllowsUnfilteredPatternToMatchOnlyRules() throws Exception {
    easyUniverseScopeQueryHelper(/*universeScope=*/ RULES_ONLY, /*query=*/ UNFILTERED);
  }

  @Test
  public void testUnfilteredUniverseAllowsUnfilteredPatternToMatchAllTargets() throws Exception {
    easyUniverseScopeQueryHelper(/*universeScope=*/ UNFILTERED, /*query=*/ UNFILTERED);
  }

  @Test
  public void testRuleOnlyUniverseAllowsRuleOnlyPatternToMatchOnlyRules() throws Exception {
    easyUniverseScopeQueryHelper(/*universeScope=*/ RULES_ONLY, /*query=*/ RULES_ONLY);
  }

  @Test
  public void testUnfilteredUniverseAllowsRuleOnlyPatternToMatchAllTargets() throws Exception {
    easyUniverseScopeQueryHelper(/*universeScope=*/ UNFILTERED, /*query=*/ RULES_ONLY);
  }

  private void easyUniverseScopeQueryHelper(String universeScope, String query) throws Exception {
    // Given a query universe based on a requested pattern,
    helper.setUniverseScope(universeScope);

    // And given a package "a" with rule "a" having output "a.out",
    writeFile("a/BUILD", "genrule(name='a', cmd='', srcs=['//a:a.txt'], outs=['a.out'])");

    // When evaluating the requested TargetsBelowDirectory query expression,
    Set<Target> targets = eval(query);
    Iterable<Label> labels = Iterables.transform(targets, TO_LABEL);

    // Then the result contains the rule with label "//a:a",
    assertThat(labels).contains(Label.parseCanonical("@//a:a"));
  }

  @Test
  public void testUniverseLoadsInUnitsOfPackages() throws Exception {
    // Given a package "a", containing rule "b" depending on package "b" and rule "c" depending
    // on package "c",
    writeFile(
        "a/BUILD",
        """
        genrule(
            name = "b",
            srcs = ["//b:b.txt"],
            outs = ["b.out"],
            cmd = "",
        )

        genrule(
            name = "c",
            srcs = ["//c:c.txt"],
            outs = ["c.out"],
            cmd = "",
        )
        """);
    writeFile("b/BUILD", "exports_files(['b.txt'])");
    writeFile("c/BUILD", "exports_files(['c.txt'])");

    // And given a universe scope of "//a:b",
    helper.setUniverseScope("//a:b");

    // When queried for targets in the transitive dependencies of "a:b",
    ImmutableList<Label> labelsThatExist =
        ImmutableList.of(Label.parseCanonical("@//a:b"), Label.parseCanonical("@//b:b.txt"));
    for (Label label : labelsThatExist) {
      // Then those targets are found.
      Set<Target> targets =
          eval(String.format("//%s:%s", label.getPackageFragment(), label.getName()));
      assertThat(Iterables.transform(targets, TO_LABEL)).contains(label);
    }

    // When queried for targets that are in the loaded packages but weren't in the transitive
    // dependencies of //a:b, then a result is returned, since we loaded the package.
    assertThat(eval("//a:c")).hasSize(1);

    assertThat(helper.getEvents()).isEmpty();
    // When queried for deps, only the element itself is returned.
    assertThat(eval("deps(//a:c)")).isEqualTo(eval("//a:c"));
    assertContainsEvent("Targets were missing from graph: [//a:c]");

    // When queried for a target that isn't in package "a" or in the transitive dependencies of
    // "a:b", like "//c:c.txt",
    EvalThrowsResult result = evalThrows("//c:c.txt", false);

    // Then the query fails.
    assertPackageLoadingCode(result.getFailureDetail(), Code.PACKAGE_MISSING);
    assertThat(result.getMessage()).startsWith("no such package 'c'");
  }

  @Test
  public void testAllRdeps() throws Exception {
    writeBuildFiles3();

    assertThat(eval("allrdeps(//d)" + getDependencyCorrection()))
        .isEqualTo(
            eval(
                "//a:a union //a:out union //b:b union //b:out union //c:c union //c:out "
                    + "union //d:d"));

    // Test depth-bounded variant:
    assertThat(eval("allrdeps(//d, 1)" + getDependencyCorrection()))
        .isEqualTo(eval("//b union //c union //d"));
  }

  @Test
  public void testAllRdepsWithRestrictedUniverse() throws Exception {
    writeBuildFiles3();
    writeFile(
        "e/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        foo_library(name = 'e', deps = ['//b:out'])
        """);
    helper.setUniverseScope("//e/...");
    assertThat(eval("allrdeps(//d)"))
        .isEqualTo(eval("//e:e union //b:b union //b:out union //d:d"));

    // Test depth-bounded variant:
    assertThat(eval("allrdeps(//d, 1)")).isEqualTo(eval("//b union //d"));
  }

  @Test
  public void testPackageRdeps() throws Exception {
    writeBuildFiles3();
    assertThat(eval("rbuildfiles(b/BUILD, c/BUILD, c/nonexist, nonexist/nonexist)"))
        .isEqualTo(eval("//b:BUILD union //c:BUILD"));
  }

  @Test
  public void testRbuildfiles_topLevelPackage() throws Exception {
    writeFile("BUILD");
    assertThat(eval("rbuildfiles(BUILD)")).isEqualTo(eval("//:BUILD"));
  }

  @Test
  public void testRbuildfilesCaresAbout_workspace() throws Exception {
    writeFile("foo/BUILD");
    assertThat(eval("rbuildfiles(WORKSPACE)")).containsAtLeastElementsIn(eval("//foo:BUILD"));
  }

  @Test
  public void testRbuildfilesCaresAbout_starlarkLoads() throws Exception {
    writeFile(
        "foo/BUILD",
        """
        load("//foo:starlark.bzl", "blah")
        load("//foo:also_starlark.scl", "blargh")
        """);
    writeFile(
        "foo/starlark.bzl",
        """
        def blah():
            pass
        """);
    writeFile(
        "foo/also_starlark.scl", //
        "def blargh():",
        "  pass");
    assertThat(eval("rbuildfiles(foo/starlark.bzl)")).isEqualTo(eval("//foo:BUILD"));
    assertThat(eval("rbuildfiles(foo/also_starlark.scl)")).isEqualTo(eval("//foo:BUILD"));
  }

  @Test
  public void testRbuildfilesCaresAbout_symlinks() throws Exception {
    writeFile("starlark/BUILD");
    writeFile(
        "starlark/target.bzl",
        """
        def blah():
            pass
        """);
    ensureSymbolicLink("starlark/symlink_to_target.bzl", "starlark/target.bzl");
    ensureSymbolicLink(
        "starlark/symlink_to_symlink_to_target.bzl", "starlark/symlink_to_target.bzl");
    writeFile("loads_target/BUILD", "load('//starlark:target.bzl', 'blah')");
    writeFile("loads_symlink_to_target/BUILD", "load('//starlark:symlink_to_target.bzl', 'blah')");
    writeFile(
        "loads_symlink_to_symlink_to_target/BUILD",
        "load('//starlark:symlink_to_symlink_to_target.bzl', 'blah')");

    assertThat(evalToListOfStrings("rbuildfiles(starlark/target.bzl)"))
        .containsExactly(
            "//loads_target:BUILD",
            "//loads_symlink_to_target:BUILD",
            "//loads_symlink_to_symlink_to_target:BUILD");
    assertThat(evalToListOfStrings("rbuildfiles(starlark/symlink_to_target.bzl)"))
        .containsExactly(
            "//loads_symlink_to_target:BUILD", "//loads_symlink_to_symlink_to_target:BUILD");
    assertThat(evalToListOfStrings("rbuildfiles(starlark/symlink_to_symlink_to_target.bzl)"))
        .containsExactly("//loads_symlink_to_symlink_to_target:BUILD");
  }

  @Test
  public void testRbuildfilesForBrokenPkg() throws Exception {
    writeFile("broken/BUILD", "break this package!\nsh_library(name = 'e', deps = ['//b:out'])");
    helper.setKeepGoing(true);
    ResultAndTargets<Target> result = helper.evaluateQuery("rbuildfiles(broken/BUILD)");
    assertThat(result.getQueryEvalResult().getSuccess()).isFalse();
    assertThat(resultSetToListOfStrings(result.getResultSet())).containsExactly("//broken:BUILD");
  }

  @Test
  public void testRbuildfilesDoesNotReturnNonexistentButReferredToSubpackages() throws Exception {
    writeFile("a/BUILD", "genrule(name='a', srcs=['//a/sub:sub.txt'], outs=['a.out'], cmd=':')");

    // By ensuring that the directory for the nonexistent package reference exists, this test
    // ensures that the empty query result is due only to the absence of the nonexistent package's
    // BUILD file.
    writeFile("a/sub/foo.txt");

    assertThat(evalToListOfStrings("rbuildfiles(a/sub/BUILD)")).isEmpty();
  }

  @Test
  public void testRbuildfilesWorksOnPackagesNotMatchingButDepenededOnByUniverse() throws Exception {
    // Package "a" contains rule "a" depending on package "b" which loads from "//c:lib.bzl".
    writeFile("a/BUILD", "genrule(name='a', cmd='', srcs=['//b:b.txt'], outs=['a.out'])");
    writeFile(
        "b/BUILD",
        """
        load("//c:lib.bzl", "blah")

        exports_files(["b.txt"])
        """);
    writeFile("c/BUILD");
    writeFile(
        "c/lib.bzl",
        """
        def blah():
            pass
        """);

    // The universe scope is "//a:a", directly matching only package "a", though "//a:a" depends on
    // packages "b" and "c".
    helper.setUniverseScope("//a:a");

    // rbuildfiles(c/lib.bzl) returns //b:BUILD despite the universe pattern not matching any
    // targets in the bzl file's package nor the package that loads the bzl file.
    assertThat(evalToListOfStrings("rbuildfiles(c/lib.bzl)")).containsExactly("//b:BUILD");
  }

  @Override
  @Test
  public void testCycleInStarlark() throws Exception {
    runCycleInStarlarkTest(/*checkFailureDetail=*/ false);
  }
  
  @Override
  @Test
  public void bzlPackageBadDueToBrokenLoad() throws Exception {
    writeFile("foo/BUILD", "load('//bar:bar.bzl', 'sym')");
    writeFile("bar/BUILD", "load('//noexist:noexist.bzl', 'bad')");
    writeFile("bar/bar.bzl", "sym = 0");
    if (helper.reportsUniverseEvaluationErrors()) {
      EvalThrowsResult result = evalThrows("buildfiles(//foo:BUILD)", true);
      assertThat(result.getFailureDetail())
          .comparingExpectedFieldsOnly()
          .isEqualTo(
              FailureDetails.FailureDetail.newBuilder()
                  .setQuery(
                      FailureDetails.Query.newBuilder()
                          .setCode(FailureDetails.Query.Code.BUILD_FILE_ERROR))
                  .build());
    } else {
      assertThat(evalToListOfStrings("buildfiles(//foo:BUILD)"))
          .containsExactly("//foo:BUILD", "//bar:bar.bzl", "//bar:BUILD");
    }
  }

  @Override
  @Test
  public void bzlPackageBadDueToBrokenSyntax() throws Exception {
    writeFile("foo/BUILD", "load('//bar:bar.bzl', 'sym')");
    writeFile("bar/BUILD", "malformed syntax");
    writeFile("bar/bar.bzl", "sym = 0");
    if (helper.reportsUniverseEvaluationErrors()) {
      EvalThrowsResult result = evalThrows("buildfiles(//foo:BUILD)", true);
      assertThat(result.getFailureDetail())
          .comparingExpectedFieldsOnly()
          .isEqualTo(
              FailureDetails.FailureDetail.newBuilder()
                  .setPackageLoading(
                      FailureDetails.PackageLoading.newBuilder().setCode(Code.SYNTAX_ERROR))
                  .build());
    } else {
      assertThat(evalToListOfStrings("buildfiles(//foo:BUILD)"))
          .containsExactly("//foo:BUILD", "//bar:bar.bzl", "//bar:BUILD");
    }
  }

  @Override
  public void boundedRdepsWithError() throws Exception {
    writeFile(
        "foo/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        foo_library(
            name = "foo",
            deps = [":dep"],
        )

        foo_library(
            name = "dep",
            deps = ["//bar:missing"],
        )
        """);
    assertThat(evalToListOfStrings("rdeps(//foo:foo, //foo:dep, 1)"))
        .containsExactly("//foo:dep", "//foo:foo");
    assertDoesNotContainEvent("no such package 'bar'");
  }

  // This test checks that packages not needed to evaluate a query expression aren't loaded, which
  // isn't true for SkyQuery, because it loads all packages in the transitive closure of the
  // universe it's supplied with.
  @Override
  @Test
  public void testWildcardsDontLoadUnnecessaryPackages() {}

  @Override
  @Test
  public void boundedDepsWithError() {}

  // SkyQuery requires unordered results, so this graph-order-sensitive test does not apply.
  @Override
  @Test
  public void testGraphOrderOfWildcards() {}

  @Test
  public void testBoundedAllrdepsStreaming() throws Exception {
    writeFile(
        "foo/BUILD",
        """
        load("//test_defs:foo_library.bzl", "foo_library")

        foo_library(
            name = "a",
            deps = [":b"],
        )

        foo_library(
            name = "b",
            deps = [":c"],
        )

        foo_library(
            name = "c",
            deps = [":d"],
        )

        foo_library(name = "d")
        """);
    assertThat(evalToString("allrdeps(//foo:d + //foo:c, 1)")).isEqualTo("//foo:b //foo:c //foo:d");
  }

  @Test
  public void testAllrdepsUnboundedWithNonDefaultDependencyFilter() throws Exception {
    runTestRdepsWithNonDefaultDependencyFilter("allrdeps(//foo:a)", "//foo:a //foo:b");
  }

  @Test
  public void testAllrdepsBoundedWithNonDefaultDependencyFilter() throws Exception {
    runTestRdepsWithNonDefaultDependencyFilter("allrdeps(//foo:a, 1)", "//foo:a //foo:b");
  }

  @Test
  public void inconsistentFilesystemExceptionProcessed() throws Exception {
    writeFile("foo/bar/BUILD");
    ((SkyframeQueryHelper) helper)
        .setSyscallCache(TestUtils.makeDisappearingFileCache("foo/bar/BUILD"));
    EvalThrowsResult result1 = evalThrows("foo/bar:BUILD", /*unconditionallyThrows=*/ false);
    assertThat(result1.getMessage()).contains("Inconsistent filesystem operations");
    assertThat(result1.getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(Code.TRANSIENT_INCONSISTENT_FILESYSTEM_ERROR);
  }

  @Test
  @Override
  public void testExternalRepo_allTargetsBelow() {
    // Fails due to http://b/145772216, should not override otherwise.
  }

  @Test
  public void testExternalRepoScope_allTargetsInPackage() throws Exception {
    helper.setUniverseScope("@my_repo//...");
    writeBzlmodBuildFiles();
    assertThat(evalToString("@my_repo//a/b:*")).isEqualTo(REPO_AB_ALL);
    assertThat(evalToString("@my_repo//a:*")).isEqualTo(REPO_A_ALL);
  }

  @Test
  public void testExternalRepoScope_allTargetsBelow() throws Exception {
    writeBzlmodBuildFiles();
    helper.setUniverseScope("@my_repo//...");
    assertThat(evalToListOfStrings("@my_repo//...:*"))
        .containsExactly(on(" ").splitToList(REPO_A_AB_ALL).toArray());
    assertThat(evalToListOfStrings("@my_repo//a/..."))
        .containsExactly(on(" ").splitToList(REPO_A_AB_RULES).toArray());
    assertThat(evalToListOfStrings("@my_repo//a/b/..."))
        .containsExactly(on(" ").splitToList(REPO_AB_RULES).toArray());
  }
}
