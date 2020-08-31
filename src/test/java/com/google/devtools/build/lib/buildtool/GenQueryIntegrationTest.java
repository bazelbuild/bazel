// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEvent;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for the 'genquery' rule.
 */
@RunWith(JUnit4.class)
public class GenQueryIntegrationTest extends BuildIntegrationTestCase {
  @Test
  public void testDoesNotFailHorribly() throws Exception {
    write(
        "fruits/BUILD",
        "sh_library(name='melon', deps=[':papaya'])",
        "sh_library(name='papaya')",
        "genquery(name='q',",
        "         scope=[':melon'],",
        "         expression='deps(//fruits:melon)')");
    assertQueryResult("//fruits:q", "//fruits:melon", "//fruits:papaya");
  }

  private void deterministicTestHelper(boolean graphless) throws Exception {
    write(
        "fruits/BUILD",
        "sh_library(name='melon', deps=[':papaya', ':apple'])",
        "sh_library(name='papaya', deps=[':banana'])",
        "sh_library(name='banana', deps=[':apple'])",
        "sh_library(name='apple', deps=[':cherry'])",
        "sh_library(name='cherry')",
        "genquery(name='q',",
        "         scope=[':melon'],",
        "         expression='deps(//fruits:melon)')");
    String firstResult = getQueryResult("//fruits:q");
    for (int i = 0; i < 10; i++) {
      createFilesAndMocks(); // Do a clean.
      if (graphless) {
        runtimeWrapper.addOptions("--experimental_genquery_use_graphless_query");
      }
      assertThat(getQueryResult("//fruits:q")).isEqualTo(firstResult);
    }
  }

  @Test
  public void testDeterministic() throws Exception {
    deterministicTestHelper(/* graphless= */ false);
  }

  @Test
  public void testDeterministicGraphless() throws Exception {
    runtimeWrapper.addOptions("--experimental_genquery_use_graphless_query");
    deterministicTestHelper(/* graphless= */ true);
  }

  @Test
  public void testiDuplicateName() throws Exception {
    write("one/BUILD", "sh_library(name='foo')");
    write("two/BUILD", "sh_library(name='foo')");
    write(
        "query/BUILD",
        "sh_library(name='common', deps=['//one:foo', '//two:foo'])",
        "genquery(name='q',",
        "         scope=['//query:common'],",
        "         expression='deps(//query:common)')");
    runtimeWrapper.addOptions("--experimental_genquery_use_graphless_query");
    assertThat(getQueryResult("//query:q").split("\n")).hasLength(3);
  }

  @Test
  public void testFailsIfGoesOutOfScope() throws Exception {
    write("vegetables/BUILD",
        "sh_library(name='tomato', deps=[':cabbage'])",
        "sh_library(name='cabbage')",
        "genquery(name='q',",
        "         scope=[':cabbage'],",
        "         expression='deps(//vegetables:tomato)')");

    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//vegetables:q"));

    assertContainsEvent(events.collector(), "is not within the scope of the query");
  }

  // Regression test for http://b/29964062.
  @Test
  public void testFailsIfGoesOutOfScopeViaSelect() throws Exception {
    write("q/BUILD",
        "genquery(name='q', expression='deps(//q:f)', scope=['f'])",
        "config_setting(name='cs', values={'define':'D=1'})",
        "filegroup(name='f', srcs=select({'cs':[], '//conditions:default':['//dne']}))");

    addOptions("--keep_going");
    addOptions("--define=D=1");
    assertThrows(BuildFailedException.class, () -> buildTarget("//q"));

    assertContainsEvent(
        events.collector(),
        "errors were encountered while computing transitive closure of the scope.");
  }

  // Regression test for http://b/34132681
  @Test
  public void testFailsIfBrokenDependencyViaSelect() throws Exception {
    write(
        "q/BUILD",
        "genquery(name='q', expression='deps(//q:f)', scope=['f'])",
        "config_setting(name='cs', values={'define':'D=1'})",
        "filegroup(name='f', srcs=select({'cs':[], '//conditions:default':['//d:d']}))");
    // d exists but is missing "srcs"
    write("d/BUILD", "sh_binary(name = 'd')");

    addOptions("--keep_going");
    addOptions("--define=D=1");
    assertThrows(BuildFailedException.class, () -> buildTarget("//q"));

    assertContainsEvent(
        events.collector(),
        "errors were encountered while computing transitive closure of the scope");
  }


  @Test
  public void testResultsAlphabetized() throws Exception {
    write(
        "fruits/BUILD",
        "sh_library(name='melon', deps=[':a', ':z', ':c', ':1', '//z:a', '//a:z', '//c:c'])",
        "sh_library(name='a')",
        "sh_library(name='z')",
        "sh_library(name='1')",
        "sh_library(name='c')",
        "genquery(name='q',",
        "         scope=[':melon'],",
        "         expression='deps(//fruits:melon)')");
    write("z/BUILD", "sh_library(name = 'a')");
    write("a/BUILD", "sh_library(name = 'z')");
    write("c/BUILD", "sh_library(name = 'c', deps = ['//z:a'])");
    assertQueryResult(
        "//fruits:q",
        // Results are ordered in reverse dependency order (nodes, then their dependencies), and in
        // reverse alphabetic order for ties.
        "//fruits:melon",
        "//fruits:z",
        "//fruits:c",
        "//fruits:a",
        "//fruits:1",
        "//c:c",
        "//z:a",
        "//a:z");
  }

  @Test
  public void testQueryReexecutedIfDepsChange() throws Exception {
    write("food/BUILD",
        "sh_library(name='fruit_salad', deps=['//fruits:tropical'])",
        "genquery(name='q',",
        "         scope=[':fruit_salad']," +
        "         expression='deps(//food:fruit_salad)')");

    write("fruits/BUILD",
        "sh_library(name='tropical', deps=[':papaya'])",
        "sh_library(name='papaya')");

    assertQueryResult("//food:q", "//food:fruit_salad", "//fruits:tropical", "//fruits:papaya");

    write("fruits/BUILD",
        "sh_library(name='tropical', deps=[':papaya', ':coconut'])",
        "sh_library(name='papaya')",
        "sh_library(name='coconut')");

    assertQueryResult(
        "//food:q",
        "//food:fruit_salad",
        "//fruits:tropical",
        "//fruits:papaya",
        "//fruits:coconut");
  }

  @Test
  public void testGenQueryEncountersAnotherGenQuery() throws Exception {
    write("spices/BUILD",
        "sh_library(name='cinnamon', deps=[':nutmeg'])",
        "sh_library(name='nutmeg')",
        "genquery(name='q',",
        "         scope=[':cinnamon'],",
        "         expression='deps(//spices:cinnamon)')");

    write("fruits/BUILD",
        "sh_library(name='pear', deps=[':plum'])",
        "sh_library(name='plum')",
        "genquery(name='q',",
        "         scope=[':pear', '//spices:q'],",
        "         expression='deps(//fruits:pear) + deps(//spices:q)')");

    assertQueryResult(
        "//fruits:q",
        "//spices:q",
        "//spices:cinnamon",
        "//spices:nutmeg",
        "//fruits:pear",
        "//fruits:plum");
  }

  /**
   * Regression test for b/14227750: genquery referring to non-existent target crashes on skyframe.
   */
  @Test
  public void testHandlesMissingTargetGracefully() throws Exception {
    write("a/BUILD",
        "genquery(name='query', scope=['//b:target'], expression='deps(//b:nosuchtarget)')");
    write("b/BUILD",
        "sh_library(name = 'target')");
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//a:query"));
    events.assertContainsError(
        "in genquery rule //a:query: query failed: no such target '//b:nosuchtarget'");
  }

  @Test
  public void testMultiplePatternsInQuery() throws Exception {
    String buildFile = "";
    String genQuery = "genquery(name = 'q', scope = [':top'], "
        + "expression = 'deps(//spices:top) ' + \n";
    String topTarget = "sh_library(name = 'top', deps = [\n";
    for (int i = 0; i < 20; i++) {
      String targetName = (i % 2 == 0 ? "in" : "out") + i;
      buildFile += "sh_library(name = '" + targetName + "')\n";
      if (i % 2 != 0) {
        genQuery += "' - //spices:" + targetName + " ' + \n";
      }
      topTarget += "    ':" + targetName + "',\n";
    }
    topTarget += "]\n)\n";
    genQuery += "'')";
    write("spices/BUILD", buildFile, topTarget, genQuery);
    List<String> expected = new ArrayList<>(11);
    for (int i = 0; i < 20; i += 2) {
      expected.add(i / 2, "//spices:in" + i);
    }
    Collections.sort(expected, Collections.reverseOrder());
    expected.add(0, "//spices:top");
    assertQueryResult("//spices:q", expected.toArray(new String[0]));
  }

  @Test
  public void testGraphOutput_factored() throws Exception {
    write(
        "fruits/BUILD",
        "sh_library(name='melon', deps=[':papaya', ':coconut', ':mango'])",
        "sh_library(name='papaya')",
        "sh_library(name = 'mango')",
        "sh_library(name = 'coconut')",
        "genquery(name='q',",
        "         scope=[':melon'],",
        "         opts = ['--output=graph'],",
        "         expression='deps(//fruits:melon)')");
    assertPartialQueryResult(
        "//fruits:q",
        "  \"//fruits:melon\"",
        "  \"//fruits:melon\" -> \"//fruits:coconut\\n//fruits:mango\\n//fruits:papaya\"",
        "  \"//fruits:coconut\\n//fruits:mango\\n//fruits:papaya\"");
  }

  @Test
  public void testGraphOutput_unfactored() throws Exception {
    write(
        "fruits/BUILD",
        "sh_library(name='melon', deps=[':papaya', ':coconut', ':mango'])",
        "sh_library(name='papaya')",
        "sh_library(name = 'mango')",
        "sh_library(name = 'coconut')",
        "genquery(name='q',",
        "         scope=[':melon'],",
        "         opts = ['--output=graph', '--nograph:factored'],",
        "         expression='deps(//fruits:melon)')");
    assertPartialQueryResult(
        "//fruits:q",
        "  \"//fruits:melon\"",
        "  \"//fruits:melon\" -> \"//fruits:coconut\"",
        "  \"//fruits:melon\" -> \"//fruits:mango\"",
        "  \"//fruits:melon\" -> \"//fruits:papaya\"",
        "  \"//fruits:papaya\"",
        "  \"//fruits:mango\"",
        "  \"//fruits:coconut\"");
  }

  @Test
  public void testDoesntAllowLocationOutputWithLoadfiles() throws Exception {
    write(
        "foo/bzl.bzl",
        "x = 2");
    write(
        "foo/BUILD",
        "load('//foo:bzl.bzl', 'x')",
        "sh_library(name='foo')",
        "genquery(",
        "  name = 'gen-loadfiles',",
        "  expression = 'loadfiles(//foo:foo)',",
        "  scope = ['//foo:foo'],",
        ")",
        "genquery(",
        "  name = 'gen-loadfiles-location',",
        "  expression = 'loadfiles(//foo:foo)',",
        "  opts = ['--output=location'],",
        "  scope = ['//foo:foo'],",
        ")");
    assertQueryResult(
        "//foo:gen-loadfiles",
        "//foo:bzl.bzl");
    assertThrows(
        ViewCreationFailedException.class, () -> buildTarget("//foo:gen-loadfiles-location"));
    events.assertContainsError(
        "in genquery rule //foo:gen-loadfiles-location: query failed: Query expressions "
            + "involving 'buildfiles' or 'loadfiles' cannot be used with --output=location");
  }

  @Test
  public void testDoesntAllowLocationOutputWithBuildfiles() throws Exception {
    write(
        "foo/bzl.bzl",
        "x = 2");
    write(
        "foo/BUILD",
        "load('//foo:bzl.bzl', 'x')",
        "sh_library(name='foo')",
        "genquery(",
        "  name = 'gen-buildfiles',",
        "  expression = 'buildfiles(//foo:foo)',",
        "  scope = ['//foo:foo'],",
        ")",
        "genquery(",
        "  name = 'gen-buildfiles-location',",
        "  expression = 'buildfiles(//foo:foo)',",
        "  opts = ['--output=location'],",
        "  scope = ['//foo:foo'],",
        ")");
    assertQueryResult(
        "//foo:gen-buildfiles",
        "//foo:bzl.bzl",
        "//foo:BUILD");
    assertThrows(
        ViewCreationFailedException.class, () -> buildTarget("//foo:gen-buildfiles-location"));
    events.assertContainsError(
        "in genquery rule //foo:gen-buildfiles-location: query failed: Query expressions "
            + "involving 'buildfiles' or 'loadfiles' cannot be used with --output=location");
  }

  /** Regression test for b/127644784. */
  @Test
  public void somepathOutputDeterministic() throws Exception {
    /*
     * This graph structure routinely reproduces the bug within 10 iterations:
     *
     *   ----------top------------
     *   |       |       |       |
     *  mid1    mid2    mid3    mid4
     *   |       |       |       |
     *   --lower--       |       |
     *       |           |       |
     *       -----bottom----------
     */
    write(
        "query/BUILD",
        "genquery(",
        "  name = 'query',",
        "  expression = 'somepath(//top, //bottom)',",
        "  scope = ['//top', '//bottom'],",
        ")");
    write("top/BUILD", "sh_library(name = 'top', deps = ['//mid1', '//mid2', '//mid3', '//mid4'])");
    write("mid1/BUILD", "sh_library(name = 'mid1', deps = ['//lower'])");
    write("mid2/BUILD", "sh_library(name = 'mid2', deps = ['//lower'])");
    write("mid3/BUILD", "sh_library(name = 'mid3', deps = ['//bottom'])");
    write("mid4/BUILD", "sh_library(name = 'mid4', deps = ['//bottom'])");
    write("lower/BUILD", "sh_library(name = 'lower', deps = ['//bottom'])");
    write("bottom/BUILD", "sh_library(name = 'bottom')");

    String firstResult = getQueryResult("//query");
    for (int i = 0; i < 10; i++) {
      createFilesAndMocks(); // Do a clean.
      assertThat(getQueryResult("//query")).isEqualTo(firstResult);
    }
  }

  private void runNodepDepsTest(String optsStringValue, boolean expectVisibilityDep)
      throws Exception {
    write(
        "foo/BUILD",
        "sh_library(name = 't1', deps = [':t2'], visibility = [':pg', '//query:__pkg__'])",
        "sh_library(name = 't2')",
        "package_group(name = 'pg')");
    write(
        "query/BUILD",
        "genquery(",
        "  name = 'gen',",
        "  expression = 'deps(//foo:t1)',",
        "  scope = ['//foo:t1'],",
        "  opts = " + optsStringValue,
        ")");

    List<String> queryResultStrings =
        ImmutableList.copyOf(getQueryResult("//query:gen").split("\n"));
    if (expectVisibilityDep) {
      assertThat(queryResultStrings).contains("//foo:pg");
    } else {
      assertThat(queryResultStrings).doesNotContain("//foo:pg");
    }
  }

  @Test
  public void testNodepDeps_defaultIsFalse() throws Exception {
    runNodepDepsTest(/*optsStringValue=*/ "[]", /*expectVisibilityDep=*/ false);
  }

  @Test
  public void testNodepDeps_false() throws Exception {
    runNodepDepsTest(/*optsStringValue=*/ "['--nodep_deps=false']", /*expectVisibilityDep=*/ false);
  }

  @Test
  public void testNodepDeps_true() throws Exception {
    runNodepDepsTest(/*optsStringValue=*/ "['--nodep_deps=true']", /*expectVisibilityDep=*/ true);
  }

  private void assertQueryResult(String queryTarget, String... expected) throws Exception {
    assertThat(getQueryResult(queryTarget).split("\n"))
        .asList()
        .containsExactlyElementsIn(ImmutableList.copyOf(expected))
        .inOrder();
  }

  private void assertPartialQueryResult(String queryTarget, String... expected) throws Exception {
    assertThat(getQueryResult(queryTarget).split("\n"))
        .asList()
        .containsAtLeastElementsIn(ImmutableList.copyOf(expected))
        .inOrder();
  }

  private String getQueryResult(String queryTarget) throws Exception {
    buildTarget(queryTarget);
    Artifact output = Iterables.getOnlyElement(getArtifacts(queryTarget));
    return readContentAsLatin1String(output);
  }
}
