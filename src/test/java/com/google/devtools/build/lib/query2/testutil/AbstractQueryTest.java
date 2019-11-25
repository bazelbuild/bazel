// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.testutil;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.TestConstants.GENRULE_SETUP;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.DotOutputVisitor;
import com.google.devtools.build.lib.graph.LabelSerializer;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.query2.engine.DigraphQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.testutil.AbstractQueryTest.QueryHelper.ResultAndTargets;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

/**
 * Tests for the query engine, generic over the result type. This allows us to share the tests
 * between the different implementations, and also parameterize it over the set of options, such as
 * {@code --keep_going}.
 *
 * @param <T> the actual target type
 */
public abstract class AbstractQueryTest<T> {

  protected static final ImmutableSet<?> EMPTY = ImmutableSet.of();

  private static final String DEFAULT_UNIVERSE = "//...:*";

  protected static final String BAD_PACKAGE_NAME =
      "package names may contain "
          + "A-Z, a-z, 0-9, or any of ' !\"#$%&'()*+,-./;<=>?[]^_`{|}~' "
          + "(most 7-bit ascii characters except 0-31, 127, ':', or '\\')";

  protected MockToolsConfig mockToolsConfig;
  protected QueryHelper<T> helper;
  protected AnalysisMock analysisMock;

  @Before
  public final void initializeQueryHelper() throws Exception {
    helper = createQueryHelper();
    helper.setUp();
    mockToolsConfig = new MockToolsConfig(helper.getRootDirectory());
    analysisMock = AnalysisMock.get();
    helper.setUniverseScope(getDefaultUniverseScope());
  }

  /**
   * By default, we load the universe (of both rules and files) for our tests. If a specific test or
   * subclass requires that only a subset of the universe is loaded, it may override this default
   * and/or specify a per-test method universe scope.
   */
  protected String getDefaultUniverseScope() {
    return DEFAULT_UNIVERSE;
  }

  protected abstract QueryHelper<T> createQueryHelper();

  /**
   * Used to disable configurable attribute queries on DepServerQueryEnvironment, which doesn't
   * support them.
   */
  protected boolean testConfigurableAttributes() {
    return true;
  }

  /** Partial query to filter out implicit dependencies. */
  protected String getDependencyCorrection() {
    return "";
  }

  /** Partial query to filter out implicit dependencies of genrules. */
  protected String getDependencyCorrectionWithGen() {
    return getDependencyCorrection() + " - deps(" + GENRULE_SETUP + ")";
  }

  protected final void writeFile(String pathName, String... lines) throws IOException {
    helper.writeFile(pathName, lines);
  }

  protected final void overwriteFile(String pathName, String... lines) throws IOException {
    helper.overwriteFile(pathName, lines);
  }

  protected void assertContainsEvent(String expectedMessage) {
    helper.assertContainsEvent(expectedMessage);
  }

  protected final void assertDoesNotContainEvent(String notExpectedMessage) {
    helper.assertDoesNotContainEvent(notExpectedMessage);
  }

  protected final void ensureSymbolicLink(String link, String target) throws IOException {
    helper.ensureSymbolicLink(link, target);
  }

  protected final void assertStartsWith(String expected, String actual) {
    if (!actual.startsWith(expected)) {
      // Call into ChattyAssertsTestCase to get the nice formatting.
      assertThat(actual).isEqualTo(expected);
    }
  }

  // Evaluate the query, assert that it is successful, and return its results.
  protected Set<T> eval(String query) throws Exception {
    ResultAndTargets<T> result = helper.evaluateQuery(query);
    assertWithMessage(
            "evaluateQuery failed: " + query + "\n" + Iterables.toString(helper.getEvents()))
        .that(result.getQueryEvalResult().getSuccess())
        .isTrue();
    return result.getResultSet();
  }

  // Like eval(), but asserts that evaluation completes abruptly with a
  // QueryException, whose message is returned.
  protected String evalThrows(String query, boolean unconditionallyThrows) throws Exception {
    try {
      helper.evaluateQuery(query);
      fail("evaluateQuery completed normally: " + query);
      return null; // unreachable
    } catch (QueryException e) {
      return e.getCause() != null ? e.getCause().getMessage() : e.getMessage();
    }
  }

  // Returns the set as a space-separated list of labels in lex order.
  protected String evalToString(String query) throws Exception {
    return Joiner.on(' ').join(evalToListOfStrings(query));
  }

  protected ImmutableList<String> evalToListOfStrings(String query) throws Exception {
    return resultSetToListOfStrings(eval(query));
  }

  protected ImmutableList<String> resultSetToListOfStrings(Set<T> results) {
    return Ordering.natural()
        .immutableSortedCopy(
            Iterables.transform(
                results,
                new Function<T, String>() {
                  @Override
                  public String apply(T node) {
                    return helper.getLabel(node);
                  }
                }));
  }

  protected void assertContains(Set<T> x, Set<T> y) throws Exception {
    if (!x.containsAll(y)) {
      fail("x is not a superset of y:\nx = " + x + "\ny = " + y);
    }
  }

  protected void assertNotContains(Set<T> x, Set<T> y) throws Exception {
    assertThat(x.containsAll(y)).isFalse();
  }

  @Test
  public void testTargetLiteralWithMissingTargets() throws Exception {
    writeFile("a/BUILD");
    assertThat(evalThrows("//a:b", false))
        .isEqualTo(
            "no such target '//a:b': target 'b' not declared in package 'a' "
                + "defined by "
                + helper.getRootDirectory().getPathString()
                + "/a/BUILD");
  }

  protected void writeBuildFiles1() throws Exception {
    // Note, these BUILD files contain no rules, only files, so we use the
    // "a/...:*" wildcard to match them.
    writeFile("a/BUILD", "exports_files(['x', 'y', 'z'])");
    writeFile("a/b/BUILD", "exports_files(['p', 'q'])");
  }

  protected static final String AB_FILES = "//a/b:BUILD //a/b:p //a/b:q";
  protected static final String A_FILES = "//a:BUILD //a:x //a:y //a:z";
  protected static final String A_AB_FILES = AB_FILES + " " + A_FILES;

  @Test
  public void testTargetLiterals() throws Exception {
    writeBuildFiles1();
    assertThat(evalToString("a/b:*")).isEqualTo(AB_FILES);
    assertThat(evalToString("a/...:*")).isEqualTo(A_AB_FILES);
    assertThat(evalToString("a:*")).isEqualTo(A_FILES);
    assertThat(evalToString("//a:x")).isEqualTo("//a:x");
  }

  @Test
  public void testBadTargetLiterals() throws Exception {
    assertThat(evalThrows("bad:*:*", false))
        .isEqualTo("Invalid package name 'bad:*': " + BAD_PACKAGE_NAME);
  }

  @Test
  public void testAlgebraicSetOperations() throws Exception {
    writeBuildFiles1();
    assertThat(evalToString("a/...:* intersect a/b/...:*")).isEqualTo(AB_FILES);
    assertThat(evalToString("a/b/...:* intersect a/...:*")).isEqualTo(AB_FILES);
    assertThat(evalToString("//a:x union a/b/...:*")).isEqualTo(AB_FILES + " //a:x");
    assertThat(evalToString("a/b/...:* union //a:x")).isEqualTo(AB_FILES + " //a:x");
    assertThat(evalToString("a/...:* except a/b/...:*")).isEqualTo(A_FILES);
    assertThat(evalToString("a/b/...:* except a/...:*")).isEmpty();

    assertThat(evalToString("(a/...:* union a/b/...:*) except //a/b:p"))
        .isEqualTo("//a/b:BUILD //a/b:q " + A_FILES);
    assertThat(evalToString("a/...:* union (a/b/...:* except //a/b:p)")).isEqualTo(A_AB_FILES);

    // Test - + ^ variants:
    assertThat(evalToString("a/...:* + (a/b/...:* - //a/b:p)")).isEqualTo(A_AB_FILES);
    assertThat(evalToString("a/...:* ^ a/b/...:*")).isEqualTo(AB_FILES);
  }

  @Test
  public void testAlgebraicSetOperations_ManyOperands() throws Exception {
    writeBuildFiles1();
    assertThat(evalToString("//a:BUILD + //a:x + //a:y + //a:z + //a/b:BUILD + //a/b:p + //a/b:q"))
        .isEqualTo(A_AB_FILES);
    assertThat(
            evalToString(
                "a/...:* - //a:BUILD - //a:x - //a:y - //a:z - //a/b:BUILD - //a/b:p - //a/b:q"))
        .isEmpty();
    assertThat(
            evalToString(
                "(//a:x + //a:y) ^ (//a:x + //a:z) ^ (//a:x + //a/b:p) ^ (//a:x + //a/b:q)"))
        .isEqualTo("//a:x");
  }

  private void writeBuildFiles2() throws Exception {
    writeFile(
        "c/BUILD",
        "genrule(name='c', srcs=['p', 'q'], outs=['r', 's'], cmd=':')",
        "cc_binary(name='d', srcs=['e.cc'], data=['r'])");
  }

  @Test
  public void testKindOperator() throws Exception {
    writeBuildFiles2();
    assertThat(evalToString("c:*"))
        .isEqualTo(
            "//c:BUILD //c:c //c:d //c:d.dwp //c:d.stripped //c:e.cc //c:p //c:q //c:r //c:s");
    assertThat(evalToString("kind(rule, c:*)")).isEqualTo("//c:c //c:d");
    assertThat(evalToString("kind(genrule, c:*)")).isEqualTo("//c:c");
    assertThat(evalToString("kind(cc.*, c:*)")).isEqualTo("//c:d");
    assertThat(evalToString("kind(file, c:*)"))
        .isEqualTo("//c:BUILD //c:d.dwp //c:d.stripped //c:e.cc //c:p //c:q //c:r //c:s");
    assertThat(evalToString("kind(gener.*, c:*)"))
        .isEqualTo("//c:d.dwp //c:d.stripped //c:r //c:s");
    assertThat(evalToString("kind(gen.*, c:*)"))
        .isEqualTo("//c:c //c:d.dwp //c:d.stripped //c:r //c:s");
    assertThat(evalToString("kind(source, c:*)")).isEqualTo("//c:BUILD //c:e.cc //c:p //c:q");
    assertThat(evalToString("kind('source file', c:*)"))
        .isEqualTo("//c:BUILD //c:e.cc //c:p //c:q");
  }

  @Test
  public void testFilterOperator() throws Exception {
    writeBuildFiles2();
    assertThat(evalToString("c:*"))
        .isEqualTo(
            "//c:BUILD //c:c //c:d //c:d.dwp //c:d.stripped //c:e.cc //c:p //c:q //c:r //c:s");
    assertThat(evalToString("filter(BUILD, c:*)")).isEqualTo("//c:BUILD");
    assertThat(evalToString("filter('\\.cc$', c:*)")).isEqualTo("//c:e.cc");
    assertThat(evalToString("filter(//c.*cc$, c:*)")).isEqualTo("//c:e.cc");
    assertThat(evalToString("filter(:.$, c:*)")).isEqualTo("//c:c //c:d //c:p //c:q //c:r //c:s");
  }

  @Test
  public void testAttrOperatorOnName() throws Exception {
    writeBuildFiles2();
    writeBuildFilesWithConfigurableAttributes();

    assertThat(evalToString("attr(name, '.*', '//c:*')")).isEqualTo("//c:c //c:d");
    assertThat(evalToString("attr(name, '.+', '//c:*')")).isEqualTo("//c:c //c:d");
    assertThat(evalToString("attr(name, '.*d.*', '//c:*')")).isEqualTo("//c:d");

    assertThat(evalToString("attr(name, '.*e.*', '//c:*')")).isEmpty();
  }

  @Test
  public void testAttrOperator() throws Exception {
    writeBuildFiles2();
    writeBuildFilesWithConfigurableAttributes();

    assertThat(evalToString("c:*"))
        .isEqualTo(
            "//c:BUILD //c:c //c:d //c:d.dwp //c:d.stripped //c:e.cc //c:p //c:q //c:r //c:s");
    assertThat(evalToString("attr(cmd,':', c:*)")).isEqualTo("//c:c");
    // Using "empty" pattern will just check existence of the attribute.
    assertThat(evalToString("attr(cmd,'', c:*)")).isEqualTo("//c:c");
    assertThat(evalToString("attr(linkshared, 0, c:*)")).isEqualTo("//c:d");
    assertThat(evalToString("attr('data', 'r', c:*)")).isEqualTo("//c:d");
    // Empty list attribute value always resolves to '[]'. If list attribute has
    // more than one value, the will be delimited with ','.
    assertThat(evalToString("attr('deps', '\\[\\]', c:*)")).isEqualTo("//c:d");
    assertThat(evalToString("attr('deps', '^..$', c:*)")).isEqualTo("//c:d");
    assertThat(evalToString("attr('srcs', '\\[[^,]+\\]', c:*)")).isEqualTo("//c:d");

    // Configurable attributes:
    if (testConfigurableAttributes()) {
      assertThat(evalToString("attr('deps', 'bdep', //configurable/...)"))
          .isEqualTo("//configurable:main");
      assertThat(evalToString("attr('deps', 'nomatch', //configurable/...)")).isEmpty();
    }
  }

  /** Regression test for b/16835016: don't crash when evaluating null-valued attributes. */
  @Test
  public void testNullAttrOperator() throws Exception {
    writeBuildFiles2();
    assertThat(evalToString("attr(deprecation, ' ', c:*)")).isEmpty();
  }

  private void writeBooleanBuildFiles() throws Exception {
    writeFile(
        "t/BUILD",
        "cc_library(name='t', srcs=['t.cc'], data=['r'], testonly=0)",
        "cc_library(name='t_test', srcs=['t.cc'], data=['r'], testonly=1)");
  }

  @Test
  public void testAttrOperatorOnBooleans() throws Exception {
    writeBooleanBuildFiles();
    // Assure that integers query correctly for BOOLEAN values.
    assertThat(evalToString("attr(testonly, 0, t:*)")).isEqualTo("//t:t");
    assertThat(evalToString("attr(testonly, 1, t:*)")).isEqualTo("//t:t_test");
  }

  @Test
  public void testSomeOperator() throws Exception {
    writeBuildFiles2();
    assertThat(eval("some(c:*)")).hasSize(1);
    assertContains(eval("c:*"), eval("some(c:*)"));
    assertThat(evalToString("some(//c:q)")).isEqualTo("//c:q");

    assertThat(evalThrows("some(//c:q intersect //c:p)", true)).isEqualTo("argument set is empty");
  }

  protected void writeBuildFiles3() throws Exception {
    writeFile(
        "a/BUILD",
        "genrule(name='a', srcs=['//b', '//c'], outs=['out'], cmd=':')",
        "exports_files(['a2'])");
    writeFile("b/BUILD", "genrule(name='b', srcs=['//d'], outs=['out'], cmd=':')");
    writeFile("c/BUILD", "genrule(name='c', srcs=['//d'], outs=['out'], cmd=':')");
    writeFile("d/BUILD", "exports_files(['d'])");
  }

  protected void writeBuildFilesWithConfigurableAttributesUnconditionally() throws Exception {
    writeFile(
        "conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'})",
        "config_setting(",
        "    name = 'b',",
        "    values = {'test_arg': 'b'})");
    writeFile(
        "configurable/BUILD",
        "cc_binary(",
        "    name = 'main',",
        "    srcs = ['main.cc'],",
        "    deps = select({",
        "        '//conditions:a': [':adep'],",
        "        '//conditions:b': [':bdep'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':defaultdep'],",
        "    }))",
        "cc_library(",
        "    name = 'adep',",
        "    srcs = ['adep.cc'])",
        "cc_library(",
        "    name = 'bdep',",
        "    srcs = ['bdep.cc'])",
        "cc_library(",
        "    name = 'defaultdep',",
        "    srcs = ['defaultdep.cc'])");
  }

  private void writeBuildFilesWithConfigurableAttributes() throws Exception {
    if (testConfigurableAttributes()) {
      writeBuildFilesWithConfigurableAttributesUnconditionally();
    }
  }

  @Test
  public void testSomePathOperator() throws Exception {
    writeBuildFiles3();
    writeBuildFilesWithConfigurableAttributes();

    assertThat(eval("somepath(//a, //a:a2)")).isEqualTo(EMPTY); // no path
    assertThat(eval("somepath(//d, //a)")).isEqualTo(EMPTY); // no path

    Set<T> somepathAToD = eval("somepath(//a, //d)");
    assertThat(somepathAToD).containsAtLeastElementsIn(eval("//a"));
    Set<T> aAndB = eval("//a + //b");
    // Contains one of {//b, //c}:
    assertThat(somepathAToD).containsAnyIn(aAndB);
    assertContains(somepathAToD, eval("//d"));

    // Configurable attributes:
    if (testConfigurableAttributes()) {
      assertThat(eval("somepath(//configurable:main, //configurable:bdep.cc)"))
          .isEqualTo(eval("//configurable:main + //configurable:bdep + //configurable:bdep.cc"));
    }
  }

  @Test
  public void testSomePathOperatorOrdering() throws Exception {
    writeFile(
        "a/BUILD",
        "genrule(name='a1', srcs=['//b', '//c'], outs=['out1'], cmd=':')",
        "genrule(name='a0', srcs=[':a1'], outs=['out0'], cmd=':')");
    writeFile("b/BUILD", "genrule(name='b', srcs=['//d'], outs=['out'], cmd=':')");
    writeFile("c/BUILD", "genrule(name='c', srcs=['//d'], outs=['out'], cmd=':')");
    writeFile("d/BUILD", "exports_files(['d'])");

    List<String> pathList1 = ImmutableList.of("//a:a0", "//a:a1", "//b:b", "//d:d");
    List<String> pathList2 = ImmutableList.of("//a:a0", "//a:a1", "//c:c", "//d:d");

    List<String> somepathAToD = evalToListOfStrings("somepath(//a:a0, //d)");
    if (somepathAToD.contains("//b:b")) {
      assertThat(pathList1).isEqualTo(somepathAToD);
    } else {
      assertThat(somepathAToD).isEqualTo(pathList2);
    }
  }

  @Test
  public void testAllPathsOperator() throws Exception {
    writeBuildFiles3();
    writeBuildFilesWithConfigurableAttributes();

    assertThat(eval("somepath(//a, //a:a2)")).isEqualTo(EMPTY); // no path
    assertThat(eval("somepath(//d, //a)")).isEqualTo(EMPTY); // no path

    Set<T> allpathsAtoD = eval("allpaths(//a, //d)");
    assertThat(allpathsAtoD).containsAtLeastElementsIn(eval("//a + //b + //c + //d"));

    // Configurable attributes:
    if (testConfigurableAttributes()) {
      assertThat(eval("allpaths(//configurable:main, //configurable:bdep.cc)"))
          .isEqualTo(eval("//configurable:main + //configurable:bdep + //configurable:bdep.cc"));
    }
  }

  @Test
  public void testPathOperatorsWithOutputFile() throws Exception {
    writeFile("a/BUILD", "genrule(name='a', outs=['out'], cmd=':')");

    assertThat(eval("somepath(//a, //a:out)")).isEqualTo(EMPTY); // no path
    assertThat(eval("allpaths(//a, //a:out)")).isEqualTo(EMPTY); // no path

    assertThat(eval("somepath(//a:out, //a)")).isEqualTo(eval("//a + //a:out"));
    assertThat(eval("allpaths(//a:out, //a)")).isEqualTo(eval("//a + //a:out"));
  }

  @Test
  public void testDeps() throws Exception {
    writeBuildFiles3();
    writeBuildFilesWithConfigurableAttributes();

    assertThat(eval("deps(//d)")).isEqualTo(eval("//d"));
    assertThat(eval("deps(//c)" + getDependencyCorrectionWithGen()))
        .isEqualTo(eval("//c union //d"));
    assertThat(eval("deps(//b)" + getDependencyCorrectionWithGen()))
        .isEqualTo(eval("//b union //d"));
    assertThat(eval("deps(//a)" + getDependencyCorrectionWithGen()))
        .isEqualTo(eval("//a union //b union //c union //d"));

    assertThat(eval("deps(//c:out)")).isEqualTo(eval("deps(//c) union //c:out"));
    assertThat(eval("deps(//b:out)")).isEqualTo(eval("deps(//b) union //b:out"));
    assertThat(eval("deps(//a:out)")).isEqualTo(eval("deps(//a) union //a:out"));

    // Test depth-bounded variant:
    assertThat(eval("deps(//a, 0)" + getDependencyCorrectionWithGen())).isEqualTo(eval("//a"));
    assertThat(eval("deps(//a, 1)" + getDependencyCorrectionWithGen()))
        .isEqualTo(eval("//a union //b union //c"));
    assertThat(eval("deps(//a, 2)" + getDependencyCorrectionWithGen()))
        .isEqualTo(eval("//a + //b + //c + //d"));

    // Regression coverage for bug #1561800:
    // "blaze query 'deps(<output file>, 1)' returns the output file,
    // not its generating rule"
    assertThat(eval("deps(//a:out, 0)")).isEqualTo(eval("//a:out"));
    assertThat(eval("deps(//a:out, 1)")).isEqualTo(eval("//a:out + //a"));
    assertThat(eval("deps(//a:out, 2)" + getDependencyCorrectionWithGen()))
        .isEqualTo(eval("//a:out + //a + //b + //c"));

    // Configurable attributes:
    if (testConfigurableAttributes()) {
      String implicitDeps = "";
      if (analysisMock.isThisBazel()) {
        implicitDeps =
            " + "
                + helper.getToolsRepository()
                + "//tools/def_parser:def_parser"
                + " + "
                + helper.getToolsRepository()
                + "//tools/cpp:grep-includes";
      }
      assertThat(eval("deps(//configurable:main, 1)" + TestConstants.CC_DEPENDENCY_CORRECTION))
          .containsExactlyElementsIn(
              eval(
                  helper.getToolsRepository()
                      + "//tools/cpp:malloc + //configurable:main + "
                      + "//configurable:main.cc + //configurable:adep + //configurable:bdep + "
                      + "//configurable:defaultdep + //conditions:a + //conditions:b"
                      + implicitDeps));
    }
  }

  @Test
  public void testDepsDoesNotIncludeBuildFiles() throws Exception {
    writeFile("deps/BUILD", "exports_files(['build_def', 'skylark.bzl'])");
    writeFile(
        "deps/skylark.bzl",
        "def macro():",
        "  native.genrule(name = 'dep2', outs = ['dep2.txt'], cmd = 'echo Hi >$@')");

    writeFile(
        "s/BUILD",
        "load('//deps:skylark.bzl', 'macro')",
        "macro()",
        "genrule(name = 'my_rule',",
        "        outs = ['my.txt'],",
        "        srcs = [':dep1.txt', ':dep2.txt'],",
        "        cmd = 'echo $(SRCS) >$@')");

    List<String> result = evalToListOfStrings("deps(//s:my_rule)");
    assertThat(result).containsAtLeast("//s:dep2", "//s:dep1.txt", "//s:dep2.txt", "//s:my_rule");
    assertThat(result)
        .containsNoneOf("//deps:BUILD", "//deps:build_def", "//deps:skylark.bzl", "//s:BUILD");
  }

  @Test
  public void testSkylarkDiamondEquality() throws Exception {
    writeFile(
        "foo/BUILD",
        "load('//foo:a.bzl', 'A')",
        "load('//foo:b.bzl', 'B')",
        "load('//foo:checker.bzl', 'check')",
        "check(A.c, B.c)",
        "check(B.a, A)",
        "sh_library(name = 'foo')");
    writeFile(
        "foo/a.bzl",
        "load('//foo:c.bzl', 'C')",
        "A = struct(c = C)",
        "# comment to make sure this formats properly");
    writeFile(
        "foo/b.bzl",
        "load('//foo:a.bzl', 'A')",
        "load('//foo:c.bzl', 'C')",
        "B = struct(a = A, c = C)");
    writeFile("foo/c.bzl", "C = struct()");
    writeFile(
        "foo/checker.bzl",
        "def check(arg1, arg2):",
        "  if arg1 != arg2:",
        "    fail('Long error message just saying that the two args passed in were not equal')");
    // Check no errors.
    assertThat(evalToString("//foo:foo")).isEqualTo("//foo:foo");
  }

  @Test
  public void testRdeps() throws Exception {
    writeBuildFiles3();
    writeBuildFilesWithConfigurableAttributes();

    assertThat(eval("rdeps(//a, //d)" + getDependencyCorrection()))
        .isEqualTo(eval("//a union //b union //c union //d"));
    assertThat(eval("rdeps(//b, //d)" + getDependencyCorrection()))
        .isEqualTo(eval("//b union //d"));
    assertThat(eval("rdeps(//b union //c, //d)" + getDependencyCorrection()))
        .isEqualTo(eval("//b union //c union //d"));
    assertThat(eval("rdeps(//a union //c, //b)" + getDependencyCorrection()))
        .isEqualTo(eval("//a union //b"));
    assertThat(eval("rdeps(//a:out union //c:out, //b)" + getDependencyCorrection()))
        .isEqualTo(eval("//a union //a:out union //b"));
    assertThat(eval("rdeps(//d, //a)" + getDependencyCorrection())).isEqualTo(EMPTY);

    // Test depth-bounded variant:
    assertThat(eval("rdeps(//a, //d, 1)" + getDependencyCorrection()))
        .isEqualTo(eval("//b union //c union //d"));
    assertThat(eval("rdeps(//a, //d, 0)" + getDependencyCorrection())).isEqualTo(eval("//d"));

    // Configurable attributes:
    if (testConfigurableAttributes()) {
      assertThat(eval("rdeps(//configurable:all, //configurable:adep.cc)"))
          .isEqualTo(eval("//configurable:main + //configurable:adep + //configurable:adep.cc"));
      assertThat(eval("rdeps(//configurable:all, //configurable:bdep.cc)"))
          .isEqualTo(eval("//configurable:main + //configurable:bdep + //configurable:bdep.cc"));
      assertThat(eval("rdeps(//configurable:all, //configurable:defaultdep.cc)"))
          .isEqualTo(
              eval(
                  "//configurable:main + //configurable:defaultdep + "
                      + "//configurable:defaultdep.cc"));
    }
  }

  @Test
  public void testLet() throws Exception {
    writeBuildFiles3();

    assertContains(
        eval("//b + //c + //d"),
        eval("let x = //a in deps($x) except $x" + getDependencyCorrectionWithGen()));
    assertThat(evalThrows("$undefined", true)).isEqualTo("undefined variable 'undefined'");
  }

  @Test
  public void testScopeOfLetExpressions() throws Exception {
    int numTargets = 1000;

    StringBuilder filesBuilder = new StringBuilder("'0'");
    for (int i = 1; i < numTargets; i++) {
      filesBuilder.append(String.format(", '%d'", i));
    }
    String files = filesBuilder.toString();
    writeFile("a/BUILD", "exports_files([" + files + "])");

    StringBuilder letQueryBuilder = new StringBuilder("(let x = //a:0 in $x)");
    for (int i = 1; i < numTargets; i++) {
      letQueryBuilder.append(String.format(" + (let x = //a:%d in $x)", i));
    }
    String letQuery = letQueryBuilder.toString();

    assertThat(eval(letQuery)).containsExactlyElementsIn(eval("//a:* - //a:BUILD"));
  }

  @Test
  public void testSubdirSymlinkCycle() throws Exception {
    writeBuildFiles1();
    helper.ensureSymbolicLink("a/s", "s");
    assertThat(evalToString("a/...:*")).isEqualTo(A_AB_FILES);
  }

  @Test
  public void testCycleInSubpackage() throws Exception {
    writeFile("a/BUILD", "sh_library(name = 'a', deps = [':dep'])", "sh_library(name = 'dep')");
    writeFile("a/subdir/BUILD", "sh_library(name = 'cycletarget', deps = ['cycletarget'])");
    assertThat(evalToListOfStrings("deps(//a:a)")).containsExactly("//a:a", "//a:dep");
  }

  protected void setupCycleInSkylarkParentDir() throws Exception {
    writeFile("a/BUILD", "load('//a:cycle1.bzl', 'C1')", "sh_library(name = 'a')");
    writeFile("a/cycle1.bzl", "load('//a:cycle2.bzl', 'C2')", "C1 = struct()");
    writeFile("a/cycle2.bzl", "load('//a:cycle1.bzl', 'C1')", "C2 = struct()");
    writeFile("a/subdir/BUILD", "sh_library(name = 'subdir')");
  }

  @Test
  public void testCycleInSkylarkParentDir() throws Exception {
    setupCycleInSkylarkParentDir();
    assertThat(evalToListOfStrings("//a/subdir:all")).containsExactly("//a/subdir:subdir");
  }

  @Test
  public void testNestedLetExpressions() throws Exception {
    writeFile("a/BUILD", "exports_files(['f1', 'f2'])");
    writeFile("b/BUILD", "exports_files(['f1', 'f2'])");
    String letQuery =
        "let x1 = //a:f1 in "
            + "let x2 = //a:f2 in "
            + "let x1 = //b:f1 in "
            + "let x2 = //b:f2 in "
            + "$x1 + $x2";
    assertThat(eval(letQuery)).containsExactlyElementsIn(eval("//b:f1 + //b:f2"));
  }

  @Test
  public void testBuildFiles() throws Exception {
    writeBuildFiles3();
    assertThat(eval("//a ^ //b")).isEqualTo(EMPTY);
    assertThat(eval("buildfiles(//a ^ //b)")).isEqualTo(EMPTY);
    assertThat(eval("buildfiles(//a)")).isEqualTo(eval("//a:BUILD"));
    assertThat(eval("buildfiles(//b)")).isEqualTo(eval("//b:BUILD"));
    assertThat(eval("buildfiles(//a + //b)")).isEqualTo(eval("//a:BUILD + //b:BUILD"));
  }

  @Test
  public void testBuildFilesDoesNotReturnVisibilityOfRule() throws Exception {
    writeFile("fruit/BUILD", "sh_library(name='fruit', visibility=['//fruit/lemon:lemon'])");
    writeFile("fruit/lemon/BUILD", "package_group(name='lemon', packages=['//fruit/...'])");
    assertThat(eval("buildfiles(//fruit:all)")).isEqualTo(eval("//fruit:BUILD"));
  }

  @Test
  public void testBuildFilesDoesNotReturnVisibilityOfBUILD() throws Exception {
    writeFile(
        "fruit/BUILD",
        "sh_library(name='fruit', srcs=['fruit.sh'])",
        "exports_files(['BUILD'], visibility=['//fruit/lemon:lemon'])");
    writeFile("fruit/lemon/BUILD", "package_group(name='lemon', packages=['//fruit/...'])");

    assertThat(eval("buildfiles(//fruit:all)")).isEqualTo(eval("//fruit:BUILD"));
  }

  @Test
  public void testNoImplicitDeps() throws Exception {
    writeFile("x/BUILD", "cc_binary(name='x', srcs=['x.cc'])");

    // Implicit dependencies:
    String hostDepsExpr = helper.getToolsRepository() + "//tools/cpp:malloc";
    String implicitDepsExpr = "";
    if (analysisMock.isThisBazel()) {
      implicitDepsExpr +=
          " + "
              + helper.getToolsRepository()
              + "//tools/def_parser:def_parser"
              + " + "
              + helper.getToolsRepository()
              + "//tools/def_parser:def_parser.exe"
              + " + "
              + helper.getToolsRepository()
              + "//tools/cpp:grep-includes";
    }

    String targetDepsExpr = "//x:x + //x:x.cc";

    // Test all combinations of --[no]host_deps and --[no]implicit_deps on //x:x
    assertEqualsFiltered(
        targetDepsExpr + " + " + hostDepsExpr + implicitDepsExpr,
        "deps(//x)" + TestConstants.CC_DEPENDENCY_CORRECTION);
    assertEqualsFiltered(
        targetDepsExpr + " + " + hostDepsExpr,
        "deps(//x)" + TestConstants.CC_DEPENDENCY_CORRECTION,
        Setting.ONLY_TARGET_DEPS);
    assertEqualsFiltered(targetDepsExpr, "deps(//x)", Setting.NO_IMPLICIT_DEPS);
    assertEqualsFiltered(
        targetDepsExpr, "deps(//x)", Setting.ONLY_TARGET_DEPS, Setting.NO_IMPLICIT_DEPS);
  }

  protected void assertEqualsFiltered(String expected, String actual, Setting... settings)
      throws Exception {
    helper.setQuerySettings(settings);
    assertThat(eval(actual)).containsExactlyElementsIn(eval(expected));
  }

  private void runNodepDepsTest(boolean expectVisibilityDep, Setting... settings) throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 't1', deps = [':t2'], visibility = [':pg'])",
        "sh_library(name = 't2')",
        "package_group(name = 'pg')");

    helper.setQuerySettings(settings);

    if (expectVisibilityDep) {
      assertThat(eval("deps(//foo:t1)")).contains(Iterables.getOnlyElement(eval("//foo:pg")));
    } else {
      assertThat(eval("deps(//foo:t1)")).doesNotContain(Iterables.getOnlyElement(eval("//foo:pg")));
    }
  }

  @Test
  public void testNodepDeps_DefaultIsTrue() throws Exception {
    runNodepDepsTest(/*expectVisibilityDep=*/ true);
  }

  @Test
  public void testNodepDeps_False() throws Exception {
    runNodepDepsTest(/*expectVisibilityDep=*/ false, Setting.NO_NODEP_DEPS);
  }

  @Test
  public void testCycleInSkylark() throws Exception {
    writeFile("a/BUILD", "load('//a:cycle1.bzl', 'C1')", "sh_library(name = 'a')");
    writeFile("a/cycle1.bzl", "load('//a:cycle2.bzl', 'C2')", "C1 = struct()");
    writeFile("a/cycle2.bzl", "load('//a:cycle1.bzl', 'C1')", "C2 = struct()");
    try {
      evalThrows("//a:all", false);
    } catch (QueryException e) {
      // Expected.
    }
  }

  @Test
  public void testLabelsOperator() throws Exception {
    writeBuildFiles3();
    writeBuildFilesWithConfigurableAttributes();
    writeFile("k/BUILD", "py_binary(name='k', srcs=['k.py'])");
    analysisMock.pySupport().setup(mockToolsConfig);

    // srcs:
    assertThat(eval("labels(srcs, //a)")).isEqualTo(eval("//b + //c"));
    assertThat(eval("labels(srcs, //b)")).isEqualTo(eval("//d"));
    assertThat(eval("labels(srcs, //b + //a)")).isEqualTo(eval("//b + //c + //d"));

    // outs:
    assertThat(eval("labels(outs, //a)")).isEqualTo(eval("//a:out"));
    assertThat(eval("labels(outs, //b)")).isEqualTo(eval("//b:out"));
    assertThat(eval("labels(outs, //d)")).isEqualTo(EMPTY); // d is a file

    // empty:
    assertThat(eval("labels(data, //b + //a)")).isEqualTo(EMPTY);

    // no such attribute
    assertThat(eval("labels(no_such_attr, //b)")).isEqualTo(EMPTY);

    // singleton LABEL:
    assertThat(eval("labels(srcs, //k)")).isEqualTo(eval("//k:k.py"));

    // Works for implicit edges too.  This is for consistency with --output
    // xml, which exposes them too.
    String toolsRepository = helper.getToolsRepository();
    assertThat(eval("labels(\"$python2to3\", //k)"))
        .isEqualTo(eval(toolsRepository + "//tools/python:2to3"));

    // Configurable deps:
    if (testConfigurableAttributes()) {
      assertThat(eval("labels(\"deps\", //configurable:main)"))
          .isEqualTo(eval("//configurable:adep + //configurable:bdep + //configurable:defaultdep"));
    }
  }

  /* tests(x) operator */

  @Test
  public void testTestsOperatorExpandsTestsAndExcludesNonTests() throws Exception {
    writeFile(
        "a/BUILD",
        "test_suite(name='a')",
        "sh_test(name='sh_test', srcs=['sh_test.sh'])",
        "py_test(name='py_test', srcs=['py_test.py'])",
        "cc_test(name='cc_test')",
        "cc_binary(name='cc_binary')");
    assertThat(eval("tests(//a)")).isEqualTo(eval("//a:sh_test + //a:py_test + //a:cc_test"));
  }

  @Test
  public void testTestsOperatorFiltersByTagSizeAndEnv() throws Exception {
    writeFile(
        "b/BUILD",
        "test_suite(name='large_tests', tags=['large'])",
        "test_suite(name='prod_tests', tags=['prod'])",
        "test_suite(name='foo_tests', tags=['foo'])",
        "sh_test(name='sh_test', size='large', srcs=['sh_test.sh'])",
        "py_test(name='py_test', tags=['prod'], srcs=['py_test.py'])",
        "cc_test(name='cc_test', tags=['foo'])");

    assertThat(eval("tests(//b:large_tests)")).isEqualTo(eval("//b:sh_test"));
    assertThat(eval("tests(//b:prod_tests)")).isEqualTo(eval("//b:py_test"));
    assertThat(eval("tests(//b:foo_tests)")).isEqualTo(eval("//b:cc_test"));
  }

  @Test
  public void testTestsOperatorFiltersByNegativeTag() throws Exception {
    writeFile(
        "b/BUILD",
        "test_suite(name='foo_tests', tags=['foo'])",
        "test_suite(name='bar_tests', tags=['bar'])",
        "test_suite(name='foo_notbar_tests', tags=['foo', '-bar'])",
        "py_test(name='py_test', tags=['blah', 'prod'], srcs=['py_test.py'])",
        "cc_test(name='cc_test', tags=['foo'])",
        "cc_test(name='cc_test2', tags=['bar'])",
        "cc_test(name='cc_test3', tags=['foo', 'bar'])");

    assertThat(eval("tests(//b:foo_notbar_tests)")).isEqualTo(eval("//b:cc_test"));
    assertThat(eval("tests(//b:foo_tests)")).isEqualTo(eval("//b:cc_test + //b:cc_test3"));
    assertThat(eval("tests(//b:bar_tests)")).isEqualTo(eval("//b:cc_test2 + //b:cc_test3"));
  }

  @Test
  public void testTestsOperatorCrossesPackages() throws Exception {
    writeFile("c/BUILD", "test_suite(name='c', tests=['//d:suite'])");
    writeFile(
        "d/BUILD", "test_suite(name='suite')", "sh_test(name='sh_test', srcs=['sh_test.sh'])");

    assertThat(eval("tests(//c)")).isEqualTo(eval("//d:sh_test"));
  }

  @Test
  public void testTestsOperatorHandlesCyclesGracefully() throws Exception {
    writeFile("c/BUILD", "test_suite(name='c', tests=['//d'])");
    writeFile("d/BUILD", "test_suite(name='d', tests=['//c'])");

    assertThat(eval("tests(//c)")).isEqualTo(EMPTY); // Doesn't crash or get stuck.
  }

  @Test
  public void testTestSuiteInTestsAttributeAndViceVersa() throws Exception {
    writeFile(
        "cherry/BUILD",
        "test_suite(name='cherry', tests=[':suite', ':direct'])",
        "test_suite(name='suite', tests=[':indirect'])",
        "sh_test(name='direct', srcs=['direct.sh'])",
        "sh_test(name='indirect', srcs=['indirect.sh'])");

    assertThat(eval("tests(//cherry:cherry)"))
        .isEqualTo(eval("//cherry:direct + //cherry:indirect"));
  }

  @Test
  public void testTestsOperatorReportsMissingTargets() throws Exception {
    writeFile("c/BUILD", "test_suite(name='c', tests=['//d'])");
    writeFile("d/BUILD");

    assertStartsWith(
        "couldn't expand 'tests' attribute of test_suite //c:c: " + "no such target '//d:d'",
        evalThrows("tests(//c)", false));
  }

  @Test
  public void testDotDotDotWithUnrelatedCycle() throws Exception {
    writeFile("a/BUILD", "sh_library(name = 'a')");
    writeFile(
        "cycle/BUILD",
        "sh_library(name = 'cycle1', deps = ['cycle2'])",
        "sh_library(name = 'cycle2', deps = ['cycle1'])");
    assertThat(eval("//a:a")).isEqualTo(eval("//a/..."));
  }

  @Test
  public void testDotDotDotWithCycle() throws Exception {
    writeFile("a/BUILD", "sh_library(name = 'a')");
    writeFile(
        "a/b/BUILD",
        "sh_library(name = 'cycle1', deps = ['cycle2'])",
        "sh_library(name = 'cycle2', deps = ['cycle1'])");
    assertThat(eval("//a:a + //a/b:cycle1 + //a/b:cycle2")).isEqualTo(eval("//a/..."));
  }

  /* set(x) operator */

  @Test
  public void testSet() throws Exception {
    writeBuildFiles3();

    assertThat(eval("set()")).isEqualTo(EMPTY);
    assertThat(eval("set(\t//a\n//b )")).isEqualTo(eval("//a + //b"));
    assertThat(eval("set(//a //b //c //d)")).isEqualTo(eval("//a + //b + //c + //d"));
  }

  /* Regression tests */

  // Regression test for bug #1153968, "CRASH in query: getTransitiveClosure
  // called without prior call to buildTransitiveClosure".
  @Test
  public void testRuleOutputAmbiguityIsntFatal() throws Exception {
    writeFile("x/BUILD", "genrule(name='x', outs=['x'], cmd='')");
    Set<T> result = eval("allpaths(x:*, //x)"); // doesn't crash
    // result = { genrule(//x) }
    assertThat(result).hasSize(1);
    T r = result.iterator().next();
    assertThat(helper.getLabel(r).toString()).isEqualTo("//x:x");
  }

  // Regression test for bug #2340261:
  // "blaze query doesn't show deps that come from the default_visibility..."
  @Test
  public void testDefaultVisibilityReturnedInDeps() throws Exception {
    writeFile(
        "kiwi/BUILD", "package(default_visibility=['//mango:mango'])", "sh_library(name='kiwi')");
    writeFile("mango/BUILD", "package_group(name='mango', packages=[])");

    Set<T> result = eval("deps(//kiwi:kiwi)" + getDependencyCorrection());
    assertThat(result).isEqualTo(eval("//mango:mango + //kiwi:kiwi"));
  }

  @Test
  public void testDefaultVisibilityReturnedInDeps_NonEmptyDependencyFilter() throws Exception {
    writeFile(
        "kiwi/BUILD", "package(default_visibility=['//mango:mango'])", "sh_library(name='kiwi')");
    writeFile("mango/BUILD", "package_group(name='mango', packages=[])");

    helper.setQuerySettings(Setting.ONLY_TARGET_DEPS);

    Set<T> result = eval("deps(//kiwi:kiwi)" + getDependencyCorrection());
    assertThat(result).isEqualTo(eval("//mango:mango + //kiwi:kiwi"));
  }

  @Test
  public void testDefaultVisibilityReturnedInDepsForInputFiles() throws Exception {
    writeFile(
        "kiwi/BUILD",
        "package(default_visibility=['//mango:mango'])",
        "sh_library(name='kiwi', srcs=['kiwi.sh'])");
    writeFile("mango/BUILD", "package_group(name='mango', packages=[])");

    Set<T> result = eval("deps(//kiwi:kiwi.sh)");
    assertThat(result).isEqualTo(eval("//mango:mango + //kiwi:kiwi.sh"));
  }

  // Regression test for bug #2827101:
  // "Package group dependencies are not taken into account by gcheckout"
  @Test
  public void testIncludesReturnedInDeps() throws Exception {
    writeFile(
        "peach/BUILD",
        "package_group(name='peach',",
        "              includes=[':seed'])",
        "package_group(name='seed',",
        "              includes=[':cyanide'])",
        "package_group(name='cyanide',",
        "              packages=['//hydrogen', '//nitrogen', '//carbon'])",
        "sh_library(name='dessert',",
        "           visibility=[':peach'])");

    Set<T> result = eval("deps(//peach:dessert)" + getDependencyCorrection());
    assertThat(result)
        .isEqualTo(eval("//peach:peach + //peach:seed + //peach:cyanide + //peach:dessert"));
  }

  // Regression test for #1267510, modification of result of subexpression
  // evaluation.
  @Test
  public void testRegression1267510() throws Exception {
    writeFile("x/BUILD");
    writeFile("y/BUILD");

    // somepath(x:BUILD, y:BUILD) returns a constant empty set.  "+" should not
    // attempt to modify its LHS operand.
    assertThat(eval("somepath(x:BUILD, y:BUILD) + x:BUILD")).isEqualTo(eval("x:BUILD"));
  }

  // Regression test for #1309697, NPE crash during Blaze query.
  @Test
  public void testRegression1309697() throws Exception {
    writeFile("x/BUILD", "cc_library(name='x', srcs=['a.cc', 'a.cc'])");
    String expectedError = "Label '//x:a.cc' is duplicated in the 'srcs' attribute of rule 'x'";
    if (helper.isKeepGoing()) {
      assertThat(evalThrows("//x", false)).isEqualTo(expectedError);
    } else {
      evalThrows("//x", false);
      assertContainsEvent(expectedError);
    }
  }

  // Private helper of testGraphOrderOfWildcards.
  private T one(String label) throws Exception {
    return eval(label).iterator().next();
  }

  private static <T> DotOutputVisitor<T> createVisitor(PrintWriter writer) {
    return new DotOutputVisitor<T>(
        writer,
        new LabelSerializer<T>() {
          @Override
          public String serialize(Node<T> node) {
            return node.getLabel().toString();
          }
        });
  }

  @Test
  public void testGraphOrderOfWildcards() throws Exception {
    // TODO(blaze-team): (2009) we could use some helpers for graph order tests.
    writeFile(
        "x/BUILD",
        "genrule(name='x', srcs=['y'], outs=['x.out'], cmd=':')",
        "genrule(name='y', outs=['y.out'], cmd=':')");
    helper.setOrderedResults(true); // This query needs a graph.

    ResultAndTargets<T> resultAndTargets = helper.evaluateQuery("//x:*");
    @SuppressWarnings("unchecked")
    DigraphQueryEvalResult<T> digraphResult =
        (DigraphQueryEvalResult<T>) resultAndTargets.getQueryEvalResult();
    Set<T> results = resultAndTargets.getResultSet();
    Digraph<T> subgraph = digraphResult.getGraph().extractSubgraph(results);

    T xBuild = one("//x:BUILD");
    T xx = one("//x:x");
    T xxout = one("//x:x.out");
    T xy = one("//x:y");
    T xyout = one("//x:y.out");

    assertThat(results).isEqualTo(ImmutableSet.of(xBuild, xx, xxout, xy, xyout));

    Digraph<T> expected = new Digraph<>();
    expected.addEdge(xyout, xy);
    expected.addEdge(xx, xy);
    expected.addEdge(xxout, xx);
    expected.createNode(xBuild);
    if (!expected.equals(subgraph)) {
      // TODO(blaze-team): (2009) make this a utility method of Digraph.
      System.err.println("Expected:");
      expected.visitNodesBeforeEdges(
          AbstractQueryTest.<T>createVisitor(
              new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.err, UTF_8)))),
          null);
      System.err.println("Was:");
      subgraph.visitNodesBeforeEdges(
          AbstractQueryTest.<T>createVisitor(
              new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.err, UTF_8)))),
          null);
      fail();
    }
  }

  // Regression test for bug #1345896, "Blaze query p:* loads more packages
  // than just p".
  @Test
  public void testWildcardsDontLoadUnnecessaryPackages() throws Exception {
    writeFile("x/BUILD", "cc_library(name='x', deps=['//y'])");
    writeFile("y/BUILD");

    eval("//x:*");
    helper.assertPackageNotLoaded("@//y");
  }

  // #1352570, "NPE crash in deps(x, n)".
  @Test
  public void testRegression1352570() throws Exception {
    writeFile(
        "x/BUILD",
        "cc_library(name='x', deps=['z'])",
        "cc_library(name='y', deps=['z'])",
        "cc_library(name='z')");
    Set<T> result = eval("deps(//x:x + //x:y, 2) intersect //x:*"); // no crash
    assertThat(result).isEqualTo(eval("//x:x + //x:y + //x:z"));
  }

  // Regression test for bug #1686119,
  // "blaze query dies with java.lang.IllegalArgumentException".
  @Test
  public void testRegressionBug1686119() throws Exception {
    writeFile(
        "x/BUILD",
        "Fileset(name='x',",
        "        entries=[FilesetEntry(files=['a'])],",
        "        out='y')");
    assertEqualsFiltered("//x:x + //x:a", "deps(//x:x)");
    assertEqualsFiltered("//x:x + //x:a", "deps(//x:x)", Setting.ONLY_TARGET_DEPS);
    assertEqualsFiltered("//x:x + //x:a", "deps(//x:x)", Setting.NO_IMPLICIT_DEPS);
  }

  @Test
  public void testFilesetPackageDeps() throws Exception {
    writeFile(
        "x/BUILD",
        "Fileset(name='glob',",
        "        entries=[FilesetEntry()],",
        "        out='glob')",
        "Fileset(name='noglob',",
        "        entries=[FilesetEntry(files=['a'])],",
        "        out='noglob')");

    Set<T> globResult = eval("deps(//x:glob)");
    Set<T> noglobResult = eval("deps(//x:noglob)");

    assertContains(globResult, eval("//x:BUILD"));
    assertNotContains(noglobResult, eval("//x:BUILD"));
  }

  /** Tests that the default_hdrs_check value is correctly propagated to individual rules. */
  @Test
  public void testHdrsCheck() throws Exception {
    writeFile(
        "x/BUILD",
        "package(default_hdrs_check='strict')",
        "cc_library(name='a')",
        "cc_library(name='b', hdrs_check='loose')");

    assertThat(eval("attr('hdrs_check', 'strict', //x:all)")).isEqualTo(eval("//x:a"));
    assertThat(eval("attr('hdrs_check', 'loose', //x:all)")).isEqualTo(eval("//x:b"));
  }

  @Test
  public void testDefaultCopts() throws Exception {
    writeFile("x/BUILD", "package(default_copts=['-a'])", "cc_library(name='a')");
    assertThat(eval("attr('$default_copts', '\\[-a\\]', //x:all)")).isEqualTo(eval("//x:a"));
  }

  @Test
  public void testTestSuiteWithFile() throws Exception {
    // Note that test_suite does not restrict the set of targets that can appear here.
    writeFile("x/BUILD", "test_suite(name='a', tests=['a.txt'])");
    assertThat(eval("tests(//x:a)")).isEmpty();
    assertThat(eval("deps(//x:a)")).isEqualTo(eval("//x:a + //x:a.txt"));
  }

  @Test
  public void testStrictTestSuiteWithFile() throws Exception {
    helper.setQuerySettings(Setting.TESTS_EXPRESSION_STRICT);
    writeFile("x/BUILD", "test_suite(name='a', tests=['a.txt'])");
    assertThat(evalThrows("tests(//x:a)", false))
        .isEqualTo(
            "The label '//x:a.txt' in the test_suite '//x:a' does not refer to a test or "
                + "test_suite rule!");
  }

  @Test
  public void testAmbiguousAllResolvesToTestSuiteNamedAll() throws Exception {
    helper.setQuerySettings(Setting.TESTS_EXPRESSION_STRICT);
    writeFile(
        "x/BUILD",
        "cc_test(name='one')",
        "cc_test(name='two')",
        "test_suite(name='all', tests=[':one'])");
    assertThat(eval("tests(//x:all)")).isEqualTo(eval("//x:one"));
    // Expect an ambiguity warning in the event handler.
    assertContainsEvent(
        "The target pattern '//x:all' is ambiguous: ':all' is both a wildcard, and "
            + "the name of an existing test_suite rule; using the latter interpretation");
  }

  // Test that long expressions can be parsed and evaluated (without stackoverflow)
  @Test
  public void testBigExpression() throws Exception {
    writeBuildFiles3();

    StringBuilder query = new StringBuilder();
    query.append("//a");
    for (int i = 1; i < 10000; i++) {
      query.append("+ //b");
    }
    assertThat(eval(query.toString())).isEqualTo(eval("//a + //b"));
  }

  @Test
  public void testSlashSlashDotDotDot() throws Exception {
    helper.clearAllFiles();
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "sh_library(name = 'a', srcs = ['a.sh'])");
    assertThat(eval("//...")).isEqualTo(eval("//a"));
  }

  @Test
  public void testQueryTimeLoadingOfTargetPatternHappyPath() throws Exception {
    // Given a workspace containing two packages, "//a" and "//a/b",
    writeFile("a/BUILD", "sh_library(name = 'a')");
    writeFile("a/b/BUILD", "sh_library(name = 'b')");

    // When the query environment is queried for "//a/b:b" which hasn't been loaded,
    Set<T> queryTimeLoadedPattern = eval("//a/b:b");

    // Then the query evaluates to that target.
    assertThat(queryTimeLoadedPattern).hasSize(1);
  }

  @Test
  public void testQueryTimeLoadingOfTargetsBelowPackageHappyPath() throws Exception {
    // Given a workspace containing three packages, "//a", "//a/b", and "//a/b/c",
    writeFile("a/BUILD", "sh_library(name = 'a')");
    writeFile("a/b/BUILD", "sh_library(name = 'b')");
    writeFile("a/b/c/BUILD", "sh_library(name = 'c')");

    // When the query environment is queried for "//a/b/..." which hasn't been loaded,
    Set<T> queryTimeLoadedPattern = eval("//a/b/...");

    // Then the query evaluates to the two targets "//a/b:b" and "//a/b/c:c".
    assertThat(queryTimeLoadedPattern).hasSize(2);
  }

  @Test
  public void testQueryTimeLoadingTargetsBelowMissingPackage() throws Exception {
    // Given a workspace containing one package, "//a",
    writeFile("a/BUILD", "sh_library(name = 'a')");

    // When the query environment is queried for targets belonging to packages beneath the
    // package "a/b", which doesn't exist,
    String missingPackage = "a/b";
    String s = evalThrows("//" + missingPackage + "/...", false);

    // Then an exception is thrown that says that the pattern matched nothing.
    assertThat(s).containsMatch("no targets found beneath '" + missingPackage + "'");
  }

  @Test
  public void testQueryTimeLoadingTargetsBelowNonPackageDirectory() throws Exception {
    // Given a workspace containing two packages, "//a/b/c", and "//a/b/c/d",
    writeFile("a/b/c/BUILD", "sh_library(name = 'c')");
    writeFile("a/b/c/d/BUILD", "sh_library(name = 'd')");

    // When the query environment is queried for "//a/b/..." which hasn't been loaded,
    Set<T> queryTimeLoadedPattern = eval("//a/b/...");

    // Then the query evaluates to the two targets "//a/b/c:c" and "//a/b/c/d:d".
    assertThat(queryTimeLoadedPattern).hasSize(2);
  }

  private void useExtendedSetOfRules() throws Exception {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(TestAspects.BASE_RULE);
    builder.addRuleDefinition(TestAspects.ASPECT_REQUIRING_RULE);
    builder.addRuleDefinition(TestAspects.EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER_RULE);
    builder.addRuleDefinition(TestAspects.HONEST_RULE);
    builder.addRuleDefinition(TestAspects.SIMPLE_RULE);
    helper.useRuleClassProvider(builder.build());
  }

  @Test
  public void testHaveDepsOnAspectsAttributes() throws Exception {
    try {
      useExtendedSetOfRules();
      writeFile(
          "a/BUILD",
          "extra_attribute_aspect_requiring_provider(name='a', foo=[':b'])",
          "honest(name='b', foo=[])");
      writeFile("extra/BUILD", "honest(name='extra', foo=[])");

      Truth.assertThat(evalToString("deps(//a:a)")).contains("//extra:extra");
    } finally {
      helper.clearAllFiles();
      helper.useRuleClassProvider(TestRuleClassProvider.getRuleClassProvider());
    }
  }

  @Test
  public void testNoDepsOnAspectAttributeWhenAspectMissing() throws Exception {
    try {
      useExtendedSetOfRules();
      writeFile(
          "a/BUILD",
          "aspect(name='a', foo=[':b'])",
          "honest(name='b', foo=[])",
          "extra_attribute_aspect_requiring_provider(name='c', foo=[':d'])",
          "simple(name='d', foo=[])");
      writeFile("extra/BUILD", "honest(name='extra', foo=[])");

      assertThat(evalToString("deps(//a:a)")).doesNotContain("//extra:extra");
      assertThat(evalToString("deps(//a:c)")).doesNotContain("//extra:extra");
    } finally {
      helper.clearAllFiles();
      helper.useRuleClassProvider(TestRuleClassProvider.getRuleClassProvider());
    }
  }

  @Test
  public void testNoDepsOnAspectAttributeWithNoImpicitDeps() throws Exception {
    try {
      useExtendedSetOfRules();
      helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
      writeFile(
          "a/BUILD",
          "extra_attribute_aspect_requiring_provider(name='a', foo=[':b'])",
          "honest(name='b', foo=[])");
      writeFile("extra/BUILD", "honest(name='extra', foo=[])");

      Truth.assertThat(evalToString("deps(//a:a)")).doesNotContain("//extra:extra");
    } finally {
      helper.clearAllFiles();
      helper.useRuleClassProvider(TestRuleClassProvider.getRuleClassProvider());
    }
  }

  public void simpleVisibilityTest(String visibility, boolean expectVisible) throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a,//b");
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "filegroup(name = 'a', srcs = ['//b:b'])");
    writeFile(
        "b/BUILD", "filegroup(name = 'b', srcs = ['b.txt'], visibility = ['" + visibility + "'])");
    String actual = evalToString("visible(//a, somepath(//a, //b))");
    if (expectVisible) {
      assertThat(actual).isEqualTo("//a:a //b:b");
    } else {
      assertThat(actual).isEqualTo("//a:a");
    }
  }

  @Test
  public void testVisible_simple_public() throws Exception {
    simpleVisibilityTest("//visibility:public", true);
  }

  @Test
  public void testVisible_simple_private() throws Exception {
    simpleVisibilityTest("//visibility:private", false);
  }

  @Test
  public void testVisible_simple_package() throws Exception {
    simpleVisibilityTest("//a:__pkg__", true);
  }

  @Test
  public void testVisible_simple_subpackages() throws Exception {
    simpleVisibilityTest("//a:__subpackages__", true);
  }

  @Test
  public void testVisible_simple_different_subpackages() throws Exception {
    simpleVisibilityTest("//c:__subpackages__", false);
  }

  @Test
  public void testVisible_private_same_package() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a:a,//a:b");
    writeFile("WORKSPACE");
    writeFile(
        "a/BUILD",
        "filegroup(name = 'a', srcs = [':b'], visibility = ['//visibility:private'])",
        "filegroup(name = 'b', srcs = ['b.txt'], visibility = ['//visibility:private'])");
    assertThat(evalToString("visible(//a:a, somepath(//a:a, //a:b))")).isEqualTo("//a:a //a:b");
  }

  @Test
  public void testVisible_package_group() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a,//b");
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "filegroup(name = 'a', srcs = ['//b:b'])");
    writeFile(
        "b/BUILD",
        "package_group(name = 'friends', packages = ['//a', '//b'])",
        "filegroup(name = 'b', srcs = ['b.txt'], visibility = [':friends'])");
    assertThat(evalToString("visible(//a, somepath(//a, //b))")).isEqualTo("//a:a //b:b");
  }

  @Test
  public void testVisible_package_group_invisible() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a,//b");
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "filegroup(name = 'a', srcs = ['//b:b'])");
    writeFile(
        "b/BUILD",
        "package_group(name = 'friends', packages = ['//c'])",
        "filegroup(name = 'b', srcs = ['b.txt'], visibility = [':friends'])");
    writeFile("c/BUILD");
    assertThat(evalToString("visible(//a, somepath(//a, //b))")).isEqualTo("//a:a");
  }

  @Test
  public void testVisible_package_group_include() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a,//b");
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "filegroup(name = 'a', srcs = ['//b:b'])");
    writeFile(
        "b/BUILD",
        "package_group(name = 'friends', packages = ['//c'], includes = [':friends_of_friends'])",
        "package_group(name = 'friends_of_friends', packages = ['//a'])",
        "filegroup(name = 'b', srcs = ['b.txt'], visibility = [':friends'])");
    writeFile("c/BUILD");
    assertThat(evalToString("visible(//a, somepath(//a, //b))")).isEqualTo("//a:a //b:b");
  }

  @Test
  public void testVisible_java_javatests() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//java/com/google/a,//javatests/com/google/a");
    writeFile("WORKSPACE");
    writeFile(
        "java/com/google/a/BUILD",
        "filegroup(name = 'a', srcs = ['a.txt'], visibility = ['//visibility:private'])");
    writeFile(
        "javatests/com/google/a/BUILD",
        "filegroup(name = 'a', srcs = ['//java/com/google/a:a'],"
            + " visibility = ['//visibility:private'])");
    assertThat(
            evalToString(
                "visible(//javatests/com/google/a,"
                    + " somepath(//javatests/com/google/a, //java/com/google/a))"))
        .isEqualTo("//java/com/google/a:a //javatests/com/google/a:a");
  }

  @Test
  public void testVisible_java_javatests_different_package() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//java/com/google/a,//javatests/com/google/b");
    writeFile("WORKSPACE");
    writeFile(
        "java/com/google/a/BUILD",
        "filegroup(name = 'a', srcs = ['a.txt'], visibility = ['//visibility:private'])");
    writeFile(
        "javatests/com/google/b/BUILD",
        "filegroup(name = 'b', srcs = ['//java/com/google/a:a'],"
            + " visibility = ['//visibility:private'])");
    assertThat(
            evalToString(
                "visible(//javatests/com/google/b,"
                    + " somepath(//javatests/com/google/b, //java/com/google/a))"))
        .isEqualTo("//javatests/com/google/b:b");
  }

  // java cannot see javatests
  @Test
  public void testVisible_javatests_java() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//java/com/google/a,//javatests/com/google/a");
    writeFile("WORKSPACE");
    writeFile(
        "javatests/com/google/a/BUILD",
        "filegroup(name = 'a', srcs = ['a.txt'], visibility = ['//visibility:private'])");
    writeFile(
        "java/com/google/a/BUILD",
        "filegroup(name = 'a', srcs = ['//javatests/com/google/a:a'],"
            + " visibility = ['//visibility:private'])");
    assertThat(
            evalToString(
                "visible(//java/com/google/a,"
                    + " somepath(//java/com/google/a, //javatests/com/google/a))"))
        .isEqualTo("//java/com/google/a:a");
  }

  @Test
  public void testVisible_default_private() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a,//b");
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "filegroup(name = 'a', srcs = ['//b'])");
    writeFile(
        "b/BUILD",
        "package(default_visibility = ['//visibility:private'])",
        "filegroup(name = 'b', srcs = ['b.txt'])");
    assertThat(evalToString("visible(//a, somepath(//a, //b))")).isEqualTo("//a:a");
  }

  @Test
  public void testVisible_default_public() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a,//b");
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "filegroup(name = 'a', srcs = ['//b'])");
    writeFile(
        "b/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "filegroup(name = 'b', srcs = ['b.txt'])");
    assertThat(evalToString("visible(//a, somepath(//a, //b))")).isEqualTo("//a:a //b:b");
  }

  @Test
  public void testPackageGroupAllBeneath() throws Exception {
    helper.clearAllFiles();
    helper.setUniverseScope("//a,//b");
    writeFile("WORKSPACE");
    writeFile("a/BUILD", "filegroup(name = 'a', srcs = ['//b:b'])");
    writeFile(
        "b/BUILD",
        "package_group(name = 'friends', packages = ['//a/...'])",
        "filegroup(name = 'b', srcs = ['b.txt'], visibility = [':friends'])");
    assertThat(evalToString("visible(//a, somepath(//a, //b))")).isEqualTo("//a:a //b:b");
  }

  @Test
  public void testBuildfilesWithDuplicates() throws Exception {
    writeFile(
        "foo/BUILD", "load('//baz:baz.bzl', 'x')", "sh_library(name = 'foo', deps = ['//baz'])");
    writeFile(
        "bar/BUILD", "load('//baz:baz.bzl', 'x')", "sh_library(name = 'bar', deps = ['//baz'])");
    writeFile("baz/BUILD", "load('//baz:baz.bzl', 'x')", "sh_library(name = 'baz')");
    writeFile("baz/baz.bzl", "x = 2");
    assertThat(evalToString("buildfiles(deps(//foo)) + buildfiles(deps(//bar))"))
        .isEqualTo("//bar:BUILD //baz:BUILD //baz:baz.bzl //foo:BUILD");
  }

  @Test
  public void testTargetsFromBuildfilesAndRealTargets() throws Exception {
    writeFile(
        "foo/BUILD", "load('//baz:baz.bzl', 'x')", "sh_library(name = 'foo', deps = ['//baz'])");
    writeFile(
        "baz/BUILD",
        "load('//baz:baz.bzl', 'x')",
        "exports_files(['baz.bzl'])",
        "sh_library(name = 'baz')");
    writeFile("baz/baz.bzl", "x = 2");
    assertThat(evalToString("buildfiles(deps(//foo)) + //baz:BUILD + //baz:baz.bzl"))
        .isEqualTo("//baz:BUILD //baz:baz.bzl //foo:BUILD");
    assertThat(evalToString("buildfiles(deps(//foo)) ^ //baz:BUILD")).isEqualTo("//baz:BUILD");
    assertThat(evalToString("buildfiles(deps(//foo)) ^ //baz:baz.bzl")).isEqualTo("//baz:baz.bzl");
  }

  @Test
  public void testBuildfilesOfBuildfiles() throws Exception {
    writeFile("foo/BUILD", "load('//baz:baz.bzl', 'x')", "sh_library(name = 'foo')");
    writeFile("baz/BUILD", "load('//bar:bar.bzl', 'x')");
    writeFile("baz/baz.bzl", "x = 1");
    writeFile("bar/BUILD");
    writeFile("bar/bar.bzl", "x = 2");
    assertThat(evalToString("buildfiles(//foo)"))
        .isEqualTo("//baz:BUILD //baz:baz.bzl //foo:BUILD");
    assertThat(evalToString("buildfiles(buildfiles(//foo))"))
        .isEqualTo("//baz:BUILD //baz:baz.bzl //foo:BUILD");
  }

  @Test
  public void testBoundedDepsStreaming() throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 'a', deps = [':b'])",
        "sh_library(name = 'b', deps = [':c'])",
        "sh_library(name = 'c', deps = [':d'])",
        "sh_library(name = 'd')");
    assertThat(evalToString("deps(//foo:a + //foo:b, 1)" + getDependencyCorrection()))
        .isEqualTo("//foo:a //foo:b //foo:c");
  }

  @Test
  public void testBoundedRdepsStreaming() throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 'a', deps = [':b'])",
        "sh_library(name = 'b', deps = [':c'])",
        "sh_library(name = 'c', deps = [':d'])",
        "sh_library(name = 'd')");
    assertThat(evalToString("rdeps(//foo:a, //foo:d + //foo:c, 1)" + getDependencyCorrection()))
        .isEqualTo("//foo:b //foo:c //foo:d");
  }

  @Test
  public void testEqualityOfOrderedThreadSafeImmutableSet() throws Exception {
    writeFile("foo/BUILD", "sh_library(name = 'a')", "sh_library(name = 'b')");

    Set<T> targets = eval("//foo:a + //foo:b");
    QueryEnvironment<T> env = helper.getQueryEnvironment();
    ThreadSafeMutableSet<T> mutableSet = env.createThreadSafeMutableSet();
    mutableSet.addAll(targets);
    assertThat(targets).isEqualTo(mutableSet);
  }

  @Test
  public void testSiblings_Simple() throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 'a')",
        "sh_library(name = 'b')",
        "sh_library(name = 'c')",
        "sh_library(name = 'd')");
    assertThat(evalToString("siblings(//foo:a)"))
        .isEqualTo("//foo:BUILD //foo:a //foo:b //foo:c //foo:d");
  }

  @Test
  public void testSiblings_DuplicatePackages() throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 'a')",
        "sh_library(name = 'b')",
        "sh_library(name = 'c')",
        "sh_library(name = 'd')");
    assertThat(evalToString("siblings(//foo:a + //foo:b + //foo:c + //foo:d)"))
        .isEqualTo("//foo:BUILD //foo:a //foo:b //foo:c //foo:d");
  }

  @Test
  public void testSiblings_SamePackageRdeps() throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 'a', deps = [':b'])",
        "sh_library(name = 'b', deps = [':c', ':d'])",
        "sh_library(name = 'c', deps = [':d'])",
        "sh_library(name = 'd')");
    writeFile(
        "bar/BUILD",
        "sh_library(name = 'e', deps = ['//foo:d'])",
        "sh_library(name = 'f', deps = ['//foo:d'])");
    assertThat(evalToString("rdeps(//foo:* + //bar:*, //foo:d, 1)"))
        .isEqualTo("//bar:e //bar:f //foo:b //foo:c //foo:d");
    assertThat(evalToString("rdeps(siblings(//foo:d), //foo:d, 1)"))
        .isEqualTo("//foo:b //foo:c //foo:d");
    // 'same_pkg_direct_rdeps(//foo:d)' is supposed to have the same semantics as
    // 'rdeps(siblings(//foo:d), //foo:d, 1) - //foo:d'
    assertThat(evalToString("same_pkg_direct_rdeps(//foo:d)")).isEqualTo("//foo:b //foo:c");
  }

  @Test
  public void testSiblings_MatchesTargetNamedAll() throws Exception {
    writeFile(
        "foo/BUILD",
        // NOTE: target named 'all' collides with, takes precedence over the ':all' wildcard
        "sh_library(name = 'all')",
        "sh_library(name = 'ball')",
        "sh_library(name = 'call')",
        "sh_library(name = 'doll')");
    assertThat(evalToString("//foo:all")).isEqualTo("//foo:all");
    assertThat(evalToString("kind(' rule', siblings(//foo:BUILD))"))
        .isEqualTo("//foo:all //foo:ball //foo:call //foo:doll");
  }

  // Explicit test for the interaction of 'siblings' on operands coming from 'buildfiles' or
  // 'loadfiles'. The behavior here of treating a load'd .bzl file as coming from the package
  // loading it, rather than the package to which it belongs, is unfortunate, but it's the only
  // thing blaze can do with the unfortunate implementation details of 'buildfiles' and 'loadfiles'
  // (see FakeLoadTarget and other tests dealing with these functions).
  @Test
  public void testSiblings_WithBuildfiles() throws Exception {
    writeFile("foo/BUILD", "load('//bar:bar.bzl', 'x')", "sh_library(name = 'foo')");
    writeFile("bar/BUILD", "sh_library(name = 'bar')");
    writeFile("bar/bar.bzl", "x = 42");
    assertThat(evalToString("siblings(buildfiles(//foo:foo))")).isEqualTo("//foo:BUILD //foo:foo");
  }

  @Test
  public void testSamePackageRdeps_simple() throws Exception {
    writeFile(
        "foo/BUILD",
        "java_library(name = 'a', srcs = ['A.java'])",
        "java_library(name = 'b', srcs = ['B.java'], deps = [':a'])",
        "java_library(name = 'c', srcs = ['C.java'], deps = [':b'])");
    assertThat(evalToString("same_pkg_direct_rdeps(//foo:A.java)")).isEqualTo("//foo:a");
  }

  @Test
  public void testSamePackageRdeps_duplicate() throws Exception {
    writeFile(
        "foo/BUILD",
        "java_library(name = 'a', srcs = ['A.java'])",
        "java_library(name = 'b', srcs = ['B.java'], deps = [':a'])",
        "java_library(name = 'c', srcs = ['C.java'], deps = [':b'])");
    assertThat(evalToString("same_pkg_direct_rdeps(//foo:A.java + //foo:A.java)"))
        .isEqualTo("//foo:a");
  }

  @Test
  public void testSamePackageRdeps_two() throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 'a', deps = [':b'])",
        "sh_library(name = 'b', deps = [':c', ':d'])",
        "sh_library(name = 'c', deps = [':d'])",
        "sh_library(name = 'd')");
    writeFile(
        "bar/BUILD",
        "sh_library(name = 'e', deps = ['//foo:d'])",
        "sh_library(name = 'f', deps = ['//foo:d'])");
    assertThat(evalToString("kind(rule, same_pkg_direct_rdeps(//foo:d))"))
        .isEqualTo("//foo:b //foo:c");
  }

  @Test
  public void testSamePackageRdeps_twoPackages() throws Exception {
    writeFile(
        "foo/BUILD",
        "java_library(name = 'a', srcs = ['A.java'])",
        "java_library(name = 'b', srcs = ['B.java'], deps = [':a'])",
        "java_library(name = 'c', srcs = ['C.java'], deps = [':b'])");
    // //bar:d directly depends on //foo:a but is in the wrong package
    writeFile("bar/BUILD", "java_library(name = 'd', srcs = ['D.java'], deps = ['//foo:a'])");
    assertThat(evalToString("kind(rule, same_pkg_direct_rdeps(//foo:a))")).isEqualTo("//foo:b");
  }

  @Test
  public void testSamePackageRdeps_crissCross() throws Exception {
    writeFile(
        "foo/BUILD", //
        "java_library(name = 'a', srcs = ['A.java'])",
        "java_library(name = 'b', srcs = ['B.java'], deps = ['//bar:a'])");
    writeFile(
        "bar/BUILD", //
        "java_library(name = 'a', srcs = ['A.java'])",
        "java_library(name = 'b', srcs = ['B.java'], deps = ['//foo:a'])");
    assertThat(evalToString("kind(rule, same_pkg_direct_rdeps(//foo:a + //bar:a))")).isEmpty();
  }

  @Test
  public void testVisibleWithNonPackageGroupVisibility() throws Exception {
    writeFile("foo/BUILD", "sh_library(name = 'foo', visibility = ['//bar:bar'])");
    writeFile("bar/BUILD", "sh_library(name = 'bar')");
    assertThat(evalToString("visible(//bar:bar, //foo:foo)")).isEmpty();
  }

  @Test
  public void testVisibleWithPackageGroupWithNonPackageGroupIncludes() throws Exception {
    writeFile(
        "foo/BUILD",
        "sh_library(name = 'foo', visibility = [':pg'])",
        "package_group(name = 'pg', includes = ['//bar:bar'])");
    writeFile("bar/BUILD", "sh_library(name = 'bar')");
    assertThat(evalToString("visible(//bar:bar, //foo:foo)")).isEmpty();
  }

  // Regression test for default visibility of output file targets being traversed even with
  // --noimplicit_deps is set.
  @Test
  public void testDefaultVisibilityOfOutputTarget_NoImplicitDeps() throws Exception {
    writeFile(
        "foo/BUILD",
        "package(default_visibility = [':pg'])",
        "genrule(name = 'gen', srcs = ['in'], outs = ['out'], cmd = 'doesntmatter')",
        "package_group(name = 'pg', includes = [':other-pg'])",
        "package_group(name = 'other-pg')");
    assertEqualsFiltered(
        "deps(//foo:gen) + //foo:out + //foo:pg + //foo:other-pg",
        "deps(//foo:out)",
        Setting.NO_IMPLICIT_DEPS);
  }

  /**
   * A helper interface that allows creating a bunch of BUILD files and running queries against
   * them. We use this rather than the existing FoundationTestCase / BuildTestCase infrastructure to
   * allow running the same test against multiple query implementations (like the deps server).
   */
  public interface QueryHelper<T> {

    /** Basic set-up; this is called once at the beginning of a test, before anything else. */
    void setUp() throws Exception;

    void setKeepGoing(boolean keepGoing);

    boolean isKeepGoing();

    void setOrderedResults(boolean orderedResults);

    void setUniverseScope(String universeScope);

    void setBlockUniverseEvaluationErrors(boolean blockUniverseEvaluationErrors);

    /** Re-initializes the query environment with the given settings. */
    void setQuerySettings(Setting... settings);

    Path getRootDirectory();

    PathFragment getBlacklistedPackagePrefixesFile();

    /** Removes all files below the package root. */
    void clearAllFiles() throws IOException;

    /** Changes the rule class provider to be used for the query evaluation. */
    void useRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider) throws Exception;

    /**
     * Create a scratch file in the given filesystem, with the given pathName, consisting of a set
     * of lines. The method returns a Path instance for the scratch file.
     */
    void writeFile(String fileName, String... lines) throws IOException;

    /** Like {@code writeFile}, but the file is written unconditionally. */
    void overwriteFile(String fileName, String... lines) throws IOException;

    /**
     * Create a symbolic link in the given filesystem from {@code link} that points to {@code
     * target}.
     */
    void ensureSymbolicLink(String link, String target) throws IOException;

    /** Return an instance of {@link QueryEnvironment} according to set-up rules. */
    QueryEnvironment<T> getQueryEnvironment();

    /** Evaluates the given query and returns the result. */
    ResultAndTargets<T> evaluateQuery(String query) throws QueryException, InterruptedException;

    default Set<T> evaluateQueryRaw(String query) throws QueryException, InterruptedException {
      return evaluateQuery(query).results;
    }

    default String getToolsRepository() {
      return "";
    }

    /**
     * Contains both the results of the query (Like if there were errors, empty result, etc.) and
     * the actual targets returned by the query.
     */
    static class ResultAndTargets<T> {

      private final QueryEvalResult queryEvalResult;
      private final Set<T> results;

      public ResultAndTargets(QueryEvalResult queryEvalResult, Set<T> results) {
        this.queryEvalResult = queryEvalResult;
        this.results = results;
      }

      public QueryEvalResult getQueryEvalResult() {
        return queryEvalResult;
      }

      public Set<T> getResultSet() {
        return results;
      }
    }

    /**
     * Clears the event storage that is used for {@link #assertContainsEvent} and {@link
     * #getFirstEvent}.
     */
    void clearEvents();

    /** Asserts that the event storage contains an event with the given message text. */
    void assertContainsEvent(String expectedMessage);

    /** Asserts that the event storage does not contain an event with the given message text. */
    void assertDoesNotContainEvent(String notExpectedMessage);

    /** Returns the message text for the first event in the event storage. */
    String getFirstEvent();

    Iterable<Event> getEvents();

    /**
     * If this implementation is backed by a package cache, this asserts that the given package is
     * not present in the cache.
     */
    void assertPackageNotLoaded(String packageName) throws Exception;

    String getLabel(T target);
  }
}
