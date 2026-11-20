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
package com.google.devtools.build.lib.query2.testutil;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.query2.FilteredDirectRdepsInUniverseExpressionMapper;
import com.google.devtools.build.lib.query2.SkyQueryEnvironment;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;

/**
 * Common low-level tests for {@link SkyQueryEnvironment#transformParsedQuery}. {@link
 * SkyQueryEnvironment}, as a whole, is tested by unit tests and integration tests.
 */
public abstract class AbstractSkyQueryEnvironmentTest {

  protected SkyQueryEnvironment makeEnv(
      ImmutableList<String> universeScope, PathFragment parserPrefix) {
    // The only parameters that matter for this test are 'universeScope' and 'parserPrefix'.
    return new SkyQueryEnvironment(
        /* keepGoing= */ false,
        /* loadingPhaseThreads= */ 1,
        /* trackIncrementalState= */ true,
        /* eventHandler= */ NullEventHandler.INSTANCE,
        /* settings= */ ImmutableSet.of(),
        /* extraFunctions= */ ImmutableList.of(),
        TargetPattern.mainRepoParser(parserPrefix),
        parserPrefix,
        /* graphFactory= */ new DummyWalkableGraphFactory(),
        UniverseScope.fromUniverseScopeList(universeScope),
        /* pkgPath= */ new PathPackageLocator(null, ImmutableList.of(), ImmutableList.of()),
        LabelPrinter.legacy());
  }

  protected void assertNoTransformation(
      String queryString, ImmutableList<String> universeScope, PathFragment parserPrefix)
      throws Exception {
    assertTransformation(queryString, queryString, universeScope, parserPrefix);
  }

  protected void assertTransformation(
      String queryString,
      String expectedTransformedQueryString,
      ImmutableList<String> universeScope,
      PathFragment parserPrefix)
      throws Exception {
    try (SkyQueryEnvironment env = makeEnv(universeScope, parserPrefix)) {
      QueryExpression query = QueryExpression.parse(queryString, env);
      QueryExpression transformedQuery = env.transformParsedQuery(query);
      assertThat(transformedQuery.toString()).isEqualTo(expectedTransformedQueryString);
    }
  }

  /**
   * Tests that a query with a candidate rdeps expression is transformed into one with an
   * appropriate allrdeps expression.
   *
   * @param skyAbsolute whether the SkyQueryEnvironment universeScope is absolute
   * @param queryAbsolute whether the universe of the rdeps expression is absolute
   * @param prefix whether to use a parserPrefix (only makes sense at least one of {@code
   *     skyAbsolute} and {@code queryAbsolute} is false)
   */
  private void runTransformParsedQueryMatchingUniverse(
      boolean skyAbsolute, boolean queryAbsolute, boolean prefix) throws Exception {
    PathFragment parserPrefix = prefix ? PathFragment.create("bar") : PathFragment.EMPTY_FRAGMENT;
    TargetPattern.Parser targetPatternParser = TargetPattern.mainRepoParser(parserPrefix);
    String skyUniversePattern = skyAbsolute ? targetPatternParser.absolutize("...") : "...";
    String queryPattern = queryAbsolute ? targetPatternParser.absolutize("...") : "...";

    // Note: The usage of let and deps here is only to make the query slightly non-trivial.
    String query = String.format("let x = deps(rdeps(%s, //foo)) in $x + $x", queryPattern);
    assertTransformation(
        query,
        "let x = deps(allrdeps(" + maybeTransformLiteral("//foo") + ")) in ($x + $x)",
        ImmutableList.of(skyUniversePattern),
        parserPrefix);
  }

  protected String maybeTransformLiteral(String targetLiteral) {
    return targetLiteral;
  }

  @Test
  public void transformParsedQuery_matchingUniverse_skyRelative_queryRelative() throws Exception {
    runTransformParsedQueryMatchingUniverse(false, false, false);
  }

  @Test
  public void transformParsedQuery_matchingUniverse_skyRelative_queryRelative_withPrefix()
      throws Exception {
    runTransformParsedQueryMatchingUniverse(false, false, true);
  }

  @Test
  public void transformParsedQuery_matchingUniverse_skyRelative_queryAbsolute() throws Exception {
    runTransformParsedQueryMatchingUniverse(false, true, false);
  }

  @Test
  public void transformParsedQuery_matchingUniverse_skyRelative_queryAbsolute_withPrefix()
      throws Exception {
    runTransformParsedQueryMatchingUniverse(false, true, true);
  }

  @Test
  public void transformParsedQuery_matchingUniverse_skyAbsolute_queryRelative() throws Exception {
    runTransformParsedQueryMatchingUniverse(true, false, false);
  }

  @Test
  public void transformParsedQuery_matchingUniverse_skyAbsolute_queryRelative_withPrefix()
      throws Exception {
    runTransformParsedQueryMatchingUniverse(true, false, true);
  }

  @Test
  public void transformParsedQuery_matchingUniverse_skyAbsolute_queryAbsolute() throws Exception {
    runTransformParsedQueryMatchingUniverse(true, true, false);
  }

  @Test
  public void transformParsedQuery_nonMatchingUniverse_tooManyPatterns() throws Exception {
    // Note: The usage of let and deps here is only to make the query slightly non-trivial.
    assertTransformation(
        "let x = deps(rdeps(//..., //foo)) in ($x + $x)",
        "let x = deps(rdeps(//..., " + maybeTransformLiteral("//foo") + ")) in ($x + $x)",
        ImmutableList.of("//...", "-//nope"),
        PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void transformParsedQuery_nonMatchingUniverse_differentPattern() throws Exception {
    // Note: The usage of let and deps here is only to make the query slightly non-trivial.
    assertTransformation(
        "let x = deps(rdeps(//..., //foo)) in ($x + $x)",
        "let x = deps(rdeps(//..., " + maybeTransformLiteral("//foo") + ")) in ($x + $x)",
        ImmutableList.of("//nope"),
        PathFragment.EMPTY_FRAGMENT);
  }

  /**
   * Tests that a query with a candidate rdeps expression (whose universe is a filter of the
   * environment's universe scope) is transformed into one with an appropriate allrdeps expression.
   *
   * @param skyAbsolute whether the SkyQueryEnvironment universeScope is absolute
   * @param queryAbsolute whether the universe of the rdeps expression is absolute
   * @param prefix whether to use a parserPrefix (only makes sense at least one of {@code
   *     skyAbsolute} and {@code queryAbsolute} is false)
   */
  private void runTransformParsedQueryMatchingFilteredUniverse(
      boolean skyAbsolute, boolean queryAbsolute, boolean prefix) throws Exception {
    PathFragment parserPrefix = prefix ? PathFragment.create("bar") : PathFragment.EMPTY_FRAGMENT;
    TargetPattern.Parser targetPatternParser = TargetPattern.mainRepoParser(parserPrefix);
    String skyUniversePattern = skyAbsolute ? targetPatternParser.absolutize("...") : "...";
    String queryPattern = queryAbsolute ? targetPatternParser.absolutize("...") : "...";

    // Note: The usage of let and deps here is only to make the query slightly non-trivial.
    String query =
        String.format(
            "let x = deps(rdeps(kind('binary', attr('tags', 'xtag', %s)), //foo, 1)) in $x + $x",
            queryPattern);
    String filteredRdepsSubquery =
        "kind('binary', attr('tags', 'xtag', allrdeps(" + maybeTransformLiteral("//foo") + ", 1)))";
    assertTransformation(
        query,
        "let x = deps(" + filteredRdepsSubquery + ") in ($x + $x)",
        ImmutableList.of(skyUniversePattern),
        parserPrefix);
  }

  @Test
  public void transformParsedQuery_matchingFilteredUniverse_skyRelative_queryRelative()
      throws Exception {
    runTransformParsedQueryMatchingFilteredUniverse(false, false, false);
  }

  @Test
  public void transformParsedQuery_matchingFilteredUniverse_skyRelative_queryRelative_withPrefix()
      throws Exception {
    runTransformParsedQueryMatchingFilteredUniverse(false, false, true);
  }

  @Test
  public void transformParsedQuery_matchingFilteredUniverse_skyRelative_queryAbsolute()
      throws Exception {
    runTransformParsedQueryMatchingFilteredUniverse(false, true, false);
  }

  @Test
  public void transformParsedQuery_matchingFilteredUniverse_skyRelative_queryAbsolute_withPrefix()
      throws Exception {
    runTransformParsedQueryMatchingFilteredUniverse(false, true, true);
  }

  @Test
  public void transformParsedQuery_matchingFilteredUniverse_skyAbsolute_queryRelative()
      throws Exception {
    runTransformParsedQueryMatchingFilteredUniverse(true, false, false);
  }

  @Test
  public void transformParsedQuery_matchingFilteredUniverse_skyAbsolute_queryRelative_withPrefix()
      throws Exception {
    runTransformParsedQueryMatchingFilteredUniverse(true, false, true);
  }

  @Test
  public void transformParsedQuery_matchingFilteredUniverse_skyAbsolute_queryAbsolute()
      throws Exception {
    runTransformParsedQueryMatchingFilteredUniverse(true, true, false);
  }

  @Test
  public void testTransformParsedQuery_differentUniverseButEligible() throws Exception {
    assertTransformation(
        "let x = deps(rdeps(kind('binary', attr('tags', 'xtag', //...)), //foo, 1)) in $x + $x",
        "let x = deps(kind('binary', attr('tags', 'xtag', kind(' rule$', allrdeps("
            + maybeTransformLiteral("//foo")
            + ", 1))))) in ($x + $x)",
        ImmutableList.of("//...:*"),
        PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void transformParsedQuery_nonMatchingFilteredUniverse_notTransformed() throws Exception {
    // Note: The usage of let and deps here is only to make the query slightly non-trivial.
    assertFilteredDirectRdepsTransformation(
        "let x = deps(rdeps(kind('k1', //baz/...), //foo, 1)) in (\"$x\" + \"$x\")",
        "let x = deps(rdeps(kind('k1', //baz/...), //foo, 1)) in ($x + $x)",
        "//...",
        PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void transformParsedQuery_filteredUniverseHasExclusion_notTransformed() throws Exception {
    // Note: The usage of let and deps here is only to make the query slightly non-trivial.
    assertFilteredDirectRdepsTransformation(
        "let x = rdeps(kind('k1', (//... - //bar/...)), //foo, 1) in (\"$x\" + \"$x\")",
        "let x = rdeps(kind('k1', (//... - //bar/...)), //foo, 1) in ($x + $x)",
        "//...",
        PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void transformParsedQuery_nonFilteringFunctionOfMatchingUniverse_notTransformed()
      throws Exception {
    // deps is non-sensical, but so are most functions applied to the universe.
    assertFilteredDirectRdepsTransformation(
        "rdeps(deps(//...), //foo, 1)",
        "rdeps(deps(//...), //foo, 1)",
        "//...",
        PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void transformParsedQuery_setOfMatchingUniverse_notTransformed() throws Exception {
    // any set expression in the rdeps' universe is a barrier to this transformation.
    assertFilteredDirectRdepsTransformation(
        "rdeps(set(//...), //foo, 1)",
        "rdeps(set(//...), //foo, 1)",
        "//...",
        PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void transformParsedQuery_letExpressionOfMatchingUniverse_notTransformed()
      throws Exception {
    // any let expression in the rdeps' universe is a barrier to this transformation.
    assertFilteredDirectRdepsTransformation(
        "rdeps(let x = //... in \"$x\", //foo, 1)",
        "rdeps(let x = //... in $x, //foo, 1)",
        "//...",
        PathFragment.EMPTY_FRAGMENT);
  }

  private void assertFilteredDirectRdepsTransformation(
      String input, String output, String universe, PathFragment parserPrefix)
      throws QuerySyntaxException {
    TargetPattern.Parser parser = TargetPattern.mainRepoParser(parserPrefix);
    TargetPattern absoluteUniverseScopePattern;
    try {
      absoluteUniverseScopePattern = parser.parse(parser.absolutize(universe));
    } catch (TargetParsingException e) {
      throw new QuerySyntaxException(e.getMessage());
    }
    FilteredDirectRdepsInUniverseExpressionMapper mapper =
        new FilteredDirectRdepsInUniverseExpressionMapper(parser, absoluteUniverseScopePattern);

    try (SkyQueryEnvironment env = makeEnv(ImmutableList.of(universe), parserPrefix)) {
      QueryExpression query = QueryExpression.parse(input, env);
      QueryExpression transformedQuery = query.accept(mapper);
      assertThat(transformedQuery.toString()).isEqualTo(output);
    }
  }
}
