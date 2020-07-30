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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static com.google.devtools.build.skyframe.WalkableGraphUtils.exists;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternsFunction}. */
@RunWith(JUnit4.class)
public class PrepareDepsOfPatternsFunctionTest extends BuildViewTestCase {

  private static SkyKey getKeyForLabel(Label label) {
    // Note that these tests used to look for TargetMarker SkyKeys before TargetMarker was
    // inlined in TransitiveTraversalFunction. Because TargetMarker is now inlined, it doesn't
    // appear in the graph. Instead, these tests now look for TransitiveTraversal keys.
    return TransitiveTraversalValue.key(label);
  }

  @Test
  public void testFunctionLoadsTargetAndNotUnspecifiedTargets() throws Exception {
    // Given a package "//foo" with independent target rules ":foo" and ":foo2",
    createFooAndFoo2(/*dependent=*/ false);

    // Given a target pattern sequence consisting of a single-target pattern for "//foo",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo");

    // When PrepareDepsOfPatternsFunction successfully completes evaluation,
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph contains a value for the target "@//foo:foo",
    assertValidValue(walkableGraph, getKeyForLabel(Label.create("@//foo", "foo")));

    // And the graph does not contain a value for the target "@//foo:foo2".
    assertThat(exists(getKeyForLabel(Label.create("@//foo", "foo2")), walkableGraph)).isFalse();
  }

  @Test
  public void testFunctionLoadsTargetDependencies() throws Exception {
    // Given a package "//foo" with target rules ":foo" and ":foo2",
    // And given ":foo" depends on ":foo2",
    createFooAndFoo2(/*dependent=*/ true);

    // Given a target pattern sequence consisting of a single-target pattern for "//foo",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo");

    // When PrepareDepsOfPatternsFunction successfully completes evaluation,
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph contains an entry for ":foo"'s dependency, ":foo2".
    assertValidValue(walkableGraph, getKeyForLabel(Label.create("@//foo", "foo2")));
  }

  @Test
  public void testFunctionExpandsTargetPatterns() throws Exception {
    // Given a package "@//foo" with independent target rules ":foo" and ":foo2",
    createFooAndFoo2(/*dependent=*/ false);

    // Given a target pattern sequence consisting of a pattern for "//foo:*",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo:*");

    // When PrepareDepsOfPatternsFunction successfully completes evaluation,
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph contains an entry for ":foo" and ":foo2".
    assertValidValue(walkableGraph, getKeyForLabel(Label.create("@//foo", "foo")));
    assertValidValue(walkableGraph, getKeyForLabel(Label.create("@//foo", "foo2")));
  }

  @Test
  public void testTargetParsingException() throws Exception {
    // Given no packages, and a target pattern sequence referring to a non-existent target,
    String nonexistentTarget = "//foo:foo";
    ImmutableList<String> patternSequence = ImmutableList.of(nonexistentTarget);

    // When PrepareDepsOfPatternsFunction completes evaluation,
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph does not contain an entry for ":foo",
    assertThat(exists(getKeyForLabel(Label.create("@//foo", "foo")), walkableGraph)).isFalse();
  }

  @Test
  public void testDependencyTraversalNoSuchPackageException() throws Exception {
    // Given a package "//foo" with a target ":foo" that has a dependency on a non-existent target
    // "//bar:bar" in a non-existent package "//bar",
    createFooWithDependencyOnMissingBarPackage();

    // Given a target pattern sequence consisting of a single-target pattern for "//foo",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo");

    // When PrepareDepsOfPatternsFunction completes evaluation,
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph contains an entry for ":foo",
    assertValidValue(
        walkableGraph,
        getKeyForLabel(Label.create("@//foo", "foo")),
        /*expectTransitiveException=*/ true);

    // And an entry with a NoSuchPackageException for "//bar:bar",
    Exception e = assertException(walkableGraph, getKeyForLabel(Label.create("@//bar", "bar")));
    assertThat(e).isInstanceOf(NoSuchPackageException.class);
  }

  @Test
  public void testDependencyTraversalNoSuchTargetException() throws Exception {
    // Given a package "//foo" with a target ":foo" that has a dependency on a non-existent target
    // "//bar:bar" in an existing package "//bar",
    createFooWithDependencyOnBarPackageWithMissingTarget();

    // Given a target pattern sequence consisting of a single-target pattern for "//foo",
    ImmutableList<String> patternSequence = ImmutableList.of("//foo");

    // When PrepareDepsOfPatternsFunction completes evaluation,
    WalkableGraph walkableGraph = getGraphFromPatternsEvaluation(patternSequence);

    // Then the graph contains an entry for ":foo" which has both a value and an exception,
    assertValidValue(
        walkableGraph,
        getKeyForLabel(Label.create("@//foo", "foo")),
        /*expectTransitiveException=*/ true);

    // And an entry with a NoSuchTargetException for "//bar:bar",
    Exception e = assertException(walkableGraph, getKeyForLabel(Label.create("@//bar", "bar")));
    assertThat(e).isInstanceOf(NoSuchTargetException.class);
  }

  @Test
  public void testParsingProblemsKeepGoing() throws Exception {
    parsingProblem(/*keepGoing=*/ true);
  }

  /**
   * PrepareDepsOfPatternsFunction always keeps going despite any target pattern parsing errors, in
   * keeping with the original behavior of {@link WalkableGraph.WalkableGraphFactory#prepareAndGet},
   * which always used {@code keepGoing=true} during target pattern parsing because it was
   * responsible for ensuring that queries had a complete graph to work on.
   */
  @Test
  public void testParsingProblemsNoKeepGoing() throws Exception {
    parsingProblem(/*keepGoing=*/ false);
  }

  private void parsingProblem(boolean keepGoing) throws Exception {
    // Given a package "//foo" with target rule ":foo",
    createFooAndFoo2(/*dependent=*/ false);

    // Given a target pattern sequence consisting of a pattern with parsing problems followed by
    // a legit target pattern,
    String bogusPattern = "//foo/....";
    ImmutableList<String> patternSequence = ImmutableList.of(bogusPattern, "//foo:foo");

    // When PrepareDepsOfPatternsFunction runs in the selected keep-going mode,
    WalkableGraph walkableGraph =
        getGraphFromPatternsEvaluation(patternSequence, /*keepGoing=*/ keepGoing);

    // Then it skips evaluation of the malformed target pattern, but logs about it,
    assertContainsEvent("Skipping '" + bogusPattern + "': ");

    // And then the graph contains a value for the legit target pattern's target "@//foo:foo".
    assertThat(exists(getKeyForLabel(Label.create("@//foo", "foo")), walkableGraph)).isTrue();
  }

  // Helpers:

  private WalkableGraph getGraphFromPatternsEvaluation(ImmutableList<String> patternSequence)
      throws InterruptedException {
    return getGraphFromPatternsEvaluation(patternSequence, /*keepGoing=*/ true);
  }

  private WalkableGraph getGraphFromPatternsEvaluation(
      ImmutableList<String> patternSequence, boolean keepGoing) throws InterruptedException {
    SkyKey independentTarget = PrepareDepsOfPatternsValue.key(patternSequence, "");
    ImmutableList<SkyKey> singletonTargetPattern = ImmutableList.of(independentTarget);

    // When PrepareDepsOfPatternsFunction completes evaluation,
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setNumThreads(LOADING_PHASE_THREADS)
            .setEventHandler(new Reporter(new EventBus(), eventCollector))
            .build();
    EvaluationResult<SkyValue> evaluationResult =
        getSkyframeExecutor().getDriver().evaluate(singletonTargetPattern, evaluationContext);
    // Currently all callers either expect success or pass keepGoing=true, which implies success,
    // since PrepareDepsOfPatternsFunction swallows all errors. Will need to be changed if a test
    // that evaluates with keepGoing=false and expects errors is added.
    assertThatEvaluationResult(evaluationResult).hasNoError();

    return Preconditions.checkNotNull(evaluationResult.getWalkableGraph());
  }

  private void createFooAndFoo2(boolean dependent) throws IOException {
    String dependencyIfAny = dependent ? "srcs = [':foo2']," : "";
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',",
        dependencyIfAny,
        "    outs = ['out.txt'],",
        "    cmd = 'touch $@')",
        "genrule(name = 'foo2',",
        "    outs = ['out2.txt'],",
        "    cmd = 'touch $@')");
  }

  private void createFooWithDependencyOnMissingBarPackage() throws IOException {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',",
        "    srcs = ['//bar:bar'],",
        "    outs = ['out.txt'],",
        "    cmd = 'touch $@')");
  }

  private void createFooWithDependencyOnBarPackageWithMissingTarget() throws IOException {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',",
        "    srcs = ['//bar:bar'],",
        "    outs = ['out.txt'],",
        "    cmd = 'touch $@')");
    scratch.file("bar/BUILD");
  }

  private static void assertValidValue(WalkableGraph graph, SkyKey key)
      throws InterruptedException {
    assertValidValue(graph, key, /*expectTransitiveException=*/ false);
  }

  /**
   * A node in the walkable graph may have both a value and an exception. This happens when one of a
   * node's transitive dependencies throws an exception, but its parent recovers from it.
   */
  private static void assertValidValue(
      WalkableGraph graph, SkyKey key, boolean expectTransitiveException)
      throws InterruptedException {
    assertThat(graph.getValue(key)).isNotNull();
    if (expectTransitiveException) {
      assertThat(graph.getException(key)).isNotNull();
    } else {
      assertThat(graph.getException(key)).isNull();
    }
  }

  private static Exception assertException(WalkableGraph graph, SkyKey key)
      throws InterruptedException {
    assertThat(graph.getValue(key)).isNull();
    Exception exception = graph.getException(key);
    assertThat(exception).isNotNull();
    return exception;
  }
}
