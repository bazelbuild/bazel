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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.WalkableGraphUtils.exists;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternValue.PrepareDepsOfPatternSkyKeysAndExceptions;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PrepareDepsOfTargetsUnderDirectoryFunction}. */
@RunWith(JUnit4.class)
public final class PrepareDepsOfPatternFunctionTest extends BuildViewTestCase {

  private static PrepareDepsOfPatternSkyKeysAndExceptions createPrepDepsKeysMaybe(
      ImmutableList<String> patterns) {
    return PrepareDepsOfPatternValue.keys(patterns, "");
  }

  private static SkyKey createPrepDepsKey(String pattern) {
    PrepareDepsOfPatternSkyKeysAndExceptions keysAndExceptions =
        PrepareDepsOfPatternValue.keys(ImmutableList.of(pattern), "");
    assertThat(keysAndExceptions.getExceptions()).isEmpty();
    return Iterables.getOnlyElement(keysAndExceptions.getValues()).getSkyKey();
  }

  private EvaluationResult<PrepareDepsOfPatternValue> getEvaluationResult(SkyKey key)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SequencedSkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHander(reporter)
            .build();
    EvaluationResult<PrepareDepsOfPatternValue> evaluationResult =
        skyframeExecutor.getDriver().evaluate(ImmutableList.of(key), evaluationContext);
    Preconditions.checkState(!evaluationResult.hasError());
    return evaluationResult;
  }

  @Test
  public void testUnparsablePattern() {
    // Given an string that can't be parsed,
    String unparsablePattern = "Not a//parsable/.../pattern/..//";
    ImmutableList<String> unparsablePatternList = ImmutableList.of(unparsablePattern);

    // When PrepareDepsOfPatternValue.keys is called with that string as an argument,
    PrepareDepsOfPatternSkyKeysAndExceptions keysAndExceptionsResult =
        createPrepDepsKeysMaybe(unparsablePatternList);

    // Then it returns a wrapped TargetParsingException.
    assertThat(keysAndExceptionsResult.getValues()).isEmpty();
    assertThat(
        Iterables.getOnlyElement(keysAndExceptionsResult.getExceptions()).getOriginalPattern())
        .isEqualTo(unparsablePattern);
  }

  @Test
  public void testSingleTargetPatternEvaluationAndTransitiveLoading() throws Exception {
    evaluatePatternAndCheckTransitiveLoading("//a", /*adExists=*/ false);
  }

  @Test
  public void testTargetsBelowDirectoryPatternEvaluationAndTransitiveLoading() throws Exception {
    evaluatePatternAndCheckTransitiveLoading("//a/...", /*adExists=*/ true);
  }

  private void evaluatePatternAndCheckTransitiveLoading(String pattern, boolean adExists)
      throws IOException, InterruptedException, LabelSyntaxException {
    // Given a package "a" with a genrule "a" that depends on a target "b.txt" in a created
    // package "b", and a package "c" with a genrule "c", and a package "a/d" with a genrule "d".
    createPackages();

    // When PrepareDepsOfPatternFunction is evaluated for the provided pattern,
    SkyKey key = createPrepDepsKey(pattern);
    EvaluationResult<PrepareDepsOfPatternValue> evaluationResult =
        getEvaluationResult(key);
    WalkableGraph graph = Preconditions.checkNotNull(evaluationResult.getWalkableGraph());

    // Then the result is not null,
    Preconditions.checkNotNull(evaluationResult.get(key));

    // And the TransitiveTraversalValue for "a:a" is evaluated,
    SkyKey aaKey = TransitiveTraversalValue.key(Label.parseAbsolute("@//a:a", ImmutableMap.of()));
    assertThat(exists(aaKey, graph)).isTrue();

    // And that TransitiveTraversalValue depends on "b:b.txt".
    Iterable<SkyKey> depsOfAa =
        Iterables.getOnlyElement(graph.getDirectDeps(ImmutableList.of(aaKey)).values());
    SkyKey bTxtKey =
        TransitiveTraversalValue.key(Label.parseAbsolute("@//b:b.txt", ImmutableMap.of()));
    assertThat(depsOfAa).contains(bTxtKey);

    // And the TransitiveTraversalValue for "b:b.txt" is evaluated.
    assertThat(exists(bTxtKey, graph)).isTrue();

    // And the TransitiveTraversalValue for "c:c" is NOT evaluated.
    SkyKey ccKey = TransitiveTraversalValue.key(Label.parseAbsolute("@//c:c", ImmutableMap.of()));
    assertThat(exists(ccKey, graph)).isFalse();

    // And the TransitiveTraversalValue for "a/d:d" is or is not evaluated depending on the provided
    // expectation.
    SkyKey adKey = TransitiveTraversalValue.key(Label.parseAbsolute("@//a/d:d", ImmutableMap.of()));
    assertThat(exists(adKey, graph)).isEqualTo(adExists);
  }

  /**
   * Creates a package "a" with a genrule "a" that depends on a target "b.txt" in a created
   * package "b", and a package "c" with a genrule "c", and a package "a/d" with a genrule "d".
   */
  private void createPackages() throws IOException {
    scratch.file("a/BUILD", "genrule(name='a', cmd='', srcs=['//b:b.txt'], outs=['a.out'])");
    scratch.file("b/BUILD", "exports_files(['b.txt'])");
    scratch.file("c/BUILD", "genrule(name='c', cmd='', srcs=['c.txt'], outs=['c.out'])");
    scratch.file("a/d/BUILD", "genrule(name='d', cmd='', srcs=['d.txt'], outs=['d.out'])");
  }
}
