// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.query2.SkyQueryEnvironment;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for {@link TotalWeightQueryExpressionVisitor}. */
@RunWith(JUnit4.class)
public class TotalWeightQueryExpressionVisitorTest {
  private final TotalWeightQueryExpressionVisitor underTest =
      new TotalWeightQueryExpressionVisitor();

  @Test
  public void basicBodyCount() throws Exception {
    assertThat(calculateQueryWeight("$x - attr('tags', 'foo', $x)")).isEqualTo(11);
  }

  @Test
  public void nestedQueryFunctions() throws Exception {
    assertThat(calculateQueryWeight("kind('pat', allrdeps(//a + //b + //c + //d + //e + //f))"))
        .isEqualTo(21);
  }

  @Test
  public void multipleLetStatements_sameVariableRebound() throws Exception {
    assertThat(
            calculateQueryWeight("let x = (let x = //... in $x - deps($x)) in $x - allrdeps($x)"))
        .isEqualTo(15);
  }

  @Test
  public void multipleLetStatements_sameVariableReboundAlsoInVariableExpr() throws Exception {
    assertThat(calculateQueryWeight("let x = $x in $x - allrdeps($x)")).isEqualTo(7);
  }

  @Test
  public void multipleLetStatements_differentVariable() throws Exception {
    assertThat(
            calculateQueryWeight("let x = (let y = //... in $x - deps($y)) in $x - allrdeps($y)"))
        .isEqualTo(15);
  }

  @Test
  public void variableInsideSetExpression() throws Exception {
    assertThat(calculateQueryWeight("$x + set($x)")).isEqualTo(4);
  }

  private long calculateQueryWeight(String expr) throws Exception {
    try (SkyQueryEnvironment env = makeEnv()) {
      return QueryExpression.parse(expr, env).accept(underTest);
    }
  }

  private static SkyQueryEnvironment makeEnv() {
    // Creates a bare-minimum SkyQueryEnvironment usable for parsing a query expression to weigh it.
    return new SkyQueryEnvironment(
        /* keepGoing= */ false,
        /* loadingPhaseThreads= */ 1,
        /* eventHandler= */ NullEventHandler.INSTANCE,
        /* settings= */ ImmutableSet.of(),
        /* extraFunctions= */ ImmutableList.of(),
        TargetPattern.mainRepoParser(PathFragment.EMPTY_FRAGMENT),
        PathFragment.EMPTY_FRAGMENT,
        /* graphFactory= */ new WalkableGraphFactory() {
          @Override
          public EvaluationResult<SkyValue> prepareAndGet(
              Set<SkyKey> roots, EvaluationContext evaluationContext) throws InterruptedException {
            return null;
          }
        },
        UniverseScope.fromUniverseScopeList(ImmutableList.of("//...")),
        /* pkgPath= */ new PathPackageLocator(null, ImmutableList.of(), ImmutableList.of()),
        LabelPrinter.legacy());
  }
}
