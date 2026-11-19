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
package com.google.devtools.build.lib.query2;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.testutil.AbstractSkyQueryEnvironmentTest;
import com.google.devtools.build.lib.query2.testutil.DummyWalkableGraphFactory;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SkyQueryEnvironmentTest extends AbstractSkyQueryEnvironmentTest {

  private SkyQueryEnvironment makeManualEnv(boolean trackIncrementalState) {
    return new SkyQueryEnvironment(
        /* keepGoing= */ false,
        /* loadingPhaseThreads= */ 1,
        trackIncrementalState,
        /* eventHandler= */ NullEventHandler.INSTANCE,
        /* settings= */ ImmutableSet.of(),
        /* extraFunctions= */ ImmutableList.of(),
        TargetPattern.mainRepoParser(PathFragment.EMPTY_FRAGMENT),
        PathFragment.EMPTY_FRAGMENT,
        /* graphFactory= */ new DummyWalkableGraphFactory(),
        UniverseScope.fromUniverseScopeList(ImmutableList.of("//...")),
        /* pkgPath= */ new PathPackageLocator(null, ImmutableList.of(), ImmutableList.of()),
        LabelPrinter.legacy());
  }

  @Test
  public void testQueryWithRdepsAndNoTrackIncrementalStateFails() throws Exception {
    runTestWithEdgeTraversingFunction("rdeps(//..., //foo)");
  }

  @Test
  public void testQueryWithAllRdepsAndNoTrackIncrementalStateFails() throws Exception {
    runTestWithEdgeTraversingFunction("allrdeps(//...)");
  }

  @Test
  public void testQueryWithDepsAndNoTrackIncrementalStateFails() throws Exception {
    runTestWithEdgeTraversingFunction("deps(//...)");
  }

  @Test
  public void testQueryWithSiblingsAndNoTrackIncrementalStateFails() throws Exception {
    runTestWithEdgeTraversingFunction("siblings(//...)");
  }

  @Test
  public void testQueryWithSamePkgDirectRdepsAndNoTrackIncrementalStateFails() throws Exception {
    runTestWithEdgeTraversingFunction("same_pkg_direct_rdeps(//...)");
  }

  @Test
  public void testQueryWithSomePathAndNoTrackIncrementalStateFails() throws Exception {
    runTestWithEdgeTraversingFunction("somepath(//..., //...)");
  }

  @Test
  public void testQueryWithAllPathsAndNoTrackIncrementalStateFails() throws Exception {
    runTestWithEdgeTraversingFunction("allpaths(//..., //...)");
  }

  @Test
  public void testQueryWithKindAndNoTrackIncrementalStateDoesNotFail() throws Exception {
    runTestWithNonEdgeTraversingFunction("kind(rule, //...)");
  }

  @Test
  public void testQueryWithFilterAndNoTrackIncrementalStateDoesNotFail() throws Exception {
    runTestWithNonEdgeTraversingFunction("filter(pattern, //...)");
  }

  @Test
  public void testQueryWithAttrAndNoTrackIncrementalStateDoesNotFail() throws Exception {
    runTestWithNonEdgeTraversingFunction("attr(tags, foo, //...)");
  }

  private void runTestWithEdgeTraversingFunction(String queryExpr) throws Exception {
    try (SkyQueryEnvironment env = makeManualEnv(false)) {
      QueryExpression expr = QueryExpression.parse(queryExpr, env);
      Assert.assertThrows(QueryException.class, () -> env.beforeEvaluateQuery(expr, null));
    }
  }

  private void runTestWithNonEdgeTraversingFunction(String queryExpr) throws Exception {
    try (SkyQueryEnvironment env = makeManualEnv(false)) {
      QueryExpression expr = QueryExpression.parse(queryExpr, env);
      // The dummy walkable graph factory returns null or throws UnsupportedOperationException
      // because the graph is not populated. What we care about is that it did not throw a
      // QueryException for ILLEGAL_FLAG_COMBINATION.
      try {
        env.beforeEvaluateQuery(expr, null);
        Assert.fail("Expected NullPointerException or UnsupportedOperationException");
      } catch (NullPointerException | UnsupportedOperationException e) {
        // Expected
      }
    }
  }
}
