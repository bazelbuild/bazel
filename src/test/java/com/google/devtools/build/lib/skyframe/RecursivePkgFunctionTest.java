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
import static com.google.devtools.build.skyframe.WalkableGraphUtils.exists;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link RecursivePkgFunction}. Unfortunately, we can't directly test
 * RecursivePkgFunction as it uses PackageValues, and PackageFunction uses legacy stuff that
 * isn't easily mockable. So our testing strategy is to make hacky calls to
 * SequencedSkyframeExecutor.
 *
 * <p>Target parsing tests already cover most of the behavior of RecursivePkgFunction, but there
 * are a couple of corner cases we need to test directly.
 */
@RunWith(JUnit4.class)
public class RecursivePkgFunctionTest extends BuildViewTestCase {

  private static SkyKey buildRecursivePkgKey(
      Path root, PathFragment rootRelativePath, ImmutableSet<PathFragment> excludedPaths) {
    RootedPath rootedPath = RootedPath.toRootedPath(Root.fromPath(root), rootRelativePath);
    return RecursivePkgValue.key(
        RepositoryName.MAIN, rootedPath, excludedPaths);
  }

  private RecursivePkgValue buildRecursivePkgValue(Path root, PathFragment rootRelativePath)
      throws Exception {
    return buildRecursivePkgValue(root, rootRelativePath, ImmutableSet.of());
  }

  private RecursivePkgValue buildRecursivePkgValue(
      Path root, PathFragment rootRelativePath, ImmutableSet<PathFragment> excludedPaths)
      throws Exception {
    SkyKey key = buildRecursivePkgKey(root, rootRelativePath, excludedPaths);
    return getEvaluationResult(key).get(key);
  }

  private EvaluationResult<RecursivePkgValue> getEvaluationResult(SkyKey key)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SequencedSkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHander(reporter)
            .build();
    EvaluationResult<RecursivePkgValue> evaluationResult =
        skyframeExecutor.getDriver().evaluate(ImmutableList.of(key), evaluationContext);
    Preconditions.checkState(!evaluationResult.hasError());
    return evaluationResult;
  }

  @Test
  public void testStartingAtBuildFile() throws Exception {
    scratch.file("a/b/c/BUILD");
    RecursivePkgValue value =
        buildRecursivePkgValue(rootDirectory, PathFragment.create("a/b/c/BUILD"));
    assertThat(value.getPackages().isEmpty()).isTrue();
  }

  @Test
  public void testPackagesUnderMultipleRoots() throws Exception {
    // PackageLoader doesn't support --package_path.
    initializeSkyframeExecutor(/*doPackageLoadingChecks=*/ false);
    skyframeExecutor = getSkyframeExecutor();

    Path root1 = rootDirectory.getRelative("root1");
    Path root2 = rootDirectory.getRelative("root2");
    scratch.file(root1 + "/WORKSPACE");
    scratch.file(root2 + "/WORKSPACE");
    scratch.file(root1 + "/a/BUILD");
    scratch.file(root2 + "/a/b/BUILD");
    setPackageCacheOptions("--package_path=" + "root1" + ":" + "root2");

    RecursivePkgValue valueForRoot1 = buildRecursivePkgValue(root1, PathFragment.create("a"));
    String root1Pkg = valueForRoot1.getPackages().getSingleton();
    assertThat(root1Pkg).isEqualTo("a");

    RecursivePkgValue valueForRoot2 = buildRecursivePkgValue(root2, PathFragment.create("a"));
    String root2Pkg = valueForRoot2.getPackages().getSingleton();
    assertThat(root2Pkg).isEqualTo("a/b");
  }

  @Test
  public void testSubdirectoryExclusion() throws Exception {
    // Given a package "a" with two packages below it, "a/b" and "a/c",
    scratch.file("a/BUILD");
    scratch.file("a/b/BUILD");
    scratch.file("a/c/BUILD");

    // When the top package is evaluated for recursive package values, and "a/b" is excluded,
    PathFragment excludedPathFragment = PathFragment.create("a/b");
    SkyKey key =
        buildRecursivePkgKey(
            rootDirectory, PathFragment.create("a"), ImmutableSet.of(excludedPathFragment));
    EvaluationResult<RecursivePkgValue> evaluationResult = getEvaluationResult(key);
    RecursivePkgValue value = evaluationResult.get(key);

    // Then the package corresponding to "a/b" is not present in the result,
    assertThat(value.getPackages().toList()).doesNotContain("a/b");

    // And the "a" package and "a/c" package are.
    assertThat(value.getPackages().toList()).contains("a");
    assertThat(value.getPackages().toList()).contains("a/c");

    // Also, the computation graph does not contain a cached value for "a/b".
    WalkableGraph graph = Preconditions.checkNotNull(evaluationResult.getWalkableGraph());
    assertThat(
            exists(
                buildRecursivePkgKey(rootDirectory, excludedPathFragment, ImmutableSet.of()),
                graph))
        .isFalse();

    // And the computation graph does contain a cached value for "a/c" with the empty set excluded,
    // because that key was evaluated.
    assertThat(
            exists(
                buildRecursivePkgKey(rootDirectory, PathFragment.create("a/c"), ImmutableSet.of()),
                graph))
        .isTrue();
  }

  @Test
  public void testExcludedSubdirectoryGettingPassedDown() throws Exception {
    // Given a package "a" with two packages below a directory below it, "a/b/c" and "a/b/d",
    scratch.file("a/BUILD");
    scratch.file("a/b/c/BUILD");
    scratch.file("a/b/d/BUILD");

    // When the top package is evaluated for recursive package values, and "a/b/c" is excluded,
    ImmutableSet<PathFragment> excludedPaths = ImmutableSet.of(PathFragment.create("a/b/c"));
    SkyKey key = buildRecursivePkgKey(rootDirectory, PathFragment.create("a"), excludedPaths);
    EvaluationResult<RecursivePkgValue> evaluationResult = getEvaluationResult(key);
    RecursivePkgValue value = evaluationResult.get(key);

    // Then the package corresponding to the excluded subdirectory is not present in the result,
    assertThat(value.getPackages().toList()).doesNotContain("a/b/c");

    // And the top package and other subsubdirectory package are.
    assertThat(value.getPackages().toList()).contains("a");
    assertThat(value.getPackages().toList()).contains("a/b/d");

    // Also, the computation graph contains a cached value for "a/b" with "a/b/c" excluded, because
    // "a/b/c" does live underneath "a/b".
    WalkableGraph graph = Preconditions.checkNotNull(evaluationResult.getWalkableGraph());
    assertThat(
            exists(
                buildRecursivePkgKey(rootDirectory, PathFragment.create("a/b"), excludedPaths),
                graph))
        .isTrue();
  }
}
