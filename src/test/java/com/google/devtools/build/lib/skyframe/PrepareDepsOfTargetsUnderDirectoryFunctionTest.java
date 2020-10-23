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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link PrepareDepsOfTargetsUnderDirectoryFunction}. Insert excuses here.
 */
@RunWith(JUnit4.class)
public class PrepareDepsOfTargetsUnderDirectoryFunctionTest extends BuildViewTestCase {

  private static SkyKey createCollectPackagesKey(
      Path root, PathFragment rootRelativePath, ImmutableSet<PathFragment> excludedPaths) {
    RootedPath rootedPath = RootedPath.toRootedPath(Root.fromPath(root), rootRelativePath);
    return CollectPackagesUnderDirectoryValue.key(
        RepositoryName.MAIN, rootedPath, excludedPaths);
  }

  private static SkyKey createPrepDepsKey(Path root, PathFragment rootRelativePath) {
    return createPrepDepsKey(root, rootRelativePath, ImmutableSet.<PathFragment>of());
  }

  private static SkyKey createPrepDepsKey(
      Path root, PathFragment rootRelativePath, ImmutableSet<PathFragment> excludedPaths) {
    RootedPath rootedPath = RootedPath.toRootedPath(Root.fromPath(root), rootRelativePath);
    return PrepareDepsOfTargetsUnderDirectoryValue.key(
        RepositoryName.MAIN, rootedPath, excludedPaths);
  }

  private static SkyKey createPrepDepsKey(
      Path root,
      PathFragment rootRelativePath,
      ImmutableSet<PathFragment> excludedPaths,
      FilteringPolicy filteringPolicy) {
    RootedPath rootedPath = RootedPath.toRootedPath(Root.fromPath(root), rootRelativePath);
    return PrepareDepsOfTargetsUnderDirectoryValue.key(
        RepositoryName.MAIN, rootedPath, excludedPaths, filteringPolicy);
  }

  private EvaluationResult<?> getEvaluationResult(SkyKey... keys) throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SequencedSkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(reporter)
            .build();
    EvaluationResult<PrepareDepsOfTargetsUnderDirectoryValue> evaluationResult =
        skyframeExecutor.getDriver().evaluate(ImmutableList.copyOf(keys), evaluationContext);
    Preconditions.checkState(!evaluationResult.hasError());
    return evaluationResult;
  }

  @Test
  public void testTransitiveLoading() throws Exception {
    // Given a package "a" with a genrule "a" that depends on a target in package "b",
    createPackages();

    // When package "a" is evaluated,
    SkyKey key = createPrepDepsKey(rootDirectory, PathFragment.create("a"));
    EvaluationResult<?> evaluationResult = getEvaluationResult(key);
    WalkableGraph graph = Preconditions.checkNotNull(evaluationResult.getWalkableGraph());

    // Then the TransitiveTraversalValue for "@//a:a" is evaluated,
    SkyKey aaKey = TransitiveTraversalValue.key(Label.create("@//a", "a"));
    assertThat(exists(aaKey, graph)).isTrue();

    // And that TransitiveTraversalValue depends on "@//b:b.txt".
    Iterable<SkyKey> depsOfAa =
        Iterables.getOnlyElement(graph.getDirectDeps(ImmutableList.of(aaKey)).values());
    SkyKey bTxtKey = TransitiveTraversalValue.key(Label.create("@//b", "b.txt"));
    assertThat(depsOfAa).contains(bTxtKey);

    // And the TransitiveTraversalValue for "b:b.txt" is evaluated.
    assertThat(exists(bTxtKey, graph)).isTrue();
  }

  @Test
  public void testTargetFilterSensitivity() throws Exception {
    // Given a package "a" with a genrule "a" that depends on a target in package "b", and a test
    // rule "aTest",
    createPackages();

    // When package "a" is evaluated under a test-only filtering policy,
    SkyKey key = createPrepDepsKey(rootDirectory, PathFragment.create("a"),
        ImmutableSet.<PathFragment>of(), FilteringPolicies.FILTER_TESTS);
    EvaluationResult<?> evaluationResult = getEvaluationResult(key);
    WalkableGraph graph = Preconditions.checkNotNull(evaluationResult.getWalkableGraph());

    // Then the TransitiveTraversalValue for "@//a:a" is not evaluated,
    SkyKey aaKey = TransitiveTraversalValue.key(Label.create("@//a", "a"));
    assertThat(exists(aaKey, graph)).isFalse();

    // But the TransitiveTraversalValue for "@//a:aTest" is.
    SkyKey aaTestKey = TransitiveTraversalValue.key(Label.create("@//a", "aTest"));
    assertThat(exists(aaTestKey, graph)).isTrue();
  }

  /**
   * Creates a package "a" with a genrule "a" that depends on a target in a created package "b",
   * and a test rule "aTest".
   */
  private void createPackages() throws IOException {
    scratch.file("a/BUILD",
        "genrule(name='a', cmd='', srcs=['//b:b.txt'], outs=['a.out'])",
        "sh_test(name='aTest', size='small', srcs=['aTest.sh'])");
    scratch.file("b/BUILD",
        "exports_files(['b.txt'])");
  }

  @Test
  public void testSubdirectoryExclusion() throws Exception {
    // Given a package "a" with two packages below it, "a/b" and "a/c",
    scratch.file("a/BUILD");
    scratch.file("a/b/BUILD");
    scratch.file("a/c/BUILD");

    // When the top package is evaluated via PrepareDepsOfTargetsUnderDirectoryValue with "a/b"
    // excluded,
    PathFragment excludedPathFragment = PathFragment.create("a/b");
    SkyKey key = createPrepDepsKey(rootDirectory, PathFragment.create("a"),
        ImmutableSet.of(excludedPathFragment));
    SkyKey collectkey =
        createCollectPackagesKey(
            rootDirectory, PathFragment.create("a"), ImmutableSet.of(excludedPathFragment));
    EvaluationResult<?> evaluationResult = getEvaluationResult(key, collectkey);
    CollectPackagesUnderDirectoryValue value =
        (CollectPackagesUnderDirectoryValue)
            evaluationResult
                .getWalkableGraph()
                .getValue(
                    createCollectPackagesKey(
                        rootDirectory,
                        PathFragment.create("a"),
                        ImmutableSet.of(excludedPathFragment)));

    // Then the value reports that "a" is a package,
    assertThat(value.isDirectoryPackage()).isTrue();

    // And only the subdirectory corresponding to "a/c" is present in the result,
    RootedPath onlySubdir =
        Iterables.getOnlyElement(
            value.getSubdirectoryTransitivelyContainsPackagesOrErrors().keySet());
    assertThat(onlySubdir.getRootRelativePath()).isEqualTo(PathFragment.create("a/c"));

    // And the "a/c" subdirectory reports a package under it.
    assertThat(value.getSubdirectoryTransitivelyContainsPackagesOrErrors().get(onlySubdir))
        .isTrue();

    // Also, the computation graph does not contain a cached value for "a/b".
    WalkableGraph graph = Preconditions.checkNotNull(evaluationResult.getWalkableGraph());
    assertThat(
            exists(
                createPrepDepsKey(
                    rootDirectory, excludedPathFragment, ImmutableSet.<PathFragment>of()),
                graph))
        .isFalse();

    // And the computation graph does contain a cached value for "a/c" with the empty set excluded,
    // because that key was evaluated.
    assertThat(
            exists(
                createPrepDepsKey(
                    rootDirectory, PathFragment.create("a/c"), ImmutableSet.<PathFragment>of()),
                graph))
        .isTrue();
  }

  @Test
  public void testExcludedSubdirectoryGettingPassedDown() throws Exception {
    // Given a package "a", and a package below it in "a/b/c", and a non-BUILD file below it in
    // "a/b/d",
    scratch.file("a/BUILD");
    scratch.file("a/b/c/BUILD");
    scratch.file("a/b/d/helloworld");

    // When the top package is evaluated for recursive package values, and "a/b/c" is excluded,
    ImmutableSet<PathFragment> excludedPaths = ImmutableSet.of(PathFragment.create("a/b/c"));
    SkyKey key = createPrepDepsKey(rootDirectory, PathFragment.create("a"), excludedPaths);
    SkyKey collectKey =
        createCollectPackagesKey(rootDirectory, PathFragment.create("a"), excludedPaths);
    EvaluationResult<?> evaluationResult = getEvaluationResult(key, collectKey);
    CollectPackagesUnderDirectoryValue value =
        (CollectPackagesUnderDirectoryValue)
            evaluationResult
                .getWalkableGraph()
                .getValue(
                    createCollectPackagesKey(
                        rootDirectory, PathFragment.create("a"), excludedPaths));

    // Then the value reports that "a" is a package,
    assertThat(value.isDirectoryPackage()).isTrue();

    // And the subdirectory corresponding to "a/b" is present in the result,
    RootedPath onlySubdir =
        Iterables.getOnlyElement(
            value.getSubdirectoryTransitivelyContainsPackagesOrErrors().keySet());
    assertThat(onlySubdir.getRootRelativePath()).isEqualTo(PathFragment.create("a/b"));

    // And the "a/b" subdirectory does not report a package under it (because it got excluded).
    assertThat(value.getSubdirectoryTransitivelyContainsPackagesOrErrors().get(onlySubdir))
        .isFalse();

    // Also, the computation graph contains a cached value for "a/b" with "a/b/c" excluded, because
    // "a/b/c" does live underneath "a/b".
    WalkableGraph graph = Preconditions.checkNotNull(evaluationResult.getWalkableGraph());
    SkyKey abKey = createCollectPackagesKey(
        rootDirectory, PathFragment.create("a/b"), excludedPaths);
    assertThat(exists(abKey, graph)).isTrue();
    CollectPackagesUnderDirectoryValue abValue =
        (CollectPackagesUnderDirectoryValue) Preconditions.checkNotNull(graph.getValue(abKey));

    // And that value says that "a/b" is not a package,
    assertThat(abValue.isDirectoryPackage()).isFalse();

    // And only the subdirectory "a/b/d" is present in that value,
    RootedPath abd =
        Iterables.getOnlyElement(
            abValue.getSubdirectoryTransitivelyContainsPackagesOrErrors().keySet());
    assertThat(abd.getRootRelativePath()).isEqualTo(PathFragment.create("a/b/d"));

    // And no package is under "a/b/d".
    assertThat(abValue.getSubdirectoryTransitivelyContainsPackagesOrErrors().get(abd)).isFalse();
  }
}
