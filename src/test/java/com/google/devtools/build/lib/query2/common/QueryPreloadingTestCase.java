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
package com.google.devtools.build.lib.query2.common;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.WalkableGraphUtils.exists;

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TransitiveTargetKey;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Before;

/** Base test class for query preloading tests. */
public abstract class QueryPreloadingTestCase extends PackageLoadingTestCase {
  // Convenience constant, so test args are readable vs true/false
  protected static final boolean KEEP_GOING = true;
  protected QueryTransitivePackagePreloader visitor = null;
  protected CustomInMemoryFs fs = new CustomInMemoryFs(new ManualClock());

  @Override
  protected FileSystem createFileSystem() {
    return fs;
  }

  /**
   * Asserts all labels in expectedLabels are visited by walking
   * the dependency trees starting at startingLabels, and no other labels are visited.
   *
   * <p>Errors are expected.  We keep going after errors are encountered.
   */
  protected void assertLabelsVisitedWithErrors(
      Set<String> expectedLabels, Set<String> startingLabels) throws Exception {
    assertLabelsVisited(expectedLabels, startingLabels, KEEP_GOING);
  }

  /**
   * Check that the expected targets were exactly those visited, and that the packages of these
   * expected targets were exactly those packages visited.
   */
  protected void assertExpectedTargets(Set<String> expectedLabels, Set<Label> startingLabels)
      throws Exception {
    Set<Label> visitedLabels = getVisitedLabels(startingLabels, getSkyframeExecutor());
    assertThat(visitedLabels).containsExactlyElementsIn(asLabelSet(expectedLabels));
  }

  /**
   * Asserts all labels in expectedLabels are visited by walking
   * the dependency trees starting at startingLabels, and no other labels are visited.
   *
   * @param expectedLabels The expected set of labels visited.
   * @param startingLabelStrings Visit the transitive closure of each of these labels.
   * @param keepGoing Whether the visitation continues after encountering
   *        errors.
   */
  protected void assertLabelsVisited(
      Set<String> expectedLabels,
      Set<String> startingLabelStrings,
      boolean keepGoing)
      throws Exception {
    Set<Label> startingLabels = asLabelSet(startingLabelStrings);

    // Spawn a lot of threads to help uncover concurrency issues
    visitor.preloadTransitiveTargets(
        reporter, startingLabels, keepGoing, /*parallelThreads=*/ 200, /*callerForError=*/ null);

    assertExpectedTargets(expectedLabels, startingLabels);
  }

  /**
   * Returns the set of labels that were visited in the loading of the given starting labels.
   * Semantics are somewhat subtle in case of errors. The returned set always contains the starting
   * labels, even if they were not successfully loaded, but does not contain other unsuccessfully
   * loaded targets.
   */
  public static ImmutableSet<Label> getVisitedLabels(
      Iterable<Label> startingLabels, SkyframeExecutor skyframeExecutor)
      throws InterruptedException {
    // Do an empty evaluation just to get access to the WalkableGraph.
    WalkableGraph graph =
        skyframeExecutor
            .getEvaluator()
            .evaluate(
                ImmutableSet.of(),
                EvaluationContext.newBuilder()
                    .setParallelism(1)
                    .setEventHandler(NullEventHandler.INSTANCE)
                    .build())
            .getWalkableGraph();
    List<SkyKey> startingKeys = new ArrayList<>();
    for (Label label : startingLabels) {
      startingKeys.add(TransitiveTargetKey.of(label));
    }
    Iterable<SkyKey> nodesToVisit = new ArrayList<>(startingKeys);
    Set<SkyKey> visitedNodes = new HashSet<>();
    while (!Iterables.isEmpty(nodesToVisit)) {
      List<SkyKey> existingNodes = new ArrayList<>();
      for (SkyKey key : nodesToVisit) {
        if (exists(key, graph) && graph.getValue(key) != null && visitedNodes.add(key)) {
          existingNodes.add(key);
        }
      }
      nodesToVisit =
          Iterables.filter(
              Iterables.concat(graph.getDirectDeps(existingNodes).values()),
              skyKey -> skyKey.functionName().equals(TransitiveTargetKey.NAME));
    }
    visitedNodes.addAll(startingKeys);
    return ImmutableSet.copyOf(
        Collections2.transform(visitedNodes, skyKey -> ((TransitiveTargetKey) skyKey).getLabel()));
  }

  /**
   * Asserts all labels in expectedLabels are visited by walking
   * the dependency trees starting at startingLabels, other labels may also be visited.
   * This is for cases where we don't care what the transitive closure of the labels is,
   * except for the labels we've specified must be within the closure.
   *
   * @param expectedLabels The expected set of labels visited.
   * @param startingLabels Visit the transitive closure of each of these labels.
   * @param keepGoing Whether the visitation continues after encountering
   *        errors.
   */
  protected void assertLabelsAreSubsetOfLabelsVisited(
      Set<String> expectedLabels,
      Set<String> startingLabels,
      boolean keepGoing)
      throws Exception {
    Set<Label> labels = asLabelSet(startingLabels);

    // Spawn a lot of threads to help uncover concurrency issues
    visitor.preloadTransitiveTargets(reporter, labels, keepGoing, 200, /*callerForError=*/ null);
    assertThat(getVisitedLabels(asLabelSet(startingLabels), skyframeExecutor))
        .containsAtLeastElementsIn(asLabelSet(expectedLabels));
  }

  protected void syncPackages() throws InterruptedException, AbruptExitException {
    syncPackages(ModifiedFileSet.EVERYTHING_MODIFIED);
  }

  protected void syncPackages(ModifiedFileSet modifiedFileSet)
      throws InterruptedException, AbruptExitException {
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter, modifiedFileSet, Root.fromPath(rootDirectory));
  }

  @Before
  public final void initializeVisitor() {
    setUpSkyframe(RuleVisibility.PRIVATE);
    this.visitor = skyframeExecutor.getQueryTransitivePackagePreloader();
  }

  protected static class CustomInMemoryFs extends InMemoryFileSystem {
    private final Map<PathFragment, FileStatus> stubbedStats = Maps.newHashMap();

    public CustomInMemoryFs(ManualClock manualClock) {
      super(manualClock, DigestHashFunction.SHA256);
    }

    @Override
    public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
      if (stubbedStats.containsKey(path)) {
        return stubbedStats.get(path);
      }
      return super.statIfFound(path, followSymlinks);
    }
  }
}
