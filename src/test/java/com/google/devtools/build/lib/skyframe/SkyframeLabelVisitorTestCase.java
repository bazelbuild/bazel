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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.packages.util.PreprocessorUtils;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.DelegatingWalkableGraph;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;

import org.junit.Before;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

abstract public class SkyframeLabelVisitorTestCase extends PackageLoadingTestCase {
  // Convenience constants, so test args are readable vs true/false
  protected static final boolean KEEP_GOING = true;
  protected static final boolean EXPECT_ERROR = true;
  protected TransitivePackageLoader visitor = null;
  protected CustomInMemoryFs fs = new CustomInMemoryFs(new ManualClock());
  protected PreprocessorUtils.MutableFactorySupplier preprocessorFactorySupplier =
      new PreprocessorUtils.MutableFactorySupplier(null);

  abstract public PackageFactory.EnvironmentExtension getPackageEnvironmentExtension();

  @Override
  protected FileSystem createFileSystem() {
    return fs;
  }

  protected Collection<Event> assertNewBuildFileConflict() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file("pkg/BUILD", "sh_library(name = 'x', deps = ['//pkg2:q/sub'])");
    scratch.file("pkg2/BUILD", "sh_library(name = 'q/sub')");

    assertLabelsVisited(
        ImmutableSet.of("//pkg:x", "//pkg2:q/sub"),
        ImmutableSet.of("//pkg:x"),
        !EXPECT_ERROR,
        !KEEP_GOING);

    scratch.file("pkg2/q/BUILD");
    syncPackages();

    EventCollector warningCollector = new EventCollector(EventKind.WARNING);
    reporter.addHandler(warningCollector);
    assertLabelsVisitedWithErrors(ImmutableSet.of("//pkg:x"), ImmutableSet.of("//pkg:x"));
    assertContainsEvent("Label '//pkg2:q/sub' crosses boundary of subpackage 'pkg2/q'");
    assertContainsEvent("no such target '//pkg2:q/sub'");
    Collection<Event> warnings = Lists.newArrayList(warningCollector);
    // Check stability (not redundant).
    assertLabelsVisitedWithErrors(ImmutableSet.of("//pkg:x"), ImmutableSet.of("//pkg:x"));
    assertContainsEvent("Label '//pkg2:q/sub' crosses boundary of subpackage 'pkg2/q'");

    return warnings;
  }

  /**
   * Asserts all labels in expectedLabels are visited by walking
   * the dependency trees starting at startingLabels, and no other labels are visited.
   *
   * <p>Errors are expected.  We keep going after errors are encountered.
   */
  protected void assertLabelsVisitedWithErrors(
      Set<String> expectedLabels, Set<String> startingLabels) throws Exception {
    assertLabelsVisited(expectedLabels, startingLabels, EXPECT_ERROR, KEEP_GOING);
  }

  /**
   * Check that the expected targets were exactly those visited, and that the packages of these
   * expected targets were exactly those packages visited.
   */
  protected void assertExpectedTargets(Set<String> expectedLabels, Set<Target> startingTargets)
      throws Exception {
    Set<Label> visitedLabels =
        getVisitedLabels(
            Iterables.transform(
                startingTargets,
                new Function<Target, Label>() {
                  @Override
                  public Label apply(Target target) {
                    return target.getLabel();
                  }
                }),
            getSkyframeExecutor());
    assertThat(visitedLabels).containsExactlyElementsIn(asLabelSet(expectedLabels));

    Set<PathFragment> expectedPkgs = new HashSet<>();
    for (Label label : visitedLabels) {
      expectedPkgs.add(label.getPackageFragment());
    }

    assertEquals(expectedPkgs, getVisitedPackageNames(startingTargets));
  }

  /**
   * Asserts all labels in expectedLabels are visited by walking
   * the dependency trees starting at startingLabels, and no other labels are visited.
   *
   * @param expectedLabels The expected set of labels visited.
   * @param startingLabels Visit the transitive closure of each of these labels.
   * @param expectError Whether the visitation should succeed.
   * @param keepGoing Whether the visitation continues after encountering
   *        errors.
   */
  protected void assertLabelsVisited(
      Set<String> expectedLabels,
      Set<String> startingLabels,
      boolean expectError,
      boolean keepGoing)
      throws Exception {
    Set<Target> startingTargets = asTargetSet(startingLabels);

    // Spawn a lot of threads to help uncover concurrency issues
    boolean result =
        visitor.sync(
            reporter, startingTargets, ImmutableSet.<Label>of(), keepGoing, 200, Integer.MAX_VALUE);

    assertNotSame(expectError, result);
    assertExpectedTargets(expectedLabels, startingTargets);
  }

  /**
   * Returns the set of labels that were visited in the loading of the given starting labels.
   * Semantics are somewhat subtle in case of errors. The returned set always contains the starting
   * labels, even if they were not successfully loaded, but does not contain other unsuccessfully
   * loaded targets.
   */
  public static Set<Label> getVisitedLabels(
      Iterable<Label> startingLabels, SkyframeExecutor skyframeExecutor) {
    final WalkableGraph graph =
        new DelegatingWalkableGraph(
            ((InMemoryMemoizingEvaluator) skyframeExecutor.getEvaluatorForTesting())
                .getGraphForTesting());
    List<SkyKey> startingKeys = new ArrayList<>();
    for (Label label : startingLabels) {
      startingKeys.add(TransitiveTargetValue.key(label));
    }
    Iterable<SkyKey> nodesToVisit = new ArrayList<>(startingKeys);
    Set<SkyKey> visitedNodes = new HashSet<>();
    while (!Iterables.isEmpty(nodesToVisit)) {
      List<SkyKey> existingNodes = new ArrayList<>();
      for (SkyKey key : nodesToVisit) {
        if (graph.exists(key) && graph.getValue(key) != null && visitedNodes.add(key)) {
          existingNodes.add(key);
        }
      }
      nodesToVisit =
          Iterables.filter(
              Iterables.concat(graph.getDirectDeps(existingNodes).values()),
              new Predicate<SkyKey>() {
                @Override
                public boolean apply(SkyKey skyKey) {
                  return skyKey.functionName().equals(SkyFunctions.TRANSITIVE_TARGET);
                }
              });
    }
    visitedNodes.addAll(startingKeys);
    return ImmutableSet.copyOf(
        Collections2.transform(
            visitedNodes,
            new Function<SkyKey, Label>() {
              @Override
              public Label apply(SkyKey skyKey) {
                return (Label) skyKey.argument();
              }
            }));
  }

  /**
   * Asserts all labels in expectedLabels are visited by walking
   * the dependency trees starting at startingLabels, other labels may also be visited.
   * This is for cases where we don't care what the transitive closure of the labels is,
   * except for the labels we've specified must be within the closure.
   *
   * @param expectedLabels The expected set of labels visited.
   * @param startingLabels Visit the transitive closure of each of these labels.
   * @param expectError Whether the visitation should succeed.
   * @param keepGoing Whether the visitation continues after encountering
   *        errors.
   */
  protected void assertLabelsAreSubsetOfLabelsVisited(
      Set<String> expectedLabels,
      Set<String> startingLabels,
      boolean expectError,
      boolean keepGoing)
      throws Exception {
    Set<Target> targets = asTargetSet(startingLabels);

    // Spawn a lot of threads to help uncover concurrency issues
    boolean result =
        visitor.sync(
            reporter, targets, ImmutableSet.<Label>of(), keepGoing, 200, Integer.MAX_VALUE);
    assertNotSame(expectError, result);
    assertThat(getVisitedLabels(asLabelSet(startingLabels), skyframeExecutor))
        .containsAllIn(asLabelSet(expectedLabels));
  }

  protected void syncPackages() throws InterruptedException {
    syncPackages(ModifiedFileSet.EVERYTHING_MODIFIED);
  }

  protected void syncPackages(ModifiedFileSet modifiedFileSet) throws InterruptedException {
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(reporter, modifiedFileSet, rootDirectory);
  }

  @Override
  protected Set<Target> asTargetSet(Iterable<String> strLabels)
      throws LabelSyntaxException, NoSuchThingException, InterruptedException {
    Set<Target> targets = new HashSet<>();
    for (String strLabel : strLabels) {
      Label label = Label.parseAbsolute(strLabel);
      targets.add(getSkyframeExecutor().getPackageManager().getTarget(reporter, label));
    }
    return targets;
  }

  protected Set<PathFragment> getVisitedPackageNames(Set<Target> startingTargets) {
    ImmutableSet.Builder<PathFragment> builder = ImmutableSet.builder();
    for (PackageIdentifier packageId : visitor.getVisitedPackageNames()) {
      builder.add(packageId.getPackageFragment());
    }
    for (Target target : startingTargets) {
      builder.add(target.getPackage().getNameFragment());
    }
    return builder.build();
  }

  @Before
  public final void initializeVisitor() throws Exception {
    skyframeExecutor = super.createSkyframeExecutor(
        ImmutableList.of(getPackageEnvironmentExtension()), preprocessorFactorySupplier,
        ConstantRuleVisibility.PRIVATE, ruleClassProvider.getDefaultsPackageContent());
    this.visitor = skyframeExecutor.pkgLoader();
  }

  protected static class CustomInMemoryFs extends InMemoryFileSystem {

    private Map<Path, FileStatus> stubbedStats = Maps.newHashMap();

    public CustomInMemoryFs(ManualClock manualClock) {
      super(manualClock);
    }

    public void stubStat(Path path, @Nullable FileStatus stubbedResult) {
      stubbedStats.put(path, stubbedResult);
    }

    @Override
    public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
      if (stubbedStats.containsKey(path)) {
        return stubbedStats.get(path);
      }
      return super.stat(path, followSymlinks);
    }
  }
}
