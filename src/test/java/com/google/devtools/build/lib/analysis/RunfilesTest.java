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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link Runfiles}.
 */
@RunWith(JUnit4.class)
public class RunfilesTest extends FoundationTestCase {

  private void checkWarning() {
    assertContainsEvent("obscured by a -> x");
    assertWithMessage("Runfiles.filterListForObscuringSymlinks should have warned once")
        .that(eventCollector.count())
        .isEqualTo(1);
    assertThat(Iterables.getOnlyElement(eventCollector).getKind()).isEqualTo(EventKind.WARNING);
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "x");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), ActionsTestUtil.createArtifact(root, "c/b"));
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadGrandParentObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "x");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b/c"), ActionsTestUtil.createArtifact(root, "b/c"));
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurerNoListener() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), ActionsTestUtil.createArtifact(root, "c/b"));
    assertThat(Runfiles.filterListForObscuringSymlinks(null, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
  }

  @Test
  public void testFilterListForObscuringSymlinksIgnoresOkObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a");
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), ActionsTestUtil.createArtifact(root, "a/b"));

    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    assertNoEvents();
  }

  @Test
  public void testFilterListForObscuringSymlinksNoObscurers() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a");
    obscuringMap.put(pathA, artifactA);
    PathFragment pathBC = PathFragment.create("b/c");
    Artifact artifactBC = ActionsTestUtil.createArtifact(root, "a/b");
    obscuringMap.put(pathBC, artifactBC);
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap)
        .entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactA),
        Maps.immutableEntry(pathBC, artifactBC));
    assertNoEvents();
  }

  private void checkConflictWarning() {
    assertContainsEvent("overwrote runfile");
    assertWithMessage("ConflictChecker.put should have warned once")
        .that(eventCollector.count())
        .isEqualTo(1);
    assertThat(Iterables.getOnlyElement(eventCollector).getKind()).isEqualTo(EventKind.WARNING);
  }

  private void checkConflictError() {
    assertContainsEvent("overwrote runfile");
    assertWithMessage("ConflictChecker.put should have errored once")
        .that(eventCollector.count())
        .isEqualTo(1);
    assertThat(Iterables.getOnlyElement(eventCollector).getKind()).isEqualTo(EventKind.ERROR);
  }

  private static final class SimpleActionLookupKey implements ActionLookupKey {
    private final String name;

    SimpleActionLookupKey(String name) {
      this.name = name;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.createHermetic(name);
    }

    @Nullable
    @Override
    public Label getLabel() {
      return null;
    }
  }

  @Test
  public void testPutDerivedArtifactWithDifferentOwnerDoesNotConflict() throws Exception {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(scratch.dir("/workspace"), RootType.Output, "out");
    PathFragment path = PathFragment.create("src/foo.cc");

    SimpleActionLookupKey owner1 = new SimpleActionLookupKey("//owner1");
    SimpleActionLookupKey owner2 = new SimpleActionLookupKey("//owner2");
    Artifact artifact1 =
        new Artifact.DerivedArtifact(root, root.getExecPath().getRelative(path), owner1);
    Artifact artifact2 =
        new Artifact.DerivedArtifact(root, root.getExecPath().getRelative(path), owner2);

    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, path, artifact1);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(path, artifact1));
    checker.put(map, path, artifact2);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(path, artifact2));
    assertNoEvents();
  }

  @Test
  public void testPutDerivedArtifactWithDifferentPathConflicts() throws Exception {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(scratch.dir("/workspace"), RootType.Output, "out");
    PathFragment path = PathFragment.create("src/foo.cc");
    PathFragment path2 = PathFragment.create("src/bar.cc");

    SimpleActionLookupKey owner1 = new SimpleActionLookupKey("//owner1");
    SimpleActionLookupKey owner2 = new SimpleActionLookupKey("//owner2");
    Artifact artifact1 =
        new Artifact.DerivedArtifact(root, root.getExecPath().getRelative(path), owner1);
    Artifact artifact2 =
        new Artifact.DerivedArtifact(root, root.getExecPath().getRelative(path2), owner2);

    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, path, artifact1);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(path, artifact1));
    checker.put(map, path, artifact2);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(path, artifact2));
    checkConflictWarning();
  }

  @Test
  public void testPutCatchesConflict() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, pathA, artifactB);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactB));
    checker.put(map, pathA, artifactC);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    checkConflictWarning();
  }

  @Test
  public void testPutReportsError() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    // Same as above but with ERROR not WARNING
    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.ERROR, reporter, null);
    checker.put(map, pathA, artifactB);
    reporter.removeHandler(failFastHandler); // So it doesn't throw AssertionError
    checker.put(map, pathA, artifactC);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    checkConflictError();
  }

  @Test
  public void testPutCatchesConflictBetweenNullAndNotNull() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, pathA, null);
    checker.put(map, pathA, artifactB);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactB));
    checkConflictWarning();
  }

  @Test
  public void testPutCatchesConflictBetweenNotNullAndNull() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    // Same as above but opposite order
    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, pathA, artifactB);
    checker.put(map, pathA, null);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, null));
    checkConflictWarning();
  }

  @Test
  public void testPutIgnoresConflict() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.IGNORE, reporter, null);
    checker.put(map, pathA, artifactB);
    checker.put(map, pathA, artifactC);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    assertNoEvents();
  }

  @Test
  public void testPutIgnoresConflictNoListener() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, null, null);
    checker.put(map, pathA, artifactB);
    checker.put(map, pathA, artifactC);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    assertNoEvents();
  }

  @Test
  public void testPutIgnoresSameArtifact() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Artifact artifactB2 = ActionsTestUtil.createArtifact(root, "b");
    assertThat(artifactB2).isEqualTo(artifactB);
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, pathA, artifactB);
    checker.put(map, pathA, artifactB2);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactB2));
    assertNoEvents();
  }

  @Test
  public void testPutIgnoresNullAndNull() {
    PathFragment pathA = PathFragment.create("a");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, pathA, null);
    // Add it again
    checker.put(map, pathA, null);
    assertThat(map.entrySet()).containsExactly(Maps.immutableEntry(pathA, null));
    assertNoEvents();
  }

  @Test
  public void testPutNoConflicts() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathA = PathFragment.create("a");
    PathFragment pathB = PathFragment.create("b");
    PathFragment pathC = PathFragment.create("c");
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b");
    Map<PathFragment, Artifact> map = new LinkedHashMap<>();

    Runfiles.ConflictChecker checker =
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null);
    checker.put(map, pathA, artifactA);
    // Add different artifact under different path
    checker.put(map, pathB, artifactB);
    // Add artifact again under different path
    checker.put(map, pathC, artifactA);
    assertThat(map.entrySet())
        .containsExactly(
            Maps.immutableEntry(pathA, artifactA),
            Maps.immutableEntry(pathB, artifactB),
            Maps.immutableEntry(pathC, artifactA))
        .inOrder();
    assertNoEvents();
  }

  @Test
  public void testBuilderMergeConflictPolicyDefault() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build();
    Runfiles r2 = new Runfiles.Builder("TESTING").merge(r1).build();
    assertThat(r2.getConflictPolicy()).isEqualTo(Runfiles.ConflictPolicy.IGNORE);
  }

  @Test
  public void testBuilderMergeConflictPolicyInherit() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build()
        .setConflictPolicy(Runfiles.ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING").merge(r1).build();
    assertThat(r2.getConflictPolicy()).isEqualTo(Runfiles.ConflictPolicy.WARN);
  }

  @Test
  public void testBuilderMergeConflictPolicyInheritStrictest() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build()
        .setConflictPolicy(Runfiles.ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING").build()
        .setConflictPolicy(Runfiles.ConflictPolicy.ERROR);
    Runfiles r3 = new Runfiles.Builder("TESTING").merge(r1).merge(r2).build();
    assertThat(r3.getConflictPolicy()).isEqualTo(Runfiles.ConflictPolicy.ERROR);
    // Swap ordering
    Runfiles r4 = new Runfiles.Builder("TESTING").merge(r2).merge(r1).build();
    assertThat(r4.getConflictPolicy()).isEqualTo(Runfiles.ConflictPolicy.ERROR);
  }

  @Test
  public void testLegacyRunfilesStructure() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment workspaceName = PathFragment.create("wsname");
    PathFragment pathB = PathFragment.create("repo/b");
    PathFragment legacyPathB = LabelConstants.EXTERNAL_PATH_PREFIX.getRelative(pathB);
    PathFragment runfilesPathB = LabelConstants.EXTERNAL_RUNFILES_PATH_PREFIX.getRelative(pathB);
    Artifact artifactB = ActionsTestUtil.createArtifactWithRootRelativePath(root, legacyPathB);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(workspaceName, true);

    Map<PathFragment, Artifact> inputManifest = Maps.newHashMap();
    inputManifest.put(runfilesPathB, artifactB);
    Runfiles.ConflictChecker checker = new Runfiles.ConflictChecker(
        Runfiles.ConflictPolicy.WARN, reporter, null);
    builder.addUnderWorkspace(inputManifest, checker);

    assertThat(builder.build().entrySet())
        .containsExactly(
            Maps.immutableEntry(workspaceName.getRelative(legacyPathB), artifactB),
            Maps.immutableEntry(PathFragment.create("repo/b"), artifactB));
    assertNoEvents();
  }

  @Test
  public void testRunfileAdded() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment workspaceName = PathFragment.create("wsname");
    PathFragment pathB = PathFragment.create("repo/b");
    PathFragment legacyPathB = LabelConstants.EXTERNAL_PATH_PREFIX.getRelative(pathB);
    PathFragment runfilesPathB = LabelConstants.EXTERNAL_RUNFILES_PATH_PREFIX.getRelative(pathB);
    Artifact artifactB = ActionsTestUtil.createArtifactWithRootRelativePath(root, legacyPathB);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(workspaceName, false);

    Map<PathFragment, Artifact> inputManifest = ImmutableMap.of(runfilesPathB, artifactB);
    Runfiles.ConflictChecker checker = new Runfiles.ConflictChecker(
        Runfiles.ConflictPolicy.WARN, reporter, null);
    builder.addUnderWorkspace(inputManifest, checker);

    assertThat(builder.build().entrySet()).containsExactly(
        Maps.immutableEntry(workspaceName.getRelative(".runfile"), null),
        Maps.immutableEntry(PathFragment.create("repo/b"), artifactB));
    assertNoEvents();
  }

  // TODO(kchodorow): remove this once the default workspace name is always set.
  @Test
  public void testConflictWithExternal() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    PathFragment pathB = PathFragment.create("repo/b");
    PathFragment externalLegacyPath = LabelConstants.EXTERNAL_PATH_PREFIX.getRelative(pathB);
    PathFragment externalRunfilesPathB =
        LabelConstants.EXTERNAL_RUNFILES_PATH_PREFIX.getRelative(pathB);
    Artifact artifactB = ActionsTestUtil.createArtifactWithRootRelativePath(root, pathB);
    Artifact artifactExternalB =
        ActionsTestUtil.createArtifactWithRootRelativePath(root, externalLegacyPath);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        PathFragment.EMPTY_FRAGMENT, false);

    Map<PathFragment, Artifact> inputManifest =
        ImmutableMap.of(pathB, artifactB, externalRunfilesPathB, artifactExternalB);
    Runfiles.ConflictChecker checker = new Runfiles.ConflictChecker(
        Runfiles.ConflictPolicy.WARN, reporter, null);
    builder.addUnderWorkspace(inputManifest, checker);

    assertThat(builder.build().entrySet()).containsExactly(
        Maps.immutableEntry(PathFragment.create("repo/b"), artifactExternalB));
    checkConflictWarning();
  }

  @Test
  public void testMergeWithSymlinks() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b/target");
    PathFragment sympathA = PathFragment.create("a/symlink");
    PathFragment sympathB = PathFragment.create("b/symlink");
    Runfiles runfilesA = new Runfiles.Builder("TESTING")
        .addSymlink(sympathA, artifactA)
        .build();
    Runfiles runfilesB = new Runfiles.Builder("TESTING")
        .addSymlink(sympathB, artifactB)
        .build();

    Runfiles runfilesC = runfilesA.merge(runfilesB);
    assertThat(runfilesC.getSymlinksAsMap(null).get(sympathA)).isEqualTo(artifactA);
    assertThat(runfilesC.getSymlinksAsMap(null).get(sympathB)).isEqualTo(artifactB);
  }

  @Test
  public void mergeAll_symlinks() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Artifact artifactB = ActionsTestUtil.createArtifact(root, "b/target");
    Artifact artifactC = ActionsTestUtil.createArtifact(root, "c/target");
    PathFragment sympathA = PathFragment.create("a/symlink");
    PathFragment sympathB = PathFragment.create("b/symlink");
    PathFragment sympathC = PathFragment.create("c/symlink");
    Runfiles runfilesA = new Runfiles.Builder("TESTING").addSymlink(sympathA, artifactA).build();
    Runfiles runfilesB = new Runfiles.Builder("TESTING").addSymlink(sympathB, artifactB).build();
    Runfiles runfilesC = new Runfiles.Builder("TESTING").addSymlink(sympathC, artifactC).build();

    Runfiles runfilesMerged = runfilesA.mergeAll(StarlarkList.immutableOf(runfilesB, runfilesC));
    assertThat(runfilesMerged.getSymlinksAsMap(null))
        .containsExactly(sympathA, artifactA, sympathB, artifactB, sympathC, artifactC);
  }

  @Test
  public void testMergeEmptyWithNonEmpty() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifactA = ActionsTestUtil.createArtifact(root, "a/target");
    Runfiles runfilesB = new Runfiles.Builder("TESTING").addArtifact(artifactA).build();
    assertThat(Runfiles.EMPTY.merge(runfilesB)).isSameInstanceAs(runfilesB);
    assertThat(runfilesB.merge(Runfiles.EMPTY)).isSameInstanceAs(runfilesB);
  }

  @Test
  public void mergeAll_emptyWithNonEmpty() throws Exception {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifact = ActionsTestUtil.createArtifact(root, "target");
    Runfiles nonEmpty = new Runfiles.Builder("TESTING").addArtifact(artifact).build();

    assertThat(Runfiles.EMPTY.mergeAll(StarlarkList.immutableOf(nonEmpty)))
        .isSameInstanceAs(nonEmpty);
    assertThat(
            Runfiles.EMPTY.mergeAll(
                StarlarkList.immutableOf(Runfiles.EMPTY, nonEmpty, Runfiles.EMPTY)))
        .isSameInstanceAs(nonEmpty);
    assertThat(nonEmpty.mergeAll(StarlarkList.immutableOf(Runfiles.EMPTY, Runfiles.EMPTY)))
        .isSameInstanceAs(nonEmpty);
    assertThat(nonEmpty.mergeAll(StarlarkList.immutableOf())).isSameInstanceAs(nonEmpty);
  }

  @Test
  public void mergeAll_emptyWithEmpty() throws Exception {
    assertThat(Runfiles.EMPTY.mergeAll(StarlarkList.immutableOf()))
        .isSameInstanceAs(Runfiles.EMPTY);
    assertThat(Runfiles.EMPTY.mergeAll(StarlarkList.immutableOf(Runfiles.EMPTY, Runfiles.EMPTY)))
        .isSameInstanceAs(Runfiles.EMPTY);
  }

  @Test
  public void testOnlyExtraMiddlemenNotConsideredEmpty() {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(
            scratch.resolve("execroot"), RootType.Middleman, PathFragment.create("out"));
    Artifact mm = ActionsTestUtil.createArtifact(root, "a-middleman");
    Runfiles runfiles = new Runfiles.Builder("TESTING").addLegacyExtraMiddleman(mm).build();
    assertThat(runfiles.isEmpty()).isFalse();
  }

  @Test
  public void testMergingExtraMiddlemen() {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(
            scratch.resolve("execroot"), RootType.Middleman, PathFragment.create("out"));
    Artifact mm1 = ActionsTestUtil.createArtifact(root, "middleman-1");
    Artifact mm2 = ActionsTestUtil.createArtifact(root, "middleman-2");
    Runfiles runfiles1 = new Runfiles.Builder("TESTING").addLegacyExtraMiddleman(mm1).build();
    Runfiles runfiles2 = new Runfiles.Builder("TESTING").addLegacyExtraMiddleman(mm2).build();
    Runfiles runfilesMerged =
        new Runfiles.Builder("TESTING").merge(runfiles1).merge(runfiles2).build();
    assertThat(runfilesMerged.getExtraMiddlemen().toList()).containsExactly(mm1, mm2);
  }

  @Test
  public void testGetEmptyFilenames() {
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.resolve("/workspace")));
    Artifact artifact = ActionsTestUtil.createArtifact(root, "my-artifact");
    Runfiles runfiles =
        new Runfiles.Builder("TESTING")
            .addArtifact(artifact)
            .addSymlink(PathFragment.create("my-symlink"), artifact)
            .addRootSymlink(PathFragment.create("my-root-symlink"), artifact)
            .setEmptyFilesSupplier(
                (manifestPaths) ->
                    manifestPaths
                        .stream()
                        .map((f) -> f.replaceName(f.getBaseName() + "-empty"))
                        .collect(ImmutableList.toImmutableList()))
            .build();
    assertThat(runfiles.getEmptyFilenames().toList())
        .containsExactly("my-artifact-empty", "my-symlink-empty");
  }
}
