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
import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link Runfiles}.
 */
@RunWith(JUnit4.class)
public class RunfilesTest extends FoundationTestCase {

  private void checkWarning() {
    assertContainsEvent("obscured by a -> /workspace/a");
    assertEquals("Runfiles.filterListForObscuringSymlinks should have warned once",
        1, eventCollector.count());
    assertEquals(EventKind.WARNING, Iterables.getOnlyElement(eventCollector).getKind());
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(PathFragment.create("a"), root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), new Artifact(PathFragment.create("c/b"),
        root));
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadGrandParentObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(PathFragment.create("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b/c"), new Artifact(PathFragment.create("b/c"),
                                                         root));
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurerNoListener() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(PathFragment.create("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), new Artifact(PathFragment.create("c/b"),
        root));
    assertThat(Runfiles.filterListForObscuringSymlinks(null, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
  }

  @Test
  public void testFilterListForObscuringSymlinksIgnoresOkObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(PathFragment.create("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(PathFragment.create("a/b"), new Artifact(PathFragment.create("a/b"),
                                                         root));

    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    assertNoEvents();
  }

  @Test
  public void testFilterListForObscuringSymlinksNoObscurers() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = PathFragment.create("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(PathFragment.create("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    PathFragment pathBC = PathFragment.create("b/c");
    Artifact artifactBC = new Artifact(PathFragment.create("a/b"),
                                       root);
    obscuringMap.put(pathBC, artifactBC);
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap)
        .entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactA),
        Maps.immutableEntry(pathBC, artifactBC));
    assertNoEvents();
  }

  private void checkConflictWarning() {
    assertContainsEvent("overwrote runfile");
    assertEquals("ConflictChecker.put should have warned once", 1, eventCollector.count());
    assertEquals(EventKind.WARNING, Iterables.getOnlyElement(eventCollector).getKind());
  }

  private void checkConflictError() {
    assertContainsEvent("overwrote runfile");
    assertEquals("ConflictChecker.put should have errored once", 1, eventCollector.count());
    assertEquals(EventKind.ERROR, Iterables.getOnlyElement(eventCollector).getKind());
  }

  @Test
  public void testPutCatchesConflict() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
    Artifact artifactC = new Artifact(PathFragment.create("c"), root);
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
    Artifact artifactC = new Artifact(PathFragment.create("c"), root);
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
    Artifact artifactC = new Artifact(PathFragment.create("c"), root);
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
    Artifact artifactC = new Artifact(PathFragment.create("c"), root);
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
    Artifact artifactB2 = new Artifact(PathFragment.create("b"), root);
    assertEquals(artifactB, artifactB2);
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = PathFragment.create("a");
    PathFragment pathB = PathFragment.create("b");
    PathFragment pathC = PathFragment.create("c");
    Artifact artifactA = new Artifact(PathFragment.create("a"), root);
    Artifact artifactB = new Artifact(PathFragment.create("b"), root);
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
    assertEquals(Runfiles.ConflictPolicy.IGNORE, r2.getConflictPolicy());
  }

  @Test
  public void testBuilderMergeConflictPolicyInherit() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build()
        .setConflictPolicy(Runfiles.ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING").merge(r1).build();
    assertEquals(Runfiles.ConflictPolicy.WARN, r2.getConflictPolicy());
  }

  @Test
  public void testBuilderMergeConflictPolicyInheritStrictest() {
    Runfiles r1 = new Runfiles.Builder("TESTING").build()
        .setConflictPolicy(Runfiles.ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING").build()
        .setConflictPolicy(Runfiles.ConflictPolicy.ERROR);
    Runfiles r3 = new Runfiles.Builder("TESTING").merge(r1).merge(r2).build();
    assertEquals(Runfiles.ConflictPolicy.ERROR, r3.getConflictPolicy());
    // Swap ordering
    Runfiles r4 = new Runfiles.Builder("TESTING").merge(r2).merge(r1).build();
    assertEquals(Runfiles.ConflictPolicy.ERROR, r4.getConflictPolicy());
  }

  @Test
  public void testLegacyRunfilesStructure() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment workspaceName = PathFragment.create("wsname");
    PathFragment pathB = PathFragment.create("external/repo/b");
    Artifact artifactB = new Artifact(pathB, root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(workspaceName, true);

    Map<PathFragment, Artifact> inputManifest = Maps.newHashMap();
    inputManifest.put(pathB, artifactB);
    Runfiles.ConflictChecker checker = new Runfiles.ConflictChecker(
        Runfiles.ConflictPolicy.WARN, reporter, null);
    builder.addUnderWorkspace(inputManifest, checker);

    assertThat(builder.build().entrySet()).containsExactly(
        Maps.immutableEntry(workspaceName.getRelative(pathB), artifactB),
        Maps.immutableEntry(PathFragment.create("repo/b"), artifactB));
    assertNoEvents();
  }

  @Test
  public void testRunfileAdded() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment workspaceName = PathFragment.create("wsname");
    PathFragment pathB = PathFragment.create("external/repo/b");
    Artifact artifactB = new Artifact(pathB, root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(workspaceName, false);

    Map<PathFragment, Artifact> inputManifest = ImmutableMap.<PathFragment, Artifact>builder()
        .put(pathB, artifactB)
        .build();
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
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathB = PathFragment.create("repo/b");
    PathFragment externalPathB = Label.EXTERNAL_PACKAGE_NAME.getRelative(pathB);
    Artifact artifactB = new Artifact(pathB, root);
    Artifact artifactExternalB = new Artifact(externalPathB, root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        PathFragment.EMPTY_FRAGMENT, false);

    Map<PathFragment, Artifact> inputManifest = ImmutableMap.<PathFragment, Artifact>builder()
        .put(pathB, artifactB)
        .put(externalPathB, artifactExternalB)
        .build();
    Runfiles.ConflictChecker checker = new Runfiles.ConflictChecker(
        Runfiles.ConflictPolicy.WARN, reporter, null);
    builder.addUnderWorkspace(inputManifest, checker);

    assertThat(builder.build().entrySet()).containsExactly(
        Maps.immutableEntry(PathFragment.create("repo/b"), artifactExternalB));
    checkConflictWarning();
  }

  @Test
  public void testMergeWithSymlinks() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(PathFragment.create("a/target"), root);
    Artifact artifactB = new Artifact(PathFragment.create("b/target"), root);
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
}
