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

import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;

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

  private Runfiles.RunfilesPath runfilesPath(String path) {
    return runfilesPath(new PathFragment(path));
  }

  private Runfiles.RunfilesPath runfilesPath(PathFragment path) {
    return Runfiles.RunfilesPath.alreadyResolved(
        path, new PathFragment(TestConstants.WORKSPACE_NAME));
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurer() throws Exception {
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"), root);
    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        null, PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath("a"), artifactA);
    builder.put(runfilesPath("a/b"), new Artifact(new PathFragment("c/b"), root));
    assertThat(builder.filterListForObscuringSymlinks(reporter, null).build().entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadGrandParentObscurer() throws Exception {
    Runfiles.ManifestBuilder obscuringMap = new Runfiles.ManifestBuilder(
        null, PathFragment.EMPTY_FRAGMENT, false);
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"), root);

    obscuringMap.put(runfilesPath("a"), artifactA);
    obscuringMap.put(runfilesPath("a/b/c"), new Artifact(new PathFragment("b/c"), root));
    assertThat(obscuringMap.filterListForObscuringSymlinks(reporter, null).build().entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurerNoListener() throws Exception {
    Runfiles.ManifestBuilder obscuringMap = new Runfiles.ManifestBuilder(
        null, PathFragment.EMPTY_FRAGMENT, false);
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"),
                                          root);
    obscuringMap.put(runfilesPath("a"), artifactA);
    obscuringMap.put(runfilesPath("a/b"), new Artifact(new PathFragment("c/b"), root));
    assertThat(obscuringMap.filterListForObscuringSymlinks(null, null).build().entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
  }

  @Test
  public void testFilterListForObscuringSymlinksIgnoresOkObscurer() throws Exception {
    Runfiles.ManifestBuilder obscuringMap = new Runfiles.ManifestBuilder(
        null, PathFragment.EMPTY_FRAGMENT, false);
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"),
                                          root);
    obscuringMap.put(runfilesPath("a"), artifactA);
    obscuringMap.put(runfilesPath("a/b"), new Artifact(new PathFragment("a/b"), root));

    assertThat(obscuringMap.filterListForObscuringSymlinks(reporter, null).build().entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    assertNoEvents();
  }

  @Test
  public void testFilterListForObscuringSymlinksNoObscurers() throws Exception {
    Runfiles.ManifestBuilder obscuringMap = new Runfiles.ManifestBuilder(
        null, PathFragment.EMPTY_FRAGMENT, false);
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"),
                                          root);
    obscuringMap.put(
        Runfiles.RunfilesPath.alreadyResolved(
            pathA, new PathFragment(TestConstants.WORKSPACE_NAME)),
        artifactA);
    PathFragment pathBC = new PathFragment("b/c");
    Artifact artifactBC = new Artifact(new PathFragment("a/b"),
                                       root);
    obscuringMap.put(runfilesPath(pathBC), artifactBC);
    assertThat(obscuringMap.filterListForObscuringSymlinks(reporter, null).build()
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);

    builder.put(runfilesPath(pathA), artifactB);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactB));
    builder.put(runfilesPath(pathA), artifactC);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    checkConflictWarning();
  }

  @Test
  public void testPutReportsError() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);

    // Same as above but with ERROR not WARNING
    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.ERROR, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), artifactB);
    reporter.removeHandler(failFastHandler); // So it doesn't throw AssertionError
    builder.put(runfilesPath(pathA), artifactC);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    checkConflictError();
  }

  @Test
  public void testPutCatchesConflictBetweenNullAndNotNull() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), null);
    builder.put(runfilesPath(pathA), artifactB);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactB));
    checkConflictWarning();
  }

  @Test
  public void testPutCatchesConflictBetweenNotNullAndNull() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);

    // Same as above but opposite order
    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), artifactB);
    builder.put(runfilesPath(pathA), null);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, null));
    checkConflictWarning();
  }

  @Test
  public void testPutIgnoresConflict() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.IGNORE, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), artifactB);
    builder.put(runfilesPath(pathA), artifactC);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    assertNoEvents();
  }

  @Test
  public void testPutIgnoresConflictNoListener() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, null, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), artifactB);
    builder.put(runfilesPath(pathA), artifactC);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactC));
    assertNoEvents();
  }

  @Test
  public void testPutIgnoresSameArtifact() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactB2 = new Artifact(new PathFragment("b"), root);
    assertEquals(artifactB, artifactB2);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), artifactB);
    builder.put(runfilesPath(pathA), artifactB2);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, artifactB2));
    assertNoEvents();
  }

  @Test
  public void testPutIgnoresNullAndNull() {
    PathFragment pathA = new PathFragment("a");

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), null);
    // Add it again
    builder.put(runfilesPath(pathA), null);
    assertThat(builder.build().entrySet()).containsExactly(Maps.immutableEntry(pathA, null));
    assertNoEvents();
  }

  @Test
  public void testPutNoConflicts() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment pathA = new PathFragment("a");
    PathFragment pathB = new PathFragment("b");
    PathFragment pathC = new PathFragment("c");
    Artifact artifactA = new Artifact(new PathFragment("a"), root);
    Artifact artifactB = new Artifact(new PathFragment("b"), root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null),
        PathFragment.EMPTY_FRAGMENT, false);
    builder.put(runfilesPath(pathA), artifactA);
    // Add different artifact under different path
    builder.put(runfilesPath(pathB), artifactB);
    // Add artifact again under different path
    builder.put(runfilesPath(pathC), artifactA);
    assertThat(builder.build().entrySet())
        .containsExactly(
            Maps.immutableEntry(pathA, artifactA),
            Maps.immutableEntry(pathB, artifactB),
            Maps.immutableEntry(pathC, artifactA))
        .inOrder();
    assertNoEvents();
  }

  @Test
  public void testBuilderMergeConflictPolicyDefault() {
    Runfiles r1 = new Runfiles.Builder("TESTING", false).build();
    Runfiles r2 = new Runfiles.Builder("TESTING", false).merge(r1).build();
    assertEquals(Runfiles.ConflictPolicy.IGNORE, r2.getConflictPolicy());
  }

  @Test
  public void testBuilderMergeConflictPolicyInherit() {
    Runfiles r1 = new Runfiles.Builder("TESTING", false).build()
        .setConflictPolicy(Runfiles.ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING", false).merge(r1).build();
    assertEquals(Runfiles.ConflictPolicy.WARN, r2.getConflictPolicy());
  }

  @Test
  public void testBuilderMergeConflictPolicyInheritStrictest() {
    Runfiles r1 = new Runfiles.Builder("TESTING", false).build()
        .setConflictPolicy(Runfiles.ConflictPolicy.WARN);
    Runfiles r2 = new Runfiles.Builder("TESTING", false).build()
        .setConflictPolicy(Runfiles.ConflictPolicy.ERROR);
    Runfiles r3 = new Runfiles.Builder("TESTING", false).merge(r1).merge(r2).build();
    assertEquals(Runfiles.ConflictPolicy.ERROR, r3.getConflictPolicy());
    // Swap ordering
    Runfiles r4 = new Runfiles.Builder("TESTING", false).merge(r2).merge(r1).build();
    assertEquals(Runfiles.ConflictPolicy.ERROR, r4.getConflictPolicy());
  }

  @Test
  public void testLegacyRunfilesStructure() {
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    PathFragment workspaceName = new PathFragment("wsname");
    PathFragment pathB = new PathFragment("repo/b");
    Artifact artifactB = new Artifact(pathB, root);

    Runfiles.ManifestBuilder builder = new Runfiles.ManifestBuilder(
        new Runfiles.ConflictChecker(Runfiles.ConflictPolicy.WARN, reporter, null),
        workspaceName,
        true);

    builder.put(runfilesPath(pathB), artifactB);

    assertThat(builder.build().entrySet()).containsExactly(
        Maps.immutableEntry(workspaceName.getRelative(".runfile"), null),
        Maps.immutableEntry(workspaceName.getRelative("external").getRelative(pathB), artifactB),
        Maps.immutableEntry(pathB, artifactB));
    assertNoEvents();
  }
}
