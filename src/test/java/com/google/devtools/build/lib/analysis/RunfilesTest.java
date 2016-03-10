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
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

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
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"), root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(new PathFragment("a/b"), new Artifact(new PathFragment("c/b"),
        root));
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadGrandParentObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(new PathFragment("a/b/c"), new Artifact(new PathFragment("b/c"),
                                                         root));
    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    checkWarning();
  }

  @Test
  public void testFilterListForObscuringSymlinksCatchesBadObscurerNoListener() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(new PathFragment("a/b"), new Artifact(new PathFragment("c/b"),
                                                         root));
    assertThat(Runfiles.filterListForObscuringSymlinks(null, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
  }

  @Test
  public void testFilterListForObscuringSymlinksIgnoresOkObscurer() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    obscuringMap.put(new PathFragment("a/b"), new Artifact(new PathFragment("a/b"),
                                                         root));

    assertThat(Runfiles.filterListForObscuringSymlinks(reporter, null, obscuringMap).entrySet())
        .containsExactly(Maps.immutableEntry(pathA, artifactA)).inOrder();
    assertNoEvents();
  }

  @Test
  public void testFilterListForObscuringSymlinksNoObscurers() throws Exception {
    Map<PathFragment, Artifact> obscuringMap = new HashMap<>();
    PathFragment pathA = new PathFragment("a");
    Root root = Root.asSourceRoot(scratch.resolve("/workspace"));
    Artifact artifactA = new Artifact(new PathFragment("a"),
                                          root);
    obscuringMap.put(pathA, artifactA);
    PathFragment pathBC = new PathFragment("b/c");
    Artifact artifactBC = new Artifact(new PathFragment("a/b"),
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactC = new Artifact(new PathFragment("c"), root);
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
    PathFragment pathA = new PathFragment("a");
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
    Artifact artifactB2 = new Artifact(new PathFragment("b"), root);
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
    PathFragment pathA = new PathFragment("a");
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
    PathFragment pathA = new PathFragment("a");
    PathFragment pathB = new PathFragment("b");
    PathFragment pathC = new PathFragment("c");
    Artifact artifactA = new Artifact(new PathFragment("a"), root);
    Artifact artifactB = new Artifact(new PathFragment("b"), root);
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
}
