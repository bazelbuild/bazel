// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.HashMap;
import java.util.Map;

/**
 * Test for {@link Runfiles}.
 */
public class RunfilesTest extends FoundationTestCase {

  private void checkWarning() {
    assertContainsEvent("obscured by a -> /workspace/a");
    assertEquals("Runfiles.filterListForObscuringSymlinks should have warned once",
                 1, eventCollector.count());
    assertEquals(EventKind.WARNING, Iterables.getOnlyElement(eventCollector).getKind());
  }

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
}
