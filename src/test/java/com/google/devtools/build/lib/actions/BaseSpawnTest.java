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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.Map;

/** Tests for BaseSpawns less trivial aspects. */
@RunWith(JUnit4.class)
public class BaseSpawnTest {

  private Root rootDir;

  @Before
  public final void setup() throws IOException {
    Scratch scratch = new Scratch();
    rootDir = Root.asDerivedRoot(scratch.dir("/fake/root/dont/matter"));
  }

  @Test
  public void testGetEnvironmentDoesntAddRunfilesVarsWhenSourcesAreEmpty() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    BaseSpawn underTest = minimalBaseSpawn(baseEnviron, ImmutableMap.<PathFragment, Artifact>of(),
        EmptyRunfilesSupplier.INSTANCE);

    assertThat(underTest.getEnvironment()).isEqualTo(baseEnviron);
  }

  @Test
  public void testGetEnvironmentAddsRunfilesWhenOnlyOneSuppliedViaManifests() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    final String runfilesDir = "runfilesdir";
    BaseSpawn underTest = minimalBaseSpawn(baseEnviron,
        ImmutableMap.of(new PathFragment(runfilesDir), mkArtifact("dontmatter", rootDir)),
        EmptyRunfilesSupplier.INSTANCE);

    Map<String, String> expected = ImmutableMap.<String, String>builder()
        .putAll(baseEnviron)
        .put("PYTHON_RUNFILES", runfilesDir)
        .put("JAVA_RUNFILES", runfilesDir)
        .build();

    assertThat(underTest.getEnvironment()).isEqualTo(expected);
  }

  @Test
  public void testGetEnvironmentAddsRunfilesWhenOnlyOneSuppliedViaRunfilesSupplier() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    final String runfilesDir = "runfilesdir";
    BaseSpawn underTest = minimalBaseSpawn(baseEnviron,
        ImmutableMap.<PathFragment, Artifact>of(),
        new RunfilesSupplierImpl(new PathFragment(runfilesDir), Runfiles.EMPTY));

    Map<String, String> expected = ImmutableMap.<String, String>builder()
        .putAll(baseEnviron)
        .put("PYTHON_RUNFILES", runfilesDir)
        .put("JAVA_RUNFILES", runfilesDir)
        .build();

    assertThat(underTest.getEnvironment()).isEqualTo(expected);
  }

  @Test
  public void testGetEnvironmentDoesntAddRunfilesWhenSupplierAndManifestsSupplied() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    BaseSpawn underTest = minimalBaseSpawn(baseEnviron,
        ImmutableMap.of(new PathFragment("runfilesdir"), mkArtifact("dontmatter", rootDir)),
        new RunfilesSupplierImpl(new PathFragment("runfilesdir2"), Runfiles.EMPTY));

    assertThat(underTest.getEnvironment()).isEqualTo(baseEnviron);
  }

  @Test
  public void testGetEnvironmentDoesntAddRunfilesWhenMultipleManifestsSupplied() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    BaseSpawn underTest = minimalBaseSpawn(baseEnviron,
        ImmutableMap.of(
            new PathFragment("runfilesdir1"), mkArtifact("dontmatter", rootDir),
            new PathFragment("runfilesdir2"), mkArtifact("stilldontmatter", rootDir)),
        EmptyRunfilesSupplier.INSTANCE);

    assertThat(underTest.getEnvironment()).isEqualTo(baseEnviron);
  }

  @Test
  public void testGetEnvironmentDoesntAddRunfilesWhenMultipleSuppliersSupplied() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    BaseSpawn underTest = minimalBaseSpawn(baseEnviron,
        ImmutableMap.<PathFragment, Artifact>of(),
        new RunfilesSupplierImpl(ImmutableMap.of(
            new PathFragment("runfilesdir1"), Runfiles.EMPTY,
            new PathFragment("runfilesdir2"), Runfiles.EMPTY)));

    assertThat(underTest.getEnvironment()).isEqualTo(baseEnviron);
  }

  private static BaseSpawn minimalBaseSpawn(
      Map<String, String> environment,
      Map<PathFragment, Artifact> runfilesManifests,
      RunfilesSupplier runfilesSupplier) {
    return new BaseSpawn(
        ImmutableList.<String>of(),
        environment,
        ImmutableMap.<String, String>of(),
        runfilesManifests,
        runfilesSupplier,
        null,
        null,
        ImmutableSet.<PathFragment>of());
  }

  private static Artifact mkArtifact(String path, Root rootDir) {
    return new Artifact(new PathFragment(path), rootDir);
  }
}
