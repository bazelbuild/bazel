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
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for BaseSpawns less trivial aspects. */
@RunWith(JUnit4.class)
public class BaseSpawnTest {

  @Test
  public void testGetEnvironmentDoesntAddRunfilesVarsWhenSourcesAreEmpty() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    BaseSpawn underTest = minimalBaseSpawn(baseEnviron, EmptyRunfilesSupplier.INSTANCE);

    assertThat(underTest.getEnvironment()).isEqualTo(baseEnviron);
  }

  @Test
  public void testGetEnvironmentAddsRunfilesWhenOnlyOneSuppliedViaRunfilesSupplier() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    final String runfilesDir = "runfilesdir";
    BaseSpawn underTest = minimalBaseSpawn(
        baseEnviron,
        new RunfilesSupplierImpl(PathFragment.create(runfilesDir), Runfiles.EMPTY));

    Map<String, String> expected = ImmutableMap.<String, String>builder()
        .putAll(baseEnviron)
        .put("PYTHON_RUNFILES", runfilesDir)
        .put("JAVA_RUNFILES", runfilesDir)
        .build();

    assertThat(underTest.getEnvironment()).isEqualTo(expected);
  }

  @Test
  public void testGetEnvironmentDoesntAddRunfilesWhenMultipleManifestsSupplied() {
    Map<String, String> baseEnviron = ImmutableMap.of("HELLO", "world");
    BaseSpawn underTest =
        minimalBaseSpawn(
            baseEnviron,
            CompositeRunfilesSupplier.of(
                new RunfilesSupplierImpl(PathFragment.create("rfdir1"), Runfiles.EMPTY),
                new RunfilesSupplierImpl(PathFragment.create("rfdir2"), Runfiles.EMPTY)));

    assertThat(underTest.getEnvironment()).isEqualTo(baseEnviron);
  }

  private static BaseSpawn minimalBaseSpawn(
      Map<String, String> environment,
      RunfilesSupplier runfilesSupplier) {
    return new BaseSpawn(
        ImmutableList.<String>of(),
        environment,
        ImmutableMap.<String, String>of(),
        runfilesSupplier,
        null,
        null);
  }
}
