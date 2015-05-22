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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.List;

/** Tests for RunfilesSupplierImpl */
@RunWith(JUnit4.class)
public class RunfilesSupplierImplTest {

  private Root rootDir;

  @Before
  public void setup() throws IOException {
    Scratch scratch = new Scratch();
    rootDir = Root.asDerivedRoot(scratch.dir("/fake/root/dont/matter"));
  }

  @Test
  public void testGetArtifactsWithSingleMapping() {
    List<Artifact> artifacts = mkArtifacts(rootDir, "thing1", "thing2");

    RunfilesSupplierImpl underTest = new RunfilesSupplierImpl(
        ImmutableMap.of(new PathFragment("notimportant"), mkRunfiles(artifacts)));

    assertThat(underTest.getArtifacts()).containsExactlyElementsIn(artifacts);
  }

  @Test
  public void testGetArtifactsWithMultipleMappings() {
    List<Artifact> artifacts1 = mkArtifacts(rootDir, "thing_1", "thing_2", "duplicated");
    List<Artifact> artifacts2 = mkArtifacts(rootDir, "thing_3", "thing_4", "duplicated");

    RunfilesSupplierImpl underTest = new RunfilesSupplierImpl(ImmutableMap.of(
        new PathFragment("notimportant"), mkRunfiles(artifacts1),
        new PathFragment("stillnotimportant"), mkRunfiles(artifacts2)));

    assertThat(underTest.getArtifacts()).containsExactlyElementsIn(
        mkArtifacts(rootDir, "thing_1", "thing_2", "thing_3", "thing_4", "duplicated"));
  }

  private static Runfiles mkRunfiles(List<Artifact> artifacts) {
    return new Runfiles.Builder().addArtifacts(artifacts).build();
  }

  private static List<Artifact> mkArtifacts(Root rootDir, String... paths) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (String path : paths) {
      builder.add(new Artifact(new PathFragment(path), rootDir));
    }
    return builder.build();
  }
}
