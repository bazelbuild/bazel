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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for RunfilesSupplierImpl */
@RunWith(JUnit4.class)
public class RunfilesSupplierImplTest {
  private ArtifactRoot rootDir;

  @Before
  public final void setRoot() {
    Scratch scratch = new Scratch();
    Path execRoot = scratch.getFileSystem().getPath("/");
    rootDir =
        ArtifactRoot.asDerivedRoot(execRoot, false, false, false, "fake", "root", "dont", "matter");
  }

  @Test
  public void testGetArtifactsWithSingleMapping() {
    List<Artifact> artifacts = mkArtifacts(rootDir, "thing1", "thing2");

    RunfilesSupplierImpl underTest =
        new RunfilesSupplierImpl(PathFragment.create("notimportant"), mkRunfiles(artifacts));

    assertThat(underTest.getArtifacts().toList()).containsExactlyElementsIn(artifacts);
  }

  @Test
  public void testGetManifestsWhenNone() {
    RunfilesSupplier underTest =
        new RunfilesSupplierImpl(PathFragment.create("ignored"), Runfiles.EMPTY);
    assertThat(underTest.getManifests()).isEmpty();
  }

  @Test
  public void testGetManifestsWhenSupplied() {
    Artifact manifest = ActionsTestUtil.createArtifact(rootDir, "manifest");
    RunfilesSupplier underTest =
        new RunfilesSupplierImpl(
            PathFragment.create("ignored"),
            Runfiles.EMPTY,
            manifest,
            /* buildRunfileLinks= */ false,
            /* runfileLinksEnabled= */ false);
    assertThat(underTest.getManifests()).containsExactly(manifest);
  }

  private static Runfiles mkRunfiles(Iterable<Artifact> artifacts) {
    return new Runfiles.Builder("TESTING", false).addArtifacts(artifacts).build();
  }

  private static List<Artifact> mkArtifacts(ArtifactRoot rootDir, String... paths) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (String path : paths) {
      builder.add(ActionsTestUtil.createArtifact(rootDir, path));
    }
    return builder.build();
  }
}
