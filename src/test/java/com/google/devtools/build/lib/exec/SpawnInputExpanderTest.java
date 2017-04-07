// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Tests for {@link SpawnInputExpander}.
 */
@RunWith(JUnit4.class)
public class SpawnInputExpanderTest {
  private FileSystem fs;
  private SpawnInputExpander expander;
  private Map<PathFragment, ActionInput> inputMappings;

  @Before
  public final void createSpawnInputExpander() throws Exception  {
    fs = new InMemoryFileSystem();
    expander = new SpawnInputExpander(/*strict=*/true);
    inputMappings = Maps.newHashMap();
  }

  private void scratchFile(String file, String... lines) throws Exception {
    Path path = fs.getPath(file);
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.writeLinesAs(path, StandardCharsets.UTF_8, lines);
  }

  @Test
  public void testEmptyRunfiles() throws Exception {
    RunfilesSupplier supplier = EmptyRunfilesSupplier.INSTANCE;
    expander.addRunfilesToInputs(inputMappings, supplier, null);
    assertThat(inputMappings).isEmpty();
  }

  @Test
  public void testRunfilesSingleFile() throws Exception {
    Artifact artifact =
        new Artifact(fs.getPath("/root/dir/file"), Root.asSourceRoot(fs.getPath("/root")));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    ActionInputFileCache mockCache = Mockito.mock(ActionInputFileCache.class);
    Mockito.when(mockCache.isFile(artifact)).thenReturn(true);

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/file"), artifact);
  }

  @Test
  public void testRunfilesDirectoryStrict() throws Exception {
    Artifact artifact =
        new Artifact(fs.getPath("/root/dir/file"), Root.asSourceRoot(fs.getPath("/root")));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    ActionInputFileCache mockCache = Mockito.mock(ActionInputFileCache.class);
    Mockito.when(mockCache.isFile(artifact)).thenReturn(false);

    try {
      expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
      fail();
    } catch (IOException expected) {
      assertThat(expected.getMessage().contains("Not a file: /root/dir/file")).isTrue();
    }
  }

  @Test
  public void testRunfilesDirectoryNonStrict() throws Exception {
    Artifact artifact =
        new Artifact(fs.getPath("/root/dir/file"), Root.asSourceRoot(fs.getPath("/root")));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    ActionInputFileCache mockCache = Mockito.mock(ActionInputFileCache.class);
    Mockito.when(mockCache.isFile(artifact)).thenReturn(false);

    expander = new SpawnInputExpander(/*strict=*/false);
    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/file"), artifact);
  }

  @Test
  public void testRunfilesTwoFiles() throws Exception {
    Artifact artifact1 =
        new Artifact(fs.getPath("/root/dir/file"), Root.asSourceRoot(fs.getPath("/root")));
    Artifact artifact2 =
        new Artifact(fs.getPath("/root/dir/baz"), Root.asSourceRoot(fs.getPath("/root")));
    Runfiles runfiles = new Runfiles.Builder("workspace")
        .addArtifact(artifact1)
        .addArtifact(artifact2)
        .build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    ActionInputFileCache mockCache = Mockito.mock(ActionInputFileCache.class);
    Mockito.when(mockCache.isFile(artifact1)).thenReturn(true);
    Mockito.when(mockCache.isFile(artifact2)).thenReturn(true);

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(2);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/file"), artifact1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/baz"), artifact2);
  }

  @Test
  public void testRunfilesSymlink() throws Exception {
    Artifact artifact =
        new Artifact(fs.getPath("/root/dir/file"), Root.asSourceRoot(fs.getPath("/root")));
    Runfiles runfiles = new Runfiles.Builder("workspace")
        .addSymlink(PathFragment.create("symlink"), artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    ActionInputFileCache mockCache = Mockito.mock(ActionInputFileCache.class);
    Mockito.when(mockCache.isFile(artifact)).thenReturn(true);

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/symlink"), artifact);
  }

  @Test
  public void testRunfilesRootSymlink() throws Exception {
    Artifact artifact =
        new Artifact(fs.getPath("/root/dir/file"), Root.asSourceRoot(fs.getPath("/root")));
    Runfiles runfiles = new Runfiles.Builder("workspace")
        .addRootSymlink(PathFragment.create("symlink"), artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    ActionInputFileCache mockCache = Mockito.mock(ActionInputFileCache.class);
    Mockito.when(mockCache.isFile(artifact)).thenReturn(true);

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(2);
    assertThat(inputMappings).containsEntry(PathFragment.create("runfiles/symlink"), artifact);
    // If there's no other entry, Runfiles adds an empty file in the workspace to make sure the
    // directory gets created.
    assertThat(inputMappings)
        .containsEntry(
            PathFragment.create("runfiles/workspace/.runfile"), SpawnInputExpander.EMPTY_FILE);
  }

  @Test
  public void testEmptyManifest() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile("/root/_foo/MANIFEST");

    Artifact artifact =
        new Artifact(fs.getPath("/root/foo"), Root.asSourceRoot(fs.getPath("/root")));
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).isEmpty();
  }

  @Test
  public void testManifestWithSingleFile() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/_foo/MANIFEST",
        "workspace/bar /dir/file",
        "<some digest>");

    Artifact artifact =
        new Artifact(fs.getPath("/root/foo"), Root.asSourceRoot(fs.getPath("/root")));
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("foo/bar"), ActionInputHelper.fromPath("/dir/file"));
  }

  @Test
  public void testManifestWithTwoFiles() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/_foo/MANIFEST",
        "workspace/bar /dir/file",
        "<some digest>",
        "workspace/baz /dir/file",
        "<some digest>");

    Artifact artifact =
        new Artifact(fs.getPath("/root/foo"), Root.asSourceRoot(fs.getPath("/root")));
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).hasSize(2);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("foo/bar"), ActionInputHelper.fromPath("/dir/file"));
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("foo/baz"), ActionInputHelper.fromPath("/dir/file"));
  }

  @Test
  public void testManifestWithDirectory() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/_foo/MANIFEST",
        "workspace/bar /some",
        "<some digest>");

    Artifact artifact =
        new Artifact(fs.getPath("/root/foo"), Root.asSourceRoot(fs.getPath("/root")));
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(
            PathFragment.create("foo/bar"), ActionInputHelper.fromPath("/some"));
  }
}
