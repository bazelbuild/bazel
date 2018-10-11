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
import static com.google.devtools.build.lib.exec.FilesetManifest.RelativeSymlinkBehavior.ERROR;
import static com.google.devtools.build.lib.exec.FilesetManifest.RelativeSymlinkBehavior.IGNORE;
import static com.google.devtools.build.lib.exec.FilesetManifest.RelativeSymlinkBehavior.RESOLVE;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link FilesetManifest}.
 */
@RunWith(JUnit4.class)
public class FilesetManifestTest {
  private FileSystem fs;
  private Path execRoot;

  @Before
  public final void createSpawnInputExpander() throws Exception  {
    fs = new InMemoryFileSystem();
    execRoot = fs.getPath("/root");
  }

  private void scratchFile(String file, String... lines) throws Exception {
    Path path = fs.getPath(file);
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeLinesAs(path, StandardCharsets.UTF_8, lines);
  }

  @Test
  public void testEmptyManifest() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile("/root/_foo/MANIFEST");

    Artifact artifact =
        new Artifact(fs.getPath("/root/foo"), ArtifactRoot.asSourceRoot(Root.fromPath(execRoot)));
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
    assertThat(manifest.getEntries()).isEmpty();
  }

  @Test
  public void testManifestWithSingleFile() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar /dir/file",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
    assertThat(manifest.getEntries())
        .containsExactly(PathFragment.create("out/foo/bar"), "/dir/file");
  }

  @Test
  public void testManifestWithTwoFiles() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar /dir/file",
        "<some digest>",
        "workspace/baz /dir/file",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
    assertThat(manifest.getEntries())
        .containsExactly(
            PathFragment.create("out/foo/bar"), "/dir/file",
            PathFragment.create("out/foo/baz"), "/dir/file");
  }

  @Test
  public void testManifestWithDirectory() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar /some",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
    assertThat(manifest.getEntries())
        .containsExactly(PathFragment.create("out/foo/bar"), "/some");
  }

  /** Regression test: code was previously crashing in this case. */
  @Test
  public void testManifestWithEmptyPath() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar ", // <-- Note the trailing whitespace!
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
    assertThat(manifest.getEntries()).containsExactly(PathFragment.create("out/foo/bar"), null);
  }

  @Test
  public void testManifestWithMissingWorkspacePrefix() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "notworkspace/bar /foo/bar",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    try {
      FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
      fail();
    } catch (IOException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("fileset manifest line must start with 'workspace': 'notworkspace/bar'");
    }
  }

  @Test
  public void testManifestWithErrorOnRelativeSymlink() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar foo",
        "<some digest>",
        "workspace/foo /foo/bar",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    try {
      FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", ERROR);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().isEqualTo("runfiles target is not absolute: foo");
    }
  }

  @Test
  public void testManifestWithIgnoredRelativeSymlink() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar foo",
        "<some digest>",
        "workspace/foo /foo/bar",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
    assertThat(manifest.getEntries())
        .containsExactly(PathFragment.create("out/foo/foo"), "/foo/bar");
  }

  @Test
  public void testManifestWithResolvedRelativeSymlink() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar foo",
        "<some digest>",
        "workspace/foo /foo/bar",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", RESOLVE);
    assertThat(manifest.getEntries())
        .containsExactly(
            PathFragment.create("out/foo/bar"), "/foo/bar",
            PathFragment.create("out/foo/foo"), "/foo/bar");
  }

  @Test
  public void testManifestWithResolvedRelativeSymlinkWithDotSlash() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar ./foo",
        "<some digest>",
        "workspace/foo /foo/bar",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", RESOLVE);
    assertThat(manifest.getEntries())
        .containsExactly(
            PathFragment.create("out/foo/bar"), "/foo/bar",
            PathFragment.create("out/foo/foo"), "/foo/bar");
  }

  @Test
  public void testManifestWithResolvedRelativeSymlinkWithDotDotSlash() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar/bar ../foo/foo",
        "<some digest>",
        "workspace/foo/foo /foo/bar",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", RESOLVE);
    assertThat(manifest.getEntries())
        .containsExactly(
            PathFragment.create("out/foo/bar/bar"), "/foo/bar",
            PathFragment.create("out/foo/foo/foo"), "/foo/bar");
  }

  @Test
  public void testManifestWithUnresolvableRelativeSymlink() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar foo",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest filesetManifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", RESOLVE);
    assertThat(filesetManifest.getEntries()).isEmpty();
    assertThat(filesetManifest.getArtifactValues()).isEmpty();
  }

  @Test
  public void testManifestWithUnresolvableRelativeSymlinkToRelativeSymlink() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar foo",
        "<some digest>",
        "workspace/foo baz",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", RESOLVE);
    assertThat(manifest.getEntries()).isEmpty();
    assertThat(manifest.getArtifactValues()).isEmpty();
  }

  /** Current behavior is first one wins. */
  @Test
  public void testDefactoBehaviorWithDuplicateEntries() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar /foo/bar",
        "<some digest>",
        "workspace/bar /baz",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    FilesetManifest manifest =
        FilesetManifest.parseManifestFile(artifact.getExecPath(), execRoot, "workspace", IGNORE);
    assertThat(manifest.getEntries())
        .containsExactly(
            PathFragment.create("out/foo/bar"), "/foo/bar");
  }
}
