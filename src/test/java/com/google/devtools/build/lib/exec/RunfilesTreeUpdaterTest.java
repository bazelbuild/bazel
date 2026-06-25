// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link RunfilesTreeUpdater#runfilesUseRealSymlinks}. */
@RunWith(JUnit4.class)
public final class RunfilesTreeUpdaterTest {

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private Path runfilesDir;
  private Path outputManifest;

  @Before
  public void setUp() throws Exception {
    runfilesDir = fs.getPath("/execroot/foo.runfiles");
    runfilesDir.createDirectoryAndParents();
    outputManifest = runfilesDir.getChild("MANIFEST");
  }

  private void writeManifest(String content) throws Exception {
    FileSystemUtils.writeContent(outputManifest, UTF_8, content);
  }

  private Path createEntry(String relativePath) throws Exception {
    Path entry = runfilesDir.getRelative(relativePath);
    entry.getParentDirectory().createDirectoryAndParents();
    return entry;
  }

  @Test
  public void symlinkEntry_returnsTrue() throws Exception {
    writeManifest("ws/foo /source/foo\n");
    FileSystemUtils.ensureSymbolicLink(
        createEntry("ws/foo"), PathFragment.create("/source/foo"));

    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isTrue();
  }

  @Test
  public void regularFileEntry_returnsFalse() throws Exception {
    writeManifest("ws/foo /source/foo\n");
    FileSystemUtils.writeContent(createEntry("ws/foo"), UTF_8, "stale content");

    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isFalse();
  }

  @Test
  public void skipsEmptyFileEntries() throws Exception {
    // Empty-file entries (no target) are always regular files even in symlink mode — skip them.
    writeManifest("ws/__init__.py \nws/foo /source/foo\n");
    FileSystemUtils.createEmptyFile(createEntry("ws/__init__.py"));
    FileSystemUtils.ensureSymbolicLink(
        createEntry("ws/foo"), PathFragment.create("/source/foo"));

    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isTrue();
  }

  @Test
  public void nothingToProbe_returnsFalse() throws Exception {
    writeManifest("");
    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isFalse();

    writeManifest("ws/__init__.py \nws/sub/__init__.py \n");
    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isFalse();

    // Manifest doesn't exist on disk.
    outputManifest.delete();
    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isFalse();
  }

  @Test
  public void missingEntry_returnsFalse() throws Exception {
    writeManifest("ws/foo /source/foo\n");

    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isFalse();
  }

  @Test
  public void danglingSymlink_returnsTrue() throws Exception {
    writeManifest("ws/foo /nonexistent/foo\n");
    FileSystemUtils.ensureSymbolicLink(
        createEntry("ws/foo"), PathFragment.create("/nonexistent/foo"));

    assertThat(RunfilesTreeUpdater.runfilesUseRealSymlinks(runfilesDir, outputManifest)).isTrue();
  }
}
