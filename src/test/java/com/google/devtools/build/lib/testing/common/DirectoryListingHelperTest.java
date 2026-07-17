// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testing.common;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testing.common.DirectoryListingHelper.directory;
import static com.google.devtools.build.lib.testing.common.DirectoryListingHelper.file;
import static com.google.devtools.build.lib.testing.common.DirectoryListingHelper.leafDirectoryEntries;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import java.io.FileNotFoundException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link com.google.devtools.build.lib.testing.common.DirectoryListingHelper}. */
@RunWith(JUnit4.class)
public final class DirectoryListingHelperTest {

  private final Scratch scratch = new Scratch();
  private final Path root = scratch.getFileSystem().getPath("/");

  @Test
  public void leafDirectoryEntries_emptyDirectory_returnsEmptyList() throws Exception {
    assertThat(leafDirectoryEntries(root)).isEmpty();
  }

  @Test
  public void leafDirectoryEntries_returnsFile() throws Exception {
    scratch.file("/file");
    assertThat(leafDirectoryEntries(root)).containsExactly(file("file"));
  }

  @Test
  public void leafDirectoryEntries_fileInSubfolders_returnsFileOnly() throws Exception {
    scratch.file("/dir1/dir2/file");
    assertThat(leafDirectoryEntries(root)).containsExactly(file("dir1/dir2/file"));
  }

  @Test
  public void leafDirectoryEntries_returnsEmptyDirectory() throws Exception {
    scratch.dir("/dir");
    assertThat(leafDirectoryEntries(root)).containsExactly(directory("dir"));
  }

  @Test
  public void leafDirectoryEntries_mixedEmptyDirectoriesAndFiles_returnsAllEntries()
      throws Exception {
    scratch.dir("/dir/empty1");
    scratch.dir("/dir/subdir/empty2");
    scratch.file("/dir2/file3");
    scratch.file("/dir2/file4");

    assertThat(leafDirectoryEntries(root))
        .containsExactly(
            directory("dir/empty1"),
            directory("dir/subdir/empty2"),
            file("dir2/file3"),
            file("dir2/file4"));
  }

  @Test
  public void leafDirectoryEntries_returnsEntriesUnderProvidedPathOnly() throws Exception {
    scratch.file("/dir/file1");
    scratch.file("/dir2/file2");
    Path dir = scratch.dir("/dir");

    assertThat(leafDirectoryEntries(dir)).containsExactly(file("file1"));
  }

  @Test
  public void leafDirectoryEntries_missingDirectory_fails() {
    Path nonexistent = scratch.getFileSystem().getPath("/nonexistent");
    assertThrows(FileNotFoundException.class, () -> leafDirectoryEntries(nonexistent));
  }
}
