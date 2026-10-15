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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AsyncDirectoryCleaner}. */
@RunWith(JUnit4.class)
public final class AsyncDirectoryCleanerTest {
  private final Scratch scratch = new Scratch();
  private Path parentDir;
  private Path targetDir;

  @Before
  public void setUp() throws Exception {
    parentDir = scratch.dir("/test_root");
    targetDir = parentDir.getChild("target");
    targetDir.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(targetDir.getChild("file1.txt"), "hello");
    Path subDir = targetDir.getChild("sub");
    subDir.createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(subDir.getChild("file2.txt"), "world");
  }

  @Test
  public void testMoveToTempDirectory_movesDirectoryAndPreservesContents() throws Exception {
    assertThat(targetDir.exists()).isTrue();

    Path tempPath = AsyncDirectoryCleaner.moveToTempDirectory(targetDir);

    assertThat(targetDir.exists()).isFalse();
    assertThat(tempPath.exists()).isTrue();
    assertThat(tempPath.getParentDirectory()).isEqualTo(parentDir);
    assertThat(tempPath.getBaseName()).startsWith("target_tmp_");
    assertThat(FileSystemUtils.readContent(tempPath.getChild("file1.txt"), UTF_8))
        .isEqualTo("hello\n");
    assertThat(FileSystemUtils.readContent(tempPath.getChild("sub").getChild("file2.txt"), UTF_8))
        .isEqualTo("world\n");
  }

  @Test
  public void testMoveToTempDirectory_failsOnRootDirectory() {
    Path rootDir = scratch.getFileSystem().getPath("/");
    assertThrows(IOException.class, () -> AsyncDirectoryCleaner.moveToTempDirectory(rootDir));
  }

  @Test
  public void testMoveToTempDirectory_failsOnNonexistentDirectory() {
    Path nonExistent = parentDir.getChild("does_not_exist");
    assertThrows(IOException.class, () -> AsyncDirectoryCleaner.moveToTempDirectory(nonExistent));
  }
}
