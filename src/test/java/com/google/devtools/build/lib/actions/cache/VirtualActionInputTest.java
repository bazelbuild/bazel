// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions.cache;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.DigestHashFunction.SHA256;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.windows.WindowsFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public class VirtualActionInputTest {
  @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

  public enum FileSystemType {
    IN_MEMORY,
    JAVA,
    NATIVE;

    FileSystem getFileSystem() {
      return switch (this) {
        case IN_MEMORY -> new InMemoryFileSystem(SHA256);
        case JAVA -> new JavaIoFileSystem(SHA256);
        case NATIVE ->
            OS.getCurrent() == OS.WINDOWS
                ? new WindowsFileSystem(SHA256, /* createSymbolicLinks= */ false)
                : new UnixFileSystem(SHA256, "hash");
      };
    }
  }

  @Test
  public void testAtomicallyWriteRelativeTo(@TestParameter FileSystemType fileSystemType)
      throws Exception {
    FileSystem fs = fileSystemType.getFileSystem();
    Path execRoot = fs.getPath(tempFolder.getRoot().getPath());

    Path outputFile = execRoot.getRelative("some/file");
    VirtualActionInput input =
        ActionsTestUtil.createVirtualActionInput(
            outputFile.relativeTo(execRoot).getPathString(), "hello");

    var digest = input.atomicallyWriteRelativeTo(execRoot);

    assertThat(outputFile.getParentDirectory().readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("file", Dirent.Type.FILE));
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("hello");
    assertThat(outputFile.isExecutable()).isTrue();
    assertThat(digest).isEqualTo(SHA256.getHashFunction().hashString("hello", UTF_8).asBytes());
  }

  @Test
  public void testAtomicallyWriteRelativeTo_concurrentRead(
      @TestParameter FileSystemType fileSystemType) throws Exception {
    FileSystem fs = fileSystemType.getFileSystem();
    Path execRoot = fs.getPath(tempFolder.getRoot().getPath());

    Path outputFile = execRoot.getRelative("some/file");
    VirtualActionInput input =
        ActionsTestUtil.createVirtualActionInput(
            outputFile.relativeTo(execRoot).getPathString(), "hello");

    input.atomicallyWriteRelativeTo(execRoot);
    byte[] digest;
    byte[] bytes;
    try (var in = outputFile.getInputStream()) {
      digest = input.atomicallyWriteRelativeTo(execRoot);
      bytes = in.readAllBytes();
    }

    assertThat(outputFile.getParentDirectory().readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("file", Dirent.Type.FILE));
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("hello");
    assertThat(outputFile.isExecutable()).isTrue();
    assertThat(digest).isEqualTo(SHA256.getHashFunction().hashString("hello", UTF_8).asBytes());
    assertThat(bytes).isEqualTo("hello".getBytes(UTF_8));
  }
}
