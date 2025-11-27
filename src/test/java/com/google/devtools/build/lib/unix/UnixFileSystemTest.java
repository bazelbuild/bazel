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
package com.google.devtools.build.lib.unix;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.TraceProfilerService;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileAccessException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SymlinkAwareFileSystemTest;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.testing.junit.testparameterinjector.TestParameter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import org.junit.Test;

/** Tests for the {@link com.google.devtools.build.lib.unix.UnixFileSystem} class. */
public class UnixFileSystemTest extends SymlinkAwareFileSystemTest {

  @Override
  protected FileSystem getFreshFileSystem(DigestHashFunction digestHashFunction) {
    return new UnixFileSystem(digestHashFunction, /*hashAttributeName=*/ "");
  }

  @Override
  public void destroyFileSystem(FileSystem fileSystem) {
    // Nothing.
  }

  // Most tests are just inherited from FileSystemTest.

  @Test
  public void testPermissions() throws Exception {
    Path file = absolutize("file");
    FileSystemUtils.createEmptyFile(file);
    for (int bits = 0; bits <= 0777; bits++) {
      String msg = "for permissions 0%s".formatted(Integer.toString(bits, 8));
      file.chmod(bits);
      assertWithMessage(msg).that(file.stat().getPermissions()).isEqualTo(bits);
      assertWithMessage(msg).that(file.isReadable()).isEqualTo((bits & 0400) != 0);
      assertWithMessage(msg).that(file.isWritable()).isEqualTo((bits & 0200) != 0);
      assertWithMessage(msg).that(file.isExecutable()).isEqualTo((bits & 0100) != 0);
    }
  }

  @Test
  public void testPermissionsError() throws Exception {
    Path file = absolutize("/");
    assertThrows(IOException.class, () -> file.chmod(0777));
  }

  @Test
  public void testCircularSymlinkFound() throws Exception {
    Path linkA = absolutize("link-a");
    Path linkB = absolutize("link-b");
    linkA.createSymbolicLink(linkB);
    linkB.createSymbolicLink(linkA);
    assertThat(linkA.exists(Symlinks.FOLLOW)).isFalse();
    assertThrows(IOException.class, () -> linkA.statIfFound(Symlinks.FOLLOW));
  }

  @Test
  public void testIsSpecialFile() throws Exception {
    Path regular = absolutize("regular");
    Path fifo = absolutize("fifo");
    FileSystemUtils.createEmptyFile(regular);
    NativePosixFiles.mkfifo(fifo.toString(), 0777);

    assertThat(regular.isFile()).isTrue();
    assertThat(regular.isSpecialFile()).isFalse();
    assertThat(regular.stat().isFile()).isTrue();
    assertThat(regular.stat().isSpecialFile()).isFalse();
    assertThat(fifo.isFile()).isTrue();
    assertThat(fifo.isSpecialFile()).isTrue();
    assertThat(fifo.stat().isFile()).isTrue();
    assertThat(fifo.stat().isSpecialFile()).isTrue();
  }

  @Test
  public void testReaddirSpecialFile() throws Exception {
    Path dir = absolutize("dir");
    Path symlink = dir.getChild("symlink");
    Path fifo = dir.getChild("fifo");
    dir.createDirectoryAndParents();
    symlink.createSymbolicLink(fifo.asFragment());
    NativePosixFiles.mkfifo(fifo.toString(), 0777);

    assertThat(dir.getDirectoryEntries()).containsExactly(symlink, fifo);

    assertThat(dir.readdir(Symlinks.NOFOLLOW))
        .containsExactly(
            new Dirent("symlink", Dirent.Type.SYMLINK), new Dirent("fifo", Dirent.Type.UNKNOWN));

    assertThat(dir.readdir(Symlinks.FOLLOW))
        .containsExactly(
            new Dirent("symlink", Dirent.Type.UNKNOWN), new Dirent("fifo", Dirent.Type.UNKNOWN));
  }

  @Test
  public void testReaddirPermissionError() throws Exception {
    Path dir = absolutize("dir");
    dir.createDirectoryAndParents();
    dir.chmod(0333); // unreadable

    assertThrows(FileAccessException.class, dir::getDirectoryEntries);
    assertThrows(FileAccessException.class, () -> dir.readdir(Symlinks.NOFOLLOW));
  }

  @Test
  public void testInputStreamIsFileInputStream(@TestParameter boolean profiling) throws Exception {
    try (var m = new MaybeWithMockProfiler(profiling);
        InputStream in = xFile.getInputStream()) {
      assertThat(in).isInstanceOf(FileInputStream.class);
    }
  }

  @Test
  public void testOutputStreamIsFileOutputStream(@TestParameter boolean profiling)
      throws Exception {
    try (var m = new MaybeWithMockProfiler(profiling);
        OutputStream out = xFile.getOutputStream()) {
      assertThat(out).isInstanceOf(FileOutputStream.class);
    }
  }

  private static class MaybeWithMockProfiler implements AutoCloseable {
    private final boolean enabled;

    MaybeWithMockProfiler(boolean enabled) {
      if (enabled) {
        TraceProfilerService mock = mock(TraceProfilerService.class);
        when(mock.isActive()).thenReturn(true);
        when(mock.isProfiling(any(ProfilerTask.class))).thenReturn(true);
        Profiler.setTraceProfilerServiceForTesting(mock);
      }
      this.enabled = enabled;
    }

    @Override
    public void close() {
      if (enabled) {
        Profiler.setTraceProfilerServiceForTesting(null);
      }
    }
  }
}
