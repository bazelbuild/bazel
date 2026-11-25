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

package com.google.devtools.build.lib.unix;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.unix.NativePosixFilesException.PosixError;
import com.google.devtools.build.lib.util.OS;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.rules.TemporaryFolder;

/** Tests for {@link NativePosixFilesException}. */
@RunWith(JUnit4.class)
public final class NativePosixFilesExceptionTest {
  @Rule public TemporaryFolder temporaryFolder = new TemporaryFolder();

  @Test
  public void clonefileCreatesIndependentCopyOnMac() throws Exception {
    Assume.assumeTrue(OS.getCurrent() == OS.DARWIN);
    Path source = temporaryFolder.newFile("source").toPath();
    Files.writeString(source, "original");
    Path target = source.resolveSibling("target");
    NativePosixFilesServiceImpl service = new NativePosixFilesServiceImpl();
    service.clonefile(source.toString(), target.toString());
    assertThat(Files.readString(target)).isEqualTo("original");
    assertThat(Files.isSymbolicLink(target)).isFalse();
    Files.writeString(target, "modified");
    assertThat(Files.readString(source)).isEqualTo("original");

    Path alias = source.resolveSibling("alias");
    Files.createSymbolicLink(alias, source.getFileName());
    Path clonedAlias = source.resolveSibling("clonedAlias");
    service.clonefile(alias.toString(), clonedAlias.toString());
    assertThat(Files.isSymbolicLink(clonedAlias)).isFalse();
    assertThat(Files.readString(clonedAlias)).isEqualTo("original");

    assertThat(
            assertThrows(
                    NativePosixFilesException.class,
                    () -> service.clonefile(source.toString(), target.toString()))
                .getMessage())
        .contains("clonefile(");
  }

  @Test
  public void clonefileClonesDirectoryTreeOnMac() throws Exception {
    Assume.assumeTrue(OS.getCurrent() == OS.DARWIN);
    Path source = temporaryFolder.newFolder("tree").toPath();
    Path nested = Files.createDirectory(source.resolve("nested"));
    Files.writeString(nested.resolve("file"), "original");
    Files.createSymbolicLink(source.resolve("link"), Path.of("nested/file"));
    Path target = source.resolveSibling("clone");

    new NativePosixFilesServiceImpl().clonefile(source.toString(), target.toString());

    assertThat(Files.isSymbolicLink(target.resolve("link"))).isTrue();
    assertThat(Files.readString(target.resolve("link"))).isEqualTo("original");
    Files.writeString(target.resolve("nested/file"), "changed");
    assertThat(Files.readString(nested.resolve("file"))).isEqualTo("original");

    Path alias = source.resolveSibling("treeAlias");
    Files.createSymbolicLink(alias, source.getFileName());
    Path clonedAlias = source.resolveSibling("clonedTreeAlias");
    new NativePosixFilesServiceImpl().clonefile(alias.toString(), clonedAlias.toString());
    assertThat(Files.isSymbolicLink(clonedAlias)).isFalse();
    assertThat(Files.readString(clonedAlias.resolve("link"))).isEqualTo("original");
  }

  @Test
  public void testErrorMapping() {
    assertThat(new NativePosixFilesException("msg", PosixError.ENOENT).getError())
        .isEqualTo(PosixError.ENOENT);
    assertThat(new NativePosixFilesException("msg", PosixError.EACCES).getError())
        .isEqualTo(PosixError.EACCES);
    assertThat(new NativePosixFilesException("msg", PosixError.ELOOP).getError())
        .isEqualTo(PosixError.ELOOP);
    assertThat(new NativePosixFilesException("msg", PosixError.ETIMEDOUT).getError())
        .isEqualTo(PosixError.ETIMEDOUT);
    assertThat(new NativePosixFilesException("msg", PosixError.OTHER).getError())
        .isEqualTo(PosixError.OTHER);
  }

  @Test
  public void testErrorMappingNullFallback() {
    assertThat(new NativePosixFilesException("msg", null).getError()).isEqualTo(PosixError.OTHER);
  }
}
