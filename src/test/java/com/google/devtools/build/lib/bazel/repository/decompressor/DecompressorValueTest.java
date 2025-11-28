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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link DecompressorValue}.
 */
@RunWith(JUnit4.class)
public class DecompressorValueTest {

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  @Test
  public void testKnownFileExtensionsDoNotThrow() throws Exception {
    Path path = fs.getPath("/foo/.external-repositories/some-repo/bar.zip");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ZipDecompressor.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.jar");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ZipDecompressor.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.zip");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ZipDecompressor.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.nupkg");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ZipDecompressor.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.whl");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ZipDecompressor.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.gz");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarGzFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tgz");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarGzFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.gz");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(GzFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.xz");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarXzFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.txz");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarXzFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.xz");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(XzFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.zst");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarZstFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tzst");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarZstFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.zst");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ZstFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.bz2");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarBz2Function.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tbz");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(TarBz2Function.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.bz2");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(Bz2Function.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.ar");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ArFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.deb");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(ArFunction.class);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.7z");
    assertThat(DecompressorValue.getDecompressor(path)).isInstanceOf(SevenZDecompressor.class);
  }

  @Test
  public void testUnknownFileExtensionsThrow() throws Exception {
    Path zipPath = fs.getPath("/foo/.external-repositories/some-repo/bar.baz");
    RepositoryFunctionException expected =
        assertThrows(
            RepositoryFunctionException.class, () -> DecompressorValue.getDecompressor(zipPath));
    assertThat(expected).hasMessageThat().contains("Expected a file with a .zip, .jar,");
  }

}
