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
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.runfiles.Runfiles;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/** Tests for {@link DecompressorValue}. */
@RunWith(JUnit4.class)
public class DecompressorValueTest {

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  @Test
  public void testKnownFileExtensionsDoNotThrow() throws Exception {
    Path path = fs.getPath("/foo/.external-repositories/some-repo/bar.zip");
    Decompressor unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.jar");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.zip");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.nupkg");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.whl");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.gz");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tgz");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.xz");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.txz");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.zst");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tzst");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tar.bz2");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.tbz");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.ar");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.deb");
    unused = DecompressorValue.getDecompressor(path);
    path = fs.getPath("/foo/.external-repositories/some-repo/bar.baz.7z");
    unused = DecompressorValue.getDecompressor(path);
  }

  @Test
  public void testUnknownFileExtensionsThrow() throws Exception {
    Path zipPath = fs.getPath("/foo/.external-repositories/some-repo/bar.baz");
    RepositoryFunctionException expected =
        assertThrows(
            RepositoryFunctionException.class, () -> DecompressorValue.getDecompressor(zipPath));
    assertThat(expected).hasMessageThat().contains("Expected a file with a .zip, .jar,");
  }

  @Test
  public void httpBzlDocumentation() throws IOException {
    String filePath = Runfiles.create().rlocation("_main/tools/build_defs/repo/http.bzl");
    String contents = Files.readString(Paths.get(filePath), StandardCharsets.UTF_8);

    // Find where the archive formats variable is initialized and parse out.
    int startVarNameIndex = contents.indexOf("SUPPORTED_ARCHIVE_FORMATS =");
    int startBracket = contents.indexOf("[", startVarNameIndex);
    int endBracket = contents.indexOf("]", startBracket);
    String formats = contents.substring(startBracket + 1, endBracket);
    List<String> observedExtensions =
        Arrays.stream(formats.split(","))
            .map(String::strip)
            .filter(s -> s.contains("\""))
            .map(s -> s.substring(1, s.length() - 1))
            .toList();

    List<String> expectedExtensions =
        DecompressorValue.allSupportedExtensions(/* prefix= */ "", /* suffix= */ "");
    if (!expectedExtensions.equals(observedExtensions)) {
      String copyPasteCode =
          "SUPPORTED_ARCHIVE_FORMATS = [\n"
              + String.join(
                  "\n",
                  DecompressorValue.allSupportedExtensions(
                      /* prefix= */ "    \"", /* suffix= */ "\","))
              + "\n]";

      fail(
          String.format(
              "Supported archive formats list is out-dated.\n\n"
                  + "Expected:\n\t%1$s\nGot:\n\t%2$s\n\n"
                  + "Copy-paste string to replace in http.bzl:\n\n%3$s\n",
              expectedExtensions, observedExtensions, copyPasteCode));
    }
  }
}
