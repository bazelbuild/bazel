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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.decompress;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Checks that Brotli compressed files can be decompressed. */
@RunWith(JUnit4.class)
public class BrFunctionTest {
  @Rule public TestName name = new TestName();

  /**
   * Byte array of a Brotli compressed file containing the word "bazel". You can generate the same
   * sequence of bytes with the following commands (sed part is linux-specific):
   *
   * <pre>{@code
   * $ echo "bazel" > file
   * $ brotli file
   * $ od -v -t d1 brotli.br | cut -c9- | sed 's/\([0-9]\+\)/\1,/g'
   * }</pre>
   */
  private static final byte[] bazelBrBytes =
      new byte[] {
        33, 20, 0, 4, 98, 97, 122, 101, 108, 10, 3,
      };

  @Test
  public void decompressBrfile()
      throws IOException, InterruptedException, RepositoryFunctionException {
    // Create an "archives" directory to hold compressed files and an "extracted" directory where
    // the extraction will occur.
    String tmpDir = Paths.get(TestUtils.tmpDir()).resolve(name.getMethodName()).toString();
    Path.of(tmpDir).toFile().mkdirs();
    File archiveDir = Paths.get(tmpDir).resolve("archives").toFile();
    assertThat(archiveDir.mkdirs()).isTrue();
    File extractionDir = Paths.get(tmpDir).resolve("extracted").toFile();
    assertThat(extractionDir.mkdirs()).isTrue();

    // Write the example compressed brotli file to the archive directory.
    OutputStream os =
        Files.newOutputStream(java.nio.file.Path.of(archiveDir.getPath()).resolve("file.br"));
    os.write(bazelBrBytes);
    os.close();

    // Decompress.
    FileSystem testFs = FileSystems.getNativeFileSystem();
    DecompressorDescriptor.Builder descriptor =
        DecompressorDescriptor.builder()
            .setDestinationPath(testFs.getPath(extractionDir.getCanonicalPath()))
            .setArchivePath(testFs.getPath(archiveDir.getCanonicalPath()).getRelative("file.br"));

    com.google.devtools.build.lib.vfs.Path fileDir = decompress(descriptor.build());
    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());

    // The decompressed file with the correct name is there with the correct contents.
    assertThat(files).containsExactly("file");
    File pathFile = fileDir.getRelative("file").getPathFile();
    assertThat(Files.readString(pathFile.toPath())).contains("bazel");
  }
}
