// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.repository.CompressedTarFunction;
import com.google.devtools.build.lib.bazel.repository.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.TarGzFunction;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.zip.GZIPInputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests decompressing archives. */
@RunWith(JUnit4.class)
public class CompressedTarFunctionTest {

  /* Regular file */
  private static final String REGULAR_FILE_NAME = "regularFile";

  /* Hard link file, created by ln <REGULAR_FILE_NAME> <HARD_LINK_FILE_NAME> */
  private static final String HARD_LINK_FILE_NAME = "hardLinkFile";

  /* Symbolic(Soft) link file, created by ln -s <REGULAR_FILE_NAME> <SYMBOLIC_LINK_FILE_NAME> */
  private static final String RELATIVE_SYMBOLIC_LINK_FILE_NAME = "relativeSymbolicLinkFile";
  private static final String ABSOLUTE_SYMBOLIC_LINK_FILE_NAME = "absoluteSymbolicLinkFile";

  private static final String PATH_TO_TEST_ARCHIVE =
      "/com/google/devtools/build/lib/rules/repository/";

  private static final String ROOT_FOLDER_NAME = "root_folder";

  private static final String INNER_FOLDER_NAME = "another_folder";

  /* Tarball, created by
   * tar -czf <ARCHIVE_NAME> <REGULAR_FILE_NAME> <HARD_LINK_FILE_NAME> <SYMBOLIC_LINK_FILE_NAME>
   *
   * The tarball has the following structure
   *
   * root_folder/
   *    another_folder/
   *        regularFile
   *        hardLinkFile hardlink to root_folder/another_folder/regularFile
   *        relativeSymbolicLinkFile -> regularFile
   *        absoluteSymbolicLinkFile -> /root_folder/another_folder/regularFile
   */
  private static final String ARCHIVE_NAME = "test_decompress_archive.tar.gz";

  private FileSystem testFS;
  private Path workingDir;
  private Path tarballPath;
  private Path outDir;
  private DecompressorDescriptor.Builder descriptorBuilder;

  @Before
  public void setUpFs() throws Exception {

    testFS =
        OS.getCurrent() == OS.WINDOWS
            ? new JavaIoFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS)
            : new UnixFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);

    tarballPath =
        testFS
            .getPath(BlazeTestUtils.runfilesDir())
            .getRelative(TestConstants.JAVATESTS_ROOT + PATH_TO_TEST_ARCHIVE + ARCHIVE_NAME);

    workingDir = testFS.getPath(new File(TestUtils.tmpDir()).getCanonicalPath());
    outDir = workingDir.getRelative("out");

    descriptorBuilder =
        DecompressorDescriptor.builder()
            .setDecompressor(TarGzFunction.INSTANCE)
            .setRepositoryPath(outDir)
            .setArchivePath(tarballPath);
  }

  /**
   * Test decompressing a tar.gz file with hard link file and symbolic link file inside without
   * stripping a prefix
   *
   * @throws Exception
   */
  @Test
  public void testDecompressWithoutPrefix() throws Exception {

    Path outputDir =
        new CompressedTarFunction() {
          @Override
          protected InputStream getDecompressorStream(DecompressorDescriptor descriptor)
              throws IOException {
            return new GZIPInputStream(
                Files.newInputStream(descriptor.archivePath().getPathFile().toPath()));
          }
        }.decompress(descriptorBuilder.build());

    assertOutputFiles(outputDir.getRelative(ROOT_FOLDER_NAME).getRelative(INNER_FOLDER_NAME));
  }

  /**
   * Test decompressing a tar.gz file with hard link file and symbolic link file inside and
   * stripping a prefix
   *
   * @throws Exception
   */
  @Test
  public void testDecompressWithPrefix() throws Exception {

    descriptorBuilder.setPrefix(ROOT_FOLDER_NAME);

    Path outputDir =
        new CompressedTarFunction() {
          @Override
          protected InputStream getDecompressorStream(DecompressorDescriptor descriptor)
              throws IOException {
            return new GZIPInputStream(
                Files.newInputStream(descriptor.archivePath().getPathFile().toPath()));
          }
        }.decompress(descriptorBuilder.build());

    assertOutputFiles(outputDir.getRelative(INNER_FOLDER_NAME));
  }

  /** Validate the content of the output directory */
  private void assertOutputFiles(Path outputDir) throws Exception {

    assertThat(outputDir.exists()).isTrue();
    assertThat(outputDir.getRelative(REGULAR_FILE_NAME).exists()).isTrue();
    assertThat(outputDir.getRelative(REGULAR_FILE_NAME).getFileSize()).isNotEqualTo(0);
    assertThat(outputDir.getRelative(REGULAR_FILE_NAME).isSymbolicLink()).isFalse();
    assertThat(outputDir.getRelative(HARD_LINK_FILE_NAME).exists()).isTrue();
    assertThat(outputDir.getRelative(HARD_LINK_FILE_NAME).getFileSize()).isNotEqualTo(0);
    assertThat(outputDir.getRelative(HARD_LINK_FILE_NAME).isSymbolicLink()).isFalse();
    assertThat(outputDir.getRelative(RELATIVE_SYMBOLIC_LINK_FILE_NAME).exists()).isTrue();
    assertThat(outputDir.getRelative(RELATIVE_SYMBOLIC_LINK_FILE_NAME).getFileSize())
        .isNotEqualTo(0);
    assertThat(outputDir.getRelative(RELATIVE_SYMBOLIC_LINK_FILE_NAME).isSymbolicLink()).isTrue();
    assertThat(outputDir.getRelative(ABSOLUTE_SYMBOLIC_LINK_FILE_NAME).exists()).isTrue();
    assertThat(outputDir.getRelative(ABSOLUTE_SYMBOLIC_LINK_FILE_NAME).getFileSize())
        .isNotEqualTo(0);
    assertThat(outputDir.getRelative(ABSOLUTE_SYMBOLIC_LINK_FILE_NAME).isSymbolicLink()).isTrue();
    assertThat(
            Files.isSameFile(
                java.nio.file.Paths.get(outputDir.getRelative(REGULAR_FILE_NAME).toString()),
                java.nio.file.Paths.get(outputDir.getRelative(HARD_LINK_FILE_NAME).toString())))
        .isTrue();
    assertThat(
            Files.isSameFile(
                java.nio.file.Paths.get(outputDir.getRelative(REGULAR_FILE_NAME).toString()),
                java.nio.file.Paths.get(
                    outputDir.getRelative(RELATIVE_SYMBOLIC_LINK_FILE_NAME).toString())))
        .isTrue();
    assertThat(
            Files.isSameFile(
                java.nio.file.Paths.get(outputDir.getRelative(REGULAR_FILE_NAME).toString()),
                java.nio.file.Paths.get(
                    outputDir.getRelative(ABSOLUTE_SYMBOLIC_LINK_FILE_NAME).toString())))
        .isTrue();
  }
}
