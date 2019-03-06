// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

/**
 * Helper class for working with test archive file.
 *
 * The archive has the following structure
 *
 * root_folder/
 *    another_folder/
 *        regularFile
 *        hardLinkFile hardlink to root_folder/another_folder/regularFile
 *        relativeSymbolicLinkFile -> regularFile
 *        absoluteSymbolicLinkFile -> /root_folder/another_folder/regularFile
 */
public class TestArchiveDescriptor {
  /* Regular file */
  private static final String REGULAR_FILE_NAME = "regularFile";

  /* Hard link file, created by ln <REGULAR_FILE_NAME> <HARD_LINK_FILE_NAME> */
  private static final String HARD_LINK_FILE_NAME = "hardLinkFile";

  /* Symbolic(Soft) link file, created by ln -s <REGULAR_FILE_NAME> <SYMBOLIC_LINK_FILE_NAME> */
  private static final String RELATIVE_SYMBOLIC_LINK_FILE_NAME = "relativeSymbolicLinkFile";
  private static final String ABSOLUTE_SYMBOLIC_LINK_FILE_NAME = "absoluteSymbolicLinkFile";
  private static final String PATH_TO_TEST_ARCHIVE =
      "/com/google/devtools/build/lib/bazel/repository/";

  static final String ROOT_FOLDER_NAME = "root_folder";
  static final String INNER_FOLDER_NAME = "another_folder";

  private final String archiveName;
  private final String outDirName;
  private final boolean withHardLinks;

  TestArchiveDescriptor(String archiveName, String outDirName, boolean withHardLinks) {
    this.archiveName = archiveName;
    this.outDirName = outDirName;
    this.withHardLinks = withHardLinks;
  }

  DecompressorDescriptor.Builder createDescriptorBuilder() throws IOException {
    FileSystem testFS = OS.getCurrent() == OS.WINDOWS
        ? new JavaIoFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS)
        : new UnixFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);

    // do not rely on TestConstants.JAVATESTS_ROOT end with slash, but ensure separators
    // are not duplicated
    String path = (TestConstants.JAVATESTS_ROOT + PATH_TO_TEST_ARCHIVE + archiveName)
        .replace("//", "/");
    Path tarballPath = testFS.getPath(Runfiles.create().rlocation(path));

    Path workingDir = testFS.getPath(new File(TestUtils.tmpDir()).getCanonicalPath());
    Path outDir = workingDir.getRelative(outDirName);

    return DecompressorDescriptor.builder()
            .setRepositoryPath(outDir)
            .setArchivePath(tarballPath);
  }

  /** Validate the content of the output directory */
  void assertOutputFiles(Path rootOutputDir, String... relativePath) throws Exception {
    assertThat(rootOutputDir.asFragment().endsWith(PathFragment.create(outDirName))).isTrue();
    Path outputDir = rootOutputDir;
    for (String part : relativePath) {
      outputDir = outputDir.getRelative(part);
    }

    assertThat(outputDir.exists()).isTrue();
    assertThat(outputDir.getRelative(REGULAR_FILE_NAME).exists()).isTrue();
    assertThat(outputDir.getRelative(REGULAR_FILE_NAME).getFileSize()).isNotEqualTo(0);
    assertThat(outputDir.getRelative(REGULAR_FILE_NAME).isSymbolicLink()).isFalse();
    assertThat(outputDir.getRelative(RELATIVE_SYMBOLIC_LINK_FILE_NAME).exists()).isTrue();
    assertThat(outputDir.getRelative(RELATIVE_SYMBOLIC_LINK_FILE_NAME).getFileSize())
        .isNotEqualTo(0);
    assertThat(outputDir.getRelative(RELATIVE_SYMBOLIC_LINK_FILE_NAME).isSymbolicLink()).isTrue();
    assertThat(outputDir.getRelative(ABSOLUTE_SYMBOLIC_LINK_FILE_NAME).exists()).isTrue();
    assertThat(outputDir.getRelative(ABSOLUTE_SYMBOLIC_LINK_FILE_NAME).getFileSize())
        .isNotEqualTo(0);
    assertThat(outputDir.getRelative(ABSOLUTE_SYMBOLIC_LINK_FILE_NAME).isSymbolicLink()).isTrue();

    if (withHardLinks) {
      assertThat(outputDir.getRelative(HARD_LINK_FILE_NAME).exists()).isTrue();
      assertThat(outputDir.getRelative(HARD_LINK_FILE_NAME).getFileSize()).isNotEqualTo(0);
      assertThat(outputDir.getRelative(HARD_LINK_FILE_NAME).isSymbolicLink()).isFalse();
      assertThat(
          Files.isSameFile(
              java.nio.file.Paths.get(outputDir.getRelative(REGULAR_FILE_NAME).toString()),
              java.nio.file.Paths.get(outputDir.getRelative(HARD_LINK_FILE_NAME).toString())))
          .isTrue();
    }
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
