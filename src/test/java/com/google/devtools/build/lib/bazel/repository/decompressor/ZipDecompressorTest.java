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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.INNER_FOLDER_NAME;
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.ROOT_FOLDER_NAME;

import com.google.devtools.build.lib.vfs.Path;
import java.util.HashMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ZipDecompressor}.
 */
@RunWith(JUnit4.class)
public class ZipDecompressorTest {

  private static final int FILE = 0100644;
  private static final int EXECUTABLE = 0100755;
  private static final int DIRECTORY = 040755;

  // External attributes hold the permissions in the higher-order bits, so the input int has to be
  // shifted.
  private static final int FILE_ATTRIBUTE = FILE << 16;
  private static final int EXECUTABLE_ATTRIBUTE = EXECUTABLE << 16;
  private static final int DIRECTORY_ATTRIBUTE = DIRECTORY << 16;

  private static final String ARCHIVE_NAME = "test_decompress_archive.zip";

  /**
   * Test decompressing a tar.gz file with hard link file and symbolic link file inside without
   * stripping a prefix
   */
  @Test
  public void testDecompressWithoutPrefix() throws Exception {
    TestArchiveDescriptor archiveDescriptor =
        new TestArchiveDescriptor(ARCHIVE_NAME, "out/inner", false);
    Path outputDir = decompress(archiveDescriptor.createDescriptorBuilder().build());

    archiveDescriptor.assertOutputFiles(outputDir, ROOT_FOLDER_NAME, INNER_FOLDER_NAME);
  }

  /**
   * Test decompressing a tar.gz file with hard link file and symbolic link file inside and
   * stripping a prefix
   */
  @Test
  public void testDecompressWithPrefix() throws Exception {
    TestArchiveDescriptor archiveDescriptor = new TestArchiveDescriptor(ARCHIVE_NAME, "out", false);
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor.createDescriptorBuilder().setPrefix(ROOT_FOLDER_NAME);
    Path outputDir = decompress(descriptorBuilder.build());

    archiveDescriptor.assertOutputFiles(outputDir, INNER_FOLDER_NAME);
  }

  /**
   * Test decompressing a zip file, with some entries being renamed during the extraction process.
   */
  @Test
  public void testDecompressWithRenamedFiles() throws Exception {
    TestArchiveDescriptor archiveDescriptor = new TestArchiveDescriptor(ARCHIVE_NAME, "out", false);
    String innerDirName = ROOT_FOLDER_NAME + "/" + INNER_FOLDER_NAME;

    HashMap<String, String> renameFiles = new HashMap<>();
    renameFiles.put(innerDirName + "/hardLinkFile", innerDirName + "/renamedFile");
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor.createDescriptorBuilder().setRenameFiles(renameFiles);
    Path outputDir = decompress(descriptorBuilder.build());

    Path innerDir = outputDir.getRelative(ROOT_FOLDER_NAME).getRelative(INNER_FOLDER_NAME);
    assertThat(innerDir.getRelative("renamedFile").exists()).isTrue();
  }

  /** Test that entry renaming is applied prior to prefix stripping. */
  @Test
  public void testDecompressWithRenamedFilesAndPrefix() throws Exception {
    TestArchiveDescriptor archiveDescriptor = new TestArchiveDescriptor(ARCHIVE_NAME, "out", false);
    String innerDirName = ROOT_FOLDER_NAME + "/" + INNER_FOLDER_NAME;

    HashMap<String, String> renameFiles = new HashMap<>();
    renameFiles.put(innerDirName + "/hardLinkFile", innerDirName + "/renamedFile");
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor
            .createDescriptorBuilder()
            .setPrefix(ROOT_FOLDER_NAME)
            .setRenameFiles(renameFiles);
    Path outputDir = decompress(descriptorBuilder.build());

    Path innerDir = outputDir.getRelative(INNER_FOLDER_NAME);
    assertThat(innerDir.getRelative("renamedFile").exists()).isTrue();
  }

  private Path decompress(DecompressorDescriptor descriptor) throws Exception {
    return ZipDecompressor.INSTANCE.decompress(descriptor);
  }

  @Test
  public void testGetPermissions() throws Exception {
    int permissions = ZipDecompressor.getPermissions(FILE_ATTRIBUTE, "foo/bar");
    assertThat(permissions).isEqualTo(FILE);
    permissions = ZipDecompressor.getPermissions(EXECUTABLE_ATTRIBUTE, "foo/bar");
    assertThat(permissions).isEqualTo(EXECUTABLE);
    permissions = ZipDecompressor.getPermissions(DIRECTORY_ATTRIBUTE, "foo/bar");
    assertThat(permissions).isEqualTo(DIRECTORY);
  }

  @Test
  public void testWindowsPermissions() throws Exception {
    int permissions =
        ZipDecompressor.getPermissions(ZipDecompressor.WINDOWS_FILE_ATTRIBUTE_DIRECTORY, "foo/bar");
    assertThat(permissions).isEqualTo(DIRECTORY);
    permissions =
        ZipDecompressor.getPermissions(ZipDecompressor.WINDOWS_FILE_ATTRIBUTE_ARCHIVE, "foo/bar");
    assertThat(permissions).isEqualTo(EXECUTABLE);
    permissions =
        ZipDecompressor.getPermissions(ZipDecompressor.WINDOWS_FILE_ATTRIBUTE_NORMAL, "foo/bar");
    assertThat(permissions).isEqualTo(EXECUTABLE);
  }

  @Test
  public void testDirectoryWithRegularFilePermissions() throws Exception {
    int permissions = ZipDecompressor.getPermissions(FILE, "foo/bar/");
    assertThat(permissions).isEqualTo(040755);
  }
}
