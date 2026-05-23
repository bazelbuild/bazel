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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ZipDecompressor}.
 */
@RunWith(JUnit4.class)
public class ZipDecompressorTest {

  @Rule public TemporaryFolder folder = new TemporaryFolder();
  private static final int FILE = 0100644;
  private static final int EXECUTABLE = 0100755;
  private static final int DIRECTORY = 040755;
  private static final int SYMLINK = 0120777;

  // External attributes hold the permissions in the higher-order bits, so the input int has to be
  // shifted.
  private static final int FILE_ATTRIBUTE = FILE << 16;
  private static final int EXECUTABLE_ATTRIBUTE = EXECUTABLE << 16;
  private static final int DIRECTORY_ATTRIBUTE = DIRECTORY << 16;
  private static final int SYMLINK_ATTRIBUTE = SYMLINK << 16;

  private static final String ARCHIVE_NAME = "test_decompress_archive.zip";

  private Path createZipFile(String entryName, String content) throws IOException {
    FileSystem fs = FileSystems.getNativeFileSystem();
    File zipFile = folder.newFile("malicious.zip");
    try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(zipFile))) {
      ZipEntry entry = new ZipEntry(entryName);
      zos.putNextEntry(entry);
      zos.write(content.getBytes(UTF_8));
      zos.closeEntry();
    }
    return fs.getPath(zipFile.getAbsolutePath());
  }

  private Path createSymlinkZipFile(String entryName, String target) throws IOException {
    FileSystem fs = FileSystems.getNativeFileSystem();
    File zipFile = folder.newFile("symlink.zip");
    byte[] targetBytes = target.getBytes(UTF_8);
    CRC32 crc = new CRC32();
    crc.update(targetBytes);

    try (ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(zipFile))) {
      ZipEntry entry = new ZipEntry(entryName);
      entry.setMethod(ZipEntry.STORED);
      entry.setTime(0);
      entry.setCrc(crc.getValue());
      entry.setSize(targetBytes.length);
      entry.setCompressedSize(targetBytes.length);
      zipOutputStream.putNextEntry(entry);
      zipOutputStream.write(targetBytes);
      zipOutputStream.closeEntry();
    }

    byte[] zipBytes = Files.readAllBytes(zipFile.toPath());
    setCentralDirectoryExternalAttributes(zipBytes, entryName, SYMLINK_ATTRIBUTE);
    Files.write(zipFile.toPath(), zipBytes);

    return fs.getPath(zipFile.getAbsolutePath());
  }

  private static void setCentralDirectoryExternalAttributes(
      byte[] zipBytes, String entryName, int externalAttributes) throws IOException {
    byte[] entryNameBytes = entryName.getBytes(UTF_8);
    for (int offset = 0; offset <= zipBytes.length - 46; ) {
      if (getLittleEndianInt(zipBytes, offset) != 0x02014b50) {
        offset++;
        continue;
      }

      int nameLength = getLittleEndianShort(zipBytes, offset + 28);
      int extraLength = getLittleEndianShort(zipBytes, offset + 30);
      int commentLength = getLittleEndianShort(zipBytes, offset + 32);
      int nameOffset = offset + 46;
      if (nameLength == entryNameBytes.length
          && startsWith(zipBytes, nameOffset, entryNameBytes)) {
        zipBytes[offset + 5] = 3; // Set "version made by" OS to Unix.
        setLittleEndianInt(zipBytes, offset + 38, externalAttributes);
        return;
      }
      offset = nameOffset + nameLength + extraLength + commentLength;
    }

    throw new IOException("Could not find central directory entry for " + entryName);
  }

  private static boolean startsWith(byte[] bytes, int offset, byte[] prefix) {
    if (offset + prefix.length > bytes.length) {
      return false;
    }
    for (int i = 0; i < prefix.length; i++) {
      if (bytes[offset + i] != prefix[i]) {
        return false;
      }
    }
    return true;
  }

  private static int getLittleEndianShort(byte[] bytes, int offset) {
    return (bytes[offset] & 0xff) | ((bytes[offset + 1] & 0xff) << 8);
  }

  private static int getLittleEndianInt(byte[] bytes, int offset) {
    return getLittleEndianShort(bytes, offset) | (getLittleEndianShort(bytes, offset + 2) << 16);
  }

  private static void setLittleEndianInt(byte[] bytes, int offset, int value) {
    bytes[offset] = (byte) value;
    bytes[offset + 1] = (byte) (value >> 8);
    bytes[offset + 2] = (byte) (value >> 16);
    bytes[offset + 3] = (byte) (value >> 24);
  }

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
   * Test decompressing a zip file with hard link file and symbolic link file inside and stripping a
   * prefix
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
   * Test decompressing a zip file with hard link file and symbolic link file inside and stripping a
   * component
   */
  @Test
  public void testDecompressWithStripComponents() throws Exception {
    TestArchiveDescriptor archiveDescriptor = new TestArchiveDescriptor(ARCHIVE_NAME, "out", false);
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor.createDescriptorBuilder().setStripComponents(1);
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

  /** Test that entry renaming is applied prior to stripping components. */
  @Test
  public void testDecompressWithRenamedFilesAndStripComponents() throws Exception {
    TestArchiveDescriptor archiveDescriptor = new TestArchiveDescriptor(ARCHIVE_NAME, "out", false);
    String innerDirName = ROOT_FOLDER_NAME + "/" + INNER_FOLDER_NAME;

    HashMap<String, String> renameFiles = new HashMap<>();
    renameFiles.put(innerDirName + "/hardLinkFile", innerDirName + "/renamedFile");
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor
            .createDescriptorBuilder()
            .setStripComponents(1)
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

  @Test
  public void testDecompressZipWithUpLevelReference() throws IOException {
    Path zipFile = createZipFile("../foo", "bar");
    DecompressorDescriptor descriptor =
        DecompressorDescriptor.builder()
            .setArchivePath(zipFile)
            .setDestinationPath(zipFile.getParentDirectory().getRelative("out"))
            .build();
    IOException thrown = assertThrows(IOException.class, () -> decompress(descriptor));
    assertThat(thrown).hasMessageThat().contains("path is escaping the destination directory");
  }

  @Test
  public void testDecompressZipWithAbsoluteSymlinkTarget() throws IOException {
    Path zipFile = createSymlinkZipFile("config.bzl", "/etc/passwd");
    DecompressorDescriptor descriptor =
        DecompressorDescriptor.builder()
            .setArchivePath(zipFile)
            .setDestinationPath(zipFile.getParentDirectory().getRelative("out"))
            .build();

    IOException thrown = assertThrows(IOException.class, () -> decompress(descriptor));
    assertThat(thrown).hasMessageThat().contains("pointing to /etc/passwd");
  }
}
