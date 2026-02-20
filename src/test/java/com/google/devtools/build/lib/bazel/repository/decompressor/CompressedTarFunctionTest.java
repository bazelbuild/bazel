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
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests decompressing archives. */
@RunWith(JUnit4.class)
public class CompressedTarFunctionTest {

  @Rule public TemporaryFolder folder = new TemporaryFolder();

  /* Tarball, created by "tar -czf <ARCHIVE_NAME> <files...>" */
  private static final String ARCHIVE_NAME = "test_decompress_archive.tar.gz";

  private TestArchiveDescriptor archiveDescriptor;

  @Before
  public void setUpFs() throws Exception {
    archiveDescriptor = new TestArchiveDescriptor(ARCHIVE_NAME, "out", true);
  }

  /**
   * Test decompressing a tar.gz file with hard link file and symbolic link file inside without
   * stripping a prefix
   */
  @Test
  public void testDecompressWithoutPrefix() throws Exception {
    Path outputDir = decompress(archiveDescriptor.createDescriptorBuilder().build());

    archiveDescriptor.assertOutputFiles(outputDir, ROOT_FOLDER_NAME, INNER_FOLDER_NAME);
  }

  /**
   * Test decompressing a tar.gz file with hard link file and symbolic link file inside and
   * stripping a prefix
   */
  @Test
  public void testDecompressWithPrefix() throws Exception {
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor.createDescriptorBuilder().setPrefix(ROOT_FOLDER_NAME);
    Path outputDir = decompress(descriptorBuilder.build());

    archiveDescriptor.assertOutputFiles(outputDir, INNER_FOLDER_NAME);
  }

  /**
   * Test decompressing a tar.gz file, with some entries being renamed during the extraction
   * process.
   */
  @Test
  public void testDecompressWithRenamedFiles() throws Exception {
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
    return new CompressedTarFunction() {
      @Override
      protected InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
          throws IOException {
        return new GZIPInputStream(compressedInputStream);
      }
    }.decompress(descriptor);
  }

  @Test
  public void testDecompressTarWithUpLevelReference() throws Exception {
    FileSystem fs = FileSystems.getNativeFileSystem();
    File tarGzFile = folder.newFile("malicious.tar.gz");
    try (FileOutputStream fos = new FileOutputStream(tarGzFile);
        GzipCompressorOutputStream gzos = new GzipCompressorOutputStream(fos);
        TarArchiveOutputStream tos = new TarArchiveOutputStream(gzos)) {
      TarArchiveEntry entry = new TarArchiveEntry("../foo");
      entry.setSize(3);
      tos.putArchiveEntry(entry);
      tos.write("bar".getBytes(UTF_8));
      tos.closeArchiveEntry();
    }
    Path tarGzPath = fs.getPath(tarGzFile.getAbsolutePath());

    DecompressorDescriptor descriptor =
        DecompressorDescriptor.builder()
            .setArchivePath(tarGzPath)
            .setDestinationPath(tarGzPath.getParentDirectory().getRelative("out"))
            .build();
    IOException thrown = assertThrows(IOException.class, () -> decompress(descriptor));
    assertThat(thrown).hasMessageThat().contains("path is escaping the destination directory");
  }

  @Test
  public void testDecompressTarWithSymlinkEscape() throws Exception {
    FileSystem fs = FileSystems.getNativeFileSystem();
    File tarGzFile = folder.newFile("malicious_symlink.tar.gz");
    try (FileOutputStream fos = new FileOutputStream(tarGzFile);
        GzipCompressorOutputStream gzos = new GzipCompressorOutputStream(fos);
        TarArchiveOutputStream tos = new TarArchiveOutputStream(gzos)) {
      TarArchiveEntry entry = new TarArchiveEntry("link", TarArchiveEntry.LF_SYMLINK);
      entry.setLinkName("../foo");
      entry.setIds(0, 0);
      entry.setNames("user", "group");
      tos.putArchiveEntry(entry);
      tos.closeArchiveEntry();
    }
    Path tarGzPath = fs.getPath(tarGzFile.getAbsolutePath());

    DecompressorDescriptor descriptor =
        DecompressorDescriptor.builder()
            .setArchivePath(tarGzPath)
            .setDestinationPath(tarGzPath.getParentDirectory().getRelative("out"))
            .build();
    IOException thrown = assertThrows(IOException.class, () -> decompress(descriptor));
    assertThat(thrown)
        .hasMessageThat()
        .contains("Tar entries cannot refer to files outside of their directory");
  }
}
