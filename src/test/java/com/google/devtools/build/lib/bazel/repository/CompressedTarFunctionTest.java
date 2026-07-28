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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.bazel.repository.TestArchiveDescriptor.INNER_FOLDER_NAME;
import static com.google.devtools.build.lib.bazel.repository.TestArchiveDescriptor.ROOT_FOLDER_NAME;

import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests decompressing archives. */
@RunWith(JUnit4.class)
public class CompressedTarFunctionTest {
  @Rule public TestName name = new TestName();
  @Rule public TemporaryFolder folder = new TemporaryFolder();
  /* Tarball, created by "tar -czf <ARCHIVE_NAME> <files...>" */
  private static final String ARCHIVE_NAME = "test_decompress_archive.tar.gz";

  private TestArchiveDescriptor archiveDescriptor;

  /** Provides a test filesystem descriptor for a test. NOTE: unique per individual test ONLY. */
  @Before
  public void setUpFs() {
    archiveDescriptor =
        new TestArchiveDescriptor(
            ARCHIVE_NAME,
            /* outDirName= */ this.getClass().getSimpleName() + "_" + name.getMethodName(),
            true);
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
   * Test decompressing a tar.gz file with hard link file and symbolic link file inside and
   * stripping the first component.
   */
  @Test
  public void testDecompressWithStripComponents() throws Exception {
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor.createDescriptorBuilder().setStripComponents(1);
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

  /** Test that entry renaming is applied prior to component stripping. */
  @Test
  public void testDecompressWithRenamedFilesAndStripComponents() throws Exception {
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


  @Test
  public void decompressTarWithSymlinkSharingStripPrefix() throws Exception {
    // Creates the example archive described in https://github.com/bazelbuild/bazel/issues/30015.
    // Also includes an absolute symlink to show it's deprefixed correctly.
    FileSystem fs =
        OS.getCurrent() == OS.WINDOWS
            ? new JavaIoFileSystem(DigestHashFunction.SHA256)
            : new UnixFileSystem(DigestHashFunction.SHA256, /*hashAttributeName=*/ "");
    File archive = folder.newFile("symlink_with_same_strip_prefix.tar.gz");
    try (FileOutputStream fos = new FileOutputStream(archive);
        GzipCompressorOutputStream gzos = new GzipCompressorOutputStream(fos);
        TarArchiveOutputStream tos = new TarArchiveOutputStream(gzos)) {

      // Create two directory entries:
      //  - a_prefix
      //  - a_prefix/a_prefix
      TarArchiveEntry directory = new TarArchiveEntry("a_prefix/");
      tos.putArchiveEntry(directory);
      tos.closeArchiveEntry();
      TarArchiveEntry subdirectory = new TarArchiveEntry("a_prefix/a_prefix/");
      tos.putArchiveEntry(subdirectory);
      tos.closeArchiveEntry();

      // Create the two regular files:
      // - a_prefix/regular_file
      // - a_prefix/a_prefix/other_file
      TarArchiveEntry regularFile = new TarArchiveEntry("a_prefix/regular_file");
      byte[] regularFileContents = "hello".getBytes();
      regularFile.setSize(regularFileContents.length);
      tos.putArchiveEntry(regularFile);
      tos.write(regularFileContents);
      tos.closeArchiveEntry();

      TarArchiveEntry otherFile = new TarArchiveEntry("a_prefix/a_prefix/other_file");
      byte[] otherFileContents = "world".getBytes();
      otherFile.setSize(otherFileContents.length);
      tos.putArchiveEntry(otherFile);
      tos.write(otherFileContents);
      tos.closeArchiveEntry();

      // Create the symbolic link pointing to other_file:
      // - a_prefix/symbolic_file -> a_prefix/other_file
      TarArchiveEntry symbolicFile =
          new TarArchiveEntry("a_prefix/symbolic_file", TarArchiveEntry.LF_SYMLINK);
      symbolicFile.setLinkName("a_prefix/other_file");
      symbolicFile.setSize(0);
      tos.putArchiveEntry(symbolicFile);
      tos.closeArchiveEntry();

      // Create an absolute symbolic link pointing to other_file:
      // - a_prefix/abs_symbolic_file -> /a_prefix/a_prefix/other_file
      TarArchiveEntry absSymbolicFile =
          new TarArchiveEntry("a_prefix/abs_symbolic_file", TarArchiveEntry.LF_SYMLINK);
      absSymbolicFile.setLinkName("/a_prefix/a_prefix/other_file");
      absSymbolicFile.setSize(0);
      tos.putArchiveEntry(absSymbolicFile);
      tos.closeArchiveEntry();
    }
    Path archivePath = fs.getPath(archive.getAbsolutePath());
    Path extractPath = fs.getPath(folder.newFolder().getAbsolutePath());

    DecompressorDescriptor descriptor =
        DecompressorDescriptor.builder()
            .setArchivePath(archivePath)
            .setDestinationPath(extractPath)
            .setPrefix("a_prefix")
            .build();
    Path outputDir = decompress(descriptor);

    // Verify the relative symbolic link.
    Path outputSymbolicFile = outputDir.getRelative("symbolic_file");
    assertWithMessage("symbolic_file should exist")
        .that(outputSymbolicFile.exists(Symlinks.NOFOLLOW))
        .isTrue();

    Path symlinkTarget = outputDir.getRelative(outputSymbolicFile.readSymbolicLink());
    assertWithMessage("symbolic_file target should exist: " + symlinkTarget.toString())
        .that(symlinkTarget.exists())
        .isTrue();

    assertWithMessage("a_prefix/other_file and symbolic_file should be the same file")
        .that(
            Files.isSameFile(
                java.nio.file.Paths.get(outputDir.getRelative("a_prefix/other_file").toString()),
                java.nio.file.Paths.get(outputDir.getRelative("symbolic_file").toString())))
        .isTrue();

    // Verify the absolute symbolic link.
    Path outputAbsSymbolicFile = outputDir.getRelative("abs_symbolic_file");
    assertWithMessage("abs_symbolic_file should exist")
        .that(outputSymbolicFile.exists(Symlinks.NOFOLLOW))
        .isTrue();

    Path absSymlinkTarget = outputDir.getRelative(outputAbsSymbolicFile.readSymbolicLink());
    assertWithMessage("abs_symbolic_file target should exist: " + symlinkTarget.toString())
        .that(absSymlinkTarget.exists())
        .isTrue();

    assertWithMessage("a_prefix/other_file and abs_symbolic_file should be the same file")
        .that(
            Files.isSameFile(
                java.nio.file.Paths.get(outputDir.getRelative("a_prefix/other_file").toString()),
                java.nio.file.Paths.get(outputDir.getRelative("abs_symbolic_file").toString())))
        .isTrue();
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
}
