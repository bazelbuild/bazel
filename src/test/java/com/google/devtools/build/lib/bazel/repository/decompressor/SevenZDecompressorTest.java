// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.INNER_FOLDER_NAME;
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.ROOT_FOLDER_NAME;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toCollection;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.List;
import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZOutputFile;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests .7z decompression. */
@RunWith(JUnit4.class)
public class SevenZDecompressorTest {
  @Rule public TestName name = new TestName();

  /**
   * .7z file, created with one file:
   *
   * <ul>
   *   <li>root_folder/another_folder/regularFile
   * </ul>
   *
   * Compressed with command "7zz a test_decompress_archive.7z root_folder"
   */
  private static final String ARCHIVE_NAME = "test_decompress_archive.7z";

  private static final String REGULAR_FILENAME = "regularFile";

  /** Provides a test filesystem descriptor for a test. NOTE: unique per individual test ONLY. */
  private TestArchiveDescriptor archiveDescriptor() throws Exception {
    return new TestArchiveDescriptor(
        ARCHIVE_NAME,
        /* outDirName= */ this.getClass().getSimpleName() + "_" + name.getMethodName(),
        /* withHardLinks= */ false);
  }

  /** Test decompressing a .7z file without stripping a prefix */
  @Test
  public void testDecompressWithoutPrefix() throws Exception {
    Path outputDir = decompress(archiveDescriptor().createDescriptorBuilder().build());

    Path fileDir = outputDir.getRelative(ROOT_FOLDER_NAME).getRelative(INNER_FOLDER_NAME);
    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());
    assertThat(files).contains(REGULAR_FILENAME);
    assertThat(fileDir.getRelative(REGULAR_FILENAME).getFileSize()).isNotEqualTo(0);
  }

  /** Test decompressing a .7z file and stripping a prefix. */
  @Test
  public void testDecompressWithPrefix() throws Exception {
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor().createDescriptorBuilder().setPrefix(ROOT_FOLDER_NAME);
    Path outputDir = decompress(descriptorBuilder.build());
    Path fileDir = outputDir.getRelative(INNER_FOLDER_NAME);

    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());
    assertThat(files).contains(REGULAR_FILENAME);
  }

  /** Test decompressing a .7z with entries being renamed during the extraction process. */
  @Test
  public void testDecompressWithRenamedFiles() throws Exception {
    String innerDirName = ROOT_FOLDER_NAME + "/" + INNER_FOLDER_NAME;

    HashMap<String, String> renameFiles = new HashMap<>();
    renameFiles.put(innerDirName + "/" + REGULAR_FILENAME, innerDirName + "/renamedFile");
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor().createDescriptorBuilder().setRenameFiles(renameFiles);
    Path outputDir = decompress(descriptorBuilder.build());

    Path fileDir = outputDir.getRelative(ROOT_FOLDER_NAME).getRelative(INNER_FOLDER_NAME);
    List<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream()
            .map(Dirent::getName)
            .collect(toCollection(ArrayList::new));
    assertThat(files).contains("renamedFile");
    assertThat(fileDir.getRelative("renamedFile").getFileSize()).isNotEqualTo(0);
  }

  /** Test that entry renaming is applied prior to prefix stripping. */
  @Test
  public void testDecompressWithRenamedFilesAndPrefix() throws Exception {
    String innerDirName = ROOT_FOLDER_NAME + "/" + INNER_FOLDER_NAME;

    HashMap<String, String> renameFiles = new HashMap<>();
    renameFiles.put(innerDirName + "/" + REGULAR_FILENAME, innerDirName + "/renamedFile");
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor()
            .createDescriptorBuilder()
            .setPrefix(ROOT_FOLDER_NAME)
            .setRenameFiles(renameFiles);
    Path outputDir = decompress(descriptorBuilder.build());

    Path fileDir = outputDir.getRelative(INNER_FOLDER_NAME);
    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());
    assertThat(files).contains("renamedFile");
    assertThat(fileDir.getRelative("renamedFile").getFileSize()).isNotEqualTo(0);
  }

  private File archiveDir;
  private File extractionDir;

  public void setUpTestDirectories() {
    // Create an "archives" directory to hold the .7z archive and an "extracted" directory where the
    // extraction will occur.
    String tmpDir =
        java.nio.file.Path.of(TestUtils.tmpDir()).resolve(name.getMethodName()).toString();
    archiveDir = java.nio.file.Path.of(tmpDir).resolve("archives").toFile();
    assertThat(archiveDir.mkdirs()).isTrue();
    extractionDir = java.nio.file.Path.of(tmpDir).resolve("extracted").toFile();
    assertThat(extractionDir.mkdirs()).isTrue();
  }

  @Test
  public void test7zFileModificationDate() throws Exception {
    setUpTestDirectories();

    // Create a test archive.
    SevenZOutputFile sevenZOutput =
        new SevenZOutputFile(new File(archiveDir.getPath(), ARCHIVE_NAME));

    // A regular entry with modification date set to 2000/02/14.
    SevenZArchiveEntry entry =
        sevenZOutput.createArchiveEntry(
            new File(TestUtils.tmpDirFile(), "test_file"),
            "root_folder/another_folder/regularFile");
    GregorianCalendar testDate = new GregorianCalendar(2000, Calendar.FEBRUARY, 14, 3, 7, 14);
    entry.setLastModifiedDate(testDate.getTime());
    sevenZOutput.putArchiveEntry(entry);
    sevenZOutput.write(
        "regular test file contents with modification date 2000/02/14\n".getBytes(UTF_8));
    sevenZOutput.closeArchiveEntry();

    // An entry that has no modification date (shouldn't crash on this).
    SevenZArchiveEntry entryWithNoModifiedDate =
        sevenZOutput.createArchiveEntry(
            new File(TestUtils.tmpDirFile(), "test_file"),
            "root_folder/another_folder/fileNoModificationDate");
    entryWithNoModifiedDate.setLastModifiedDate(null);
    sevenZOutput.putArchiveEntry(entryWithNoModifiedDate);
    sevenZOutput.write("entry has no modification date\n".getBytes(UTF_8));
    sevenZOutput.closeArchiveEntry();
    sevenZOutput.finish();

    FileSystem testFs = TestArchiveDescriptor.getFileSystem();
    DecompressorDescriptor.Builder descriptor =
        DecompressorDescriptor.builder()
            .setDestinationPath(testFs.getPath(extractionDir.getCanonicalPath()))
            .setArchivePath(
                testFs.getPath(archiveDir.getCanonicalPath()).getRelative(ARCHIVE_NAME));

    // Decompression should not crash and set the correct modification date.
    Path outputDir = decompress(descriptor.build());

    Path fileDir = outputDir.getRelative(ROOT_FOLDER_NAME).getRelative(INNER_FOLDER_NAME);
    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());

    assertThat(files).containsExactly("fileNoModificationDate", "regularFile");
    assertThat(fileDir.getRelative("regularFile").getLastModifiedTime())
        .isEqualTo(testDate.getTimeInMillis());
  }

  /** Check that we throw when handling nameless 7z entries. */
  @Test
  public void test7zEntriesWithNoNameThrows() throws Exception {
    setUpTestDirectories();
    // Create a test archive.
    SevenZOutputFile sevenZOutput =
        new SevenZOutputFile(new File(archiveDir.getPath(), ARCHIVE_NAME));

    SevenZArchiveEntry entryWithNoName =
        sevenZOutput.createArchiveEntry(new File(TestUtils.tmpDirFile(), "test_file"), "");
    sevenZOutput.putArchiveEntry(entryWithNoName);
    sevenZOutput.write("entry without a name\n".getBytes(UTF_8));
    sevenZOutput.closeArchiveEntry();
    SevenZArchiveEntry entryWithNoName2 =
        sevenZOutput.createArchiveEntry(new File(TestUtils.tmpDirFile(), "test_file"), "");
    sevenZOutput.putArchiveEntry(entryWithNoName2);
    sevenZOutput.write("entry without a name2\n".getBytes(UTF_8));
    sevenZOutput.closeArchiveEntry();

    sevenZOutput.finish();
    FileSystem testFs = TestArchiveDescriptor.getFileSystem();
    DecompressorDescriptor.Builder descriptor =
        DecompressorDescriptor.builder()
            .setDestinationPath(testFs.getPath(extractionDir.getCanonicalPath()))
            .setArchivePath(
                testFs.getPath(archiveDir.getCanonicalPath()).getRelative(ARCHIVE_NAME));

    IOException e = assertThrows(IOException.class, () -> decompress(descriptor.build()));
    assertThat(e).hasMessageThat().isEqualTo("7z archive contains unnamed entry");
  }

  private Path decompress(DecompressorDescriptor descriptor) throws Exception {
    return new SevenZDecompressor().decompress(descriptor);
  }
}
