package com.google.devtools.build.lib.bazel.repository.decompressor;

import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.INNER_FOLDER_NAME;
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.ROOT_FOLDER_NAME;

/** Tests .7z decompression. */
@RunWith(JUnit4.class)
public class SevenZDecompressorTest {
  /**
   * .7z file, created with two files:
   *
   * <ul>
   *   <li>root_folder/another_folder/regular_file
   *   <li>root_folder/another_folder/ünïcödëFïlë.txt
   * </ul>
   *
   * Compressed with command "7zz a test_decompress_archive.7z root_folder"
   */
  private static final String ARCHIVE_NAME = "test_decompress_archive.7z";

  private static final String UNICODE_FILENAME = "ünïcödëFïlë.txt";

  private TestArchiveDescriptor archiveDescriptor;

  @Before
  public void setUpFs() throws Exception {
    archiveDescriptor =
        new TestArchiveDescriptor(
            ARCHIVE_NAME, /* outDirName= */ "out", /* withHardLinks= */ false);
  }

  /** Test decompressing a .7z file without stripping a prefix */
  @Test
  public void testDecompressWithoutPrefix() throws Exception {
    Path outputDir = decompress(archiveDescriptor.createDescriptorBuilder().build());

    archiveDescriptor.assertOutputFiles(outputDir, ROOT_FOLDER_NAME, INNER_FOLDER_NAME);
  }

  /** Test decompressing a .7z file and stripping a prefix. */
  @Test
  public void testDecompressWithPrefix() throws Exception {
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor.createDescriptorBuilder().setPrefix(ROOT_FOLDER_NAME);
    Path outputDir = decompress(descriptorBuilder.build());

    archiveDescriptor.assertOutputFiles(outputDir, INNER_FOLDER_NAME);
  }

  /** Test decompressing a .7z with entries being renamed during the extraction process. */
  @Test
  public void testDecompressWithRenamedFiles() throws Exception {
    String innerDirName = ROOT_FOLDER_NAME + "/" + INNER_FOLDER_NAME;

    HashMap<String, String> renameFiles = new HashMap<>();
    renameFiles.put(innerDirName + "/regular_file", innerDirName + "/renamedFile");
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
    renameFiles.put(innerDirName + "/regular_file", innerDirName + "/renamedFile");
    DecompressorDescriptor.Builder descriptorBuilder =
        archiveDescriptor
            .createDescriptorBuilder()
            .setPrefix(ROOT_FOLDER_NAME)
            .setRenameFiles(renameFiles);
    Path outputDir = decompress(descriptorBuilder.build());

    Path innerDir = outputDir.getRelative(INNER_FOLDER_NAME);
    assertThat(innerDir.getRelative("renamedFile").exists()).isTrue();
  }

  /** Test that Unicode filenames are handled. **/
  @Test
  public void testUnicodeFilename() throws Exception {
    Path outputDir = decompress(archiveDescriptor.createDescriptorBuilder().build());

    Path unicodeFile =
        outputDir
            .getRelative(ROOT_FOLDER_NAME)
            .getRelative(INNER_FOLDER_NAME)
            .getRelative(StringEncoding.unicodeToInternal(UNICODE_FILENAME));
    assertThat(unicodeFile.exists()).isTrue();
    assertThat(unicodeFile.getFileSize()).isNotEqualTo(0);
  }

  private Path decompress(DecompressorDescriptor descriptor) throws Exception {
    return new SevenZDecompressor().decompress(descriptor);
  }
}
