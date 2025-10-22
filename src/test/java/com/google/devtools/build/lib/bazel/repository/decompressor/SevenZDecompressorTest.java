package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.INNER_FOLDER_NAME;
import static com.google.devtools.build.lib.bazel.repository.decompressor.TestArchiveDescriptor.ROOT_FOLDER_NAME;

import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
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
    List<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream()
            .map(Dirent::getName)
            .collect(Collectors.toList());
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

    List<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream()
            .map(Dirent::getName)
            .collect(Collectors.toList());
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
            .map((Dirent::getName))
            .collect(Collectors.toList());
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
    List<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream()
            .map((Dirent::getName))
            .collect(Collectors.toList());
    assertThat(files).contains("renamedFile");
    assertThat(fileDir.getRelative("renamedFile").getFileSize()).isNotEqualTo(0);
  }

  private Path decompress(DecompressorDescriptor descriptor) throws Exception {
    return new SevenZDecompressor().decompress(descriptor);
  }
}
