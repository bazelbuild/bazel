package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.commons.compress.compressors.CompressorOutputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.apache.commons.compress.compressors.xz.XZCompressorOutputStream;
import org.apache.commons.compress.compressors.zstandard.ZstdCompressorOutputStream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class CompressedFunctionTest {
  @Rule public TestName name = new TestName();

  private final String compressedFileName;
  private final Class<?> clazz;

  private File archiveDir;
  private File extractionDir;
  private FileSystem testFS;

  public CompressedFunctionTest(Class<?> clazz, String compressedFileName) {
    this.clazz = clazz;
    this.compressedFileName = compressedFileName;
  }

  public static final String EXTRACTED_FILE_NAME = "archive.txt";

  @Parameterized.Parameters
  public static Collection<Object[]> data() {
    return Arrays.asList(
        new Object[][] {
          {Bz2Function.class, EXTRACTED_FILE_NAME + ".bz2"},
          {GzFunction.class, EXTRACTED_FILE_NAME + ".gz"},
          {XzFunction.class, EXTRACTED_FILE_NAME + ".xz"},
          {ZstFunction.class, EXTRACTED_FILE_NAME + ".zst"},
        });
  }

  @Before
  public void setUp() throws IOException, CompressorException {
    // Create an "archives" directory to hold compressed files and an "extracted" directory where
    // the extraction will occur.
    String tmpDir = Paths.get(TestUtils.tmpDir()).resolve(name.getMethodName()).toString();
    archiveDir = Paths.get(tmpDir).resolve("archives").toFile();
    assertThat(archiveDir.mkdirs()).isTrue();
    extractionDir = Paths.get(tmpDir).resolve("extracted").toFile();
    assertThat(extractionDir.mkdirs()).isTrue();

    OutputStream out =
        Files.newOutputStream(
            java.nio.file.Paths.get(archiveDir.getPath()).resolve(compressedFileName));
    CompressorOutputStream cos;
    if (clazz == Bz2Function.class) {
      cos = new BZip2CompressorOutputStream(out);
    } else if (clazz == GzFunction.class) {
      cos = new GzipCompressorOutputStream(out);
    } else if (clazz == XzFunction.class) {
      cos = new XZCompressorOutputStream(out);
    } else if (clazz == ZstFunction.class) {
      cos = new ZstdCompressorOutputStream(out);
    } else {
      throw new IllegalArgumentException("Unknown compressor class passed: " + clazz.toString());
    }
    cos.write(("test compressed " + compressedFileName + " file contents\n").getBytes());
    cos.close();

    testFS = TestArchiveDescriptor.getFileSystem();
  }

  /** Basic decompression. Verifies that the uncompressed file name and contents are correct. */
  @Test
  public void testDecompress() throws Exception {
    DecompressorDescriptor.Builder descriptor =
        DecompressorDescriptor.builder()
            .setDestinationPath(testFS.getPath(extractionDir.getCanonicalPath()))
            .setArchivePath(
                testFS.getPath(archiveDir.getCanonicalPath()).getRelative(compressedFileName));

    Path fileDir = decompress(descriptor.build());
    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());

    assertThat(files).containsExactly(EXTRACTED_FILE_NAME);
    File pathFile = fileDir.getRelative(EXTRACTED_FILE_NAME).getPathFile();
    assertThat(Files.readString(pathFile.toPath()))
        .contains("test compressed " + compressedFileName + " file contents\n");
  }

  /**
   * Prefixes are ignored, so setting one will not throw and everything still works as the regular
   * decompression.
   */
  @Test
  public void testDecompressWithPrefixIsIgnored() throws Exception {
    DecompressorDescriptor.Builder descriptor =
        DecompressorDescriptor.builder()
            .setDestinationPath(testFS.getPath(extractionDir.getCanonicalPath()))
            .setPrefix("archive")
            .setArchivePath(
                testFS.getPath(archiveDir.getCanonicalPath()).getRelative(compressedFileName));

    Path fileDir = decompress(descriptor.build());
    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());

    assertThat(files).containsExactly(EXTRACTED_FILE_NAME);
    File pathFile = fileDir.getRelative(EXTRACTED_FILE_NAME).getPathFile();
    assertThat(Files.readString(pathFile.toPath()))
        .contains("test compressed " + compressedFileName + " file contents\n");
  }

  /** Test renaming the single compressed file. */
  @Test
  public void testDecompressWithRenamedFiles() throws Exception {
    FileSystem testFS = TestArchiveDescriptor.getFileSystem();
    HashMap<String, String> renameFiles = new HashMap<>();
    renameFiles.put(EXTRACTED_FILE_NAME, "renamedFile");
    DecompressorDescriptor.Builder descriptor =
        DecompressorDescriptor.builder()
            .setDestinationPath(testFS.getPath(extractionDir.getCanonicalPath()))
            .setRenameFiles(renameFiles)
            .setArchivePath(
                testFS.getPath(archiveDir.getCanonicalPath()).getRelative(compressedFileName));

    Path fileDir = decompress(descriptor.build());
    ImmutableList<String> files =
        fileDir.readdir(Symlinks.NOFOLLOW).stream().map(Dirent::getName).collect(toImmutableList());

    assertThat(files).containsExactly("renamedFile");
    File pathFile = fileDir.getRelative("renamedFile").getPathFile();
    assertThat(Files.readString(pathFile.toPath()))
        .contains("test compressed " + compressedFileName + " file contents\n");
  }

  private Path decompress(DecompressorDescriptor descriptor) throws Exception {
    return ((Decompressor) clazz.getConstructor().newInstance()).decompress(descriptor);
  }
}
