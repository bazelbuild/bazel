package com.google.devtools.build.lib.bazel.repository.decompressor;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collection;
import java.util.GregorianCalendar;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.apache.commons.compress.compressors.gzip.GzipParameters;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;

/**
 * Tests "non-core" .gz decompression code. For the "core" decompression, see
 * CompressedFunctionTest.
 */
@RunWith(Enclosed.class)
public class GzFunctionTest {
  @Rule public TestName name = new TestName();

  /**
   * Creates a simple gzip file.
   *
   * @param testName used as a directory name in the {@link TestUtils#tmpDir()} where the gzip file
   *     will be placed.
   * @param fileName the name of the gzip file to write
   * @param parameters gzip-specific metadata parameters
   * @return Path to the gzip file
   * @throws IOException
   */
  private static Path createTestGzipFile(
      String testName, String fileName, GzipParameters parameters) throws IOException {
    String tmpDir = Paths.get(TestUtils.tmpDir()).resolve(testName).toString();
    Paths.get(tmpDir).toFile().mkdirs();
    Path compressedFile = Paths.get(tmpDir, fileName);
    OutputStream outputStream = Files.newOutputStream(compressedFile);

    GzipCompressorOutputStream compressedOutputStream =
        new GzipCompressorOutputStream(outputStream, parameters);
    compressedOutputStream.write("test file contents\n".getBytes());
    compressedOutputStream.close();
    compressedOutputStream.finish();
    return compressedFile;
  }

  @RunWith(Parameterized.class)
  public static class FileNameTest {
    @Rule public TestName name = new TestName();

    public static final String ARCHIVE_NAME = "archive.txt.gz";
    public static final String BASE_NAME = "archive.txt";

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
      return Arrays.asList(
          new Object[][] {
            {"originalFilename", "originalFilename"},
            {null, BASE_NAME},
            {"   ", BASE_NAME},
            {"fake/path/to/originalFilename", "originalFilename"},
            {"../../path/to/originalFilename", "originalFilename"},
          });
    }

    // The filename as stored in the Gzip metadata.
    private final String metadataFileName;

    // The expected filename when uncompressed.
    private final String expectedUncompressedFilename;

    public FileNameTest(String metadataFileName, String expectedUncompressedFilename) {
      this.metadataFileName = metadataFileName;
      this.expectedUncompressedFilename = expectedUncompressedFilename;
    }

    /**
     * If the Gzip metadata parameters have an original filename, it will be used as the
     * uncompressed filename.
     */
    @Test
    public void uncompressesOriginalFilename() throws IOException {
      GzipParameters parameters = new GzipParameters();
      parameters.setFileName(metadataFileName);
      Path testGzipFile = createTestGzipFile(name.getMethodName(), ARCHIVE_NAME, parameters);

      com.google.devtools.build.lib.bazel.repository.decompressor.GzFunction fn =
          new com.google.devtools.build.lib.bazel.repository.decompressor.GzFunction();
      try (InputStream decompressorStream =
          fn.getDecompressorStream(
              new BufferedInputStream(Files.newInputStream(testGzipFile), 32))) {
        String uncompressedFilename = fn.getUncompressedFileName(decompressorStream, ARCHIVE_NAME);
        assertThat(uncompressedFilename).isEqualTo(expectedUncompressedFilename);
      }
    }
  }

  @RunWith(JUnit4.class)
  public static class FileAttributes {
    @Rule public TestName name = new TestName();

    @Test
    public void setLastModifiedTime() throws IOException {
      FileSystem testFs = TestArchiveDescriptor.getFileSystem();
      com.google.devtools.build.lib.vfs.Path tmpDir = TestUtils.createUniqueTmpDir(testFs);
      File testFile = new File(tmpDir.getPathFile(), "test_file");
      assertThat(testFile.createNewFile()).isTrue();

      // Set the modified time gzip metadata.
      GregorianCalendar testDate = new GregorianCalendar(2000, Calendar.FEBRUARY, 14, 3, 7, 14);
      GzipParameters parameters = new GzipParameters();
      // Expects unix time in seconds.
      long unixTimeSeconds = testDate.getTimeInMillis() / 1000;
      parameters.setModificationTime(unixTimeSeconds);

      // Create the gzip file with the above metadata.
      Path testGzipFile = createTestGzipFile(name.getMethodName(), "test.txt.gz", parameters);
      com.google.devtools.build.lib.bazel.repository.decompressor.GzFunction fn =
          new com.google.devtools.build.lib.bazel.repository.decompressor.GzFunction();
      try (InputStream decompressorStream =
          fn.getDecompressorStream(
              new BufferedInputStream(Files.newInputStream(testGzipFile), 32))) {
        // Calling set attributes will set the modified time according to the metadata.
        fn.setFileAttributes(decompressorStream, testFs.getPath(testFile.getCanonicalPath()));
      }

      // There was an error in Apache Commons Compress where the time was improperly divided by
      // 1000, thus losing 3 digits of precision. This replicates that wrong behavior until we
      // upgrade the Apache Commons Compress library to 1.28. At which point, you can replace
      // hackExpectedTimeSeconds below with unixTimeSeconds.
      // See https://github.com/apache/commons-compress/pull/624
      long hackExpectedTimeSeconds = unixTimeSeconds / 1000 * 1000;
      // Time should be in epoch milliseconds.
      assertThat(testFile.lastModified()).isEqualTo(hackExpectedTimeSeconds * 1000);
    }
  }
}
