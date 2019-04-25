package org.brotli.wrapper.enc;

import static org.junit.Assert.assertEquals;

import org.brotli.integration.BrotliJniTestBase;
import org.brotli.integration.BundleHelper;
import org.brotli.wrapper.dec.BrotliInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.junit.runner.RunWith;
import org.junit.runners.AllTests;

/** Tests for {@link org.brotli.wrapper.enc.BrotliOutputStream}. */
@RunWith(AllTests.class)
public class BrotliOutputStreamTest extends BrotliJniTestBase {

  private enum TestMode {
    WRITE_ALL,
    WRITE_CHUNKS,
    WRITE_BYTE
  }

  private static final int CHUNK_SIZE = 256;

  static InputStream getBundle() throws IOException {
    return new FileInputStream(System.getProperty("TEST_BUNDLE"));
  }

  /** Creates a test suite. */
  public static TestSuite suite() throws IOException {
    TestSuite suite = new TestSuite();
    InputStream bundle = getBundle();
    try {
      List<String> entries = BundleHelper.listEntries(bundle);
      for (String entry : entries) {
        suite.addTest(new StreamTestCase(entry, TestMode.WRITE_ALL));
        suite.addTest(new StreamTestCase(entry, TestMode.WRITE_CHUNKS));
        suite.addTest(new StreamTestCase(entry, TestMode.WRITE_BYTE));
      }
    } finally {
      bundle.close();
    }
    return suite;
  }

  /** Test case with a unique name. */
  static class StreamTestCase extends TestCase {
    final String entryName;
    final TestMode mode;
    StreamTestCase(String entryName, TestMode mode) {
      super("BrotliOutputStreamTest." + entryName + "." + mode.name());
      this.entryName = entryName;
      this.mode = mode;
    }

    @Override
    protected void runTest() throws Throwable {
      BrotliOutputStreamTest.run(entryName, mode);
    }
  }

  private static void run(String entryName, TestMode mode) throws Throwable {
    InputStream bundle = getBundle();
    byte[] original;
    try {
      original = BundleHelper.readEntry(bundle, entryName);
    } finally {
      bundle.close();
    }
    if (original == null) {
      throw new RuntimeException("Can't read bundle entry: " + entryName);
    }

    if ((mode == TestMode.WRITE_CHUNKS) && (original.length <= CHUNK_SIZE)) {
      return;
    }

    ByteArrayOutputStream dst = new ByteArrayOutputStream();
    OutputStream encoder = new BrotliOutputStream(dst);
    try {
      switch (mode) {
        case WRITE_ALL:
          encoder.write(original);
          break;

        case WRITE_CHUNKS:
          for (int offset = 0; offset < original.length; offset += CHUNK_SIZE) {
            encoder.write(original, offset, Math.min(CHUNK_SIZE, original.length - offset));
          }
          break;

        case WRITE_BYTE:
          for (byte singleByte : original) {
            encoder.write(singleByte);
          }
          break;
      }
    } finally {
      encoder.close();
    }

    InputStream decoder = new BrotliInputStream(new ByteArrayInputStream(dst.toByteArray()));
    try {
      long originalCrc = BundleHelper.fingerprintStream(new ByteArrayInputStream(original));
      long crc = BundleHelper.fingerprintStream(decoder);
      assertEquals(originalCrc, crc);
    } finally {
      decoder.close();
    }
  }
}
