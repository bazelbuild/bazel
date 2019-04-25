package org.brotli.wrapper.enc;

import static org.junit.Assert.assertEquals;

import org.brotli.integration.BrotliJniTestBase;
import org.brotli.integration.BundleHelper;
import org.brotli.wrapper.dec.BrotliInputStream;
import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.junit.runner.RunWith;
import org.junit.runners.AllTests;

/** Tests for {@link org.brotli.wrapper.enc.Encoder}. */
@RunWith(AllTests.class)
public class EncoderTest extends BrotliJniTestBase {
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
        suite.addTest(new EncoderTestCase(entry));
      }
    } finally {
      bundle.close();
    }
    return suite;
  }

  /** Test case with a unique name. */
  static class EncoderTestCase extends TestCase {
    final String entryName;
    EncoderTestCase(String entryName) {
      super("EncoderTest." + entryName);
      this.entryName = entryName;
    }

    @Override
    protected void runTest() throws Throwable {
      EncoderTest.run(entryName);
    }
  }

  private static void run(String entryName) throws Throwable {
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

    for (int window = 10; window <= 22; window++) {
      byte[] compressed =
          Encoder.compress(original, new Encoder.Parameters().setQuality(6).setWindow(window));

      InputStream decoder = new BrotliInputStream(new ByteArrayInputStream(compressed));
      try {
        long originalCrc = BundleHelper.fingerprintStream(new ByteArrayInputStream(original));
        long crc = BundleHelper.fingerprintStream(decoder);
        assertEquals(originalCrc, crc);
      } finally {
        decoder.close();
      }
    }
  }
}
