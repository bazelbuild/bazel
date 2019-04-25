/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.dec;

import static org.junit.Assert.assertEquals;

import org.brotli.integration.BrotliJniTestBase;
import org.brotli.integration.BundleHelper;
import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.junit.runner.RunWith;
import org.junit.runners.AllTests;

/** Tests for {@link org.brotli.wrapper.dec.Decoder}. */
@RunWith(AllTests.class)
public class DecoderTest extends BrotliJniTestBase {

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
        suite.addTest(new DecoderTestCase(entry));
      }
    } finally {
      bundle.close();
    }
    return suite;
  }

  /** Test case with a unique name. */
  static class DecoderTestCase extends TestCase {
    final String entryName;
    DecoderTestCase(String entryName) {
      super("DecoderTest." + entryName);
      this.entryName = entryName;
    }

    @Override
    protected void runTest() throws Throwable {
      DecoderTest.run(entryName);
    }
  }

  private static void run(String entryName) throws Throwable {
    InputStream bundle = getBundle();
    byte[] compressed;
    try {
      compressed = BundleHelper.readEntry(bundle, entryName);
    } finally {
      bundle.close();
    }
    if (compressed == null) {
      throw new RuntimeException("Can't read bundle entry: " + entryName);
    }

    byte[] decompressed = Decoder.decompress(compressed);

    long crc = BundleHelper.fingerprintStream(new ByteArrayInputStream(decompressed));
    assertEquals(BundleHelper.getExpectedFingerprint(entryName), crc);
  }
}
