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

/** Tests for {@link org.brotli.wrapper.dec.BrotliInputStream}. */
@RunWith(AllTests.class)
public class BrotliInputStreamTest extends BrotliJniTestBase {

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
        suite.addTest(new StreamTestCase(entry));
      }
    } finally {
      bundle.close();
    }
    return suite;
  }

  /** Test case with a unique name. */
  static class StreamTestCase extends TestCase {
    final String entryName;
    StreamTestCase(String entryName) {
      super("BrotliInputStreamTest." + entryName);
      this.entryName = entryName;
    }

    @Override
    protected void runTest() throws Throwable {
      BrotliInputStreamTest.run(entryName);
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

    InputStream src = new ByteArrayInputStream(compressed);
    InputStream decoder = new BrotliInputStream(src);
    long crc;
    try {
      crc = BundleHelper.fingerprintStream(decoder);
    } finally {
      decoder.close();
    }
    assertEquals(BundleHelper.getExpectedFingerprint(entryName), crc);
  }
}
