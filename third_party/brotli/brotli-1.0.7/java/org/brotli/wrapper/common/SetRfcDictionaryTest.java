/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.common;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.brotli.dec.Dictionary;
import org.brotli.integration.BrotliJniTestBase;
import org.brotli.wrapper.dec.BrotliInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link BrotliCommon}.
 */
@RunWith(JUnit4.class)
public class SetRfcDictionaryTest extends BrotliJniTestBase {

  @Test
  public void testRfcDictionaryChecksums() throws NoSuchAlgorithmException {
    System.err.println(Dictionary.getData().slice().remaining());
    MessageDigest md5 = MessageDigest.getInstance("MD5");
    md5.update(Dictionary.getData().slice());
    assertTrue(BrotliCommon.checkDictionaryDataMd5(md5.digest()));

    MessageDigest sha1 = MessageDigest.getInstance("SHA-1");
    sha1.update(Dictionary.getData().slice());
    assertTrue(BrotliCommon.checkDictionaryDataSha1(sha1.digest()));

    MessageDigest sha256 = MessageDigest.getInstance("SHA-256");
    sha256.update(Dictionary.getData().slice());
    assertTrue(BrotliCommon.checkDictionaryDataSha256(sha256.digest()));
  }

  @Test
  public void testSetRfcDictionary() throws IOException {
    /* "leftdatadataleft" encoded with dictionary words. */
    byte[] data = {27, 15, 0, 0, 0, 0, -128, -29, -76, 13, 0, 0, 7, 91, 38, 49, 64, 2, 0, -32, 78,
        27, 65, -128, 32, 80, 16, 36, 8, 6};
    BrotliCommon.setDictionaryData(Dictionary.getData());

    BrotliInputStream decoder = new BrotliInputStream(new ByteArrayInputStream(data));
    byte[] output = new byte[17];
    int offset = 0;
    try {
      int bytesRead;
      while ((bytesRead = decoder.read(output, offset, 17 - offset)) != -1) {
        offset += bytesRead;
      }
    } finally {
      decoder.close();
    }
    assertEquals(16, offset);
    byte[] expected = {
      'l', 'e', 'f', 't',
      'd', 'a', 't', 'a',
      'd', 'a', 't', 'a',
      'l', 'e', 'f', 't',
      0
    };
    assertArrayEquals(expected, output);
  }
}
