/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.common;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.brotli.integration.BrotliJniTestBase;
import org.brotli.wrapper.dec.BrotliInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link BrotliCommon}.
 */
@RunWith(JUnit4.class)
public class SetZeroDictionaryTest extends BrotliJniTestBase {

  @Test
  public void testZeroDictionary() throws IOException {
    /* "leftdatadataleft" encoded with dictionary words. */
    byte[] data = {27, 15, 0, 0, 0, 0, -128, -29, -76, 13, 0, 0, 7, 91, 38, 49, 64, 2, 0, -32, 78,
        27, 65, -128, 32, 80, 16, 36, 8, 6};
    byte[] dictionary = new byte[BrotliCommon.RFC_DICTIONARY_SIZE];
    BrotliCommon.setDictionaryData(dictionary);

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
    assertArrayEquals(new byte[17], output);
  }
}
