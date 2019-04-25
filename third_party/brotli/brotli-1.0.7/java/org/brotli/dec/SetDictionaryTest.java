/* Copyright 2016 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.dec;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Dictionary}.
 */
@RunWith(JUnit4.class)
public class SetDictionaryTest {

  /** See {@link SynthTest} */
  private static final byte[] BASE_DICT_WORD = {
      (byte) 0x1b, (byte) 0x03, (byte) 0x00, (byte) 0x00, (byte) 0x00, (byte) 0x00, (byte) 0x80,
      (byte) 0xe3, (byte) 0xb4, (byte) 0x0d, (byte) 0x00, (byte) 0x00, (byte) 0x07, (byte) 0x5b,
      (byte) 0x26, (byte) 0x31, (byte) 0x40, (byte) 0x02, (byte) 0x00, (byte) 0xe0, (byte) 0x4e,
      (byte) 0x1b, (byte) 0x41, (byte) 0x02
    };

  /** See {@link SynthTest} */
  private static final byte[] ONE_COMMAND = {
      (byte) 0x1b, (byte) 0x02, (byte) 0x00, (byte) 0x00, (byte) 0x00, (byte) 0x00, (byte) 0x80,
      (byte) 0xe3, (byte) 0xb4, (byte) 0x0d, (byte) 0x00, (byte) 0x00, (byte) 0x07, (byte) 0x5b,
      (byte) 0x26, (byte) 0x31, (byte) 0x40, (byte) 0x02, (byte) 0x00, (byte) 0xe0, (byte) 0x4e,
      (byte) 0x1b, (byte) 0x11, (byte) 0x86, (byte) 0x02
    };

  @Test
  public void testSetDictionary() throws IOException {
    byte[] buffer = new byte[16];
    BrotliInputStream decoder;

    // No dictionary set; still decoding should succeed, if no dictionary entries are used.
    decoder = new BrotliInputStream(new ByteArrayInputStream(ONE_COMMAND));
    assertEquals(3, decoder.read(buffer, 0, buffer.length));
    assertEquals("aaa", new String(buffer, 0, 3, "US-ASCII"));
    decoder.close();

    // Decoding of dictionary item must fail.
    decoder = new BrotliInputStream(new ByteArrayInputStream(BASE_DICT_WORD));
    boolean decodingFailed = false;
    try {
      decoder.read(buffer, 0, buffer.length);
    } catch (IOException ex) {
      decodingFailed = true;
    }
    assertEquals(true, decodingFailed);
    decoder.close();

    // Load dictionary data.
    FileChannel dictionaryChannel =
        new FileInputStream(System.getProperty("RFC_DICTIONARY")).getChannel();
    ByteBuffer dictionary = dictionaryChannel.map(FileChannel.MapMode.READ_ONLY, 0, 122784).load();
    Dictionary.setData(dictionary);

    // Retry decoding of dictionary item.
    decoder = new BrotliInputStream(new ByteArrayInputStream(BASE_DICT_WORD));
    assertEquals(4, decoder.read(buffer, 0, buffer.length));
    assertEquals("time", new String(buffer, 0, 4, "US-ASCII"));
    decoder.close();
  }
}
