/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.dec;

import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link BitReader}.
 */
@RunWith(JUnit4.class)
public class BitReaderTest {

  @Test
  public void testReadAfterEos() {
    State reader = new State();
    Decode.initState(reader, new ByteArrayInputStream(new byte[1]));
    BitReader.readBits(reader, 9);
    try {
      BitReader.checkHealth(reader, 0);
    } catch (BrotliRuntimeException ex) {
      // This exception is expected.
      return;
    }
    fail("BrotliRuntimeException should have been thrown by BitReader.checkHealth");
  }
}
