/* Copyright 2018 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.enc;

import static org.junit.Assert.assertEquals;

import org.brotli.integration.BrotliJniTestBase;
import org.brotli.wrapper.dec.Decoder;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link org.brotli.wrapper.enc.Encoder}. */
@RunWith(JUnit4.class)
public class EmptyInputTest extends BrotliJniTestBase {
  @Test
  public void testEmptyInput() throws IOException {
    byte[] data = new byte[0];
    byte[] encoded = Encoder.compress(data);
    assertEquals(1, encoded.length);
    byte[] decoded = Decoder.decompress(encoded);
    assertEquals(0, decoded.length);
  }
}
