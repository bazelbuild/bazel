/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.dec;

import static org.junit.Assert.assertEquals;

import org.brotli.integration.BrotliJniTestBase;
import java.io.IOException;
import java.io.InputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link org.brotli.wrapper.dec.BrotliInputStream}. */
@RunWith(JUnit4.class)
public class EagerStreamTest extends BrotliJniTestBase {

  @Test
  public void testEagerReading() throws IOException {
    final StringBuilder log = new StringBuilder();
    final byte[] data = {0, 0, 16, 42, 3};
    InputStream source = new InputStream() {
      int index;

      @Override
      public int read() {
        if (index < data.length) {
          log.append("<").append(index);
          return data[index++];
        } else {
          log.append("<#");
          return -1;
        }
      }

      @Override
      public int read(byte[] b) throws IOException {
        return read(b, 0, b.length);
      }

      @Override
      public int read(byte[] b, int off, int len) throws IOException {
        if (len < 1) {
          return 0;
        }
        int d = read();
        if (d == -1) {
          return 0;
        }
        b[off] = (byte) d;
        return 1;
      }
    };
    BrotliInputStream reader = new BrotliInputStream(source);
    reader.setEager(true);
    int count = 0;
    while (true) {
      log.append("^").append(count);
      int b = reader.read();
      if (b == -1) {
        log.append(">#");
        break;
      } else {
        log.append(">").append(count++);
      }
    }
    // Lazy log:  ^0<0<1<2<3<4>0^1>#
    assertEquals("^0<0<1<2<3>0^1<4>#", log.toString());
  }

}
