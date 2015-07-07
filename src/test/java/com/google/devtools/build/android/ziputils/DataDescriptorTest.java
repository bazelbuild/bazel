// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.android.ziputils;

import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTCRC;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTLEN;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTSIG;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTSIZ;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.ZipInputStream;

/**
 * Unit tests for {@link DataDescriptor}.
 */
@RunWith(JUnit4.class)
public class DataDescriptorTest {

  /**
   * Test of viewOf method, of class DataDescriptor.
   */
  @Test
  public void testViewOf() {
    int[] markers = { 12345678, DataDescriptor.SIGNATURE, 0};
    for (int marker : markers) {
      ByteBuffer buffer = ByteBuffer.allocate(50).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 50; i++) {
        buffer.put((byte) i);
      }
      int offset = 20;
      buffer.putInt(offset, marker);
      buffer.position(offset);
      DataDescriptor view = DataDescriptor.viewOf(buffer);
      int expMark = marker == DataDescriptor.SIGNATURE ? (int) ZipInputStream.EXTSIG : -1;
      int expSize = marker == DataDescriptor.SIGNATURE ? ZipInputStream.EXTHDR
          : ZipInputStream.EXTHDR - 4;
      int expPos = 0;
      assertEquals("not based at current position[" + marker + "]", expMark, view.get(EXTSIG));
      assertEquals("Not slice with position 0[" + marker + "]", expPos, view.buffer.position());
      assertEquals("Not sized with comment[" + marker + "]", expSize, view.getSize());
      assertEquals("Not limited to size[" + marker + "]", expSize, view.buffer.limit());
    }
  }

  /**
   * Test of view method, of class DataDescriptor.
   */
  @Test
  public void testView_0args() {
    DataDescriptor view = DataDescriptor.allocate();
    int expSize = ZipInputStream.EXTHDR;
    int expPos = 0;
    int expMarker = (int) ZipInputStream.EXTSIG;
    assertTrue("no marker", view.hasMarker());
    assertEquals("No marker", expMarker, view.get(EXTSIG));
    assertEquals("Not at position 0", expPos, view.buffer.position());
    assertEquals("Not sized correctly", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
  }

  /**
   * Test of view method, of class DataDescriptor.
   */
  @Test
  public void testView_ByteBuffer() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    buffer.position(50);
    DataDescriptor view = DataDescriptor.view(buffer);
    int expMark = (int) ZipInputStream.EXTSIG;
    int expSize = ZipInputStream.EXTHDR;
    int expPos = 0;
    assertEquals("not based at current position", expMark, view.get(EXTSIG));
    assertEquals("Not slice with position 0", expPos, view.buffer.position());
    assertEquals("Not sized with comment", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
  }

  /**
   * Test of copy method, of class DataDescriptor.
   */
  @Test
  public void testCopy() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    DataDescriptor view = DataDescriptor.allocate();
    view.copy(buffer);
    int expSize = view.getSize();
    assertEquals("buffer not advanced as expected", expSize, buffer.position());
    buffer.position(0);
    DataDescriptor clone = DataDescriptor.viewOf(buffer);
    assertEquals("Fail to copy mark", view.get(EXTSIG), clone.get(EXTSIG));
  }

  /**
   * Test of with and get methods.
   */
  @Test
  public void testWithAndGetMethods() {
    int crc = 0x12345678;
    int compressed = 0x357f1d5;
    int uncompressed = 0x74813159;
    DataDescriptor view = DataDescriptor.allocate()
        .set(EXTCRC, crc)
        .set(EXTSIZ, compressed)
        .set(EXTLEN, uncompressed);
    assertEquals("CRC", crc, view.get(EXTCRC));
    assertEquals("Compressed size", compressed, view.get(EXTSIZ));
    assertEquals("Uncompressed size", uncompressed, view.get(EXTLEN));
  }
}
