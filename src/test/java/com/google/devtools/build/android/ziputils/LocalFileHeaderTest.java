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

import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCCRC;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCFLG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCHOW;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCLEN;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCSIG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCSIZ;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCTIM;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCVER;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.ZipInputStream;

/**
 * Unit tests for {@link LocalFileHeader}.
 */
@RunWith(JUnit4.class)
public class LocalFileHeaderTest {

  @Test
  public void testViewOf() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    int offset = 20;
    int filenameLength = 10;
    int extraLength = 25;
    int marker = LocalFileHeader.SIGNATURE;
    buffer.putShort(offset + ZipInputStream.LOCNAM, (short) filenameLength); // filename length
    buffer.putShort(offset + ZipInputStream.LOCEXT, (short) extraLength); // extra data length
    buffer.putInt(offset, marker); // need to zero filename length to have predictable size
    buffer.position(offset);
    LocalFileHeader view = LocalFileHeader.viewOf(buffer);
    int expMark = (int) ZipInputStream.LOCSIG;
    int expSize = ZipInputStream.LOCHDR + filenameLength + extraLength; // fixed + comment
    int expPos = 0;
    assertEquals("not based at current position", expMark, view.get(LOCSIG));
    assertEquals("Not slice with position 0", expPos, view.buffer.position());
    assertEquals("Not sized with comment", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
  }

  @Test
  public void testView_String_byteArr() {
    String filename = "pkg/foo.class";
    byte[] extraData = { 1, 2, 3, 4, 5, 6, 7, 8};
    int expSize = ZipInputStream.LOCHDR + filename.getBytes(UTF_8).length
        + extraData.length;
    int expPos = 0;
    LocalFileHeader view = LocalFileHeader.allocate(filename, extraData);
    assertEquals("Incorrect filename", filename, view.getFilename());
    Assert.assertArrayEquals("Incorrect extra data", extraData, view.getExtraData());
    assertEquals("Not at position 0", expPos, view.buffer.position());
    assertEquals("Not sized correctly", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
  }

  @Test
  public void testView_3Args() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    int offset = 20;
    buffer.position(offset);
    String filename = "pkg/foo.class";
    int expMark = LocalFileHeader.SIGNATURE;
    int expSize = ZipInputStream.LOCHDR + filename.getBytes(UTF_8).length;
    int expPos = 0;
    LocalFileHeader view = LocalFileHeader.view(buffer, filename, null);
    assertEquals("not based at current position", expMark, view.get(LOCSIG));
    assertEquals("Not slice with position 0", expPos, view.buffer.position());
    assertEquals("Not sized with filename", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
    assertEquals("Incorrect filename", filename, view.getFilename());
  }

  @Test
  public void testCopy() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    LocalFileHeader view = LocalFileHeader.allocate("pkg/foo.class", null);
    view.copy(buffer);
    int expSize = view.getSize();
    assertEquals("buffer not advanced as expected", expSize, buffer.position());
    buffer.position(0);
    LocalFileHeader clone = LocalFileHeader.viewOf(buffer);
    assertEquals("Fail to copy mark", view.get(LOCSIG), view.get(LOCSIG));
    assertEquals("Fail to copy comment", view.getFilename(), clone.getFilename());
  }

  @Test
  public void testWithAndGetMethods() {
    int crc = 0x12345678;
    int compressed = 0x357f1d5;
    int uncompressed = 0x74813159;
    short flags = 0x7a61;
    short method = 0x3b29;
    int time = 0x12c673e1;
    short version = 0x1234;
    LocalFileHeader view = LocalFileHeader.allocate("pkg/foo.class", null)
        .set(LOCCRC, crc)
        .set(LOCSIZ, compressed)
        .set(LOCLEN, uncompressed)
        .set(LOCFLG, flags)
        .set(LOCHOW, method)
        .set(LOCTIM, time)
        .set(LOCVER, version);
    assertEquals("CRC", crc, view.get(LOCCRC));
    assertEquals("Compressed size", compressed, view.get(LOCSIZ));
    assertEquals("Uncompressed size", uncompressed, view.get(LOCLEN));
    assertEquals("Flags", flags, view.get(LOCFLG));
    assertEquals("Method", method, view.get(LOCHOW));
    assertEquals("Modified time", time, view.get(LOCTIM));
    assertEquals("Version needed", version, view.get(LOCVER));
  }
}
