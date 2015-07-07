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

import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENATT;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENATX;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENCRC;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENDSK;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENFLG;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENHOW;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENOFF;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIG;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENTIM;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENVEM;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENVER;
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
 * Unit tests for {@link DirectoryEntry}.
 */
@RunWith(JUnit4.class)
public class DirectoryEntryTest {

  /**
   * Test of viewOf method, of class DirectoryEntry.
   */
  @Test
  public void testViewOf() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    int offset = 20;
    int filenameLength = 10;
    int extraLength = 6;
    int commentLength = 8;
    int marker = DirectoryEntry.SIGNATURE;
    buffer.putShort(offset + ZipInputStream.CENNAM, (short) filenameLength); // filename length
    buffer.putShort(offset + ZipInputStream.CENEXT, (short) extraLength); // extra data length
    buffer.putShort(offset + ZipInputStream.CENCOM, (short) commentLength); // comment length
    buffer.putInt(20, marker); // any marker
    buffer.position(offset);
    DirectoryEntry view = DirectoryEntry.viewOf(buffer);
    int expMark = (int) ZipInputStream.CENSIG;
    int expSize = ZipInputStream.CENHDR + filenameLength + extraLength + commentLength;
    int expPos = 0;
    assertEquals("not based at current position", expMark, view.get(CENSIG));
    assertEquals("Not slice with position 0", expPos, view.buffer.position());
    assertEquals("Not sized with comment", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
  }

  /**
   * Test of view method, of class DirectoryEntry.
   */
  @Test
  public void testView_3Args() {
    String filename = "pkg/foo.class";
    String comment = "got milk";
    byte[] extraData = { 1, 2, 3, 4, 5, 6, 7, 8};
    int expSize = ZipInputStream.CENHDR + filename.getBytes(UTF_8).length
        + extraData.length + comment.getBytes(UTF_8).length;
    int expPos = 0;
    DirectoryEntry view = DirectoryEntry.allocate(filename, extraData, comment);
    assertEquals("Incorrect filename", filename, view.getFilename());
    Assert.assertArrayEquals("Incorrect extra data", extraData, view.getExtraData());
    assertEquals("Incorrect comment", comment, view.getComment());
    assertEquals("Not at position 0", expPos, view.buffer.position());
    assertEquals("Not sized correctly", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
  }

  /**
   * Test of view method, of class DirectoryEntry.
   */
  @Test
  public void testView_4Args() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    int offset = 20;
    buffer.position(offset);
    String filename = "pkg/foo.class";
    byte[] extraData = { 1, 2, 3, 4, 5};
    String comment = "c";
    int expMark = (int) ZipInputStream.CENSIG;
    int expSize = 46 + filename.getBytes(UTF_8).length + extraData.length
        + comment.getBytes(UTF_8).length;
    int expPos = 0;
    DirectoryEntry view = DirectoryEntry.view(buffer, filename, extraData, comment);
    assertEquals("not based at current position", expMark, view.get(CENSIG));
    assertEquals("Not slice with position 0", expPos, view.buffer.position());
    assertEquals("Not sized with filename", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
    assertEquals("Incorrect filename", filename, view.getFilename());
    Assert.assertArrayEquals("Incorrect extra data", extraData, view.getExtraData());
    assertEquals("Incorrect comment", comment, view.getComment());
  }

  /**
   * Test of copy method, of class DirectoryEntry.
   */
  @Test
  public void testCopy() {
    String filename = "pkg/foo.class";
    byte[] extraData = {};
    String comment = "always comment!";
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    DirectoryEntry view = DirectoryEntry.allocate(filename, extraData, comment);
    view.copy(buffer);
    int expSize = view.getSize();
    assertEquals("buffer not advanced as expected", expSize, buffer.position());
    buffer.position(0);
    DirectoryEntry clone = DirectoryEntry.viewOf(buffer);
    assertEquals("Fail to copy mark", view.get(CENSIG), clone.get(CENSIG));
    assertEquals("Fail to copy comment", view.getFilename(), clone.getFilename());
    Assert.assertArrayEquals("Fail to copy comment", view.getExtraData(), clone.getExtraData());
    assertEquals("Fail to copy comment", view.getComment(), clone.getComment());
  }

  /**
   * Test of with and get methods.
   */
  @Test
  public void testWithAndGetMethods() {
    int crc = 0x12345678;
    int compressed = 0x357f1d5;
    int uncompressed = 0x74813159;
    short flags = 0x7a61;
    short method = 0x3b29;
    int time = 0x12312345;
    short version = 0x1234;
    short versionMadeBy = 0x27a1;
    short disk = 0x5a78;
    int extAttr = 0x73b27a15;
    short intAttr = 0x37cc;
    int offset = 0x74c93ac1;
    DirectoryEntry view = DirectoryEntry.allocate("pkg/foo.class", null, "")
        .set(CENCRC, crc)
        .set(CENSIZ, compressed)
        .set(CENLEN, uncompressed)
        .set(CENFLG, flags)
        .set(CENHOW, method)
        .set(CENTIM, time)
        .set(CENVER, version)
        .set(CENVEM, versionMadeBy)
        .set(CENDSK, disk)
        .set(CENATX, extAttr)
        .set(CENATT, intAttr)
        .set(CENOFF, offset);
    assertEquals("CRC", crc, view.get(CENCRC));
    assertEquals("Compressed size", compressed, view.get(CENSIZ));
    assertEquals("Uncompressed size", uncompressed, view.get(CENLEN));
    assertEquals("Flags", flags, view.get(CENFLG));
    assertEquals("Method", method, view.get(CENHOW));
    assertEquals("Modified time", time, view.get(CENTIM));
    assertEquals("Version needed", version, view.get(CENVER));
    assertEquals("Version made by", versionMadeBy, view.get(CENVEM));
    assertEquals("Disk", disk, view.get(CENDSK));
    assertEquals("External attributes", extAttr, view.get(CENATX));
    assertEquals("Internal attributes", intAttr, view.get(CENATT));
    assertEquals("Offset", offset, view.get(CENOFF));
  }
}
