// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertWithMessage;
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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.ZipInputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link DirectoryEntry}. */
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
    assertWithMessage("not based at current position").that(view.get(CENSIG)).isEqualTo(expMark);
    assertWithMessage("Not slice with position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized with comment").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
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
    assertWithMessage("Incorrect filename").that(view.getFilename()).isEqualTo(filename);
    assertWithMessage("Incorrect extra data").that(view.getExtraData()).isEqualTo(extraData);
    assertWithMessage("Incorrect comment").that(view.getComment()).isEqualTo(comment);
    assertWithMessage("Not at position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized correctly").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
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
    assertWithMessage("not based at current position").that(view.get(CENSIG)).isEqualTo(expMark);
    assertWithMessage("Not slice with position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized with filename").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
    assertWithMessage("Incorrect filename").that(view.getFilename()).isEqualTo(filename);
    assertWithMessage("Incorrect extra data").that(view.getExtraData()).isEqualTo(extraData);
    assertWithMessage("Incorrect comment").that(view.getComment()).isEqualTo(comment);
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
    assertWithMessage("buffer not advanced as expected").that(buffer.position()).isEqualTo(expSize);
    buffer.position(0);
    DirectoryEntry clone = DirectoryEntry.viewOf(buffer);
    assertWithMessage("Fail to copy mark").that(clone.get(CENSIG)).isEqualTo(view.get(CENSIG));
    assertWithMessage("Fail to copy comment")
        .that(clone.getFilename())
        .isEqualTo(view.getFilename());
    assertWithMessage("Fail to copy comment")
        .that(clone.getExtraData())
        .isEqualTo(view.getExtraData());
    assertWithMessage("Fail to copy comment").that(clone.getComment()).isEqualTo(view.getComment());
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
    assertWithMessage("CRC").that(view.get(CENCRC)).isEqualTo(crc);
    assertWithMessage("Compressed size").that(view.get(CENSIZ)).isEqualTo(compressed);
    assertWithMessage("Uncompressed size").that(view.get(CENLEN)).isEqualTo(uncompressed);
    assertWithMessage("Flags").that(view.get(CENFLG)).isEqualTo(flags);
    assertWithMessage("Method").that(view.get(CENHOW)).isEqualTo(method);
    assertWithMessage("Modified time").that(view.get(CENTIM)).isEqualTo(time);
    assertWithMessage("Version needed").that(view.get(CENVER)).isEqualTo(version);
    assertWithMessage("Version made by").that(view.get(CENVEM)).isEqualTo(versionMadeBy);
    assertWithMessage("Disk").that(view.get(CENDSK)).isEqualTo(disk);
    assertWithMessage("External attributes").that(view.get(CENATX)).isEqualTo(extAttr);
    assertWithMessage("Internal attributes").that(view.get(CENATT)).isEqualTo(intAttr);
    assertWithMessage("Offset").that(view.get(CENOFF)).isEqualTo(offset);
  }
}
