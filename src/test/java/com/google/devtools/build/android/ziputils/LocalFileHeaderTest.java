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
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCCRC;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCFLG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCHOW;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCLEN;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCSIG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCSIZ;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCTIM;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCVER;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.ZipInputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LocalFileHeader}. */
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
    assertWithMessage("not based at current position").that(view.get(LOCSIG)).isEqualTo(expMark);
    assertWithMessage("Not slice with position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized with comment").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
  }

  @Test
  public void testView_String_byteArr() {
    String filename = "pkg/foo.class";
    byte[] extraData = { 1, 2, 3, 4, 5, 6, 7, 8};
    int expSize = ZipInputStream.LOCHDR + filename.getBytes(UTF_8).length
        + extraData.length;
    int expPos = 0;
    LocalFileHeader view = LocalFileHeader.allocate(filename, extraData);
    assertWithMessage("Incorrect filename").that(view.getFilename()).isEqualTo(filename);
    assertWithMessage("Incorrect extra data").that(view.getExtraData()).isEqualTo(extraData);
    assertWithMessage("Not at position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized correctly").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
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
    assertWithMessage("not based at current position").that(view.get(LOCSIG)).isEqualTo(expMark);
    assertWithMessage("Not slice with position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized with filename").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
    assertWithMessage("Incorrect filename").that(view.getFilename()).isEqualTo(filename);
  }

  @Test
  public void testCopy() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    LocalFileHeader view = LocalFileHeader.allocate("pkg/foo.class", null);
    view.copy(buffer);
    int expSize = view.getSize();
    assertWithMessage("buffer not advanced as expected").that(buffer.position()).isEqualTo(expSize);
    buffer.position(0);
    LocalFileHeader clone = LocalFileHeader.viewOf(buffer);
    assertWithMessage("Fail to copy mark").that(view.get(LOCSIG)).isEqualTo(view.get(LOCSIG));
    assertWithMessage("Fail to copy comment")
        .that(clone.getFilename())
        .isEqualTo(view.getFilename());
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
    assertWithMessage("CRC").that(view.get(LOCCRC)).isEqualTo(crc);
    assertWithMessage("Compressed size").that(view.get(LOCSIZ)).isEqualTo(compressed);
    assertWithMessage("Uncompressed size").that(view.get(LOCLEN)).isEqualTo(uncompressed);
    assertWithMessage("Flags").that(view.get(LOCFLG)).isEqualTo(flags);
    assertWithMessage("Method").that(view.get(LOCHOW)).isEqualTo(method);
    assertWithMessage("Modified time").that(view.get(LOCTIM)).isEqualTo(time);
    assertWithMessage("Version needed").that(view.get(LOCVER)).isEqualTo(version);
  }
}
