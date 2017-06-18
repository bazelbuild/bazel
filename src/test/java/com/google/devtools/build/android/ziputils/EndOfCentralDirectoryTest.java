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
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDDCD;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDDSK;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDOFF;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIG;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSUB;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDTOT;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.ZipInputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link EndOfCentralDirectory}. */
@RunWith(JUnit4.class)
public class EndOfCentralDirectoryTest {
  @Test
  public void testViewOf() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    int offset = 50;
    int marker = EndOfCentralDirectory.SIGNATURE;
    int comLength = 8;
    buffer.putInt(offset, marker);
    buffer.putShort(offset + ZipInputStream.ENDCOM, (short) comLength);
    buffer.position(offset);
    EndOfCentralDirectory view = EndOfCentralDirectory.viewOf(buffer);
    int expMark = (int) ZipInputStream.ENDSIG;
    int expSize = ZipInputStream.ENDHDR + comLength; // fixed + comment
    int expPos = 0;
    assertWithMessage("not based at current position").that(view.get(ENDSIG)).isEqualTo(expMark);
    assertWithMessage("Not slice with position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized with comment").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
  }

  @Test
  public void testView_String() {
    String[] comments = { "hello world", "", null};

    for (String comment : comments) {
      String expComment = comment != null ? comment : "";
      EndOfCentralDirectory view = EndOfCentralDirectory.allocate(comment);
      String commentResult = view.getComment();
      assertWithMessage("Incorrect comment").that(commentResult).isEqualTo(expComment);
      int expSize = ZipInputStream.ENDHDR + (comment != null ? comment.getBytes(UTF_8).length : 0);
      int expPos = 0;
      assertWithMessage("Not at position 0").that(view.buffer.position()).isEqualTo(expPos);
      assertWithMessage("Not sized correctly").that(view.getSize()).isEqualTo(expSize);
      assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
    }
  }

  @Test
  public void testView_ByteBuffer_String() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    int offset = 50;
    buffer.position(offset);
    String comment = "this is a comment";
    EndOfCentralDirectory view = EndOfCentralDirectory.view(buffer, comment);
    int expMark = (int) ZipInputStream.ENDSIG;
    int expSize = ZipInputStream.ENDHDR + comment.length();
    int expPos = 0;
    assertWithMessage("not based at current position").that(view.get(ENDSIG)).isEqualTo(expMark);
    assertWithMessage("Not slice with position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Not sized with comment").that(view.getSize()).isEqualTo(expSize);
    assertWithMessage("Not limited to size").that(view.buffer.limit()).isEqualTo(expSize);
    assertWithMessage("Incorrect comment").that(view.getComment()).isEqualTo(comment);
  }

  @Test
  public void testCopy() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    EndOfCentralDirectory view = EndOfCentralDirectory.allocate("comment");
    view.copy(buffer);
    int expSize = view.getSize();
    assertWithMessage("buffer not advanced as expected").that(buffer.position()).isEqualTo(expSize);
    buffer.position(0);
    EndOfCentralDirectory clone = EndOfCentralDirectory.viewOf(buffer);
    assertWithMessage("Fail to copy mark").that(clone.get(ENDSIG)).isEqualTo(view.get(ENDSIG));
    assertWithMessage("Fail to copy comment").that(clone.getComment()).isEqualTo(view.getComment());
  }

  @Test
  public void testWithAndGetMethods() {
    short cdDisk = (short) 0x36c2;
    int cdOffset = 0x924ac255;
    int cdSize = 0x138ca234;
    short disk = (short) 0x5c12;
    short local = (short) 0x4ae1;
    short total = (short) 0x63be;
    EndOfCentralDirectory view = EndOfCentralDirectory.allocate("Hello World!")
        .set(ENDDCD, cdDisk)
        .set(ENDOFF, cdOffset)
        .set(ENDSIZ, cdSize)
        .set(ENDDSK, disk)
        .set(ENDSUB, local)
        .set(ENDTOT, total);
    assertWithMessage("Central directory start disk").that(view.get(ENDDCD)).isEqualTo(cdDisk);
    assertWithMessage("Central directory file offset").that(view.get(ENDOFF)).isEqualTo(cdOffset);
    assertWithMessage("Central directory size").that(view.get(ENDSIZ)).isEqualTo(cdSize);
    assertWithMessage("This disk number").that(view.get(ENDDSK)).isEqualTo(disk);
    assertWithMessage("Number of records on this disk").that(view.get(ENDSUB)).isEqualTo(local);
    assertWithMessage("Total number of central directory records")
        .that(view.get(ENDTOT))
        .isEqualTo(total);
  }
}
