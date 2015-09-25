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

import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDDCD;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDDSK;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDOFF;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIG;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSUB;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDTOT;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.ZipInputStream;

/**
 * Unit tests for {@link EndOfCentralDirectory}.
 */
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
    assertEquals("not based at current position", expMark, view.get(ENDSIG));
    assertEquals("Not slice with position 0", expPos, view.buffer.position());
    assertEquals("Not sized with comment", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
  }

  @Test
  public void testView_String() {
    String[] comments = { "hello world", "", null};

    for (String comment : comments) {
      String expComment = comment != null ? comment : "";
      EndOfCentralDirectory view = EndOfCentralDirectory.allocate(comment);
      String commentResult = view.getComment();
      assertEquals("Incorrect comment", expComment, commentResult);
      int expSize = ZipInputStream.ENDHDR + (comment != null ? comment.getBytes(UTF_8).length : 0);
      int expPos = 0;
      assertEquals("Not at position 0", expPos, view.buffer.position());
      assertEquals("Not sized correctly", expSize, view.getSize());
      assertEquals("Not limited to size", expSize, view.buffer.limit());
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
    assertEquals("not based at current position", expMark, view.get(ENDSIG));
    assertEquals("Not slice with position 0", expPos, view.buffer.position());
    assertEquals("Not sized with comment", expSize, view.getSize());
    assertEquals("Not limited to size", expSize, view.buffer.limit());
    assertEquals("Incorrect comment", comment, view.getComment());
  }

  @Test
  public void testCopy() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    EndOfCentralDirectory view = EndOfCentralDirectory.allocate("comment");
    view.copy(buffer);
    int expSize = view.getSize();
    assertEquals("buffer not advanced as expected", expSize, buffer.position());
    buffer.position(0);
    EndOfCentralDirectory clone = EndOfCentralDirectory.viewOf(buffer);
    assertEquals("Fail to copy mark", view.get(ENDSIG), clone.get(ENDSIG));
    assertEquals("Fail to copy comment", view.getComment(), clone.getComment());
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
    assertEquals("Central directory start disk", cdDisk, view.get(ENDDCD));
    assertEquals("Central directory file offset", cdOffset, view.get(ENDOFF));
    assertEquals("Central directory size", cdSize, view.get(ENDSIZ));
    assertEquals("This disk number", disk, view.get(ENDDSK));
    assertEquals("Number of records on this disk", local, view.get(ENDSUB));
    assertEquals("Total number of central directory records", total, view.get(ENDTOT));
  }
}
