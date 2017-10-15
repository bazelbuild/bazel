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
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENTIM;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link CentralDirectory}. */
@RunWith(JUnit4.class)
public class CentralDirectoryTest {

  /**
   * Test of viewOf method, of class CentralDirectory.
   */
  @Test
  public void testViewOf() {
    ByteBuffer buffer = ByteBuffer.allocate(100).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 100; i++) {
      buffer.put((byte) i);
    }
    buffer.position(20);
    buffer.limit(90);
    CentralDirectory view = CentralDirectory.viewOf(buffer);
    int expPos = 0;
    int expLimit = 90;
    // expect the buffer to have been reset to 0 (CentralDirectory does NOT slice).
    assertWithMessage("View not at position 0").that(view.buffer.position()).isEqualTo(expPos);
    assertWithMessage("Buffer not at position 0").that(buffer.position()).isEqualTo(expPos);
    assertWithMessage("Buffer limit changed").that(view.buffer.limit()).isEqualTo(expLimit);
    assertWithMessage("Buffer limit changed").that(buffer.limit()).isEqualTo(expLimit);
  }

  /**
   * Test of parse method, of class CentralDirectory.
   */
  @Test
  public void testParse() {
    // First fill it with some entries
    ByteBuffer inputBuffer = ByteBuffer.allocate(10000).order(ByteOrder.LITTLE_ENDIAN);
    String comment = null;
    byte[] extra = null;
    String filename = "pkg/0.txt";
    DirectoryEntry entry = DirectoryEntry.view(inputBuffer, filename, extra , comment);
    int expSize = entry.getSize();
    comment = "";
    extra = new byte[]{};
    for (int i = 1; i < 20; i++) {
      filename = "pkg/" + i + ".txt";
      entry = DirectoryEntry.view(inputBuffer, filename, extra , comment);
      expSize += entry.getSize();
      extra = new byte[extra.length + 1];
      comment = comment + "," + i;
    }
    // Parse the entries.
    CentralDirectory cdir = CentralDirectory.viewOf(inputBuffer).at(0).parse();
    assertWithMessage("Count").that(cdir.getCount()).isEqualTo(20);
    assertWithMessage("Position after parse").that(cdir.buffer.position()).isEqualTo(expSize);
    assertWithMessage("Limit after parse").that(cdir.buffer.limit()).isEqualTo(10000);
    cdir.buffer.flip();
    assertWithMessage("Position after finish").that(cdir.buffer.position()).isEqualTo(0);
    assertWithMessage("Limit after finish").that(cdir.buffer.limit()).isEqualTo(expSize);
  }

  /**
   * Test of nextEntry method, of class CentralDirectory.
   */
  @Test
  public void testNextEntry() {
    ByteBuffer outputBuffer = ByteBuffer.allocate(10000).order(ByteOrder.LITTLE_ENDIAN);
    CentralDirectory cdir = CentralDirectory.viewOf(outputBuffer);
    String comment = null;
    byte[] extra = null;
    String filename = "pkg/0.txt";
    DirectoryEntry entry = DirectoryEntry.allocate(filename, extra , comment);
    cdir.nextEntry(entry).set(CENTIM, 0);
    int expSize = entry.getSize();
    comment = "";
    extra = new byte[]{};
    for (int i = 1; i < 20; i++) {
      filename = "pkg/" + i + ".txt";
      entry = DirectoryEntry.allocate(filename, extra , comment);
      cdir.nextEntry(entry).set(CENTIM, 0);
      int size = entry.getSize();
      expSize += size;
      extra = new byte[extra.length + 1];
      comment = comment + "," + i;
    }
    assertWithMessage("Count").that(cdir.getCount()).isEqualTo(20);
    assertWithMessage("Position after build").that(cdir.buffer.position()).isEqualTo(expSize);
    assertWithMessage("Limit after build").that(cdir.buffer.limit()).isEqualTo(10000);
    cdir.buffer.flip();
    assertWithMessage("Position after finish build").that(cdir.buffer.position()).isEqualTo(0);
    assertWithMessage("Limit after finish build").that(cdir.buffer.limit()).isEqualTo(expSize);

    // now try to parse the directory we just created.
    cdir.at(0).parse();
    assertWithMessage("Count").that(cdir.getCount()).isEqualTo(20);
    assertWithMessage("Position after re-parse").that(cdir.buffer.position()).isEqualTo(expSize);
    assertWithMessage("Limit after re-parse").that(cdir.buffer.limit()).isEqualTo(expSize);
    cdir.buffer.flip();
    assertWithMessage("Position after finish parse").that(cdir.buffer.position()).isEqualTo(0);
    assertWithMessage("Limit after finish parse").that(cdir.buffer.limit()).isEqualTo(expSize);
  }
}
