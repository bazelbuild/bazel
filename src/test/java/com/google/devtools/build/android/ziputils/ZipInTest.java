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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENHOW;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENOFF;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDOFF;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIG;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSUB;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDTOT;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.android.ziputils.ZipIn.ZipEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.zip.ZipInputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ZipIn}.
 */
@RunWith(JUnit4.class)
public class ZipInTest {

  private static final int ENTRY_COUNT = 1000;
  private FakeFileSystem fileSystem;

  @Before
  public void setUp() throws Exception {
    fileSystem = new FakeFileSystem();
  }

  @Test
  public void testEndOfCentralDirectory_found() throws Exception {
    String filename = "test.zip";
    // Find it, even if it's the only useful thing in the file.
    String subcase = " EOCD found it, ";
    byte[] bytes =
        new byte[] {
          0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        };
    fileSystem.addFile(filename, bytes);
    ZipIn zipIn = newZipIn(filename);
    EndOfCentralDirectory result = zipIn.endOfCentralDirectory();
    assertWithMessage(subcase + "found").that(result).isNotNull();
  }

  @Test
  public void testEndOfCentralDirectory_notPresent() throws Exception {
    String filename = "test.zip";
    String subcase = " EOCD not there at all, ";
    byte[] bytes =
        new byte[] {
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        };
    fileSystem.addFile(filename, bytes);
    ZipIn zipIn = newZipIn(filename);
    Exception ex =
        assertThrows(
            subcase + "expected IllegalStateException",
            Exception.class,
            () -> zipIn.endOfCentralDirectory());
    assertWithMessage(subcase + "caught exception")
        .that(ex.getClass())
        .isSameInstanceAs(IllegalStateException.class);
  }

  @Test
  public void testEndOfCentralDirectory_tooLateToRead() throws Exception {
    String filename = "test.zip";
    // If we can't read it, it's not there
    String subcase = " EOCD too late to read, ";
    byte[] bytes =
        new byte[] {
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        };
    fileSystem.addFile(filename, bytes);
    ZipIn zipIn = newZipIn(filename);
    Exception ex =
        assertThrows(
            subcase + "expected IndexOutOfBoundsException",
            Exception.class,
            () -> zipIn.endOfCentralDirectory());
    assertWithMessage(subcase + "caught exception")
        .that(ex.getClass())
        .isSameInstanceAs(IndexOutOfBoundsException.class);
  }

  @Test
  public void testEndOfCentralDirectory_goodHidenByBad() throws Exception {
    String filename = "test.zip";
    // Current implementation doesn't know to scan past a bad EOCD record.
    // I'm not sure if it should.
    String subcase = " EOCD good hidden by bad, ";
    byte[] bytes =
        new byte[] {
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        };
    fileSystem.addFile(filename, bytes);
    ZipIn zipIn = newZipIn(filename);
    Exception ex =
        assertThrows(
            subcase + "expected IndexOutOfBoundsException",
            Exception.class,
            () -> zipIn.endOfCentralDirectory());
    assertWithMessage(subcase + "caught exception")
        .that(ex.getClass())
        .isSameInstanceAs(IndexOutOfBoundsException.class);
  }

  @Test
  public void testEndOfCentralDirectory_truncatedComment() throws Exception {
    String filename = "test.zip";
    // Minimal format checking here, assuming the EndOfDirectoryTest class
    // test for that.

    String subcase = " EOCD truncated comment, ";
    byte[] bytes = new byte[100];
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    String comment = "optional file comment";
    int commentLen = comment.getBytes(UTF_8).length;
    int offset = bytes.length - ZipInputStream.ENDHDR - commentLen;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    byte[] truncated = Arrays.copyOf(bytes, bytes.length - 5);
    fileSystem.addFile(filename, truncated);
    ZipIn zipIn = newZipIn(filename);
    Exception ex =
        assertThrows(
            subcase + "expected IllegalArgumentException",
            Exception.class,
            () -> zipIn.endOfCentralDirectory());
    assertWithMessage(subcase + "caught exception")
        .that(ex.getClass())
        .isSameInstanceAs(IllegalArgumentException.class);
  }

  @Test
  public void testEndOfCentralDirectory_noComment() throws Exception {
    String filename = "test.zip";
    String subcase = " EOCD no comment, ";
    byte[] bytes = new byte[100];
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    String comment = null;
    int commentLen = 0;
    int offset = bytes.length - ZipInputStream.ENDHDR - commentLen;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    fileSystem.addFile(filename, bytes);
    ZipIn zipIn = newZipIn(filename);
    EndOfCentralDirectory result = zipIn.endOfCentralDirectory();
    assertWithMessage(subcase + "found").that(result).isNotNull();
    assertWithMessage(subcase + "comment").that(result.getComment()).isEqualTo("");
    assertWithMessage(subcase + "marker")
        .that((int) result.get(ENDSIG))
        .isEqualTo(ZipInputStream.ENDSIG);
  }

  @Test
  public void testEndOfCentralDirectory_comment() throws Exception {
    String filename = "test.zip";
    String subcase = " EOCD comment, ";
    byte[] bytes = new byte[100];
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    String comment = "optional file comment";
    int commentLen = comment.getBytes(UTF_8).length;
    int offset = bytes.length - ZipInputStream.ENDHDR - commentLen;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    assertWithMessage(subcase + "setup")
        .that(new String(bytes, bytes.length - commentLen, commentLen, UTF_8))
        .isEqualTo(comment);
    fileSystem.addFile(filename, bytes);
    ZipIn zipIn = newZipIn(filename);
    EndOfCentralDirectory result = zipIn.endOfCentralDirectory();
    assertWithMessage(subcase + "found").that(result).isNotNull();
    assertWithMessage(subcase + "comment").that(result.getComment()).isEqualTo(comment);
    assertWithMessage(subcase + "marker")
        .that((int) result.get(ENDSIG))
        .isEqualTo(ZipInputStream.ENDSIG);
  }

  @Test
  public void testEndOfCentralDirectory_extraData() throws Exception {
    String filename = "test.zip";

    String subcase = " EOCD extra data, ";
    byte[] bytes = new byte[100];
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    String comment = null;
    int commentLen = 0;
    int offset = bytes.length - ZipInputStream.ENDHDR - commentLen - 10;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    fileSystem.addFile(filename, bytes);
    ZipIn zipIn = newZipIn(filename);
    EndOfCentralDirectory result = zipIn.endOfCentralDirectory();
    assertWithMessage(subcase + "found").that(result).isNotNull();
    assertWithMessage(subcase + "comment").that(result.getComment()).isEqualTo("");
    assertWithMessage(subcase + "marker")
        .that((int) result.get(ENDSIG))
        .isEqualTo(ZipInputStream.ENDSIG);
  }

  /**
   * Test of centralDirectory method, of class ZipIn.
   */
  @Test
  public void testCentralDirectory() throws Exception {
    String filename = "test.zip";
    ByteBuffer buffer;
    int offset;
    ZipIn zipIn;
    String subcase;
    subcase = " EOCD extra data, ";
    String commonName = "thisIsNotNormal.txt";
    int filenameLen = commonName.getBytes(UTF_8).length;
    int count = ENTRY_COUNT;
    int dirEntry = ZipInputStream.CENHDR;
    int before = count;
    int between = 0; // implementation doesn't tolerate data between dir entries, does the spec?
    int after = 20;
    int eocd = ZipInputStream.ENDHDR;
    int total = before + (count * (dirEntry + filenameLen)) + ((count - 1) * between)
        + after + eocd;
    byte[] bytes = new byte[total];
    offset = before;
    for (int i = 0; i < count; i++) {
      if (i > 0) {
        offset += between;
      }
      buffer = ByteBuffer.wrap(bytes, offset, bytes.length - offset);
      DirectoryEntry.view(buffer, commonName, null, null)
          .set(CENHOW, (short) 8)
          .set(CENSIZ, before)
          .set(CENLEN, 2 * before)
          .set(CENOFF, i); // Not valid of course, but we're only testing central dir parsing.
          // and there are currently no checks in the parser to see if offset makes sense.
      offset += dirEntry + filenameLen;
    }
    offset += after;
    buffer = ByteBuffer.wrap(bytes, offset, bytes.length - offset);
    EndOfCentralDirectory.view(buffer, null)
        .set(ENDOFF, before)
        .set(ENDSIZ, offset - before - after)
        .set(ENDTOT, (short) count)
        .set(ENDSUB, (short) count);

    fileSystem.addFile(filename, bytes);
    zipIn = newZipIn(filename);
    CentralDirectory result = zipIn.centralDirectory();
    assertWithMessage(subcase + "found").that(result).isNotNull();
    List<DirectoryEntry> list = result.list();
    assertWithMessage(subcase + "size").that(list.size()).isEqualTo(count);
    for (int i = 0; i < list.size(); i++) {
      assertWithMessage(subcase + "offset check[" + i + "]")
          .that(list.get(i).get(CENOFF))
          .isEqualTo(i);
    }
  }

  /**
   * Test of scanEntries method, of class ZipIn.
   */
  @Test
  public void testScanEntries() throws Exception {
    int count = ENTRY_COUNT * 100;
    String filename = "test.jar";

    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);

    final ZipIn zipIn = newZipIn(filename);
    zipIn.scanEntries(
        new EntryHandler() {
          int count = 0;

          @Override
          public void handle(
              ZipIn in, LocalFileHeader header, DirectoryEntry dirEntry, ByteBuffer data)
              throws IOException {
            assertThat(in).isSameInstanceAs(zipIn);
            String filename = "pkg/f" + count + ".class";
            assertThat(header.getFilename()).isEqualTo(filename);
            assertThat(dirEntry.getFilename()).isEqualTo(filename);
            count++;
          }
        });
  }

  /**
   * Test of nextHeaderFrom method, of class ZipIn.
   */
  @Test
  public void testNextHeaderFrom_long() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.endOfCentralDirectory();
    count = 0;
    int offset = 0;
    LocalFileHeader header;
    do {
      header = zipIn.nextHeaderFrom(offset);
      String name = "pkg/f" + count + ".class";
      if (header != null) {
        assertThat(header.getFilename()).isEqualTo(name);
        count++;
        offset = (int) header.fileOffset() + 4;
      }
    } while(header != null);
    assertThat(count).isEqualTo(ENTRY_COUNT);
  }

  /**
   * Test of nextHeaderFrom method, of class ZipIn.
   */
  @Test
  public void testNextHeaderFrom_DirectoryEntry() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.centralDirectory();
    List<DirectoryEntry> list = zipIn.centralDirectory().list();
    count = 0;
    String name;
    LocalFileHeader header = zipIn.nextHeaderFrom(null);
    for (DirectoryEntry dirEntry : list) {
      name = "pkg/f" + count + ".class";
      assertThat(dirEntry.getFilename()).isEqualTo(name);
      assertThat(header.getFilename()).isEqualTo(name);
      header = zipIn.nextHeaderFrom(dirEntry);
      count++;
    }
    assertThat(header).isNull();
  }

  /**
   * Test of localHeaderFor method, of class ZipIn.
   */
  @Test
  public void testLocalHeaderFor() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.centralDirectory();
    List<DirectoryEntry> list = zipIn.centralDirectory().list();
    count = 0;
    String name;
    LocalFileHeader header;
    for (DirectoryEntry dirEntry : list) {
      name = "pkg/f" + count + ".class";
      header = zipIn.localHeaderFor(dirEntry);
      assertThat(dirEntry.getFilename()).isEqualTo(name);
      assertThat(header.getFilename()).isEqualTo(name);
      count++;
    }
  }

  /**
   * Test of localHeaderAt method, of class ZipIn.
   */
  @Test
  public void testLocalHeaderAt() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.centralDirectory();
    List<DirectoryEntry> list = zipIn.centralDirectory().list();
    count = 0;
    String name;
    LocalFileHeader header;
    for (DirectoryEntry dirEntry : list) {
      name = "pkg/f" + count + ".class";
      header = zipIn.localHeaderAt(dirEntry.get(CENOFF));
      assertThat(dirEntry.getFilename()).isEqualTo(name);
      assertThat(header.getFilename()).isEqualTo(name);
      count++;
    }
  }

  /**
   * Test of nextFrom method, of class ZipIn.
   */
  @Test
  public void testNextFrom_long() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.centralDirectory();
    count = 0;
    int offset = 0;
    ZipEntry zipEntry;
    do {
      zipEntry = zipIn.nextFrom(offset);
      String name = "pkg/f" + count + ".class";
      if (zipEntry.getCode() != ZipEntry.Status.ENTRY_NOT_FOUND) {
        assertThat(zipEntry.getHeader()).isNotNull();
        assertThat(zipEntry.getDirEntry()).isNotNull();
        assertThat(zipEntry.getHeader().getFilename()).isEqualTo(name);
        assertThat(zipEntry.getDirEntry().getFilename()).isEqualTo(name);
        count++;
        offset = (int) zipEntry.getHeader().fileOffset() + 4;
      }
    } while(zipEntry.getCode() != ZipEntry.Status.ENTRY_NOT_FOUND);
    assertThat(count).isEqualTo(ENTRY_COUNT);
  }

  /**
   * Test of nextFrom method, of class ZipIn.
   */
  @Test
  public void testNextFrom_DirectoryEntry() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.centralDirectory();
    List<DirectoryEntry> list = zipIn.centralDirectory().list();
    count = 0;
    String name;
    ZipEntry zipEntry = zipIn.nextFrom(null);
    for (DirectoryEntry dirEntry : list) {
      if (zipEntry.getCode() == ZipEntry.Status.ENTRY_NOT_FOUND) {
        break;
      }
      name = "pkg/f" + count + ".class";
      assertThat(zipEntry.getHeader()).isNotNull();
      assertThat(zipEntry.getDirEntry()).isNotNull();
      assertThat(zipEntry.getHeader().getFilename()).isEqualTo(name);
      assertThat(zipEntry.getDirEntry().getFilename()).isEqualTo(name);
      zipEntry = zipIn.nextFrom(dirEntry);
      count++;
    }
    assertThat(count).isEqualTo(ENTRY_COUNT);
  }

  /**
   * Test of entryAt method, of class ZipIn.
   */
  @Test
  public void testEntryAt() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.centralDirectory();
    List<DirectoryEntry> list = zipIn.centralDirectory().list();
    count = 0;
    String name;
    ZipEntry zipEntry;
    for (DirectoryEntry dirEntry : list) {
      zipEntry = zipIn.entryAt(dirEntry.get(CENOFF));
      name = "pkg/f" + count + ".class";
      assertThat(zipEntry.getHeader()).isNotNull();
      assertThat(zipEntry.getDirEntry()).isNotNull();
      assertThat(zipEntry.getHeader().getFilename()).isEqualTo(name);
      assertThat(zipEntry.getDirEntry().getFilename()).isEqualTo(name);
      count++;
    }
    assertThat(count).isEqualTo(ENTRY_COUNT);
  }

  /**
   * Test of entryWith method, of class ZipIn.
   */
  @Test
  public void testEntryWith() throws Exception {
    int count = ENTRY_COUNT;
    String filename = "test.jar";
    ZipFileBuilder builder = new ZipFileBuilder();
    for (int i = 0; i < count; i++) {
      builder.add("pkg/f" + i + ".class", "All day long");
    }
    builder.create(filename);
    final ZipIn zipIn = newZipIn(filename);
    zipIn.centralDirectory();
    count = 0;
    int offset = 0;
    LocalFileHeader header;
    do {
      header = zipIn.nextHeaderFrom(offset);
      String name = "pkg/f" + count + ".class";
      if (header != null) {
        ZipEntry zipEntry = zipIn.entryWith(header);
        assertThat(zipEntry.getDirEntry()).isNotNull();
        assertThat(zipEntry.getHeader()).isSameInstanceAs(header);
        assertThat(zipEntry.getHeader().getFilename()).isEqualTo(name);
        assertThat(zipEntry.getDirEntry().getFilename()).isEqualTo(name);
        assertThat(header.getFilename()).isEqualTo(name);
        count++;
        offset = (int) header.fileOffset() + 4;
      }
    } while(header != null);
    assertThat(count).isEqualTo(ENTRY_COUNT);
  }

  private ZipIn newZipIn(String filename) throws IOException {
    return new ZipIn(fileSystem.getInputChannel(filename), filename);
  }
}
