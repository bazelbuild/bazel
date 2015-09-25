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

import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENHOW;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENOFF;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDOFF;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIG;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSUB;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDTOT;
import static com.google.devtools.build.android.ziputils.ZipIn.ZipEntry;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.zip.ZipInputStream;

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

  /**
   * Test of endOfCentralDirectory method, of class ZipIn.
   */
  @Test
  public void testEndOfCentralDirectory() throws Exception {

    String filename = "test.zip";
    byte[] bytes;
    ByteBuffer buffer;
    String comment;
    int commentLen;
    int offset;
    ZipIn zipIn;
    EndOfCentralDirectory result;
    String subcase;

    // Find it, even if it's the only useful thing in the file.
    subcase = " EOCD found it, ";
    bytes = new byte[] {
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
    zipIn = newZipIn(filename);
    result = zipIn.endOfCentralDirectory();
    assertNotNull(subcase + "found", result);

    subcase = " EOCD not there at all, ";
    bytes = new byte[]{
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
    zipIn = newZipIn(filename);
    try {
      zipIn.endOfCentralDirectory();
      fail(subcase + "expected IllegalStateException");
    } catch (Exception ex) {
      assertSame(subcase + "caught exception", IllegalStateException.class, ex.getClass());
    }

    // If we can't read it, it's not there
    subcase = " EOCD too late to read, ";
    bytes = new byte[] {
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
    zipIn = newZipIn(filename);
    try {
      zipIn.endOfCentralDirectory();
      fail(subcase + "expected IndexOutOfBoundsException");
    } catch (Exception ex) {
      assertSame(subcase + "caught exception", IndexOutOfBoundsException.class, ex.getClass());
    }

    // Current implementation doesn't know to scan past a bad EOCD record.
    // I'm not sure if it should.
    subcase = " EOCD good hidden by bad, ";
    bytes = new byte[] {
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
    zipIn = newZipIn(filename);
    try {
      zipIn.endOfCentralDirectory();
      fail(subcase + "expected IndexOutOfBoundsException");
    } catch (Exception ex) {
      assertSame(subcase + "caught exception", IndexOutOfBoundsException.class, ex.getClass());
    }

    // Minimal format checking here, assuming the EndOfDirectoryTest class
    // test for that.

    subcase = " EOCD truncated comment, ";
    bytes = new byte[100];
    buffer = ByteBuffer.wrap(bytes);
    comment = "optional file comment";
    commentLen = comment.getBytes(UTF_8).length;
    offset = bytes.length - ZipInputStream.ENDHDR - commentLen;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    byte[] truncated = Arrays.copyOf(bytes, bytes.length - 5);
    fileSystem.addFile(filename, truncated);
    zipIn = newZipIn(filename);
    try { // not sure this is the exception we want!
      zipIn.endOfCentralDirectory();
      fail(subcase + "expected IllegalArgumentException");
    } catch (Exception ex) {
      assertSame(subcase + "caught exception", IllegalArgumentException.class, ex.getClass());
    }

    subcase = " EOCD no comment, ";
    bytes = new byte[100];
    buffer = ByteBuffer.wrap(bytes);
    comment = null;
    commentLen = 0;
    offset = bytes.length - ZipInputStream.ENDHDR - commentLen;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    fileSystem.addFile(filename, bytes);
    zipIn = newZipIn(filename);
    result = zipIn.endOfCentralDirectory();
    assertNotNull(subcase + "found", result);
    assertEquals(subcase + "comment", "", result.getComment());
    assertEquals(subcase + "marker", ZipInputStream.ENDSIG, (int) result.get(ENDSIG));

    subcase = " EOCD comment, ";
    bytes = new byte[100];
    buffer = ByteBuffer.wrap(bytes);
    comment = "optional file comment";
    commentLen = comment.getBytes(UTF_8).length;
    offset = bytes.length - ZipInputStream.ENDHDR - commentLen;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    assertEquals(subcase + "setup", comment,
        new String(bytes, bytes.length - commentLen, commentLen, UTF_8));
    fileSystem.addFile(filename, bytes);
    zipIn = newZipIn(filename);
    result = zipIn.endOfCentralDirectory();
    assertNotNull(subcase + "found", result);
    assertEquals(subcase + "comment", comment, result.getComment());
    assertEquals(subcase + "marker", ZipInputStream.ENDSIG, (int) result.get(ENDSIG));

    subcase = " EOCD extra data, ";
    bytes = new byte[100];
    buffer = ByteBuffer.wrap(bytes);
    comment = null;
    commentLen = 0;
    offset = bytes.length - ZipInputStream.ENDHDR - commentLen - 10;
    buffer.position(offset);
    EndOfCentralDirectory.view(buffer, comment);
    fileSystem.addFile(filename, bytes);
    zipIn = newZipIn(filename);
    result = zipIn.endOfCentralDirectory();
    assertNotNull(subcase + "found", result);
    assertEquals(subcase + "comment", "", result.getComment());
    assertEquals(subcase + "marker", ZipInputStream.ENDSIG, (int) result.get(ENDSIG));
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
    assertNotNull(subcase + "found", result);
    List<DirectoryEntry> list = result.list();
    assertEquals(subcase + "size", count, list.size());
    for (int i = 0; i < list.size(); i++) {
      assertEquals(subcase + "offset check[" + i + "]", i, list.get(i).get(CENOFF));
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
    zipIn.scanEntries(new EntryHandler() {
      int count = 0;
      @Override
      public void handle(ZipIn in, LocalFileHeader header, DirectoryEntry dirEntry,
          ByteBuffer data) throws IOException {
        assertSame(zipIn, in);
        String filename = "pkg/f" + count + ".class";
        assertEquals(filename, header.getFilename());
        assertEquals(filename, dirEntry.getFilename());
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
        assertEquals(name, header.getFilename());
        count++;
        offset = (int) header.fileOffset() + 4;
      }
    } while(header != null);
    assertEquals(ENTRY_COUNT, count);
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
      assertEquals(name, dirEntry.getFilename());
      assertEquals(name, header.getFilename());
      header = zipIn.nextHeaderFrom(dirEntry);
      count++;
    }
    assertNull(header);
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
      assertEquals(name, dirEntry.getFilename());
      assertEquals(name, header.getFilename());
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
      assertEquals(name, dirEntry.getFilename());
      assertEquals(name, header.getFilename());
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
        assertNotNull(zipEntry.getHeader());
        assertNotNull(zipEntry.getDirEntry());
        assertEquals(name, zipEntry.getHeader().getFilename());
        assertEquals(name, zipEntry.getDirEntry().getFilename());
        count++;
        offset = (int) zipEntry.getHeader().fileOffset() + 4;
      }
    } while(zipEntry.getCode() != ZipEntry.Status.ENTRY_NOT_FOUND);
    assertEquals(ENTRY_COUNT, count);
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
      assertNotNull(zipEntry.getHeader());
      assertNotNull(zipEntry.getDirEntry());
      assertEquals(name, zipEntry.getHeader().getFilename());
      assertEquals(name, zipEntry.getDirEntry().getFilename());
      zipEntry = zipIn.nextFrom(dirEntry);
      count++;
    }
    assertEquals(ENTRY_COUNT, count);
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
      assertNotNull(zipEntry.getHeader());
      assertNotNull(zipEntry.getDirEntry());
      assertEquals(name, zipEntry.getHeader().getFilename());
      assertEquals(name, zipEntry.getDirEntry().getFilename());
      count++;
    }
    assertEquals(ENTRY_COUNT, count);
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
        assertNotNull(zipEntry.getDirEntry());
        assertSame(header, zipEntry.getHeader());
        assertEquals(name, zipEntry.getHeader().getFilename());
        assertEquals(name, zipEntry.getDirEntry().getFilename());
        assertEquals(name, header.getFilename());
        count++;
        offset = (int) header.fileOffset() + 4;
      }
    } while(header != null);
    assertEquals(ENTRY_COUNT, count);
  }

  private ZipIn newZipIn(String filename) throws IOException {
    return new ZipIn(fileSystem.getInputChannel(filename), filename);
  }
}
