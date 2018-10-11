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

package com.google.devtools.build.zip;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import com.google.common.primitives.Bytes;
import com.google.devtools.build.zip.ZipFileEntry.Compression;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Calendar;
import java.util.Random;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ZipWriterTest {
  @Rule public TemporaryFolder tmp = new TemporaryFolder();
  @Rule public ExpectedException thrown = ExpectedException.none();

  private Random rand;
  private Calendar cal;
  private CRC32 crc;
  private Deflater deflater;
  private File test;

  @Before public void setup() throws IOException {
    rand = new Random();
    cal = Calendar.getInstance();
    cal.clear();
    cal.set(Calendar.YEAR, rand.nextInt(128) + 1980); // Zip files have 7-bit year resolution.
    cal.set(Calendar.MONTH, rand.nextInt(12));
    cal.set(Calendar.DAY_OF_MONTH, rand.nextInt(29));
    cal.set(Calendar.HOUR_OF_DAY, rand.nextInt(24));
    cal.set(Calendar.MINUTE, rand.nextInt(60));
    cal.set(Calendar.SECOND, rand.nextInt(30) * 2); // Zip files have 2 second resolution.

    crc = new CRC32();
    deflater = new Deflater(Deflater.DEFAULT_COMPRESSION, true);
    test = tmp.newFile("test.zip");
  }

  @Test public void testEmpty() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
    }

    try (ZipFile zipFile = new ZipFile(test)) {
      assertThat(zipFile.entries().hasMoreElements()).isFalse();
    }
  }

  @Test public void testComment() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.setComment("test comment");
    }

    try (ZipFile zipFile = new ZipFile(test)) {
      assertThat(zipFile.entries().hasMoreElements()).isFalse();
      assertThat(zipFile.getComment()).isEqualTo("test comment");
    }
  }

  @Test public void testFileDataBeforeEntry() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.write(new byte[] { 0xf, 0xa, 0xb });
      fail("Expected ZipException");
    } catch (ZipException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Cannot write zip contents without first setting a"
                  + " ZipEntry or starting a prefix file.");
    }

    try (ZipFile zipFile = new ZipFile(test)) {
      assertThat(zipFile.entries().hasMoreElements()).isFalse();
    }
  }

  @Test public void testSingleEntry() throws IOException {
    byte[] content = "content".getBytes(UTF_8);
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      crc.update(content);
      ZipFileEntry entry = new ZipFileEntry("foo");
      entry.setSize(content.length);
      entry.setCompressedSize(content.length);
      entry.setCrc(crc.getValue());
      entry.setTime(cal.getTimeInMillis());

      writer.putNextEntry(entry);
      writer.write(content);
      writer.closeEntry();
    }

    byte[] buf = new byte[128];
    try (ZipFile zipFile = new ZipFile(test)) {
      ZipEntry foo = zipFile.getEntry("foo");
      assertThat(foo.getMethod()).isEqualTo(ZipEntry.STORED);
      assertThat(foo.getSize()).isEqualTo(content.length);
      assertThat(foo.getCompressedSize()).isEqualTo(content.length);
      assertThat(foo.getCrc()).isEqualTo(crc.getValue());
      assertThat(foo.getTime()).isEqualTo(cal.getTimeInMillis());
      zipFile.getInputStream(foo).read(buf);
      assertThat(Bytes.indexOf(buf, content)).isEqualTo(0);
    }
  }

  @Test public void testMultipleEntry() throws IOException {
    byte[] fooContent = "content".getBytes(UTF_8);
    byte[] barContent = "stuff".getBytes(UTF_8);
    long fooCrc = -1;
    long barCrc = -1;
    int deflatedSize = -1;
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.setComment("file comment");

      crc.update(fooContent);
      fooCrc = crc.getValue();
      ZipFileEntry rawFoo = new ZipFileEntry("foo");
      rawFoo.setMethod(Compression.STORED);
      rawFoo.setSize(fooContent.length);
      rawFoo.setCompressedSize(fooContent.length);
      rawFoo.setCrc(crc.getValue());
      rawFoo.setTime(cal.getTimeInMillis());
      rawFoo.setComment("foo comment");

      writer.putNextEntry(rawFoo);
      writer.write(fooContent);
      writer.closeEntry();

      byte[] deflatedBarContent = new byte[128];
      crc.reset();
      crc.update(barContent);
      barCrc = crc.getValue();
      deflater.setInput(barContent);
      deflater.finish();
      deflatedSize = deflater.deflate(deflatedBarContent);
      ZipFileEntry rawBar = new ZipFileEntry("bar");
      rawBar.setMethod(Compression.DEFLATED);
      rawBar.setSize(barContent.length);
      rawBar.setCompressedSize(deflatedSize);
      rawBar.setCrc(barCrc);
      rawBar.setTime(cal.getTimeInMillis());

      writer.putNextEntry(rawBar);
      writer.write(deflatedBarContent, 0, deflatedSize);
      writer.closeEntry();
    }

    byte[] buf = new byte[128];
    try (ZipFile zipFile = new ZipFile(test)) {
      assertThat(zipFile.getComment()).isEqualTo("file comment");

      ZipEntry foo = zipFile.getEntry("foo");
      assertThat(foo.getMethod()).isEqualTo(ZipEntry.STORED);
      assertThat(foo.getSize()).isEqualTo(fooContent.length);
      assertThat(foo.getCompressedSize()).isEqualTo(fooContent.length);
      assertThat(foo.getCrc()).isEqualTo(fooCrc);
      assertThat(foo.getTime()).isEqualTo(cal.getTimeInMillis());
      assertThat(foo.getComment()).isEqualTo("foo comment");
      zipFile.getInputStream(foo).read(buf);
      assertThat(Bytes.indexOf(buf, fooContent)).isEqualTo(0);

      ZipEntry bar = zipFile.getEntry("bar");
      assertThat(bar.getMethod()).isEqualTo(ZipEntry.DEFLATED);
      assertThat(bar.getSize()).isEqualTo(barContent.length);
      assertThat(bar.getCompressedSize()).isEqualTo(deflatedSize);
      assertThat(bar.getCrc()).isEqualTo(barCrc);
      assertThat(bar.getTime()).isEqualTo(cal.getTimeInMillis());
      zipFile.getInputStream(bar).read(buf);
      assertThat(Bytes.indexOf(buf, barContent)).isEqualTo(0);
    }
  }

  @Test public void testWrongSizeContent() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      byte[] content = "content".getBytes(UTF_8);
      crc.update(content);
      ZipFileEntry entry = new ZipFileEntry("foo");
      entry.setSize(content.length);
      entry.setCompressedSize(content.length);
      entry.setCrc(crc.getValue());
      entry.setTime(cal.getTimeInMillis());

      writer.putNextEntry(entry);
      writer.write("some other content".getBytes(UTF_8));
      thrown.expect(ZipException.class);
      thrown.expectMessage("Number of bytes written for the entry");
      writer.closeEntry();
    }
  }

  @Test public void testRawZipEntry() throws IOException {
    byte[] content = "content".getBytes(UTF_8);
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      crc.update(content);
      ZipFileEntry entry = new ZipFileEntry("foo");
      entry.setVersion((short) 1);
      entry.setVersionNeeded((short) 2);
      entry.setSize(content.length);
      entry.setCompressedSize(content.length);
      entry.setCrc(crc.getValue());
      entry.setTime(cal.getTimeInMillis());
      entry.setFlags(ZipUtil.get16(new byte[] {0x08, 0x00}, 0));
      entry.setInternalAttributes(ZipUtil.get16(new byte[] {0x34, 0x12}, 0));
      entry.setExternalAttributes(ZipUtil.get32(new byte[] {0x0a, 0x09, 0x78, 0x56}, 0));
      entry.setLocalHeaderOffset(rand.nextInt(Integer.MAX_VALUE));

      writer.putNextEntry(entry);
      writer.write(content);
      writer.closeEntry();
    }

    byte[] buf = new byte[128];
    try (ZipFile zipFile = new ZipFile(test)) {
      ZipEntry foo = zipFile.getEntry("foo");
      assertThat(foo.getMethod()).isEqualTo(ZipEntry.STORED);
      assertThat(foo.getSize()).isEqualTo(content.length);
      assertThat(foo.getCompressedSize()).isEqualTo(content.length);
      assertThat(foo.getCrc()).isEqualTo(crc.getValue());
      assertThat(foo.getTime()).isEqualTo(cal.getTimeInMillis());
      zipFile.getInputStream(foo).read(buf);
      assertThat(Bytes.indexOf(buf, content)).isEqualTo(0);
    }

    try (ZipReader zipFile = new ZipReader(test)) {
      ZipFileEntry foo = zipFile.getEntry("foo");
      // Versions should be increased to minimum required for STORED compression.
      assertThat(foo.getVersion()).isEqualTo((short) 0xa);
      assertThat(foo.getVersionNeeded()).isEqualTo((short) 0xa);
      assertThat(foo.getFlags()).isEqualTo((short) 0); // Data descriptor bit should be cleared.
      assertThat(foo.getInternalAttributes()).isEqualTo((short) 4660);
      assertThat(foo.getExternalAttributes()).isEqualTo(1450707210);
    }
  }

  @Test public void testPrefixFile() throws IOException, InterruptedException {
    byte[] content = "content".getBytes(UTF_8);
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.startPrefixFile();
      writer.write("#!/bin/bash\necho 'hello world'\n".getBytes(UTF_8));
      writer.endPrefixFile();

      crc.update(content);
      ZipFileEntry entry = new ZipFileEntry("foo");
      entry.setSize(content.length);
      entry.setCompressedSize(content.length);
      entry.setCrc(crc.getValue());
      entry.setTime(cal.getTimeInMillis());

      writer.putNextEntry(entry);
      writer.write(content);
      writer.closeEntry();
    }

    byte[] buf = new byte[128];
    try (ZipFile zipFile = new ZipFile(test)) {
      ZipEntry foo = zipFile.getEntry("foo");
      assertThat(foo.getMethod()).isEqualTo(ZipEntry.STORED);
      assertThat(foo.getSize()).isEqualTo(content.length);
      assertThat(foo.getCompressedSize()).isEqualTo(content.length);
      assertThat(foo.getCrc()).isEqualTo(crc.getValue());
      assertThat(foo.getTime()).isEqualTo(cal.getTimeInMillis());
      zipFile.getInputStream(foo).read(buf);
      assertThat(Bytes.indexOf(buf, content)).isEqualTo(0);
    }

    Process pr = new ProcessBuilder("chmod", "750", test.getAbsolutePath()).start();
    pr.waitFor();
    pr = new ProcessBuilder(test.getAbsolutePath()).start();
    pr.getInputStream().read(buf);
    pr.waitFor();
    assertThat(Bytes.indexOf(buf, "hello world".getBytes(UTF_8))).isEqualTo(0);
  }

  @Test public void testPrefixFileAfterZip() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      byte[] content = "content".getBytes(UTF_8);
      crc.update(content);
      ZipFileEntry entry = new ZipFileEntry("foo");
      entry.setSize(content.length);
      entry.setCompressedSize(content.length);
      entry.setCrc(crc.getValue());
      entry.setTime(cal.getTimeInMillis());

      writer.putNextEntry(entry);
      thrown.expect(ZipException.class);
      thrown.expectMessage("Cannot add a prefix file after the zip contents have been started.");
      writer.startPrefixFile();
      writer.write("#!/bin/bash\necho 'hello world'\n".getBytes(UTF_8));
      writer.endPrefixFile();
    }
  }

  @Test public void testPrefixAfterFinish() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.finish();
      thrown.expect(IllegalStateException.class);
      writer.startPrefixFile();
      writer.write("#!/bin/bash\necho 'hello world'\n".getBytes(UTF_8));
      writer.endPrefixFile();
    }
  }

  @Test public void testPutEntryAfterFinish() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.finish();
      thrown.expect(IllegalStateException.class);
      writer.putNextEntry(new ZipFileEntry("foo"));
    }
  }

  @Test public void testCloseEntryAfterFinish() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      byte[] content = "content".getBytes(UTF_8);
      crc.update(content);
      ZipFileEntry entry = new ZipFileEntry("foo");
      entry.setSize(content.length);
      entry.setCompressedSize(content.length);
      entry.setCrc(crc.getValue());
      entry.setTime(cal.getTimeInMillis());

      writer.putNextEntry(entry);
      writer.write(content);
      writer.finish();
      thrown.expect(IllegalStateException.class);
      writer.closeEntry();
    }
  }

  @Test public void testFinishAfterFinish() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.finish();
      thrown.expect(IllegalStateException.class);
      writer.finish();
    }
  }

  @Test public void testWriteAfterFinish() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8)) {
      writer.finish();
      thrown.expect(IllegalStateException.class);
      writer.write("content".getBytes(UTF_8));
    }
  }

  @Test public void testZip64_FileCount_32BitMax() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8, true)) {
      ZipFileEntry template = new ZipFileEntry("template");
      template.setSize(0);
      template.setCompressedSize(0);
      template.setCrc(0);
      template.setTime(cal.getTimeInMillis());
      for (int i = 0; i < 0xffff; i++) {
        ZipFileEntry entry = new ZipFileEntry(template);
        entry.setName("entry" + i);
        writer.putNextEntry(entry);
      }
    }
    try (ZipReader reader = new ZipReader(test, UTF_8, true)) {
      assertThat(reader.size()).isEqualTo(0xffff);
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.size()).isEqualTo(0xffff);
    }
  }

  @Test public void testZip64_FileCount_Zip64Range() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8, true)) {
      ZipFileEntry template = new ZipFileEntry("template");
      template.setSize(0);
      template.setCompressedSize(0);
      template.setCrc(0);
      template.setTime(cal.getTimeInMillis());
      for (int i = 0; i < 0x100ff; i++) {
        ZipFileEntry entry = new ZipFileEntry(template);
        entry.setName("entry" + i);
        writer.putNextEntry(entry);
      }
    }
    try (ZipReader reader = new ZipReader(test, UTF_8, true)) {
      assertThat(reader.size()).isEqualTo(0x100ff);
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.size()).isEqualTo(0x100ff);
    }
  }

  @Test public void testZip64_FileCount_Zip64Range_ForceZip32() throws IOException {
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8, false)) {
      ZipFileEntry template = new ZipFileEntry("template");
      template.setSize(0);
      template.setCompressedSize(0);
      template.setCrc(0);
      template.setTime(cal.getTimeInMillis());
      for (int i = 0; i < 0x100ff; i++) {
        ZipFileEntry entry = new ZipFileEntry(template);
        entry.setName("entry" + i);
        writer.putNextEntry(entry);
      }
    }
    try (ZipReader reader = new ZipReader(test, UTF_8, true)) {
      assertThat(reader.size()).isEqualTo(0x00ff);
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.size()).isEqualTo(0x100ff);
    }
  }

  @Test public void testZip64_FileSize_32BitMax() throws IOException {
    long size = 0xffffffffL;
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8, true)) {
      ZipFileEntry entry = new ZipFileEntry("big");
      entry.setCompressedSize(size);
      entry.setSize(size);
      entry.setCrc(0);
      entry.setTime(cal.getTimeInMillis());
      writer.putNextEntry(entry);
      byte[] chunk = new byte[1024];
      for (int i = 0; i < size / chunk.length; i++) {
        writer.write(chunk);
      }
      writer.write(chunk, 0, (int) (size % chunk.length));
      writer.closeEntry();
    }
    try (ZipFile file = new ZipFile(test)) {
      ZipEntry entry = file.getEntry("big");
      assertThat(entry.getSize()).isEqualTo(size);
      assertThat(entry.getCompressedSize()).isEqualTo(size);
    }
  }

  @Test public void testZip64_FileSize_Zip64Range() throws IOException {
    long size = 0x1000000ffL;
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8, true)) {
      ZipFileEntry entry = new ZipFileEntry("big");
      entry.setCompressedSize(size);
      entry.setSize(size);
      entry.setCrc(0);
      entry.setTime(cal.getTimeInMillis());
      writer.putNextEntry(entry);
      byte[] chunk = new byte[1024];
      for (int i = 0; i < size / chunk.length; i++) {
        writer.write(chunk);
      }
      writer.write(chunk, 0, (int) (size % chunk.length));
      writer.closeEntry();
    }
    try (ZipFile file = new ZipFile(test)) {
      ZipEntry entry = file.getEntry("big");
      assertThat(entry.getSize()).isEqualTo(size);
      assertThat(entry.getCompressedSize()).isEqualTo(size);
    }
  }

  @Test public void testZip64_FileSize_Zip64Range_ForceZip32() throws IOException {
    long size = 0x1000000ffL;
    try (ZipWriter writer = new ZipWriter(Files.newOutputStream(test.toPath()), UTF_8, false)) {
      ZipFileEntry entry = new ZipFileEntry("big");
      entry.setCompressedSize(size);
      entry.setSize(size);
      entry.setCrc(0);
      entry.setTime(cal.getTimeInMillis());
      thrown.expect(ZipException.class);
      thrown.expectMessage("Writing an entry of size");
      thrown.expectMessage("without Zip64 extensions is not supported.");
      writer.putNextEntry(entry);
    }
  }
}
