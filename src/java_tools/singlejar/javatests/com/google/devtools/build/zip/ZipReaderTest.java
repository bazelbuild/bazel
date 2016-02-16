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

import com.google.common.io.ByteStreams;
import com.google.devtools.build.zip.ZipFileEntry.Compression;
import com.google.devtools.build.zip.ZipFileEntry.Feature;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.util.Calendar;
import java.util.Collection;
import java.util.Date;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipOutputStream;

@RunWith(JUnit4.class)
public class ZipReaderTest {
  @Rule public TemporaryFolder tmp = new TemporaryFolder();
  @Rule public ExpectedException thrown = ExpectedException.none();

  private File test;

  private void assertDateWithin(Date testDate, Date start, Date end) {
    if (testDate.before(start) || testDate.after(end)) {
      fail();
    }
  }

  private void assertDateAboutNow(Date testDate) {
    Date now = new Date();
    Calendar cal = Calendar.getInstance();
    cal.setTime(now);
    cal.add(Calendar.MINUTE, -30);
    Date start = cal.getTime();
    cal.add(Calendar.HOUR, 1);
    Date end = cal.getTime();
    assertDateWithin(testDate, start, end);
  }

  @Before public void setup() throws IOException {
    test = tmp.newFile("test.zip");
  }

  @Test public void testMalformed_Empty() throws IOException {
    try (FileOutputStream out = new FileOutputStream(test)) {
    }
    thrown.expect(ZipException.class);
    thrown.expectMessage("is malformed. It does not contain an end of central directory record.");
    new ZipReader(test, UTF_8).close();
  }

  @Test public void testMalformed_ShorterThanSignature() throws IOException {
    try (FileOutputStream out = new FileOutputStream(test)) {
      out.write(new byte[] { 1, 2, 3 });
    }
    thrown.expect(ZipException.class);
    thrown.expectMessage("is malformed. It does not contain an end of central directory record.");
    new ZipReader(test, UTF_8).close();
  }

  @Test public void testMalformed_SignatureLength() throws IOException {
    try (FileOutputStream out = new FileOutputStream(test)) {
      out.write(new byte[] { 1, 2, 3, 4 });
    }
    thrown.expect(ZipException.class);
    thrown.expectMessage("is malformed. It does not contain an end of central directory record.");
    new ZipReader(test, UTF_8).close();
  }

  @Test public void testEmpty() throws IOException {
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.entries()).isEmpty();
    }
  }

  @Test public void testFileComment() throws IOException {
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      zout.setComment("test comment");
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.entries()).isEmpty();
      assertThat(reader.getComment()).isEqualTo("test comment");
    }
  }

  @Test public void testFileCommentWithSignature() throws IOException {
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      zout.setComment("test comment\u0050\u004b\u0005\u0006abcdefghijklmnopqrstuvwxyz");
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.entries()).isEmpty();
      assertThat(reader.getComment())
          .isEqualTo("test comment\u0050\u004b\u0005\u0006abcdefghijklmnopqrstuvwxyz");
    }
  }

  @Test public void testSingleEntry() throws IOException {
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      zout.putNextEntry(new ZipEntry("test"));
      zout.write("foo".getBytes(UTF_8));
      zout.closeEntry();
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.entries()).hasSize(1);
    }
  }

  @Test public void testMultipleEntries() throws IOException {
    String[] names = new String[] { "test", "foo", "bar", "baz" };
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      for (String name : names) {
        zout.putNextEntry(new ZipEntry(name));
        zout.write(name.getBytes(UTF_8));
        zout.closeEntry();
      }
    }

    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.entries()).hasSize(names.length);
      int i = 0;
      for (ZipFileEntry entry : reader.entries()) {
        assertThat(entry.getName()).isEqualTo(names[i++]);
      }
      assertThat(i).isEqualTo(names.length);
    }
  }

  @Test public void testZipEntryFields() throws IOException {
    CRC32 crc = new CRC32();
    Deflater deflater = new Deflater(Deflater.DEFAULT_COMPRESSION, true);
    long date = 791784306000L; // 2/3/1995 04:05:06
    byte[] extra = new ExtraData((short) 0xaa, new byte[] { (byte) 0xbb, (byte) 0xcd }).getBytes();
    byte[] tmp = new byte[128];
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {

      ZipEntry foo = new ZipEntry("foo");
      foo.setComment("foo comment.");
      foo.setMethod(ZipEntry.DEFLATED);
      foo.setTime(date);
      foo.setExtra(extra);
      zout.putNextEntry(foo);
      zout.write("foo".getBytes(UTF_8));
      zout.closeEntry();

      ZipEntry bar = new ZipEntry("bar");
      bar.setComment("bar comment.");
      bar.setMethod(ZipEntry.STORED);
      bar.setSize("bar".length());
      bar.setCompressedSize("bar".length());
      crc.reset();
      crc.update("bar".getBytes(UTF_8));
      bar.setCrc(crc.getValue());
      zout.putNextEntry(bar);
      zout.write("bar".getBytes(UTF_8));
      zout.closeEntry();
    }

    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      ZipFileEntry fooEntry = reader.getEntry("foo");
      assertThat(fooEntry.getName()).isEqualTo("foo");
      assertThat(fooEntry.getComment()).isEqualTo("foo comment.");
      assertThat(fooEntry.getMethod()).isEqualTo(Compression.DEFLATED);
      assertThat(fooEntry.getVersion()).isEqualTo(Compression.DEFLATED.getMinVersion());
      assertThat(fooEntry.getTime()).isEqualTo(date);
      assertThat(fooEntry.getSize()).isEqualTo("foo".length());
      deflater.reset();
      deflater.setInput("foo".getBytes(UTF_8));
      deflater.finish();
      assertThat(fooEntry.getCompressedSize()).isEqualTo(deflater.deflate(tmp));
      crc.reset();
      crc.update("foo".getBytes(UTF_8));
      assertThat(fooEntry.getCrc()).isEqualTo(crc.getValue());
      assertThat(fooEntry.getExtra().getBytes()).isEqualTo(extra);

      ZipFileEntry barEntry = reader.getEntry("bar");
      assertThat(barEntry.getName()).isEqualTo("bar");
      assertThat(barEntry.getComment()).isEqualTo("bar comment.");
      assertThat(barEntry.getMethod()).isEqualTo(Compression.STORED);
      assertThat(barEntry.getVersion()).isEqualTo(Compression.STORED.getMinVersion());
      assertDateAboutNow(new Date(barEntry.getTime()));
      assertThat(barEntry.getSize()).isEqualTo("bar".length());
      assertThat(barEntry.getCompressedSize()).isEqualTo("bar".length());
      crc.reset();
      crc.update("bar".getBytes(UTF_8));
      assertThat(barEntry.getCrc()).isEqualTo(crc.getValue());
      assertThat(barEntry.getExtra().getBytes()).isEqualTo(new byte[] {});
    }
  }

  @Test public void testZipEntryInvalidTime() throws IOException {
    long date = 312796800000L; // 11/30/1979 00:00:00, which is also 0 in DOS format
    byte[] extra = new ExtraData((short) 0xaa, new byte[] { (byte) 0xbb, (byte) 0xcd }).getBytes();
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      ZipEntry foo = new ZipEntry("foo");
      foo.setComment("foo comment.");
      foo.setMethod(ZipEntry.DEFLATED);
      foo.setTime(date);
      foo.setExtra(extra);
      zout.putNextEntry(foo);
      zout.write("foo".getBytes(UTF_8));
      zout.closeEntry();
    }

    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      ZipFileEntry fooEntry = reader.getEntry("foo");
      assertThat(fooEntry.getTime()).isEqualTo(ZipUtil.DOS_EPOCH);
    }
  }

  @Test public void testRawFileData() throws IOException {
    CRC32 crc = new CRC32();
    Deflater deflator = new Deflater(Deflater.DEFAULT_COMPRESSION, true);
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      ZipEntry foo = new ZipEntry("foo");
      foo.setComment("foo comment.");
      foo.setMethod(ZipEntry.DEFLATED);
      zout.putNextEntry(foo);
      zout.write("foo".getBytes(UTF_8));
      zout.closeEntry();

      ZipEntry bar = new ZipEntry("bar");
      bar.setComment("bar comment.");
      bar.setMethod(ZipEntry.STORED);
      bar.setSize("bar".length());
      bar.setCompressedSize("bar".length());
      crc.reset();
      crc.update("bar".getBytes(UTF_8));
      bar.setCrc(crc.getValue());
      zout.putNextEntry(bar);
      zout.write("bar".getBytes(UTF_8));
      zout.closeEntry();
    }

    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      ZipFileEntry fooEntry = reader.getEntry("foo");
      InputStream fooIn = reader.getRawInputStream(fooEntry);
      byte[] fooData = new byte[10];
      fooIn.read(fooData);
      byte[] expectedFooData = new byte[10];
      deflator.reset();
      deflator.setInput("foo".getBytes(UTF_8));
      deflator.finish();
      deflator.deflate(expectedFooData);
      assertThat(fooData).isEqualTo(expectedFooData);

      ZipFileEntry barEntry = reader.getEntry("bar");
      InputStream barIn = reader.getRawInputStream(barEntry);
      byte[] barData = new byte[3];
      barIn.read(barData);
      byte[] expectedBarData = "bar".getBytes(UTF_8);
      assertThat(barData).isEqualTo(expectedBarData);

      assertThat(barIn.read()).isEqualTo(-1);
      assertThat(barIn.read(barData)).isEqualTo(-1);
      assertThat(barIn.read(barData, 0, 3)).isEqualTo(-1);

      thrown.expect(IOException.class);
      thrown.expectMessage("Reset is not supported on this type of stream.");
      barIn.reset();
    }
  }

  @Test public void testFileData() throws IOException {
    CRC32 crc = new CRC32();
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      ZipEntry foo = new ZipEntry("foo");
      foo.setComment("foo comment.");
      foo.setMethod(ZipEntry.DEFLATED);
      zout.putNextEntry(foo);
      zout.write("foo".getBytes(UTF_8));
      zout.closeEntry();

      ZipEntry bar = new ZipEntry("bar");
      bar.setComment("bar comment.");
      bar.setMethod(ZipEntry.STORED);
      bar.setSize("bar".length());
      bar.setCompressedSize("bar".length());
      crc.reset();
      crc.update("bar".getBytes(UTF_8));
      bar.setCrc(crc.getValue());
      zout.putNextEntry(bar);
      zout.write("bar".getBytes(UTF_8));
      zout.closeEntry();
    }

    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      ZipFileEntry fooEntry = reader.getEntry("foo");
      InputStream fooIn = reader.getInputStream(fooEntry);
      byte[] fooData = new byte[3];
      fooIn.read(fooData);
      byte[] expectedFooData = "foo".getBytes(UTF_8);
      assertThat(fooData).isEqualTo(expectedFooData);

      assertThat(fooIn.read()).isEqualTo(-1);
      assertThat(fooIn.read(fooData)).isEqualTo(-1);
      assertThat(fooIn.read(fooData, 0, 3)).isEqualTo(-1);

      ZipFileEntry barEntry = reader.getEntry("bar");
      InputStream barIn = reader.getInputStream(barEntry);
      byte[] barData = new byte[3];
      barIn.read(barData);
      byte[] expectedBarData = "bar".getBytes(UTF_8);
      assertThat(barData).isEqualTo(expectedBarData);

      assertThat(barIn.read()).isEqualTo(-1);
      assertThat(barIn.read(barData)).isEqualTo(-1);
      assertThat(barIn.read(barData, 0, 3)).isEqualTo(-1);

      thrown.expect(IOException.class);
      thrown.expectMessage("Reset is not supported on this type of stream.");
      barIn.reset();
    }
  }

  @Test public void testSimultaneousReads() throws IOException {
    byte[] expectedFooData = "This is file foo. It contains a foo.".getBytes(UTF_8);
    byte[] expectedBarData = "This is a different file bar. It contains only a bar."
        .getBytes(UTF_8);
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      ZipEntry foo = new ZipEntry("foo");
      foo.setComment("foo comment.");
      foo.setMethod(ZipEntry.DEFLATED);
      zout.putNextEntry(foo);
      zout.write(expectedFooData);
      zout.closeEntry();

      ZipEntry bar = new ZipEntry("bar");
      bar.setComment("bar comment.");
      bar.setMethod(ZipEntry.DEFLATED);
      zout.putNextEntry(bar);
      zout.write(expectedBarData);
      zout.closeEntry();
    }

    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      ZipFileEntry fooEntry = reader.getEntry("foo");
      ZipFileEntry barEntry = reader.getEntry("bar");
      InputStream fooIn = reader.getInputStream(fooEntry);
      InputStream barIn = reader.getInputStream(barEntry);
      byte[] fooData = new byte[expectedFooData.length];
      byte[] barData = new byte[expectedBarData.length];
      fooIn.read(fooData, 0, 10);
      barIn.read(barData, 0, 10);
      fooIn.read(fooData, 10, 10);
      barIn.read(barData, 10, 10);
      fooIn.read(fooData, 20, fooData.length - 20);
      barIn.read(barData, 20, barData.length - 20);
      assertThat(fooData).isEqualTo(expectedFooData);
      assertThat(barData).isEqualTo(expectedBarData);
    }
  }

  @Test public void testSlowRead() throws IOException {
    byte[] expectedFooData = "This is file foo. It contains a foo.".getBytes(UTF_8);
    byte[] expectedBarData = "This is a different file bar. It contains only a bar."
        .getBytes(UTF_8);
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      ZipEntry foo = new ZipEntry("foo");
      foo.setComment("foo comment.");
      foo.setMethod(ZipEntry.DEFLATED);
      zout.putNextEntry(foo);
      zout.write(expectedFooData);
      zout.closeEntry();

      ZipEntry bar = new ZipEntry("bar");
      bar.setComment("bar comment.");
      bar.setMethod(ZipEntry.DEFLATED);
      zout.putNextEntry(bar);
      zout.write(expectedBarData);
      zout.closeEntry();
    }

    try (ZipReader reader = new SlowZipReader(test, UTF_8)) {
      ZipFileEntry fooEntry = reader.getEntry("foo");
      ZipFileEntry barEntry = reader.getEntry("bar");
      InputStream fooIn = reader.getInputStream(fooEntry);
      InputStream barIn = reader.getInputStream(barEntry);
      byte[] fooData = new byte[expectedFooData.length];
      byte[] barData = new byte[expectedBarData.length];
      ZipUtil.readFully(fooIn, fooData, 0, 10);
      ZipUtil.readFully(barIn, barData, 0, 10);
      ZipUtil.readFully(fooIn, fooData, 10, 10);
      ZipUtil.readFully(barIn, barData, 10, 10);
      ZipUtil.readFully(fooIn, fooData, 20, fooData.length - 20);
      ZipUtil.readFully(barIn, barData, 20, barData.length - 20);
      assertThat(fooData).isEqualTo(expectedFooData);
      assertThat(barData).isEqualTo(expectedBarData);
    }
  }

  @Test public void testZip64() throws IOException {
    // Generated with: 'echo "foo" > entry; zip -fz -q out.zip entry'
    byte[] data = new byte[] {
        0x50, 0x4b, 0x03, 0x04, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5d, (byte) 0x86, (byte) 0xa6,
        0x46, (byte) 0xa8, 0x65, 0x32, 0x7e, (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff,
        (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff, 0x05, 0x00, 0x30, 0x00, 0x65, 0x6e,
        0x74, 0x72, 0x79, 0x55, 0x54, 0x09, 0x00, 0x03, (byte) 0xb2, 0x7e, 0x4a, 0x55, (byte) 0xb2,
        0x7e, 0x4a, 0x55, 0x75, 0x78, 0x0b, 0x00, 0x01, 0x04, 0x46, 0x3a, 0x04, 0x00, 0x04,
        (byte) 0x88, 0x13, 0x00, 0x00, 0x01, 0x00, 0x10, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x66, 0x6f, 0x6f, 0x0a, 0x50,
        0x4b, 0x01, 0x02, 0x1e, 0x03, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5d, (byte) 0x86,
        (byte) 0xa6, 0x46, (byte) 0xa8, 0x65, 0x32, 0x7e, 0x04, 0x00, 0x00, 0x00, (byte) 0xff,
        (byte) 0xff, (byte) 0xff, (byte) 0xff, 0x05, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, (byte) 0xa0, (byte) 0x81, 0x00, 0x00, 0x00, 0x00, 0x65, 0x6e, 0x74, 0x72,
        0x79, 0x55, 0x54, 0x05, 0x00, 0x03, (byte) 0xb2, 0x7e, 0x4a, 0x55, 0x75, 0x78, 0x0b, 0x00,
        0x01, 0x04, 0x46, 0x3a, 0x04, 0x00, 0x04, (byte) 0x88, 0x13, 0x00, 0x00, 0x01, 0x00, 0x08,
        0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x4b, 0x06, 0x06, 0x2c, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x03, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x57, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x57, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x4b, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, (byte) 0xae,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x50, 0x4b, 0x05, 0x06,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x57, 0x00, 0x00, 0x00, (byte) 0xff,
        (byte) 0xff, (byte) 0xff, (byte) 0xff, 0x00, 0x00
      };

    try (FileOutputStream out = new FileOutputStream(test)) {
      out.write(data);
    }
    String foo = "foo\n";
    byte[] expectedFooData = foo.getBytes(UTF_8);
    ExtraDataList extras = new ExtraDataList();
    extras.add(new ExtraData((short) 0x0001, ZipUtil.longToLittleEndian(expectedFooData.length)));
    byte[] extra = extras.getBytes();
    CRC32 crc = new CRC32();
    crc.reset();
    crc.update(expectedFooData);
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      ZipFileEntry fooEntry = reader.getEntry("entry");
      InputStream fooIn = reader.getInputStream(fooEntry);
      byte[] fooData = new byte[expectedFooData.length];
      fooIn.read(fooData);
      assertThat(fooData).isEqualTo(expectedFooData);
      assertThat(fooEntry.getName()).isEqualTo("entry");
      assertThat(fooEntry.getComment()).isEqualTo("");
      assertThat(fooEntry.getMethod()).isEqualTo(Compression.STORED);
      assertThat(fooEntry.getVersionNeeded()).isEqualTo(Feature.ZIP64_SIZE.getMinVersion());
      assertThat(fooEntry.getSize()).isEqualTo(expectedFooData.length);
      assertThat(fooEntry.getCompressedSize()).isEqualTo(expectedFooData.length);
      assertThat(fooEntry.getCrc()).isEqualTo(crc.getValue());
      assertThat(fooEntry.getExtra().get((short) 0x0001).getBytes()).isEqualTo(extra);
    }
  }

  @Test public void testZip64_Potential() throws IOException {
    try (ZipWriter writer = new ZipWriter(new FileOutputStream(test), UTF_8, true)) {
      ZipFileEntry template = new ZipFileEntry("template");
      template.setSize(0);
      template.setCompressedSize(0);
      template.setCrc(0);
      template.setTime(ZipUtil.DOS_EPOCH);
      for (int i = 0; i < 0xffff; i++) {
        ZipFileEntry entry = new ZipFileEntry(template);
        entry.setName("entry" + i);
        writer.putNextEntry(entry);
      }
    }
    try (ZipReader reader = new ZipReader(test, UTF_8, true)) {
      Collection<ZipFileEntry> entries = reader.entries();
      assertThat(entries).hasSize(0xffff);
    }
  }

  @Test public void testZip64_NumFiles() throws IOException {
    try (ZipWriter writer = new ZipWriter(new FileOutputStream(test), UTF_8, true)) {
      ZipFileEntry template = new ZipFileEntry("template");
      template.setSize(0);
      template.setCompressedSize(0);
      template.setCrc(0);
      template.setTime(ZipUtil.DOS_EPOCH);
      for (int i = 0; i < 0x100ff; i++) {
        ZipFileEntry entry = new ZipFileEntry(template);
        entry.setName("entry" + i);
        writer.putNextEntry(entry);
      }
    }
    try (ZipReader reader = new ZipReader(test, UTF_8, true)) {
      Collection<ZipFileEntry> entries = reader.entries();
      assertThat(entries).hasSize(0x100ff);
    }
  }

  @Test public void testZip64_Max32BitSizeFile() throws IOException {
    File bigFile = tmp.newFile("big");
    try (RandomAccessFile bigOut = new RandomAccessFile(bigFile, "rw")) {
      bigOut.setLength(0xffffffffL);
    }
    try (ZipWriter writer = new ZipWriter(new FileOutputStream(test), UTF_8, true)) {
      ZipFileEntry bigEntry = new ZipFileEntry(bigFile.getName());
      bigEntry.setSize(0xffffffffL);
      bigEntry.setCompressedSize(0xffffffffL);
      bigEntry.setCrc(0);
      bigEntry.setTime(ZipUtil.DOS_EPOCH);
      writer.putNextEntry(bigEntry);
      ByteStreams.copy(new BufferedInputStream(new FileInputStream(bigFile)), writer);
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      Collection<ZipFileEntry> entries = reader.entries();
      assertThat(entries).hasSize(1);
      ZipFileEntry bigEntry = reader.getEntry(bigFile.getName());
      assertThat(bigEntry.getSize()).isEqualTo(0xffffffffL);
    }
  }

  @Test public void testZip64_Zip64SizeFile() throws IOException {
    File biggerFile = tmp.newFile("big");
    try (RandomAccessFile biggerOut = new RandomAccessFile(biggerFile, "rw")) {
      biggerOut.setLength(0x1000000ffL);
    }
    try (ZipWriter writer = new ZipWriter(new FileOutputStream(test), UTF_8, true)) {
      ZipFileEntry bigEntry = new ZipFileEntry(biggerFile.getName());
      bigEntry.setSize(0x1000000ffL);
      bigEntry.setCompressedSize(0x1000000ffL);
      bigEntry.setCrc(0);
      bigEntry.setTime(ZipUtil.DOS_EPOCH);
      writer.putNextEntry(bigEntry);
      ByteStreams.copy(new BufferedInputStream(new FileInputStream(biggerFile)), writer);
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      Collection<ZipFileEntry> entries = reader.entries();
      assertThat(entries).hasSize(1);
      ZipFileEntry bigEntry = reader.getEntry(biggerFile.getName());
      assertThat(bigEntry.getSize()).isEqualTo(0x1000000ffL);
    }
  }

  @Test public void testZip64_FileCount_Zip64Range_ForceZip32() throws IOException {
    try (ZipWriter writer = new ZipWriter(new FileOutputStream(test), UTF_8, false)) {
      ZipFileEntry template = new ZipFileEntry("template");
      template.setSize(0);
      template.setCompressedSize(0);
      template.setCrc(0);
      template.setTime(ZipUtil.DOS_EPOCH);
      for (int i = 0; i < 0x100ff; i++) {
        ZipFileEntry entry = new ZipFileEntry(template);
        entry.setName("entry" + i);
        writer.putNextEntry(entry);
      }
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.size()).isEqualTo(0x100ff);
    }
    try (ZipReader reader = new ZipReader(test, UTF_8, true)) {
      assertThat(reader.size()).isEqualTo(0x00ff);
    }
  }
}
