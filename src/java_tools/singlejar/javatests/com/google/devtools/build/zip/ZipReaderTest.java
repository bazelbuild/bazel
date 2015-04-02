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

package com.google.devtools.build.zip;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import com.google.devtools.build.zip.ZipFileEntry.Compression;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Calendar;
import java.util.Date;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipOutputStream;

@RunWith(JUnit4.class)
public class ZipReaderTest {
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

  @Rule public TemporaryFolder tmp = new TemporaryFolder();
  @Rule public ExpectedException thrown = ExpectedException.none();

  @Test public void testMalformed_Empty() throws IOException {
    File test = tmp.newFile("test.zip");
    try (FileOutputStream out = new FileOutputStream(test)) {
    }
    thrown.expect(ZipException.class);
    thrown.expectMessage("is malformed. It does not contain an end of central directory record.");
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
    }
  }

  @Test public void testMalformed_ShorterThanSignature() throws IOException {
    File test = tmp.newFile("test.zip");
    try (FileOutputStream out = new FileOutputStream(test)) {
      out.write(new byte[] { 1, 2, 3 });
    }
    thrown.expect(ZipException.class);
    thrown.expectMessage("is malformed. It does not contain an end of central directory record.");
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
    }
  }

  @Test public void testMalformed_SignatureLength() throws IOException {
    File test = tmp.newFile("test.zip");
    try (FileOutputStream out = new FileOutputStream(test)) {
      out.write(new byte[] { 1, 2, 3, 4 });
    }
    thrown.expect(ZipException.class);
    thrown.expectMessage("is malformed. It does not contain an end of central directory record.");
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
    }
  }

  @Test public void testEmpty() throws IOException {
    File test = tmp.newFile("test.zip");
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.entries()).isEmpty();
    }
  }

  @Test public void testFileComment() throws IOException {
    File test = tmp.newFile("test.zip");
    try (ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(test))) {
      zout.setComment("test comment");
    }
    try (ZipReader reader = new ZipReader(test, UTF_8)) {
      assertThat(reader.entries()).isEmpty();
      assertThat(reader.getComment()).isEqualTo("test comment");
    }
  }

  @Test public void testFileCommentWithSignature() throws IOException {
    File test = tmp.newFile("test.zip");
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
    File test = tmp.newFile("test.zip");
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
    File test = tmp.newFile("test.zip");
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
    File test = tmp.newFile("test.zip");
    CRC32 crc = new CRC32();
    Deflater deflater = new Deflater(Deflater.DEFAULT_COMPRESSION, true);
    long date = 791784306000L; // 2/3/1995 04:05:06
    byte[] extra = new byte[] { (byte) 0xaa, (byte) 0xbb, (byte) 0xcd };
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
      assertThat(fooEntry.getExtra()).isEqualTo(extra);

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
      assertThat(barEntry.getExtra()).isEqualTo(new byte[] {});
    }
  }

  @Test public void testZipEntryInvalidTime() throws IOException {
    File test = tmp.newFile("test.zip");
    long date = 312796800000L; // 11/30/1979 00:00:00, which is also 0 in DOS format
    byte[] extra = new byte[] { (byte) 0xaa, (byte) 0xbb, (byte) 0xcd };
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
    File test = tmp.newFile("test.zip");
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
    File test = tmp.newFile("test.zip");
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
    File test = tmp.newFile("test.zip");
    byte[] expectedFooData = "This if file foo. It contains a foo.".getBytes(UTF_8);
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
}
