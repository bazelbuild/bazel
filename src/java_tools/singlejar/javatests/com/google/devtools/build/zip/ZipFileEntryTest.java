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

import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.zip.ZipFileEntry.Compression;
import com.google.devtools.build.zip.ZipFileEntry.Feature;
import com.google.devtools.build.zip.ZipFileEntry.Flag;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ZipFileEntryTest {
  @Rule public ExpectedException thrown = ExpectedException.none();

  @Test public void testNulls() {
    NullPointerTester tester = new NullPointerTester();
    tester.testAllPublicConstructors(ZipFileEntry.class);
    tester.testAllPublicInstanceMethods(new ZipFileEntry("foo"));
  }

  @Test public void testCrc() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setCrc(32);
  }

  @Test public void testCrc_Negative() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("invalid entry crc-32");
    foo.setCrc(-1);
  }

  @Test public void testCrc_Large() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("invalid entry crc-32");
    foo.setCrc(0x100000000L);
  }

  @Test public void testSize() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setSize(32);
  }

  @Test public void testSize_Negative() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("invalid entry size");
    foo.setSize(-1);
  }

  @Test public void testSize_Large() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setSize(0x100000000L);
    assertThat(foo.getVersion()).isEqualTo(Feature.ZIP64_SIZE.getMinVersion());
  }

  @Test public void testCompressedSize() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setCompressedSize(32);
  }

  @Test public void testCompressedSize_Negative() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("invalid entry size");
    foo.setCompressedSize(-1);
  }

  @Test public void testCompressedSize_Large() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setCompressedSize(0x100000000L);
    assertThat(foo.getVersion()).isEqualTo(Feature.ZIP64_CSIZE.getMinVersion());
  }

  @Test public void testMinVersion() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    assertThat(foo.getVersion()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setVersion((short) 0x14);
    assertThat(foo.getVersion()).isEqualTo((short) 0x14);
  }

  @Test public void testMinVersion_MethodUpdated() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    assertThat(foo.getVersion()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setMethod(Compression.DEFLATED);
    assertThat(foo.getVersion()).isEqualTo(Feature.DEFLATED.getMinVersion());
  }

  @Test public void testMinVersion_Zip64Updated() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    assertThat(foo.getVersion()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setSize(0xfffffffeL);
    assertThat(foo.getVersion()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setSize(0xffffffffL);
    assertThat(foo.getVersion()).isEqualTo(Feature.ZIP64_SIZE.getMinVersion());
  }

  @Test public void testMinVersion_BelowRequired() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setVersion((short) 0);
    assertThat(foo.getVersion()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setMethod(Compression.DEFLATED);
    foo.setVersion(Compression.STORED.getMinVersion());
    assertThat(foo.getVersion()).isEqualTo(Feature.DEFLATED.getMinVersion());
  }

  @Test public void testMinVersionNeeded() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setVersionNeeded((short) 0x14);
    assertThat(foo.getVersionNeeded()).isEqualTo((short) 0x14);
  }

  @Test public void testMinVersionNeeded_MethodUpdated() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setMethod(Compression.DEFLATED);
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.DEFLATED.getMinVersion());
  }

  @Test public void testMinVersionNeeded_Zip64Updated() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setSize(0xfffffffeL);
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setSize(0xffffffffL);
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.ZIP64_SIZE.getMinVersion());
  }

  @Test public void testMinVersionNeeded_BelowRequired() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setVersionNeeded((short) 0);
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.DEFAULT.getMinVersion());
    foo.setMethod(Compression.DEFLATED);
    foo.setVersionNeeded(Compression.STORED.getMinVersion());
    assertThat(foo.getVersionNeeded()).isEqualTo(Feature.DEFLATED.getMinVersion());
  }

  @Test public void testSetFlag() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setFlag(Flag.DATA_DESCRIPTOR, true);
    assertThat(foo.getFlags()).isEqualTo((short) 0x08);
    foo.setFlag(Flag.DATA_DESCRIPTOR, true);
    assertThat(foo.getFlags()).isEqualTo((short) 0x08);
    foo.setFlag(Flag.DATA_DESCRIPTOR, false);
    assertThat(foo.getFlags()).isEqualTo((short) 0x00);
    foo.setFlag(Flag.DATA_DESCRIPTOR, false);
    assertThat(foo.getFlags()).isEqualTo((short) 0x00);
  }

  @Test public void testLocalHeaderOffset() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setLocalHeaderOffset(32);
  }

  @Test public void testLocalHeaderOffset_Negative() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("invalid local header offset");
    foo.setLocalHeaderOffset(-1);
  }

  @Test public void testLocalHeaderOffset_Large() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setLocalHeaderOffset(0x100000000L);
    assertThat(foo.getVersion()).isEqualTo(Feature.ZIP64_OFFSET.getMinVersion());
  }

  @Test public void testExtra() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    foo.setExtra(new ExtraDataList(new byte[32]));
  }

  @Test public void testExtra_Large() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("invalid extra field length");
    foo.setExtra(new ExtraDataList(new byte[0x10000]));
  }

  @Test public void testExtraData() {
    ZipFileEntry foo = new ZipFileEntry("foo");
    ExtraDataList extra = new ExtraDataList();
    extra.add(new ExtraData((short) 0xCAFE, new byte[] { 0x01, 0x02 }));
    extra.add(new ExtraData((short) 0xDEAD, new byte[] { (byte) 0xBE, (byte) 0xEF }));
    foo.setExtra(extra);
    // Expect 2 records: 0xCAFE 0x0002 0x01 0x02, 0xDEAD 0x0002 0xBE 0xEF in little endian
    assertThat(foo.getExtra().getBytes()).isEqualTo(new byte[] {
        (byte) 0xFE, (byte) 0xCA, 0x02, 0x00, 0x01, 0x02,
        (byte) 0xAD, (byte) 0xDE, 0x02, 0x00, (byte) 0xBE, (byte) 0xEF });
  }
}
