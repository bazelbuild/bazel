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

import com.google.common.testing.NullPointerTester;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.zip.ZipException;

@RunWith(JUnit4.class)
public class ZipFileDataTest {
  @Rule public ExpectedException thrown = ExpectedException.none();

  private ZipFileData data;

  @Before public void setup() {
    data = new ZipFileData(UTF_8);
  }

  @Test public void testNulls() {
    NullPointerTester tester = new NullPointerTester();
    tester.testAllPublicConstructors(ZipFileData.class);
    tester.testAllPublicInstanceMethods(data);
  }

  @Test public void testCharset() {
    assertThat(new ZipFileData(UTF_8).getCharset()).isEqualTo(UTF_8);
  }

  @Test public void testComment() throws ZipException {
    data.setComment("foo");
    assertThat(data.getComment()).isEqualTo("foo");
  }

  @Test public void testSetComment_TooLong() throws ZipException {
    String comment = new String(new byte[0x100ff], UTF_8);
    thrown.expect(ZipException.class);
    thrown.expectMessage("File comment too long. Is 65791; max 65535.");
    data.setComment(comment);
  }

  @Test public void testSetComment_FromBytes() throws ZipException {
    String comment = "foo";
    byte[] bytes = comment.getBytes(UTF_8);
    data.setComment(bytes);
    assertThat(data.getComment()).isEqualTo(comment);
  }

  @Test public void testSetComment_FromBytes_TooLong() throws ZipException {
    byte[] comment = new byte[0x100ff];
    thrown.expect(ZipException.class);
    thrown.expectMessage("File comment too long. Is 65791; max 65535.");
    data.setComment(comment);
  }

  @Test public void testZip64Setting() {
    assertThat(data.isZip64()).isFalse();
    data.setCentralDirectorySize(0xffffffffL);
    assertThat(data.isZip64()).isFalse();
    data.setCentralDirectorySize(0x100000000L);
    assertThat(data.isZip64()).isTrue();

    data.setZip64(false);
    assertThat(data.isZip64()).isFalse();
    data.setCentralDirectoryOffset(0xffffffffL);
    assertThat(data.isZip64()).isFalse();
    data.setCentralDirectoryOffset(0x100000000L);
    assertThat(data.isZip64()).isTrue();

    data.setZip64(false);
    assertThat(data.isZip64()).isFalse();
    data.setExpectedEntries(0xffff);
    assertThat(data.isZip64()).isFalse();
    data.setExpectedEntries(0x10000L);
    assertThat(data.isZip64()).isTrue();

    data.setZip64(false);
    assertThat(data.isZip64()).isFalse();
    data.setZip64EndOfCentralDirectoryOffset(0);
    assertThat(data.isZip64()).isTrue();

    data.setZip64(false);
    assertThat(data.isZip64()).isFalse();
    ZipFileEntry template = new ZipFileEntry("template");
    for (int i = 0; i < 0xffff; i++) {
      ZipFileEntry entry = new ZipFileEntry(template);
      entry.setName("entry" + i);
      data.addEntry(entry);
    }
    assertThat(data.isZip64()).isFalse();
    data.addEntry(template);
    assertThat(data.isZip64()).isTrue();
  }

  @Test public void testSetZip64SetsMaybeZip64() {
    assertThat(data.isMaybeZip64()).isFalse();
    data.setZip64(true);
    assertThat(data.isMaybeZip64()).isTrue();
  }
}
