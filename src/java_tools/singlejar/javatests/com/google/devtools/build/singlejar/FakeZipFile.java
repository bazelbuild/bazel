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

package com.google.devtools.build.singlejar;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * A fake zip file to assert that a given {@link ZipInputStream} contains
 * specified entries in a specified order. Just for unit testing.
 */
public final class FakeZipFile {

  private enum EntryMode {
    DONT_CARE,
    EXPECT_DEFLATE,
    EXPECT_STORED;
  }

  /**
   * Validates an input provided as a byte array.
   */
  public static interface ByteValidator {
    /**
     * Check if {@code object} is the expected input. If {@code object} does not match the expected
     * pattern, an assertion should fails with the necessary message.
     */
    void validate(byte[] object);
  }

  private static void assertSameByteArray(byte[] expected, byte[] actual) {
    if (expected == null) {
      assertThat(actual).isNull();
    } else {
      assertThat(actual).isEqualTo(expected);
    }
  }

  private static byte[] readZipEntryContent(ZipInputStream zipInput) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buffer = new byte[1024];
    int bytesCopied;
    while ((bytesCopied = zipInput.read(buffer)) != -1) {
      out.write(buffer, 0, bytesCopied);
    }
    return out.toByteArray();
  }

  private static final class PlainByteValidator implements ByteValidator {
    private final byte[] expected;

    private PlainByteValidator(String expected) {
      this.expected = expected == null ? new byte[0] : expected.getBytes(UTF_8);
    }

    @Override
    public void validate(byte[] object) {
      assertSameByteArray(expected, object);
    }

  }

  private static final class FakeZipEntry {

    private final String name;
    private final ByteValidator content;
    private final Date date;
    private final byte[] extra;
    private final EntryMode mode;

    private FakeZipEntry(String name, Date date, String content, byte[] extra, EntryMode mode) {
      this.name = name;
      this.date = date;
      this.content = new PlainByteValidator(content);
      this.extra = extra;
      this.mode = mode;
    }

    private FakeZipEntry(String name, Date date, ByteValidator content, byte[] extra,
        EntryMode mode) {
      this.name = name;
      this.date = date;
      this.content = content;
      this.extra = extra;
      this.mode = mode;
    }

    public void assertNext(ZipInputStream zipInput) throws IOException {
      ZipEntry zipEntry = zipInput.getNextEntry();
      assertThat(zipEntry).isNotNull();
      switch (mode) {
        case EXPECT_DEFLATE:
          assertThat(zipEntry.getMethod()).isEqualTo(ZipEntry.DEFLATED);
          break;
        case EXPECT_STORED:
          assertThat(zipEntry.getMethod()).isEqualTo(ZipEntry.STORED);
          break;
        default:
          // we don't care.
          break;
      }
      assertThat(zipEntry.getName()).isEqualTo(name);
      if (date != null) {
        assertThat(zipEntry.getTime()).isEqualTo(date.getTime());
      }
      assertSameByteArray(extra, zipEntry.getExtra());
      content.validate(readZipEntryContent(zipInput));
    }
  }

  private final List<FakeZipEntry> entries = new ArrayList<>();

  @CanIgnoreReturnValue
  public FakeZipFile addEntry(String name, String content, boolean compressed) {
    entries.add(new FakeZipEntry(name, null, content, null,
        compressed ? EntryMode.EXPECT_DEFLATE : EntryMode.EXPECT_STORED));
    return this;
  }

  private byte[] preamble = null;

  private int getUnsignedShort(byte[] source, int offset) {
    int a = source[offset + 0] & 0xff;
    int b = source[offset + 1] & 0xff;
    return (b << 8) | a;
  }

  public void assertSame(byte[] data) throws IOException {
    int offset = 0;
    int length = data.length;
    if (preamble != null) {
      offset += preamble.length;
      length -= offset;
      byte[] maybePreamble = Arrays.copyOfRange(data, 0, offset);
      assertThat(maybePreamble).isEqualTo(preamble);
    }
    ZipInputStream zipInput = new ZipInputStream(new ByteArrayInputStream(data, offset, length));
    for (FakeZipEntry entry : entries) {
      entry.assertNext(zipInput);
    }
    assertThat(zipInput.getNextEntry()).isNull();
    // Verify that the end of central directory data is correct.
    // This assumes that the end of directory is at the end of input and that there is no zip file
    // comment.
    int count = getUnsignedShort(data, data.length-14);
    assertThat(count).isEqualTo(entries.size());
    count = getUnsignedShort(data, data.length-12);
    assertThat(count).isEqualTo(entries.size());
  }
}
