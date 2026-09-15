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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * A helper class to create zip files for testing.
 */
public class ZipFactory {

  static class Entry {
    private final String name;
    private final byte[] content;
    private final boolean compressed;
    private Entry(String name, byte[] content, boolean compressed) {
      this.name = name;
      this.content = content;
      this.compressed = compressed;
    }
  }

  private final List<Entry> entries = new ArrayList<>();

  // Assumes that content was created locally. Does not perform a defensive copy!
  private void addEntry(String name, byte[] content, boolean compressed) {
    entries.add(new Entry(name, content, compressed));
  }

  @CanIgnoreReturnValue
  public ZipFactory addFile(String name, String content) {
    addEntry(name, content.getBytes(ISO_8859_1), true);
    return this;
  }

  @CanIgnoreReturnValue
  public ZipFactory addFile(String name, byte[] content) {
    addEntry(name, content.clone(), true);
    return this;
  }

  @CanIgnoreReturnValue
  public ZipFactory addFile(String name, String content, boolean compressed) {
    addEntry(name, content.getBytes(ISO_8859_1), compressed);
    return this;
  }

  public byte[] toByteArray() {
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      ZipOutputStream zipper = new ZipOutputStream(out);
      for (Entry entry : entries) {
        ZipEntry zipEntry = new ZipEntry(entry.name);
        if (entry.compressed) {
          zipEntry.setMethod(ZipEntry.DEFLATED);
        } else {
          zipEntry.setMethod(ZipEntry.STORED);
          zipEntry.setSize(entry.content.length);
          zipEntry.setCrc(calculateCrc32(entry.content));
        }
        zipEntry.setTime(ZipCombiner.DOS_EPOCH.getTime());
        zipper.putNextEntry(zipEntry);
        zipper.write(entry.content);
        zipper.closeEntry();
      }
      zipper.close();
      return out.toByteArray();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public InputStream toInputStream() {
    return new ByteArrayInputStream(toByteArray());
  }

  public static long calculateCrc32(byte[] content) {
    CRC32 crc = new CRC32();
    crc.update(content);
    return crc.getValue();
  }
}
