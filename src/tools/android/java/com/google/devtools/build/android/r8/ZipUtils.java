// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import com.android.tools.r8.ByteDataView;
import com.google.common.io.ByteStreams;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Predicate;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/** Utilities for working with zip files. */
public class ZipUtils {
  public static void addEntry(String name, InputStream stream, ZipOutputStream zip)
      throws IOException {
    ZipUtils.addEntry(name, ByteStreams.toByteArray(stream), ZipEntry.STORED, zip);
  }

  public static void addEntry(String name, byte[] bytes, int compressionMethod, ZipOutputStream zip)
      throws IOException {
    CRC32 crc = new CRC32();
    crc.update(bytes);
    ZipEntry entry = createEntry(name, bytes.length, crc.getValue(), compressionMethod);
    zip.putNextEntry(entry);
    zip.write(bytes);
    zip.closeEntry();
  }

  public static void copyEntries(
      Path input, ZipOutputStream zipOutputStream, Predicate<String> exclude) throws IOException {
    try (ZipInputStream zipInputStream =
        new ZipInputStream(new BufferedInputStream(Files.newInputStream(input)))) {
      ZipEntry zipEntry;
      while ((zipEntry = zipInputStream.getNextEntry()) != null) {
        if (!exclude.test(zipEntry.getName())) {
          zipOutputStream.putNextEntry(zipEntry);
          ByteStreams.copy(zipInputStream, zipOutputStream);
        }
      }
    }
  }

  public static void writeToZipStream(
      String name, ByteDataView content, int compressionMethod, ZipOutputStream zip)
      throws IOException {
    byte[] buffer = content.getBuffer();
    int offset = content.getOffset();
    int length = content.getLength();
    CRC32 crc = new CRC32();
    crc.update(buffer, offset, length);
    ZipEntry entry = createEntry(name, length, crc.getValue(), compressionMethod);
    zip.putNextEntry(entry);
    zip.write(buffer, offset, length);
    zip.closeEntry();
  }

  private static ZipEntry createEntry(String name, int length, long crc, int compressionMethod) {
    ZipEntry entry = new ZipEntry(name);
    entry.setMethod(compressionMethod);
    entry.setSize(length);
    entry.setCrc(crc);
    entry.setTime(0);
    return entry;
  }

  private ZipUtils() {}
}
