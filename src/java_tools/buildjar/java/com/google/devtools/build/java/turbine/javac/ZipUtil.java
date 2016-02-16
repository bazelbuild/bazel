// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import java.io.IOException;
import java.util.GregorianCalendar;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Static utility methods for working with ZipOutputStream.
 */
public abstract class ZipUtil {

  /** The earliest date representable in a zip file, the DOS epoch. */
  private static final long DOS_EPOCH =
      new GregorianCalendar(1980, 0, 1, 0, 0, 0).getTimeInMillis();

  /**
   * This is a helper method for adding an uncompressed entry to a
   * ZipOutputStream.  The entry timestamp is also set to a fixed value.
   *
   * @param name filename to use within the zip file
   * @param content file contents
   * @param zip the ZipOutputStream to which this entry will be appended
   */
  public static void storeEntry(String name, byte[] content, ZipOutputStream zip)
      throws IOException {
    ZipEntry entry = new ZipEntry(name);
    entry.setMethod(ZipEntry.STORED);
    entry.setTime(DOS_EPOCH);
    entry.setSize(content.length);
    CRC32 crc32 = new CRC32();
    crc32.update(content);
    entry.setCrc(crc32.getValue());
    zip.putNextEntry(entry);
    zip.write(content);
    zip.closeEntry();
  }
}
