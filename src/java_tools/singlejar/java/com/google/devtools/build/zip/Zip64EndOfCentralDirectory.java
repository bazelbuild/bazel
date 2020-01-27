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

import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipException;

class Zip64EndOfCentralDirectory {
  static final int SIGNATURE = 0x06064b50;
  static final int FIXED_DATA_SIZE = 56;
  static final int SIGNATURE_OFFSET = 0;
  static final int SIZE_OFFSET = 4;
  static final int VERSION_OFFSET = 12;
  static final int VERSION_NEEDED_OFFSET = 14;
  static final int DISK_NUMBER_OFFSET = 16;
  static final int CD_DISK_OFFSET = 20;
  static final int DISK_ENTRIES_OFFSET = 24;
  static final int TOTAL_ENTRIES_OFFSET = 32;
  static final int CD_SIZE_OFFSET = 40;
  static final int CD_OFFSET_OFFSET = 48;

  /**
   * Read the Zip64 end of central directory record from the input stream and parse additional
   * {@link ZipFileData} from it.
   */
  static ZipFileData read(InputStream in, ZipFileData file) throws IOException {
    if (file == null) {
      throw new NullPointerException();
    }

    byte[] fixedSizeData = new byte[FIXED_DATA_SIZE];
    if (ZipUtil.readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
      throw new ZipException(
          "Unexpected end of file while reading Zip64 End of Central Directory Record.");
    }
    if (!ZipUtil.arrayStartsWith(fixedSizeData, ZipUtil.intToLittleEndian(SIGNATURE))) {
      throw new ZipException(String.format(
          "Malformed Zip64 End of Central Directory; does not start with %08x", SIGNATURE));
    }
    file.setZip64(true);
    file.setCentralDirectoryOffset(ZipUtil.getUnsignedLong(fixedSizeData, CD_OFFSET_OFFSET));
    file.setExpectedEntries(ZipUtil.getUnsignedLong(fixedSizeData, TOTAL_ENTRIES_OFFSET));
    return file;
  }

  /**
   * Generates the raw byte data of the Zip64 end of central directory record for the file.
   */
  static byte[] create(ZipFileData file) {
    byte[] buf = new byte[FIXED_DATA_SIZE];
    ZipUtil.intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
    ZipUtil.longToLittleEndian(buf, SIZE_OFFSET, FIXED_DATA_SIZE - 12);
    ZipUtil.shortToLittleEndian(buf, VERSION_OFFSET, (short) 0x2d);
    ZipUtil.shortToLittleEndian(buf, VERSION_NEEDED_OFFSET, (short) 0x2d);
    ZipUtil.intToLittleEndian(buf, DISK_NUMBER_OFFSET, 0);
    ZipUtil.intToLittleEndian(buf, CD_DISK_OFFSET, 0);
    ZipUtil.longToLittleEndian(buf, DISK_ENTRIES_OFFSET, file.getNumEntries());
    ZipUtil.longToLittleEndian(buf, TOTAL_ENTRIES_OFFSET, file.getNumEntries());
    ZipUtil.longToLittleEndian(buf, CD_SIZE_OFFSET, file.getCentralDirectorySize());
    ZipUtil.longToLittleEndian(buf, CD_OFFSET_OFFSET, file.getCentralDirectoryOffset());
    return buf;
  }
}
