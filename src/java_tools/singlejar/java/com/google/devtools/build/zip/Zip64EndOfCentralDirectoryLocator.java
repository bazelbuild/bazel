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

class Zip64EndOfCentralDirectoryLocator {
  static final int SIGNATURE = 0x07064b50;
  static final int FIXED_DATA_SIZE = 20;
  static final int SIGNATURE_OFFSET = 0;
  static final int ZIP64_EOCD_DISK_OFFSET = 4;
  static final int ZIP64_EOCD_OFFSET_OFFSET = 8;
  static final int DISK_NUMBER_OFFSET = 16;

  /**
   * Read the Zip64 end of central directory locator from the input stream and parse additional
   * {@link ZipFileData} from it.
   */
  static ZipFileData read(InputStream in, ZipFileData file) throws IOException {
    if (file == null) {
      throw new NullPointerException();
    }

    byte[] fixedSizeData = new byte[FIXED_DATA_SIZE];
    if (ZipUtil.readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
      throw new ZipException(
          "Unexpected end of file while reading Zip64 End of Central Directory Locator.");
    }
    if (!ZipUtil.arrayStartsWith(fixedSizeData, ZipUtil.intToLittleEndian(SIGNATURE))) {
      throw new ZipException(String.format(
          "Malformed Zip64 Central Directory Locator; does not start with %08x", SIGNATURE));
    }
    file.setZip64(true);
    file.setZip64EndOfCentralDirectoryOffset(
        ZipUtil.getUnsignedLong(fixedSizeData, ZIP64_EOCD_OFFSET_OFFSET));
    return file;
  }

  /**
   * Generates the raw byte data of the Zip64 end of central directory locator for the file.
   */
  static byte[] create(ZipFileData file) {
    byte[] buf = new byte[FIXED_DATA_SIZE];
    ZipUtil.intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
    ZipUtil.intToLittleEndian(buf, ZIP64_EOCD_DISK_OFFSET, 0);
    ZipUtil.longToLittleEndian(buf, ZIP64_EOCD_OFFSET_OFFSET,
        file.getZip64EndOfCentralDirectoryOffset());
    ZipUtil.intToLittleEndian(buf, DISK_NUMBER_OFFSET, 1);
    return buf;
  }
}
