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

class EndOfCentralDirectoryRecord {
  static final int SIGNATURE = 0x06054b50;
  static final int FIXED_DATA_SIZE = 22;
  static final int SIGNATURE_OFFSET = 0;
  static final int DISK_NUMBER_OFFSET = 4;
  static final int CD_DISK_OFFSET = 6;
  static final int DISK_ENTRIES_OFFSET = 8;
  static final int TOTAL_ENTRIES_OFFSET = 10;
  static final int CD_SIZE_OFFSET = 12;
  static final int CD_OFFSET_OFFSET = 16;
  static final int COMMENT_LENGTH_OFFSET = 20;

  /**
   * Read the end of central directory record from the input stream and parse {@link ZipFileData}
   * from it.
   */
  static ZipFileData read(InputStream in, ZipFileData file) throws IOException {
    if (file == null) {
      throw new NullPointerException();
    }

    byte[] fixedSizeData = new byte[FIXED_DATA_SIZE];
    if (ZipUtil.readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
      throw new ZipException(
          "Unexpected end of file while reading End of Central Directory Record.");
    }
    if (!ZipUtil.arrayStartsWith(fixedSizeData, ZipUtil.intToLittleEndian(SIGNATURE))) {
      throw new ZipException(String.format(
          "Malformed End of Central Directory Record; does not start with %08x", SIGNATURE));
    }

    byte[] comment = new byte[ZipUtil.getUnsignedShort(fixedSizeData, COMMENT_LENGTH_OFFSET)];
    if (comment.length > 0 && ZipUtil.readFully(in, comment) != comment.length) {
      throw new ZipException(
          "Unexpected end of file while reading End of Central Directory Record.");
    }
    short diskNumber = ZipUtil.get16(fixedSizeData, DISK_NUMBER_OFFSET);
    short centralDirectoryDisk = ZipUtil.get16(fixedSizeData, CD_DISK_OFFSET);
    short entriesOnDisk = ZipUtil.get16(fixedSizeData, DISK_ENTRIES_OFFSET);
    short totalEntries = ZipUtil.get16(fixedSizeData, TOTAL_ENTRIES_OFFSET);
    int centralDirectorySize = ZipUtil.get32(fixedSizeData, CD_SIZE_OFFSET);
    int centralDirectoryOffset = ZipUtil.get32(fixedSizeData, CD_OFFSET_OFFSET);
    if (diskNumber == -1 || centralDirectoryDisk == -1 || entriesOnDisk == -1
        || totalEntries == -1 || centralDirectorySize == -1 || centralDirectoryOffset == -1) {
      file.setMaybeZip64(true);
    }
    file.setComment(comment);
    file.setCentralDirectorySize(ZipUtil.getUnsignedInt(fixedSizeData, CD_SIZE_OFFSET));
    file.setCentralDirectoryOffset(ZipUtil.getUnsignedInt(fixedSizeData, CD_OFFSET_OFFSET));
    file.setExpectedEntries(ZipUtil.getUnsignedShort(fixedSizeData, TOTAL_ENTRIES_OFFSET));
    return file;
  }

  /**
   * Generates the raw byte data of the end of central directory record for the specified
   * {@link ZipFileData}.
   * @throws ZipException if the file comment is too long
   */
  static byte[] create(ZipFileData file, boolean allowZip64) throws ZipException {
    byte[] comment = file.getBytes(file.getComment());

    byte[] buf = new byte[FIXED_DATA_SIZE + comment.length];

    // Allow writing of Zip file without Zip64 extensions for large archives as a special case
    // since many reading implementations can handle this.
    short numEntries = (short) (file.getNumEntries() > 0xffff && allowZip64
        ? -1 : file.getNumEntries());
    int cdSize = (int) (file.getCentralDirectorySize() > 0xffffffffL && allowZip64
        ? -1 : file.getCentralDirectorySize());
    int cdOffset = (int) (file.getCentralDirectoryOffset() > 0xffffffffL && allowZip64
        ? -1 : file.getCentralDirectoryOffset());
    ZipUtil.intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
    ZipUtil.shortToLittleEndian(buf, DISK_NUMBER_OFFSET, (short) 0);
    ZipUtil.shortToLittleEndian(buf, CD_DISK_OFFSET, (short) 0);
    ZipUtil.shortToLittleEndian(buf, DISK_ENTRIES_OFFSET, numEntries);
    ZipUtil.shortToLittleEndian(buf, TOTAL_ENTRIES_OFFSET, numEntries);
    ZipUtil.intToLittleEndian(buf, CD_SIZE_OFFSET, cdSize);
    ZipUtil.intToLittleEndian(buf, CD_OFFSET_OFFSET, cdOffset);
    ZipUtil.shortToLittleEndian(buf, COMMENT_LENGTH_OFFSET, (short) comment.length);
    System.arraycopy(comment, 0, buf, FIXED_DATA_SIZE, comment.length);

    return buf;
  }
}
