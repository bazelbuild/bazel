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

import com.google.devtools.build.zip.ZipFileEntry.Feature;

import java.io.IOException;
import java.util.EnumSet;
import java.util.zip.ZipException;

class LocalFileHeader {
  static final int SIGNATURE = 0x04034b50;
  static final int FIXED_DATA_SIZE = 30;
  static final int SIGNATURE_OFFSET = 0;
  static final int VERSION_OFFSET = 4;
  static final int FLAGS_OFFSET = 6;
  static final int METHOD_OFFSET = 8;
  static final int MOD_TIME_OFFSET = 10;
  static final int CRC_OFFSET = 14;
  static final int COMPRESSED_SIZE_OFFSET = 18;
  static final int UNCOMPRESSED_SIZE_OFFSET = 22;
  static final int FILENAME_LENGTH_OFFSET = 26;
  static final int EXTRA_FIELD_LENGTH_OFFSET = 28;
  static final int VARIABLE_DATA_OFFSET = 30;

  /**
   * Generates the raw byte data of the local file header for the {@link ZipFileEntry}. Uses the
   * specified {@link ZipFileData} to encode the file name and comment.
   * @throws IOException 
   */
  static byte[] create(ZipFileEntry entry, ZipFileData file, boolean allowZip64)
      throws IOException {
    byte[] name = entry.getName().getBytes(file.getCharset());

    // We don't do a defensive copy here so that later, when we write the central directory entry,
    // the changes we make here take effect.
    // TODO(bazel-team): This seems like a bug. Investigate.
    ExtraDataList extra = entry.getExtra();

    EnumSet<Feature> features = entry.getFeatureSet();
    int size = (int) entry.getSize();
    int csize = (int) entry.getCompressedSize();

    if (features.contains(Feature.ZIP64_SIZE) || features.contains(Feature.ZIP64_CSIZE)) {
      if (!allowZip64) {
        throw new ZipException(String.format("Writing an entry of size %d(%d) without Zip64"
            + " extensions is not supported.", entry.getSize(), entry.getCompressedSize()));
      }
      extra.remove((short) 0x0001);
      int extraSize = 0;
      if (features.contains(Feature.ZIP64_SIZE)) {
        size = -1;
        extraSize += 8;
      }
      if (features.contains(Feature.ZIP64_CSIZE)) {
        csize = -1;
        extraSize += 8;
      }
      byte[] zip64Extra = new byte[ExtraData.FIXED_DATA_SIZE + extraSize];
      ZipUtil.shortToLittleEndian(zip64Extra, ExtraData.ID_OFFSET, (short) 0x0001);
      ZipUtil.shortToLittleEndian(zip64Extra, ExtraData.LENGTH_OFFSET, (short) extraSize);
      int offset = ExtraData.FIXED_DATA_SIZE;
      if (features.contains(Feature.ZIP64_SIZE)) {
        ZipUtil.longToLittleEndian(zip64Extra, offset, entry.getSize());
        offset += 8;
      }
      if (features.contains(Feature.ZIP64_CSIZE)) {
        ZipUtil.longToLittleEndian(zip64Extra, offset, entry.getCompressedSize());
        offset += 8;
      }
      extra.add(new ExtraData(zip64Extra, 0));
    } else {
      extra.remove((short) 0x0001);
    }

    extra.remove(ExtraDataList.EXTENDED_TIMESTAMP);
    extra.remove(ExtraDataList.INFOZIP_UNIX_NEW);

    byte[] buf = new byte[FIXED_DATA_SIZE + name.length + extra.getLength()];
    ZipUtil.intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
    ZipUtil.shortToLittleEndian(buf, VERSION_OFFSET, entry.getVersionNeeded());
    ZipUtil.shortToLittleEndian(buf, FLAGS_OFFSET, entry.getFlags());
    ZipUtil.shortToLittleEndian(buf, METHOD_OFFSET, entry.getMethod().getValue());
    ZipUtil.intToLittleEndian(buf, MOD_TIME_OFFSET, ZipUtil.unixToDosTime(entry.getTime()));
    ZipUtil.intToLittleEndian(buf, CRC_OFFSET, (int) (entry.getCrc() & 0xffffffff));
    ZipUtil.intToLittleEndian(buf, COMPRESSED_SIZE_OFFSET, csize);
    ZipUtil.intToLittleEndian(buf, UNCOMPRESSED_SIZE_OFFSET, size);
    ZipUtil.shortToLittleEndian(buf, FILENAME_LENGTH_OFFSET, (short) name.length);
    ZipUtil.shortToLittleEndian(buf, EXTRA_FIELD_LENGTH_OFFSET, (short) extra.getLength());
    System.arraycopy(name, 0, buf, FIXED_DATA_SIZE, name.length);
    ZipUtil.readFully(extra.getByteStream(), buf, FIXED_DATA_SIZE + name.length, extra.getLength());

    return buf;
  }
}
