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

import com.google.devtools.build.zip.ZipFileEntry.Compression;
import com.google.devtools.build.zip.ZipFileEntry.Feature;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.util.EnumSet;
import java.util.zip.ZipException;

class CentralDirectoryFileHeader {
  static final int SIGNATURE = 0x02014b50;
  static final int FIXED_DATA_SIZE = 46;
  static final int SIGNATURE_OFFSET = 0;
  static final int VERSION_OFFSET = 4;
  static final int VERSION_NEEDED_OFFSET = 6;
  static final int FLAGS_OFFSET = 8;
  static final int METHOD_OFFSET = 10;
  static final int MOD_TIME_OFFSET = 12;
  static final int CRC_OFFSET = 16;
  static final int COMPRESSED_SIZE_OFFSET = 20;
  static final int UNCOMPRESSED_SIZE_OFFSET = 24;
  static final int FILENAME_LENGTH_OFFSET = 28;
  static final int EXTRA_FIELD_LENGTH_OFFSET = 30;
  static final int COMMENT_LENGTH_OFFSET = 32;
  static final int DISK_START_OFFSET = 34;
  static final int INTERNAL_ATTRIBUTES_OFFSET = 36;
  static final int EXTERNAL_ATTRIBUTES_OFFSET = 38;
  static final int LOCAL_HEADER_OFFSET_OFFSET = 42;

  /**
   * Reads a {@link ZipFileEntry} from the input stream, using the specified {@link Charset} to
   * decode the filename and comment.
   */
  static ZipFileEntry read(InputStream in, Charset charset)
      throws IOException {
    byte[] fixedSizeData = new byte[FIXED_DATA_SIZE];

    if (ZipUtil.readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
      throw new ZipException(
          "Unexpected end of file while reading Central Directory File Header.");
    }
    if (!ZipUtil.arrayStartsWith(fixedSizeData, ZipUtil.intToLittleEndian(SIGNATURE))) {
      throw new ZipException(String.format(
          "Malformed Central Directory File Header; does not start with %08x", SIGNATURE));
    }

    byte[] name = new byte[ZipUtil.getUnsignedShort(fixedSizeData, FILENAME_LENGTH_OFFSET)];
    byte[] extraField = new byte[
        ZipUtil.getUnsignedShort(fixedSizeData, EXTRA_FIELD_LENGTH_OFFSET)];
    byte[] comment = new byte[ZipUtil.getUnsignedShort(fixedSizeData, COMMENT_LENGTH_OFFSET)];

    if (name.length > 0 && ZipUtil.readFully(in, name) != name.length) {
      throw new ZipException(
          "Unexpected end of file while reading Central Directory File Header.");
    }
    if (extraField.length > 0 && ZipUtil.readFully(in, extraField) != extraField.length) {
      throw new ZipException(
          "Unexpected end of file while reading Central Directory File Header.");
    }
    if (comment.length > 0 && ZipUtil.readFully(in, comment) != comment.length) {
      throw new ZipException(
          "Unexpected end of file while reading Central Directory File Header.");
    }

    ExtraDataList extra = new ExtraDataList(extraField);

    long csize = ZipUtil.getUnsignedInt(fixedSizeData, COMPRESSED_SIZE_OFFSET);
    long size = ZipUtil.getUnsignedInt(fixedSizeData, UNCOMPRESSED_SIZE_OFFSET);
    long offset = ZipUtil.getUnsignedInt(fixedSizeData, LOCAL_HEADER_OFFSET_OFFSET);
    if (csize == 0xffffffffL || size == 0xffffffffL || offset == 0xffffffffL) {
      ExtraData zip64Extra = extra.get((short) 0x0001);
      if (zip64Extra != null) {
        int index = 0;
        if (size == 0xffffffffL) {
          size = ZipUtil.getUnsignedLong(zip64Extra.getData(), index);
          index += 8;
        }
        if (csize == 0xffffffffL) {
          csize = ZipUtil.getUnsignedLong(zip64Extra.getData(), index);
          index += 8;
        }
        if (offset == 0xffffffffL) {
          offset = ZipUtil.getUnsignedLong(zip64Extra.getData(), index);
          index += 8;
        }
      }
    }

    ZipFileEntry entry = new ZipFileEntry(new String(name, charset));
    entry.setVersion(ZipUtil.get16(fixedSizeData, VERSION_OFFSET));
    entry.setVersionNeeded(ZipUtil.get16(fixedSizeData, VERSION_NEEDED_OFFSET));
    entry.setFlags(ZipUtil.get16(fixedSizeData, FLAGS_OFFSET));
    entry.setMethod(Compression.fromValue(ZipUtil.get16(fixedSizeData, METHOD_OFFSET)));
    long time = ZipUtil.dosToUnixTime(ZipUtil.get32(fixedSizeData, MOD_TIME_OFFSET));
    entry.setTime(ZipUtil.isValidInDos(time) ? time : ZipUtil.DOS_EPOCH);
    entry.setCrc(ZipUtil.getUnsignedInt(fixedSizeData, CRC_OFFSET));
    entry.setCompressedSize(csize);
    entry.setSize(size);
    entry.setInternalAttributes(ZipUtil.get16(fixedSizeData, INTERNAL_ATTRIBUTES_OFFSET));
    entry.setExternalAttributes(ZipUtil.get32(fixedSizeData, EXTERNAL_ATTRIBUTES_OFFSET));
    entry.setLocalHeaderOffset(offset);
    entry.setExtra(extra);
    entry.setComment(new String(comment, charset));

    return entry;
  }

  /**
   * Generates the raw byte data of the central directory file header for the ZipEntry. Uses the
   * specified {@link ZipFileData} to encode the file name and comment.
   * @throws ZipException 
   */
  static byte[] create(ZipFileEntry entry, ZipFileData file, boolean allowZip64)
      throws ZipException {
    if (allowZip64) {
      addZip64Extra(entry);
    } else {
      entry.getExtra().remove((short) 0x0001);
    }
    byte[] name = file.getBytes(entry.getName());
    byte[] extra = entry.getExtra().getBytes();
    byte[] comment = entry.getComment() != null
        ? file.getBytes(entry.getComment()) : new byte[]{};

    byte[] buf = new byte[FIXED_DATA_SIZE + name.length + extra.length + comment.length];

    fillFixedSizeData(buf, entry, name.length, extra.length, comment.length, allowZip64);
    System.arraycopy(name, 0, buf, FIXED_DATA_SIZE, name.length);
    System.arraycopy(extra, 0, buf, FIXED_DATA_SIZE + name.length, extra.length);
    System.arraycopy(comment, 0, buf, FIXED_DATA_SIZE + name.length + extra.length,
        comment.length);

    return buf;
  }

  /**
   * Writes the central directory file header for the ZipEntry to an output stream. Uses the
   * specified {@link ZipFileData} to encode the file name and comment.
   */
  static int write(ZipFileEntry entry, ZipFileData file, boolean allowZip64, byte[] buf,
      OutputStream stream) throws IOException {
    if (buf == null || buf.length < FIXED_DATA_SIZE) {
      buf = new byte[FIXED_DATA_SIZE];
    }

    ExtraDataList extra = new ExtraDataList(entry.getExtra());
    if (allowZip64) {
      addZip64Extra(entry);
    } else {
      extra.remove((short) 0x0001);
    }

    extra.remove(ExtraDataList.EXTENDED_TIMESTAMP);
    extra.remove(ExtraDataList.INFOZIP_UNIX_NEW);

    byte[] name = entry.getName().getBytes(file.getCharset());
    byte[] extraBytes = extra.getBytes();
    byte[] comment = entry.getComment() != null
        ? entry.getComment().getBytes(file.getCharset()) : new byte[]{};

    fillFixedSizeData(buf, entry, name.length, extraBytes.length, comment.length, allowZip64);
    stream.write(buf, 0, FIXED_DATA_SIZE);
    stream.write(name);
    stream.write(extraBytes);
    stream.write(comment);

    return FIXED_DATA_SIZE + name.length + extraBytes.length + comment.length;
  }

  /**
   * Write the fixed size data portion for the specified ZIP entry to the buffer.
   * @throws ZipException
   */
  private static void fillFixedSizeData(byte[] buf, ZipFileEntry entry, int nameLength,
      int extraLength, int commentLength, boolean allowZip64) throws ZipException {
    if (!allowZip64 && entry.getFeatureSet().contains(Feature.ZIP64_CSIZE)) {
      throw new ZipException(String.format("Writing an entry with compressed size %d without"
          + " Zip64 extensions is not supported.", entry.getCompressedSize()));
    }
    if (!allowZip64 && entry.getFeatureSet().contains(Feature.ZIP64_SIZE)) {
      throw new ZipException(String.format("Writing an entry of size %d without"
          + " Zip64 extensions is not supported.", entry.getSize()));
    }
    if (!allowZip64 && entry.getFeatureSet().contains(Feature.ZIP64_OFFSET)) {
      throw new ZipException(String.format("Writing an entry with local header offset %d without"
          + " Zip64 extensions is not supported.", entry.getLocalHeaderOffset()));
    }
    int csize = (int) (entry.getFeatureSet().contains(Feature.ZIP64_CSIZE)
        ? -1 : entry.getCompressedSize());
    int size = (int) (entry.getFeatureSet().contains(Feature.ZIP64_SIZE)
        ? -1 : entry.getSize());
    int offset = (int) (entry.getFeatureSet().contains(Feature.ZIP64_OFFSET)
        ? -1 : entry.getLocalHeaderOffset());
    ZipUtil.intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
    ZipUtil.shortToLittleEndian(buf, VERSION_OFFSET, entry.getVersion());
    ZipUtil.shortToLittleEndian(buf, VERSION_NEEDED_OFFSET, entry.getVersionNeeded());
    ZipUtil.shortToLittleEndian(buf, FLAGS_OFFSET, entry.getFlags());
    ZipUtil.shortToLittleEndian(buf, METHOD_OFFSET, entry.getMethod().getValue());
    ZipUtil.intToLittleEndian(buf, MOD_TIME_OFFSET, ZipUtil.unixToDosTime(entry.getTime()));
    ZipUtil.intToLittleEndian(buf, CRC_OFFSET, (int) (entry.getCrc() & 0xffffffff));
    ZipUtil.intToLittleEndian(buf, COMPRESSED_SIZE_OFFSET, csize);
    ZipUtil.intToLittleEndian(buf, UNCOMPRESSED_SIZE_OFFSET, size);
    ZipUtil.shortToLittleEndian(buf, FILENAME_LENGTH_OFFSET, (short) (nameLength & 0xffff));
    ZipUtil.shortToLittleEndian(buf, EXTRA_FIELD_LENGTH_OFFSET, (short) (extraLength & 0xffff));
    ZipUtil.shortToLittleEndian(buf, COMMENT_LENGTH_OFFSET, (short) (commentLength & 0xffff));
    ZipUtil.shortToLittleEndian(buf, DISK_START_OFFSET, (short) 0);
    ZipUtil.shortToLittleEndian(buf, INTERNAL_ATTRIBUTES_OFFSET, entry.getInternalAttributes());
    ZipUtil.intToLittleEndian(buf, EXTERNAL_ATTRIBUTES_OFFSET, entry.getExternalAttributes());
    ZipUtil.intToLittleEndian(buf, LOCAL_HEADER_OFFSET_OFFSET, offset);
  }

  /**
   * Update the extra data fields to contain a Zip64 extended information field if required
   */
  private static void addZip64Extra(ZipFileEntry entry) {
    EnumSet<Feature> features = entry.getFeatureSet();
    ExtraDataList extra = entry.getExtra();
    int extraSize = 0;
    if (features.contains(Feature.ZIP64_SIZE)) {
      extraSize += 8;
    }
    if (features.contains(Feature.ZIP64_CSIZE)) {
      extraSize += 8;
    }
    if (features.contains(Feature.ZIP64_OFFSET)) {
      extraSize += 8;
    }
    if (extraSize > 0) {
      extra.remove((short) 0x0001);
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
      if (features.contains(Feature.ZIP64_OFFSET)) {
        ZipUtil.longToLittleEndian(zip64Extra, offset, entry.getLocalHeaderOffset());
      }
      extra.add(new ExtraData(zip64Extra, 0));
    }
  }
}
