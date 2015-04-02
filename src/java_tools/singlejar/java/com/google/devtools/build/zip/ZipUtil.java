// Copyright 2015 Google Inc. All rights reserved.
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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Collection;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.zip.ZipException;

/** A utility class for reading and writing {@link ZipFileEntry}s from byte arrays. */
public class ZipUtil {

  /**
   * Midnight Jan 1st 1980. Uses the current time zone as the DOS format does not support time zones
   * and will always assume the current zone.
   */
  public static final long DOS_EPOCH =
      new GregorianCalendar(1980, Calendar.JANUARY, 1, 0, 0, 0).getTimeInMillis();

  /** 23:59:59 Dec 31st 2107. The maximum date representable in DOS format. */
  public static final long MAX_DOS_DATE =
      new GregorianCalendar(2107, Calendar.DECEMBER, 31, 23, 59, 59).getTimeInMillis();

  /** Converts a integral value to the corresponding little endian array. */
  private static byte[] integerToLittleEndian(byte[] buf, int offset, long value, int numBytes) {
    for (int i = 0; i < numBytes; i++) {
      buf[i + offset] = (byte) ((value & (0xffL << (i * 8))) >> (i * 8));
    }
    return buf;
  }

  /** Converts a short to the corresponding 2-byte little endian array. */
  static byte[] shortToLittleEndian(short value) {
    return integerToLittleEndian(new byte[2], 0, value, 2);
  }

  /** Writes a short to the buffer as a 2-byte little endian array starting at offset. */
  static byte[] shortToLittleEndian(byte[] buf, int offset, short value) {
    return integerToLittleEndian(buf, offset, value, 2);
  }

  /** Converts an int to the corresponding 4-byte little endian array. */
  static byte[] intToLittleEndian(int value) {
    return integerToLittleEndian(new byte[4], 0, value, 4);
  }

  /** Writes an int to the buffer as a 4-byte little endian array starting at offset. */
  static byte[] intToLittleEndian(byte[] buf, int offset, int value) {
    return integerToLittleEndian(buf, offset, value, 4);
  }

  /** Converts a long to the corresponding 8-byte little endian array. */
  static byte[] longToLittleEndian(long value) {
    return integerToLittleEndian(new byte[8], 0, value, 8);
  }

  /** Writes a long to the buffer as a 8-byte little endian array starting at offset. */
  static byte[] longToLittleEndian(byte[] buf, int offset, long value) {
    return integerToLittleEndian(buf, offset, value, 8);
  }

  /** Reads 16 bits in little-endian byte order from the buffer at the given offset. */
  static short get16(byte[] source, int offset) {
    int a = source[offset + 0] & 0xff;
    int b = source[offset + 1] & 0xff;
    return (short) ((b << 8) | a);
  }

  /** Reads 32 bits in little-endian byte order from the buffer at the given offset. */
  static int get32(byte[] source, int offset) {
    int a = source[offset + 0] & 0xff;
    int b = source[offset + 1] & 0xff;
    int c = source[offset + 2] & 0xff;
    int d = source[offset + 3] & 0xff;
    return (d << 24) | (c << 16) | (b << 8) | a;
  }

  /**
   * Reads an unsigned short in little-endian byte order from the buffer at the given offset.
   * Casts to an int to allow proper numerical comparison.
   */
  static int getUnsignedShort(byte[] source, int offset) {
    return get16(source, offset) & 0xffff;
  }

  /**
   * Reads an unsigned int in little-endian byte order from the buffer at the given offset.
   * Casts to a long to allow proper numerical comparison.
   */
  static long getUnsignedInt(byte[] source, int offset) {
    return get32(source, offset) & 0xffffffffL;
  }

  /** Checks if the timestamp is representable as a valid DOS timestamp. */
  private static boolean isValidInDos(long timestamp) {
    Calendar time = Calendar.getInstance();
    time.setTimeInMillis(timestamp);
    Calendar minTime = Calendar.getInstance();
    minTime.setTimeInMillis(DOS_EPOCH);
    Calendar maxTime = Calendar.getInstance();
    maxTime.setTimeInMillis(MAX_DOS_DATE);
    return (!time.before(minTime) && !time.after(maxTime));
  }

  /** Converts a unix timestamp into a 32-bit DOS timestamp. */
  static int unixToDosTime(long timestamp) {
    Calendar time = Calendar.getInstance();
    time.setTimeInMillis(timestamp);

    if (!isValidInDos(timestamp)) {
      DateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
      throw new IllegalArgumentException(String.format("%s is not representable in the DOS time"
          + " format. It must be in the range %s to %s", df.format(time.getTime()),
          df.format(new Date(DOS_EPOCH)), df.format(new Date(MAX_DOS_DATE))));
    }

    int dos = time.get(Calendar.SECOND) >> 1;
    dos |= time.get(Calendar.MINUTE) << 5;
    dos |= time.get(Calendar.HOUR_OF_DAY) << 11;
    dos |= time.get(Calendar.DAY_OF_MONTH) << 16;
    dos |= (time.get(Calendar.MONTH) + 1) << 21;
    dos |= (time.get(Calendar.YEAR) - 1980) << 25;
    return dos;
  }

  /** Converts a 32-bit DOS timestamp into a unix timestamp. */
  static long dosToUnixTime(int timestamp) {
    Calendar time = Calendar.getInstance();
    time.clear();
    time.set(Calendar.SECOND, (timestamp << 1) & 0x3e);
    time.set(Calendar.MINUTE, (timestamp >> 5) & 0x3f);
    time.set(Calendar.HOUR_OF_DAY, (timestamp >> 11) & 0x1f);
    time.set(Calendar.DAY_OF_MONTH, (timestamp >> 16) & 0x1f);
    time.set(Calendar.MONTH, ((timestamp >> 21) & 0x0f) - 1);
    time.set(Calendar.YEAR, ((timestamp >> 25) & 0x7f) + 1980);
    return time.getTimeInMillis();
  }

  /** Checks if array starts with target. */
  static boolean arrayStartsWith(byte[] array, byte[] target) {
    if (array == null) {
      return false;
    }
    if (target == null) {
      return true;
    }
    if (target.length > array.length) {
      return false;
    }
    for (int i = 0; i < target.length; i++) {
      if (array[i] != target[i]) {
        return false;
      }
    }
    return true;
  }

  static class LocalFileHeader {
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
     * Generates the raw byte data of the local file header for the ZipEntry. Uses the specified
     * charset to encode the file name and comment.
     */
    static byte[] create(ZipFileEntry entry, Charset charset) {
      byte[] name = entry.getName().getBytes(charset);
      byte[] extra = entry.getExtra() != null ? entry.getExtra() : new byte[]{};

      byte[] buf = new byte[FIXED_DATA_SIZE + name.length + extra.length];

      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      shortToLittleEndian(buf, VERSION_OFFSET, entry.getVersionNeeded());
      shortToLittleEndian(buf, FLAGS_OFFSET, entry.getFlags());
      shortToLittleEndian(buf, METHOD_OFFSET, (short) (entry.getMethod().getValue() & 0xffff));
      intToLittleEndian(buf, MOD_TIME_OFFSET, unixToDosTime(entry.getTime()));
      intToLittleEndian(buf, CRC_OFFSET, (int) (entry.getCrc() & 0xffffffff));
      intToLittleEndian(buf, COMPRESSED_SIZE_OFFSET,
          (int) (entry.getCompressedSize() & 0xffffffff));
      intToLittleEndian(buf, UNCOMPRESSED_SIZE_OFFSET, (int) (entry.getSize() & 0xffffffff));
      shortToLittleEndian(buf, FILENAME_LENGTH_OFFSET, (short) name.length);
      shortToLittleEndian(buf, EXTRA_FIELD_LENGTH_OFFSET, (short) extra.length);
      System.arraycopy(name, 0, buf, FIXED_DATA_SIZE, name.length);
      System.arraycopy(extra, 0, buf, FIXED_DATA_SIZE + name.length, extra.length);

      return buf;
    }
  }

  static class CentralDirectoryFileHeader {
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
     * Reads a {@link ZipFileEntry} from the input stream, using the specified charset to decode the
     * filename and comment.
     */
    static ZipFileEntry read(InputStream in, Charset charset)
        throws IOException {
      byte[] fixedSizeData = new byte[FIXED_DATA_SIZE];

      if (in.read(fixedSizeData) != FIXED_DATA_SIZE) {
        throw new ZipException(
            "Unexpected end of file while reading Central Directory File Header.");
      }
      if (!arrayStartsWith(fixedSizeData, intToLittleEndian(SIGNATURE))) {
        throw new ZipException(String.format(
            "Malformed Central Directory File Header; does not start with %08x", SIGNATURE));
      }

      byte[] name = new byte[getUnsignedShort(fixedSizeData, FILENAME_LENGTH_OFFSET)];
      byte[] extraField = new byte[getUnsignedShort(fixedSizeData, EXTRA_FIELD_LENGTH_OFFSET)];
      byte[] comment = new byte[getUnsignedShort(fixedSizeData, COMMENT_LENGTH_OFFSET)];

      if (name.length > 0 && in.read(name) != name.length) {
        throw new ZipException(
            "Unexpected end of file while reading Central Directory File Header.");
      }
      if (extraField.length > 0 && in.read(extraField) != extraField.length) {
        throw new ZipException(
            "Unexpected end of file while reading Central Directory File Header.");
      }
      if (comment.length > 0 && in.read(comment) != comment.length) {
        throw new ZipException(
            "Unexpected end of file while reading Central Directory File Header.");
      }

      ZipFileEntry entry = new ZipFileEntry(new String(name, charset));
      entry.setVersion(get16(fixedSizeData, VERSION_OFFSET));
      entry.setVersionNeeded(get16(fixedSizeData, VERSION_NEEDED_OFFSET));
      entry.setFlags(get16(fixedSizeData, FLAGS_OFFSET));
      entry.setMethod(Compression.fromValue(get16(fixedSizeData, METHOD_OFFSET)));
      long time = dosToUnixTime(get32(fixedSizeData, MOD_TIME_OFFSET));
      entry.setTime(isValidInDos(time) ? time : DOS_EPOCH);
      entry.setCrc(getUnsignedInt(fixedSizeData, CRC_OFFSET));
      entry.setCompressedSize(getUnsignedInt(fixedSizeData, COMPRESSED_SIZE_OFFSET));
      entry.setSize(getUnsignedInt(fixedSizeData, UNCOMPRESSED_SIZE_OFFSET));
      entry.setInternalAttributes(get16(fixedSizeData, INTERNAL_ATTRIBUTES_OFFSET));
      entry.setExternalAttributes(get32(fixedSizeData, EXTERNAL_ATTRIBUTES_OFFSET));
      entry.setLocalHeaderOffset(getUnsignedInt(fixedSizeData, LOCAL_HEADER_OFFSET_OFFSET));
      entry.setExtra(extraField);
      entry.setComment(new String(comment, charset));

      return entry;
    }

    /**
     * Generates the raw byte data of the central directory file header for the ZipEntry. Uses the
     * specified charset to encode the file name and comment.
     */
    static byte[] create(ZipFileEntry entry, Charset charset) {
      byte[] name = entry.getName().getBytes(charset);
      byte[] extra = entry.getExtra() != null ? entry.getExtra() : new byte[]{};
      byte[] comment = entry.getComment() != null
          ? entry.getComment().getBytes(charset) : new byte[]{};

      byte[] buf = new byte[FIXED_DATA_SIZE + name.length + extra.length + comment.length];

      fillFixedSizeData(buf, entry, name.length, extra.length, comment.length);
      System.arraycopy(name, 0, buf, FIXED_DATA_SIZE, name.length);
      System.arraycopy(extra, 0, buf, FIXED_DATA_SIZE + name.length, extra.length);
      System.arraycopy(comment, 0, buf, FIXED_DATA_SIZE + name.length + extra.length,
          comment.length);

      return buf;
    }

    /**
     * Writes the central directory file header for the ZipEntry to an output stream. Uses the
     * specified charset to encode the file name and comment.
     */
    static int write(ZipFileEntry entry, Charset charset, byte[] buf,
        OutputStream stream) throws IOException {
      if (buf == null || buf.length < FIXED_DATA_SIZE) {
        buf = new byte[FIXED_DATA_SIZE];
      }

      byte[] name = entry.getName().getBytes(charset);
      byte[] extra = entry.getExtra() != null ? entry.getExtra() : new byte[]{};
      byte[] comment = entry.getComment() != null
          ? entry.getComment().getBytes(charset) : new byte[]{};

      fillFixedSizeData(buf, entry, name.length, extra.length, comment.length);
      stream.write(buf, 0, FIXED_DATA_SIZE);
      stream.write(name);
      stream.write(extra);
      stream.write(comment);

      return FIXED_DATA_SIZE + name.length + extra.length + comment.length;
    }

    /**
     * Write the fixed size data portion for the specified ZIP entry to the buffer.
     */
    private static void fillFixedSizeData(byte[] buf, ZipFileEntry entry, int nameLength,
        int extraLength, int commentLength) {
      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      shortToLittleEndian(buf, VERSION_OFFSET, entry.getVersion());
      shortToLittleEndian(buf, VERSION_NEEDED_OFFSET, entry.getVersionNeeded());
      shortToLittleEndian(buf, FLAGS_OFFSET, entry.getFlags());
      shortToLittleEndian(buf, METHOD_OFFSET, (short) (entry.getMethod().getValue() & 0xffff));
      intToLittleEndian(buf, MOD_TIME_OFFSET, unixToDosTime(entry.getTime()));
      intToLittleEndian(buf, CRC_OFFSET, (int) (entry.getCrc() & 0xffffffff));
      intToLittleEndian(buf, COMPRESSED_SIZE_OFFSET,
          (int) (entry.getCompressedSize() & 0xffffffff));
      intToLittleEndian(buf, UNCOMPRESSED_SIZE_OFFSET, (int) (entry.getSize() & 0xffffffff));
      shortToLittleEndian(buf, FILENAME_LENGTH_OFFSET, (short) (nameLength & 0xffff));
      shortToLittleEndian(buf, EXTRA_FIELD_LENGTH_OFFSET, (short) (extraLength & 0xffff));
      shortToLittleEndian(buf, COMMENT_LENGTH_OFFSET, (short) (commentLength & 0xffff));
      shortToLittleEndian(buf, DISK_START_OFFSET, (short) 0);
      shortToLittleEndian(buf, INTERNAL_ATTRIBUTES_OFFSET, entry.getInternalAttributes());
      intToLittleEndian(buf, EXTERNAL_ATTRIBUTES_OFFSET, entry.getExternalAttributes());
      intToLittleEndian(buf, LOCAL_HEADER_OFFSET_OFFSET,
          (int) (entry.getLocalHeaderOffset() & 0xffffffff));
    }
  }

  static class EndOfCentralDirectoryRecord {
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
     * Generates the raw byte data of the end of central directory record, given the specifics of
     * the ZIP file.
     */
    static byte[] create(long fileOffset, long size, long numEntries, String fileComment,
        Charset charset) {
      byte[] comment = fileComment.getBytes(charset);

      byte[] buf = new byte[FIXED_DATA_SIZE + comment.length];

      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      shortToLittleEndian(buf, DISK_NUMBER_OFFSET, (short) 0);
      shortToLittleEndian(buf, CD_DISK_OFFSET, (short) 0);
      shortToLittleEndian(buf, DISK_ENTRIES_OFFSET, (short) (numEntries & 0xffff));
      shortToLittleEndian(buf, TOTAL_ENTRIES_OFFSET, (short) (numEntries & 0xffff));
      intToLittleEndian(buf, CD_SIZE_OFFSET, (int) (size & 0xffffffff));
      intToLittleEndian(buf, CD_OFFSET_OFFSET, (int) (fileOffset & 0xffffffff));
      shortToLittleEndian(buf, COMMENT_LENGTH_OFFSET, (short) (comment.length & 0xffff));
      System.arraycopy(comment, 0, buf, FIXED_DATA_SIZE, comment.length);

      return buf;
    }
  }

  static class CentralDirectory {

    /**
     * Writes the central directory to an output stream, given the specifics of the ZIP file.
     */
    static void write(Collection<ZipFileEntry> entries, String fileComment, long fileOffset,
        Charset charset, OutputStream stream) throws IOException {
      long directorySize = 0;
      byte[] buf = new byte[CentralDirectoryFileHeader.FIXED_DATA_SIZE];
      for (ZipFileEntry entry : entries) {
        directorySize += CentralDirectoryFileHeader.write(entry, charset, buf, stream);
      }
      stream.write(EndOfCentralDirectoryRecord.create(fileOffset, directorySize, entries.size(),
          fileComment != null ? fileComment : "", charset));
    }
  }
}
