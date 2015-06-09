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
import com.google.devtools.build.zip.ZipFileEntry.Feature;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.EnumSet;
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

  /** Reads 64 bits in little-endian byte order from the buffer at the given offset. */
  static long get64(byte[] source, int offset) {
    long a = source[offset + 0] & 0xffL;
    long b = source[offset + 1] & 0xffL;
    long c = source[offset + 2] & 0xffL;
    long d = source[offset + 3] & 0xffL;
    long e = source[offset + 4] & 0xffL;
    long f = source[offset + 5] & 0xffL;
    long g = source[offset + 6] & 0xffL;
    long h = source[offset + 7] & 0xffL;
    return (h << 56) | (g << 48) | (f << 40) | (e << 32) | (d << 24) | (c << 16) | (b << 8) | a;
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

  /**
   * Reads an unsigned long in little-endian byte order from the buffer at the given offset.
   * Performs bounds checking to see if the unsigned long will be properly represented in Java's
   * signed value.
   */
  static long getUnsignedLong(byte[] source, int offset) throws ZipException {
    long result = get64(source, offset);
    if (result < 0) {
      throw new ZipException("The requested unsigned long value is too large for Java's signed"
          + "values. This Zip file is unsupported");
    }
    return result;
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

  /** Read from the input stream into the array until it is full. */
  static int readFully(InputStream in, byte[] b) throws IOException {
    return readFully(in, b, 0, b.length);
  }

  /** Read from the input stream into the array starting at off until len bytes have been read. */
  static int readFully(InputStream in, byte[] b, int off, int len) throws IOException {
    if (len < 0) {
      throw new IndexOutOfBoundsException();
    }
    int n = 0;
    while (n < len) {
      int count = in.read(b, off + n, len - n);
      if (count < 0) {
        return n;
      }
      n += count;
    }
    return n;
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
     * Generates the raw byte data of the local file header for the {@link ZipFileEntry}. Uses the
     * specified {@link ZipFileData} to encode the file name and comment.
     * @throws IOException 
     */
    static byte[] create(ZipFileEntry entry, ZipFileData file, boolean allowZip64)
        throws IOException {
      byte[] name = entry.getName().getBytes(file.getCharset());
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
        shortToLittleEndian(zip64Extra, ExtraData.ID_OFFSET, (short) 0x0001);
        shortToLittleEndian(zip64Extra, ExtraData.LENGTH_OFFSET, (short) extraSize);
        int offset = ExtraData.FIXED_DATA_SIZE;
        if (features.contains(Feature.ZIP64_SIZE)) {
          longToLittleEndian(zip64Extra, offset, entry.getSize());
          offset += 8;
        }
        if (features.contains(Feature.ZIP64_CSIZE)) {
          longToLittleEndian(zip64Extra, offset, entry.getCompressedSize());
          offset += 8;
        }
        extra.add(new ExtraData(zip64Extra, 0));
      } else {
        extra.remove((short) 0x0001);
      }

      byte[] buf = new byte[FIXED_DATA_SIZE + name.length + extra.getLength()];
      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      shortToLittleEndian(buf, VERSION_OFFSET, entry.getVersionNeeded());
      shortToLittleEndian(buf, FLAGS_OFFSET, entry.getFlags());
      shortToLittleEndian(buf, METHOD_OFFSET, entry.getMethod().getValue());
      intToLittleEndian(buf, MOD_TIME_OFFSET, unixToDosTime(entry.getTime()));
      intToLittleEndian(buf, CRC_OFFSET, (int) (entry.getCrc() & 0xffffffff));
      intToLittleEndian(buf, COMPRESSED_SIZE_OFFSET, csize);
      intToLittleEndian(buf, UNCOMPRESSED_SIZE_OFFSET, size);
      shortToLittleEndian(buf, FILENAME_LENGTH_OFFSET, (short) name.length);
      shortToLittleEndian(buf, EXTRA_FIELD_LENGTH_OFFSET, (short) extra.getLength());
      System.arraycopy(name, 0, buf, FIXED_DATA_SIZE, name.length);
      readFully(extra.getByteStream(), buf, FIXED_DATA_SIZE + name.length, extra.getLength());

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
     * Reads a {@link ZipFileEntry} from the input stream, using the specified {@link Charset} to
     * decode the filename and comment.
     */
    static ZipFileEntry read(InputStream in, Charset charset)
        throws IOException {
      byte[] fixedSizeData = new byte[FIXED_DATA_SIZE];

      if (readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
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

      if (name.length > 0 && readFully(in, name) != name.length) {
        throw new ZipException(
            "Unexpected end of file while reading Central Directory File Header.");
      }
      if (extraField.length > 0 && readFully(in, extraField) != extraField.length) {
        throw new ZipException(
            "Unexpected end of file while reading Central Directory File Header.");
      }
      if (comment.length > 0 && readFully(in, comment) != comment.length) {
        throw new ZipException(
            "Unexpected end of file while reading Central Directory File Header.");
      }

      ExtraDataList extra = new ExtraDataList(extraField);

      long csize = getUnsignedInt(fixedSizeData, COMPRESSED_SIZE_OFFSET);
      long size = getUnsignedInt(fixedSizeData, UNCOMPRESSED_SIZE_OFFSET);
      long offset = getUnsignedInt(fixedSizeData, LOCAL_HEADER_OFFSET_OFFSET);
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
      entry.setVersion(get16(fixedSizeData, VERSION_OFFSET));
      entry.setVersionNeeded(get16(fixedSizeData, VERSION_NEEDED_OFFSET));
      entry.setFlags(get16(fixedSizeData, FLAGS_OFFSET));
      entry.setMethod(Compression.fromValue(get16(fixedSizeData, METHOD_OFFSET)));
      long time = dosToUnixTime(get32(fixedSizeData, MOD_TIME_OFFSET));
      entry.setTime(isValidInDos(time) ? time : DOS_EPOCH);
      entry.setCrc(getUnsignedInt(fixedSizeData, CRC_OFFSET));
      entry.setCompressedSize(csize);
      entry.setSize(size);
      entry.setInternalAttributes(get16(fixedSizeData, INTERNAL_ATTRIBUTES_OFFSET));
      entry.setExternalAttributes(get32(fixedSizeData, EXTERNAL_ATTRIBUTES_OFFSET));
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

      if (allowZip64) {
        addZip64Extra(entry);
      } else {
        entry.getExtra().remove((short) 0x0001);
      }
      byte[] name = entry.getName().getBytes(file.getCharset());
      byte[] extra = entry.getExtra().getBytes();
      byte[] comment = entry.getComment() != null
          ? entry.getComment().getBytes(file.getCharset()) : new byte[]{};

      fillFixedSizeData(buf, entry, name.length, extra.length, comment.length, allowZip64);
      stream.write(buf, 0, FIXED_DATA_SIZE);
      stream.write(name);
      stream.write(extra);
      stream.write(comment);

      return FIXED_DATA_SIZE + name.length + extra.length + comment.length;
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
      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      shortToLittleEndian(buf, VERSION_OFFSET, entry.getVersion());
      shortToLittleEndian(buf, VERSION_NEEDED_OFFSET, entry.getVersionNeeded());
      shortToLittleEndian(buf, FLAGS_OFFSET, entry.getFlags());
      shortToLittleEndian(buf, METHOD_OFFSET, entry.getMethod().getValue());
      intToLittleEndian(buf, MOD_TIME_OFFSET, unixToDosTime(entry.getTime()));
      intToLittleEndian(buf, CRC_OFFSET, (int) (entry.getCrc() & 0xffffffff));
      intToLittleEndian(buf, COMPRESSED_SIZE_OFFSET, csize);
      intToLittleEndian(buf, UNCOMPRESSED_SIZE_OFFSET, size);
      shortToLittleEndian(buf, FILENAME_LENGTH_OFFSET, (short) (nameLength & 0xffff));
      shortToLittleEndian(buf, EXTRA_FIELD_LENGTH_OFFSET, (short) (extraLength & 0xffff));
      shortToLittleEndian(buf, COMMENT_LENGTH_OFFSET, (short) (commentLength & 0xffff));
      shortToLittleEndian(buf, DISK_START_OFFSET, (short) 0);
      shortToLittleEndian(buf, INTERNAL_ATTRIBUTES_OFFSET, entry.getInternalAttributes());
      intToLittleEndian(buf, EXTERNAL_ATTRIBUTES_OFFSET, entry.getExternalAttributes());
      intToLittleEndian(buf, LOCAL_HEADER_OFFSET_OFFSET, offset);
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
        shortToLittleEndian(zip64Extra, ExtraData.ID_OFFSET, (short) 0x0001);
        shortToLittleEndian(zip64Extra, ExtraData.LENGTH_OFFSET, (short) extraSize);
        int offset = ExtraData.FIXED_DATA_SIZE;
        if (features.contains(Feature.ZIP64_SIZE)) {
          longToLittleEndian(zip64Extra, offset, entry.getSize());
          offset += 8;
        }
        if (features.contains(Feature.ZIP64_CSIZE)) {
          longToLittleEndian(zip64Extra, offset, entry.getCompressedSize());
          offset += 8;
        }
        if (features.contains(Feature.ZIP64_OFFSET)) {
          longToLittleEndian(zip64Extra, offset, entry.getLocalHeaderOffset());
        }
        extra.add(new ExtraData(zip64Extra, 0));
      }
    }
  }

  static class Zip64EndOfCentralDirectory {
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
      if (readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
        throw new ZipException(
            "Unexpected end of file while reading Zip64 End of Central Directory Record.");
      }
      if (!arrayStartsWith(fixedSizeData, intToLittleEndian(SIGNATURE))) {
        throw new ZipException(String.format(
            "Malformed Zip64 End of Central Directory; does not start with %08x", SIGNATURE));
      }
      file.setZip64(true);
      file.setCentralDirectoryOffset(getUnsignedLong(fixedSizeData, CD_OFFSET_OFFSET));
      file.setExpectedEntries(getUnsignedLong(fixedSizeData, TOTAL_ENTRIES_OFFSET));
      return file;
    }

    /**
     * Generates the raw byte data of the Zip64 end of central directory record for the file.
     */
    static byte[] create(ZipFileData file) {
      byte[] buf = new byte[FIXED_DATA_SIZE];
      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      longToLittleEndian(buf, SIZE_OFFSET, FIXED_DATA_SIZE - 12);
      shortToLittleEndian(buf, VERSION_OFFSET, (short) 0x2d);
      shortToLittleEndian(buf, VERSION_NEEDED_OFFSET, (short) 0x2d);
      intToLittleEndian(buf, DISK_NUMBER_OFFSET, 0);
      intToLittleEndian(buf, CD_DISK_OFFSET, 0);
      longToLittleEndian(buf, DISK_ENTRIES_OFFSET, file.getNumEntries());
      longToLittleEndian(buf, TOTAL_ENTRIES_OFFSET, file.getNumEntries());
      longToLittleEndian(buf, CD_SIZE_OFFSET, file.getCentralDirectorySize());
      longToLittleEndian(buf, CD_OFFSET_OFFSET, file.getCentralDirectoryOffset());
      return buf;
    }
  }

  static class Zip64EndOfCentralDirectoryLocator {
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
      if (readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
        throw new ZipException(
            "Unexpected end of file while reading Zip64 End of Central Directory Locator.");
      }
      if (!arrayStartsWith(fixedSizeData, intToLittleEndian(SIGNATURE))) {
        throw new ZipException(String.format(
            "Malformed Zip64 Central Directory Locator; does not start with %08x", SIGNATURE));
      }
      file.setZip64(true);
      file.setZip64EndOfCentralDirectoryOffset(
          getUnsignedLong(fixedSizeData, ZIP64_EOCD_OFFSET_OFFSET));
      return file;
    }

    /**
     * Generates the raw byte data of the Zip64 end of central directory locator for the file.
     */
    static byte[] create(ZipFileData file) {
      byte[] buf = new byte[FIXED_DATA_SIZE];
      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      intToLittleEndian(buf, ZIP64_EOCD_DISK_OFFSET, 0);
      longToLittleEndian(buf, ZIP64_EOCD_OFFSET_OFFSET, file.getZip64EndOfCentralDirectoryOffset());
      intToLittleEndian(buf, DISK_NUMBER_OFFSET, 1);
      return buf;
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
     * Read the end of central directory record from the input stream and parse {@link ZipFileData}
     * from it.
     */
    static ZipFileData read(InputStream in, ZipFileData file) throws IOException {
      if (file == null) {
        throw new NullPointerException();
      }

      byte[] fixedSizeData = new byte[FIXED_DATA_SIZE];
      if (readFully(in, fixedSizeData) != FIXED_DATA_SIZE) {
        throw new ZipException(
            "Unexpected end of file while reading End of Central Directory Record.");
      }
      if (!arrayStartsWith(fixedSizeData, intToLittleEndian(SIGNATURE))) {
        throw new ZipException(String.format(
            "Malformed End of Central Directory Record; does not start with %08x", SIGNATURE));
      }

      byte[] comment = new byte[getUnsignedShort(fixedSizeData, COMMENT_LENGTH_OFFSET)];
      if (comment.length > 0 && readFully(in, comment) != comment.length) {
        throw new ZipException(
            "Unexpected end of file while reading End of Central Directory Record.");
      }
      short diskNumber = get16(fixedSizeData, DISK_NUMBER_OFFSET);
      short centralDirectoryDisk = get16(fixedSizeData, CD_DISK_OFFSET);
      short entriesOnDisk = get16(fixedSizeData, DISK_ENTRIES_OFFSET);
      short totalEntries = get16(fixedSizeData, TOTAL_ENTRIES_OFFSET);
      int centralDirectorySize = get32(fixedSizeData, CD_SIZE_OFFSET);
      int centralDirectoryOffset = get32(fixedSizeData, CD_OFFSET_OFFSET);
      if (diskNumber == -1 || centralDirectoryDisk == -1 || entriesOnDisk == -1
          || totalEntries == -1 || centralDirectorySize == -1 || centralDirectoryOffset == -1) {
        file.setMaybeZip64(true);
      }
      file.setComment(comment);
      file.setCentralDirectorySize(getUnsignedInt(fixedSizeData, CD_SIZE_OFFSET));
      file.setCentralDirectoryOffset(getUnsignedInt(fixedSizeData, CD_OFFSET_OFFSET));
      file.setExpectedEntries(getUnsignedShort(fixedSizeData, TOTAL_ENTRIES_OFFSET));
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
      intToLittleEndian(buf, SIGNATURE_OFFSET, SIGNATURE);
      shortToLittleEndian(buf, DISK_NUMBER_OFFSET, (short) 0);
      shortToLittleEndian(buf, CD_DISK_OFFSET, (short) 0);
      shortToLittleEndian(buf, DISK_ENTRIES_OFFSET, numEntries);
      shortToLittleEndian(buf, TOTAL_ENTRIES_OFFSET, numEntries);
      intToLittleEndian(buf, CD_SIZE_OFFSET, cdSize);
      intToLittleEndian(buf, CD_OFFSET_OFFSET, cdOffset);
      shortToLittleEndian(buf, COMMENT_LENGTH_OFFSET, (short) comment.length);
      System.arraycopy(comment, 0, buf, FIXED_DATA_SIZE, comment.length);

      return buf;
    }
  }

  static class CentralDirectory {
    /**
     * Writes the central directory to an output stream for the specified {@link ZipFileData}.
     */
    static void write(ZipFileData file, boolean allowZip64, OutputStream stream)
        throws IOException {
      long directorySize = 0;
      byte[] buf = new byte[CentralDirectoryFileHeader.FIXED_DATA_SIZE];
      for (ZipFileEntry entry : file.getEntries()) {
        directorySize += CentralDirectoryFileHeader.write(entry, file, allowZip64, buf, stream);
      }
      file.setCentralDirectorySize(directorySize);
      if (file.isZip64() && allowZip64) {
        file.setZip64EndOfCentralDirectoryOffset(file.getCentralDirectoryOffset()
            + file.getCentralDirectorySize());
        stream.write(Zip64EndOfCentralDirectory.create(file));
        stream.write(Zip64EndOfCentralDirectoryLocator.create(file));
      }
      stream.write(EndOfCentralDirectoryRecord.create(file, allowZip64));
    }
  }
}
