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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.zip.ZipFileEntry.Compression;
import com.google.devtools.build.zip.ZipUtil.CentralDirectoryFileHeader;
import com.google.devtools.build.zip.ZipUtil.EndOfCentralDirectoryRecord;
import com.google.devtools.build.zip.ZipUtil.LocalFileHeader;

import java.io.BufferedInputStream;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.channels.Channels;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

/**
 * A ZIP file reader.
 *
 * <p>This class provides entry data in the form of {@link ZipFileEntry}, which provides more detail
 * about the entry than the JDK equivalent {@link ZipEntry}. In addition to providing
 * {@link InputStream}s for entries, similar to JDK {@link ZipFile#getInputStream(ZipEntry)}, it
 * also provides access to the raw byte entry data via {@link #getRawInputStream(ZipFileEntry)}.
 *
 * <p>Using the raw access capabilities allows for more efficient ZIP file processing, such as
 * merging, by not requiring each entry's data to be decompressed when read.
 *
 * <p><em>NOTE:</em> The entries are read from the central directory. If the entry is not listed
 * there, it will not be returned from {@link #entries()} or {@link #getEntry(String)}.
 */
public class ZipReader implements Closeable, AutoCloseable {

  /** An input stream for reading the file data of a ZIP file entry. */
  private class ZipEntryInputStream extends InputStream {
    private InputStream stream;
    private long rem;

    /**
     * Opens an input stream for reading at the beginning of the ZIP file entry's content.
     *
     * @param zipEntry the ZIP file entry to open the input stream for
     * @throws ZipException if a ZIP format error has occurred
     * @throws IOException if an I/O error has occurred
     */
    private ZipEntryInputStream(ZipFileEntry zipEntry) throws IOException {
      stream = new BufferedInputStream(Channels.newInputStream(
          in.getChannel().position(zipEntry.getLocalHeaderOffset())));

      byte[] fileHeader = new byte[LocalFileHeader.FIXED_DATA_SIZE];
      stream.read(fileHeader);

      if (!ZipUtil.arrayStartsWith(fileHeader,
          ZipUtil.intToLittleEndian(LocalFileHeader.SIGNATURE))) {
        throw new ZipException(String.format("The file '%s' is not a correctly formatted zip file: "
            + "Expected a File Header at file offset %d, but was not present.",
            file.getName(), zipEntry.getLocalHeaderOffset()));
      }

      int nameLength = ZipUtil.getUnsignedShort(fileHeader,
          LocalFileHeader.FILENAME_LENGTH_OFFSET);
      int extraFieldLength = ZipUtil.getUnsignedShort(fileHeader,
          LocalFileHeader.EXTRA_FIELD_LENGTH_OFFSET);
      stream.skip(nameLength + extraFieldLength);
      rem = zipEntry.getSize();
      if (zipEntry.getMethod() == Compression.DEFLATED) {
        stream = new InflaterInputStream(stream, new Inflater(true));
      }
    }

    @Override public int available() throws IOException {
      return rem > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) rem;
    }

    @Override public void close() throws IOException {
    }

    @Override public void mark(int readlimit) {
    }

    @Override public boolean markSupported() {
      return false;
    }

    @Override public int read() throws IOException {
      byte[] b = new byte[1];
      if (read(b, 0, 1) == 1) {
        return b[0] & 0xff;
      } else {
        return -1;
      }
    }

    @Override public int read(byte[] b) throws IOException {
      return read(b, 0, b.length);
    }

    @Override public int read(byte[] b, int off, int len) throws IOException {
      if (rem == 0) {
        return -1;
      }
      if (len > rem) {
        len = available();
      }
      len = stream.read(b, off, len);
      rem -= len;
      return len;
    }

    @Override public long skip(long n) throws IOException {
      if (n > rem) {
        n = rem;
      }
      n = stream.skip(n);
      rem -= n;
      return n;
    }

    @Override public void reset() throws IOException {
      throw new IOException("Reset is not supported on this type of stream.");
    }
  }

  /** An input stream for reading the raw file data of a ZIP file entry. */
  private class RawZipEntryInputStream extends InputStream {
    private InputStream stream;
    private long rem;

    /**
     * Opens an input stream for reading at the beginning of the ZIP file entry's content.
     *
     * @param zipEntry the ZIP file entry to open the input stream for
     * @throws ZipException if a ZIP format error has occurred
     * @throws IOException if an I/O error has occurred
     */
    private RawZipEntryInputStream(ZipFileEntry zipEntry) throws IOException {
      stream = new BufferedInputStream(Channels.newInputStream(
          in.getChannel().position(zipEntry.getLocalHeaderOffset())));

      byte[] fileHeader = new byte[LocalFileHeader.FIXED_DATA_SIZE];
      stream.read(fileHeader);

      if (!ZipUtil.arrayStartsWith(fileHeader,
          ZipUtil.intToLittleEndian(LocalFileHeader.SIGNATURE))) {
        throw new ZipException(String.format("The file '%s' is not a correctly formatted zip file: "
            + "Expected a File Header at file offset %d, but was not present.",
            file.getName(), zipEntry.getLocalHeaderOffset()));
      }

      int nameLength = ZipUtil.getUnsignedShort(fileHeader,
          LocalFileHeader.FILENAME_LENGTH_OFFSET);
      int extraFieldLength = ZipUtil.getUnsignedShort(fileHeader,
          LocalFileHeader.EXTRA_FIELD_LENGTH_OFFSET);
      stream.skip(nameLength + extraFieldLength);
      rem = zipEntry.getCompressedSize();
    }

    @Override public int available() throws IOException {
      return rem > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) rem;
    }

    @Override public void close() throws IOException {
    }

    @Override public void mark(int readlimit) {
    }

    @Override public boolean markSupported() {
      return false;
    }

    @Override public int read() throws IOException {
      byte[] b = new byte[1];
      if (read(b, 0, 1) == 1) {
        return b[0] & 0xff;
      } else {
        return -1;
      }
    }

    @Override public int read(byte[] b) throws IOException {
      return read(b, 0, b.length);
    }

    @Override public int read(byte[] b, int off, int len) throws IOException {
      if (rem == 0) {
        return -1;
      }
      if (len > rem) {
        len = available();
      }
      len = stream.read(b, off, len);
      rem -= len;
      return len;
    }

    @Override public long skip(long n) throws IOException {
      if (n > rem) {
        n = rem;
      }
      n = stream.skip(n);
      rem -= n;
      return n;
    }

    @Override public void reset() throws IOException {
      throw new IOException("Reset is not supported on this type of stream.");
    }
  }

  private File file;
  private Charset charset;
  private RandomAccessFile in;
  private Map<String, ZipFileEntry> zipEntries;
  private String comment;

  /**
   * Opens a zip file for raw acceess.
   *
   * <p>The UTF-8 charset is used to decode the entry names and comments.
   *
   * @param file the zip file
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  public ZipReader(File file) throws IOException {
    this(file, UTF_8);
  }

  /**
   * Opens a zip file for raw acceess.
   *
   * @param file the zip file
   * @param charset the charset to use to decode the entry names and comments.
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  public ZipReader(File file, Charset charset) throws IOException {
    if (file == null || charset == null) {
      throw new NullPointerException();
    }
    this.file = file;
    this.charset = charset;
    this.in = new RandomAccessFile(file, "r");
    this.zipEntries = readCentralDirectory();
  }

  /**
   * Returns the ZIP file comment.
   *
   * @return the ZIP file comment
   */
  public String getComment() {
    return comment;
  }

  /**
   * Returns a collection of the ZIP file entries.
   *
   * @return a collection of the ZIP file entries
   */
  public Collection<ZipFileEntry> entries() {
    return zipEntries.values();
  }

  /**
   * Returns the ZIP file entry for the specified name, or null if not found.
   *
   * @param name the name of the entry
   * @return the ZIP file entry, or null if not found
   */
  public ZipFileEntry getEntry(String name) {
    if (zipEntries.containsKey(name)) {
      return zipEntries.get(name);
    } else {
      return null;
    }
  }

  /**
   * Returns an input stream for reading the contents of the specified ZIP file entry.
   *
   * <p>Closing this ZIP file will, in turn, close all input streams that have been returned by
   * invocations of this method.
   *
   * @param entry the ZIP file entry
   * @return the input stream for reading the contents of the specified zip file entry
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  public InputStream getInputStream(ZipFileEntry entry) throws IOException {
    if (!zipEntries.get(entry.getName()).equals(entry)) {
      throw new ZipException(String.format(
          "Zip file '%s' does not contain the requested entry '%s'.", file.getName(),
          entry.getName()));
    }
    return new ZipEntryInputStream(entry);
  }

  /**
   * Returns an input stream for reading the raw contents of the specified ZIP file entry.
   *
   * <p><em>NOTE:</em> No inflating will take place; The data read from the input stream will be
   * the exact byte content of the ZIP file entry on disk.
   *
   * <p>Closing this ZIP file will, in turn, close all input streams that have been returned by
   * invocations of this method.
   *
   * @param entry the ZIP file entry
   * @return the input stream for reading the contents of the specified zip file entry
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  public InputStream getRawInputStream(ZipFileEntry entry) throws IOException {
    if (!zipEntries.get(entry.getName()).equals(entry)) {
      throw new ZipException(String.format(
          "Zip file '%s' does not contain the requested entry '%s'.", file.getName(),
          entry.getName()));
    }
    return new RawZipEntryInputStream(entry);
  }

  /**
   * Closes the ZIP file.
   *
   * <p>Closing this ZIP file will close all of the input streams previously returned by invocations
   * of the {@link #getRawInputStream(ZipFileEntry)} method.
   */
  @Override public void close() throws IOException {
    in.close();
  }

  /**
   * Finds, reads and parses ZIP file entries from the central directory.
   *
   * @return a map of all ZIP file entries read from the central directory and their names
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  private Map<String, ZipFileEntry> readCentralDirectory() throws IOException {
    byte[] eocdRecord = readEndOfCentralDirectoryRecord();

    int commentLength = ZipUtil.getUnsignedShort(eocdRecord,
        EndOfCentralDirectoryRecord.COMMENT_LENGTH_OFFSET);
    this.comment = new String(Arrays.copyOfRange(eocdRecord,
        EndOfCentralDirectoryRecord.FIXED_DATA_SIZE,
        EndOfCentralDirectoryRecord.FIXED_DATA_SIZE + commentLength), charset);

    int totalEntries = ZipUtil.getUnsignedShort(eocdRecord,
        EndOfCentralDirectoryRecord.TOTAL_ENTRIES_OFFSET);
    long cdOffset = ZipUtil.getUnsignedInt(eocdRecord,
        EndOfCentralDirectoryRecord.CD_OFFSET_OFFSET);

    return readCentralDirectoryFileHeaders(totalEntries, cdOffset);
  }

  /**
   * Looks for the target sub array in the buffer scanning backwards starting at offset. Returns the
   * index where the target is found or -1 if not found.
   *
   * @param target the sub array to find
   * @param buffer the array to scan
   * @param offset the index of where to begin scanning
   * @return the index of target within buffer or -1 if not found
   */
  private int scanBackwards(byte[] target, byte[] buffer, int offset) {
    int start = Math.min(offset, buffer.length - target.length);
    for (int i = start; i >= 0; i--) {
      for (int j = 0; j < target.length; j++) {
        if (buffer[i + j] != target[j]) {
          break;
        } else if (j == target.length - 1) {
          return i;
        }
      }
    }
    return -1;
  }

  /**
   * Finds and returns the byte array of the end of central directory record.
   *
   * @return the byte array of the end of central directory record
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  private byte[] readEndOfCentralDirectoryRecord() throws IOException {
    byte[] signature = ZipUtil.intToLittleEndian(EndOfCentralDirectoryRecord.SIGNATURE);
    byte[] buffer = new byte[(int) Math.min(64, in.length())];

    int bytesRead = 0;
    while (true) {
      in.seek(in.length() - buffer.length);
      in.readFully(buffer, 0, buffer.length - bytesRead);

      int signatureLocation = scanBackwards(signature, buffer, buffer.length - bytesRead - 1);
      while (signatureLocation != -1) {
        int eocdSize = buffer.length - signatureLocation;
        if (eocdSize >= EndOfCentralDirectoryRecord.FIXED_DATA_SIZE) {
          int commentLength = ZipUtil.getUnsignedShort(buffer, signatureLocation
              + EndOfCentralDirectoryRecord.COMMENT_LENGTH_OFFSET);
          int readCommentLength = buffer.length - signatureLocation
              - EndOfCentralDirectoryRecord.FIXED_DATA_SIZE;
          if (commentLength == readCommentLength) {
            byte[] eocdRecord = new byte[eocdSize];
            System.arraycopy(buffer, signatureLocation, eocdRecord, 0, eocdSize);
            return eocdRecord;
          }
        }
        signatureLocation = scanBackwards(signature, buffer, signatureLocation - 1);
      }
      // expand buffer
      bytesRead = buffer.length;
      int newLength = (int) Math.min(buffer.length * 2, in.length());
      if (newLength == buffer.length) {
        break;
      }
      byte[] newBuf = new byte[newLength];
      System.arraycopy(buffer, 0, newBuf, newBuf.length - buffer.length, buffer.length);
      buffer = newBuf;
    }
    throw new ZipException(String.format("Zip file '%s' is malformed. It does not contain an end"
        + " of central directory record.", file.getName()));
  }

  /**
   * Reads and parses ZIP file entries from the central directory.
   *
   * @param count the number of entries in the central directory
   * @param fileOffset the file offset of the start of the central directory
   * @return a map of all ZIP file entries read from the central directory and their names
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  private Map<String, ZipFileEntry> readCentralDirectoryFileHeaders(int count, long fileOffset)
      throws IOException {

    InputStream centralDirectory = new BufferedInputStream(
        Channels.newInputStream(in.getChannel().position(fileOffset)));

    Map<String, ZipFileEntry> entries = new LinkedHashMap<>(count);
    for (int i = 0; i < count; i++) {
      ZipFileEntry entry = CentralDirectoryFileHeader.read(centralDirectory, charset);
      entries.put(entry.getName(), entry);
    }
    return entries;
  }
}
