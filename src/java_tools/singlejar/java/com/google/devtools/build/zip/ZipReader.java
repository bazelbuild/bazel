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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.BufferedInputStream;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.channels.Channels;
import java.nio.charset.Charset;
import java.util.Collection;
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

  private final File file;
  private final RandomAccessFile in;
  private final ZipFileData zipData;

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
   * @param charset the charset to use to decode the entry names and comments
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  public ZipReader(File file, Charset charset) throws IOException {
    this(file, charset, false);
  }

  /**
   * Opens a zip file for raw acceess.
   *
   * @param file the zip file
   * @param charset the charset to use to decode the entry names and comments
   * @param strictEntries force parsing to use the number of entries recorded in the end of
   *     central directory as the correct value, not as an estimate
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  public ZipReader(File file, Charset charset, boolean strictEntries) throws IOException {
    if (file == null || charset == null) {
      throw new NullPointerException();
    }
    this.file = file;
    this.in = new RandomAccessFile(file, "r");
    this.zipData = new ZipFileData(charset);
    readCentralDirectory(strictEntries);
  }

  /**
   * Returns the zip file's name.
   */
  public String getFilename() {
    return file.getName();
  }

  /**
   * Returns the ZIP file comment.
   */
  public String getComment() {
    return zipData.getComment();
  }

  /**
   * Returns a collection of the ZIP file entries.
   */
  public Collection<ZipFileEntry> entries() {
    return zipData.getEntries();
  }

  /**
   * Returns the ZIP file entry for the specified name, or null if not found.
   */
  public ZipFileEntry getEntry(String name) {
    return zipData.getEntry(name);
  }

  /**
   * Returns the number of entries in the ZIP file.
   */
  public long size() {
    return zipData.getNumEntries();
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
    if (!zipData.getEntry(entry.getName()).equals(entry)) {
      throw new ZipException(String.format(
          "Zip file '%s' does not contain the requested entry '%s'.", file.getName(),
          entry.getName()));
    }
    return new ZipEntryInputStream(this, entry, /* raw */ false);
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
    if (!zipData.getEntry(entry.getName()).equals(entry)) {
      throw new ZipException(String.format(
          "Zip file '%s' does not contain the requested entry '%s'.", file.getName(),
          entry.getName()));
    }
    return new ZipEntryInputStream(this, entry, /* raw */ true);
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
   * @param strictEntries force parsing to use the number of entries recorded in the end of
   *     central directory as the correct value, not as an estimate
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  private void readCentralDirectory(boolean strictEntries) throws IOException {
    long eocdLocation = findEndOfCentralDirectoryRecord();
    InputStream stream = getStreamAt(eocdLocation);
    EndOfCentralDirectoryRecord.read(stream, zipData);

    if (zipData.isMaybeZip64()) {
      try {
        stream = getStreamAt(eocdLocation - Zip64EndOfCentralDirectoryLocator.FIXED_DATA_SIZE);
        Zip64EndOfCentralDirectoryLocator.read(stream, zipData);

        stream = getStreamAt(zipData.getZip64EndOfCentralDirectoryOffset());
        Zip64EndOfCentralDirectory.read(stream, zipData);
      } catch (ZipException e) {
        // expected if not in Zip64 format
      }
    }

    if (zipData.isZip64() || strictEntries) {
      // If in Zip64 format or using strict entry numbers, use the parsed information as is to read
      // the central directory file headers.
      readCentralDirectoryFileHeaders(zipData.getExpectedEntries(),
          zipData.getCentralDirectoryOffset());
    } else {
      // If not in Zip64 format, compute central directory offset by end of central directory record
      // offset and central directory size to allow reading large non-compliant Zip32 directories.
      long centralDirectoryOffset = eocdLocation - zipData.getCentralDirectorySize();
      // If the lower 4 bytes match, the above calculation is correct; otherwise fallback to
      // reported offset.
      if ((int) centralDirectoryOffset == (int) zipData.getCentralDirectoryOffset()) {
        readCentralDirectoryFileHeaders(centralDirectoryOffset);
      } else {
        readCentralDirectoryFileHeaders(zipData.getExpectedEntries(),
            zipData.getCentralDirectoryOffset());
      }
    }
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
   * Finds the file offset of the end of central directory record.
   *
   * @return the file offset of the end of central directory record
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  private long findEndOfCentralDirectoryRecord() throws IOException {
    byte[] signature = ZipUtil.intToLittleEndian(EndOfCentralDirectoryRecord.SIGNATURE);
    byte[] buffer = new byte[(int) Math.min(64, in.length())];
    int readLength = buffer.length;
    if (readLength < EndOfCentralDirectoryRecord.FIXED_DATA_SIZE) {
      throw new ZipException(String.format("Zip file '%s' is malformed. It does not contain an end"
          + " of central directory record.", file.getName()));
    }

    long offset = in.length() - buffer.length;
    while (offset >= 0) {
      in.seek(offset);
      in.readFully(buffer, 0, readLength);
      int signatureLocation = scanBackwards(signature, buffer, buffer.length);
      while (signatureLocation != -1) {
        long eocdSize = in.length() - offset - signatureLocation;
        if (eocdSize >= EndOfCentralDirectoryRecord.FIXED_DATA_SIZE) {
          int commentLength = ZipUtil.getUnsignedShort(buffer, signatureLocation
              + EndOfCentralDirectoryRecord.COMMENT_LENGTH_OFFSET);
          long readCommentLength = eocdSize - EndOfCentralDirectoryRecord.FIXED_DATA_SIZE;
          if (commentLength == readCommentLength) {
            return offset + signatureLocation;
          }
        }
        signatureLocation = scanBackwards(signature, buffer, signatureLocation - 1);
      }
      readLength = buffer.length - 3;
      buffer[buffer.length - 3] = buffer[0];
      buffer[buffer.length - 2] = buffer[1];
      buffer[buffer.length - 1] = buffer[2];
      offset -= readLength;
    }
    throw new ZipException(String.format("Zip file '%s' is malformed. It does not contain an end"
        + " of central directory record.", file.getName()));
  }

  /**
   * Reads and parses ZIP file entries from the central directory.
   *
   * @param count the number of entries in the central directory
   * @param fileOffset the file offset of the start of the central directory
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  private void readCentralDirectoryFileHeaders(long count, long fileOffset) throws IOException {
    InputStream centralDirectory = getStreamAt(fileOffset);
    for (long i = 0; i < count; i++) {
      ZipFileEntry entry = CentralDirectoryFileHeader.read(centralDirectory, zipData.getCharset());
      zipData.addEntry(entry);
    }
  }

  /**
   * Reads and parses ZIP file entries from the central directory.
   *
   * @param fileOffset the file offset of the start of the central directory
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  private void readCentralDirectoryFileHeaders(long fileOffset) throws IOException {
    CountingInputStream centralDirectory = new CountingInputStream(getStreamAt(fileOffset));
    while (centralDirectory.getCount() < zipData.getCentralDirectorySize()) {
      ZipFileEntry entry = CentralDirectoryFileHeader.read(centralDirectory, zipData.getCharset());
      zipData.addEntry(entry);
    }
  }

  /**
   * Returns a new {@link InputStream} positioned at fileOffset.
   *
   * @throws IOException if an I/O error has occurred
   */
  protected InputStream getStreamAt(long fileOffset) throws IOException {
    return new BufferedInputStream(Channels.newInputStream(in.getChannel().position(fileOffset)));
  }
}
