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

package com.google.devtools.build.singlejar;


import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * A helper class to validate zip files and provide reasonable diagnostics (better than what zip
 * does). We might want to make this into a fully-fledged binary some day.
 */
final class ZipTester {

  // The following constants are ZIP-specific.
  private static final int LOCAL_FILE_HEADER_MARKER = 0x04034b50;
  private static final int DATA_DESCRIPTOR_MARKER = 0x08074b50;
  private static final int CENTRAL_DIRECTORY_MARKER = 0x02014b50;
  private static final int END_OF_CENTRAL_DIRECTORY_MARKER = 0x06054b50;

  private static final int FILE_HEADER_BUFFER_SIZE = 26; // without marker
  private static final int DATA_DESCRIPTOR_BUFFER_SIZE = 12; // without marker

  private static final int DIRECTORY_ENTRY_BUFFER_SIZE = 42; // without marker
  private static final int END_OF_CENTRAL_DIRECTORY_BUFFER_SIZE = 18; // without marker

  // Set if the size, compressed size and CRC are set to zero, and present in
  // the data descriptor after the data.
  private static final int SIZE_MASKED_FLAG = 1 << 3;

  private static final int STORED_METHOD = 0;
  private static final int DEFLATE_METHOD = 8;

  private static final int VERSION_STORED = 10; // Version 1.0
  private static final int VERSION_DEFLATE = 20; // Version 2.0

  private static class Entry {
    private final long pos;
    private final String name;
    private final int flags;
    private final int method;
    private final int dosTime;
    Entry(long pos, String name, int flags, int method, int dosTime) {
      this.pos = pos;
      this.name = name;
      this.flags = flags;
      this.method = method;
      this.dosTime = dosTime;
    }
  }

  private final InputStream in;
  private final byte[] buffer = new byte[1024];
  private int bufferLength;
  private int bufferOffset;
  private long pos;

  private List<Entry> entries = new ArrayList<Entry>();

  public ZipTester(InputStream in) {
    this.in = in;
  }

  public ZipTester(byte[] data) {
    this(new ByteArrayInputStream(data));
  }

  private void warn(String msg) {
    System.err.println("WARNING: " + msg);
  }

  private void readMoreData(String action) throws IOException {
    if ((bufferLength > 0) && (bufferOffset > 0)) {
      System.arraycopy(buffer, bufferOffset, buffer, 0, bufferLength);
    }
    if (bufferLength >= buffer.length) {
      // The buffer size is specifically chosen to avoid this situation.
      throw new AssertionError("Internal error: buffer overrun.");
    }
    bufferOffset = 0;
    int bytesRead = in.read(buffer, bufferLength, buffer.length - bufferLength);
    if (bytesRead <= 0) {
      throw new IOException("Unexpected end of file, while " + action);
    }
    bufferLength += bytesRead;
  }

  private int readByte(String action) throws IOException {
    if (bufferLength == 0) {
      readMoreData(action);
    }
    byte result = buffer[bufferOffset];
    bufferOffset++; bufferLength--;
    pos++;
    return result & 0xff;
  }

  private long getUnsignedInt(String action) throws IOException {
    int a = readByte(action);
    int b = readByte(action);
    int c = readByte(action);
    int d = readByte(action);
    return ((d << 24) | (c << 16) | (b << 8) | a) & 0xffffffffL;
  }

  private long getUnsignedInt(byte[] source, int offset) {
    int a = source[offset + 0] & 0xff;
    int b = source[offset + 1] & 0xff;
    int c = source[offset + 2] & 0xff;
    int d = source[offset + 3] & 0xff;
    return ((d << 24) | (c << 16) | (b << 8) | a) & 0xffffffffL;
  }

  private void readFully(byte[] buffer, String action) throws IOException {
    for (int i = 0; i < buffer.length; i++) {
      buffer[i] = (byte) readByte(action);
    }
  }

  private void skip(long length, String action) throws IOException {
    for (long i = 0; i < length; i++) {
      readByte(action);
    }
  }

  private int getUnsignedShort(byte[] source, int offset) {
    int a = source[offset + 0] & 0xff;
    int b = source[offset + 1] & 0xff;
    return (b << 8) | a;
  }

  private class DeflateInputStream extends InputStream {

    private final byte[] singleByteBuffer = new byte[1];
    private int consumedBytes;
    private final Inflater inflater = new Inflater(true);
    private long totalBytesRead;

    private int inflateData(byte[] dest, int off, int len)
        throws IOException {
      consumedBytes = 0;
      int bytesProduced = 0;
      int bytesConsumed = 0;
      while ((bytesProduced == 0) && !inflater.finished()) {
        inflater.setInput(buffer, bufferOffset + bytesConsumed, bufferLength - bytesConsumed);
        int remainingBefore = inflater.getRemaining();
        try {
          bytesProduced = inflater.inflate(dest, off, len);
        } catch (DataFormatException e) {
          throw new IOException("Invalid deflate stream in ZIP file.", e);
        }
        bytesConsumed += remainingBefore - inflater.getRemaining();
        consumedBytes = bytesConsumed;
        if (bytesProduced == 0) {
          if (inflater.needsDictionary()) {
            // The DEFLATE algorithm as used in the ZIP file format does not
            // require an additional dictionary.
            throw new AssertionError("Inflater unexpectedly requires a dictionary.");
          } else if (inflater.needsInput()) {
            readMoreData("need more data for deflate");
          } else if (inflater.finished()) {
            return 0;
          } else {
            // According to the Inflater specification, this cannot happen.
            throw new AssertionError("Inflater unexpectedly produced no output.");
          }
        }
      }
      return bytesProduced;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
      if (inflater.finished()) {
        return -1;
      }
      int length = inflateData(b, off, len);
      totalBytesRead += consumedBytes;
      bufferLength -= consumedBytes;
      bufferOffset += consumedBytes;
      pos += consumedBytes;
      return length == 0 ? -1 : length;
    }

    @Override
    public int read() throws IOException {
      int bytesRead = read(singleByteBuffer, 0, 1);
      return (bytesRead == -1) ? -1 : (singleByteBuffer[0] & 0xff);
    }
  }

  private void readEntry() throws IOException {
    long entrypos = pos - 4;
    String entryDesc = "file entry at " + Long.toHexString(entrypos);
    byte[] entryBuffer = new byte[FILE_HEADER_BUFFER_SIZE];
    readFully(entryBuffer, "reading file header");
    int versionToExtract = getUnsignedShort(entryBuffer, 0);
    int flags = getUnsignedShort(entryBuffer, 2);
    int method = getUnsignedShort(entryBuffer, 4);
    int dosTime = (int) getUnsignedInt(entryBuffer, 6);
    int crc32 = (int) getUnsignedInt(entryBuffer, 10);
    long compressedSize = getUnsignedInt(entryBuffer, 14);
    long uncompressedSize = getUnsignedInt(entryBuffer, 18);
    int filenameLength = getUnsignedShort(entryBuffer, 22);
    int extraLength = getUnsignedShort(entryBuffer, 24);

    byte[] filename = new byte[filenameLength];
    readFully(filename, "reading file name");
    skip(extraLength, "skipping extra data");

    String name = new String(filename, "UTF-8");
    for (int i = 0; i < filename.length; i++) {
      if ((filename[i] < ' ')) {
        warn(entryDesc + ": file name has unexpected non-ascii characters");
      }
    }
    entryDesc = "file entry '" + name + "' at " + Long.toHexString(entrypos);

    if ((method != STORED_METHOD) && (method != DEFLATE_METHOD)) {
      throw new IOException(entryDesc + ": unknown method " + method);
    }
    if ((flags != 0) && (flags != SIZE_MASKED_FLAG)) {
      throw new IOException(entryDesc + ": unknown flags " + flags);
    }
    if ((method == STORED_METHOD) && (versionToExtract != VERSION_STORED)) {
      warn(entryDesc + ": unexpected version to extract for stored entry " + versionToExtract);
    }
    if ((method == DEFLATE_METHOD) && (versionToExtract != VERSION_DEFLATE)) {
//      warn(entryDesc + ": unexpected version to extract for deflated entry " + versionToExtract);
    }

    if (method == STORED_METHOD) {
      if (compressedSize != uncompressedSize) {
        throw new IOException(entryDesc + ": stored entries should have identical compressed and "
            + "uncompressed sizes");
      }
      skip(compressedSize, entryDesc + "skipping data");
    } else {
      // No OS resources are actually allocated.
      @SuppressWarnings("resource") DeflateInputStream deflater = new DeflateInputStream();
      long generatedBytes = 0;
      byte[] deflated = new byte[1024];
      int readBytes;
      CRC32 crc = new CRC32();
      while ((readBytes = deflater.read(deflated)) > 0) {
        crc.update(deflated, 0, readBytes);
        generatedBytes += readBytes;
      }
      int actualCrc32 = (int) crc.getValue();
      long consumedBytes = deflater.totalBytesRead;
      if (flags == SIZE_MASKED_FLAG) {
        long id = getUnsignedInt("reading footer marker");
        if (id != DATA_DESCRIPTOR_MARKER) {
          throw new IOException(entryDesc + ": expected footer at " + Long.toHexString(pos - 4)
              + ", but found " + Long.toHexString(id));
        }
        byte[] footer = new byte[DATA_DESCRIPTOR_BUFFER_SIZE];
        readFully(footer, "reading footer");
        crc32 = (int) getUnsignedInt(footer, 0);
        compressedSize = getUnsignedInt(footer, 4);
        uncompressedSize = getUnsignedInt(footer, 8);
      }

      if (consumedBytes != compressedSize) {
        throw new IOException(entryDesc + ": amount of compressed data does not match value "
            + "specified in the zip (specified: " + compressedSize + ", actual: " + consumedBytes
            + ")");
      }
      if (generatedBytes != uncompressedSize) {
        throw new IOException(entryDesc + ": amount of uncompressed data does not match value "
            + "specified in the zip (specified: " + uncompressedSize + ", actual: "
            + generatedBytes + ")");
      }
      if (crc32 != actualCrc32) {
        throw new IOException(entryDesc + ": specified crc checksum does not match actual check "
            + "sum");
      }
    }
    entries.add(new Entry(entrypos, name, flags, method, dosTime));
  }

  @SuppressWarnings("unused") // A couple of unused local variables.
  private void validateCentralDirectoryEntry(Entry entry) throws IOException {
    long entrypos = pos - 4;
    String entryDesc = "file directory entry '" + entry.name + "' at " + Long.toHexString(entrypos);

    byte[] entryBuffer = new byte[DIRECTORY_ENTRY_BUFFER_SIZE];
    readFully(entryBuffer, "reading central directory entry");
    int versionMadeBy = getUnsignedShort(entryBuffer, 0);
    int versionToExtract = getUnsignedShort(entryBuffer, 2);
    int flags = getUnsignedShort(entryBuffer, 4);
    int method = getUnsignedShort(entryBuffer, 6);
    int dosTime = (int) getUnsignedInt(entryBuffer, 8);
    int crc32 = (int) getUnsignedInt(entryBuffer, 12);
    long compressedSize = getUnsignedInt(entryBuffer, 16);
    long uncompressedSize = getUnsignedInt(entryBuffer, 20);
    int filenameLength = getUnsignedShort(entryBuffer, 24);
    int extraLength = getUnsignedShort(entryBuffer, 26);
    int commentLength = getUnsignedShort(entryBuffer, 28);
    int diskNumberStart = getUnsignedShort(entryBuffer, 30);
    int internalAttributes = getUnsignedShort(entryBuffer, 32);
    int externalAttributes = (int) getUnsignedInt(entryBuffer, 34);
    long offset = getUnsignedInt(entryBuffer, 38);

    byte[] filename = new byte[filenameLength];
    readFully(filename, "reading file name");
    skip(extraLength, "skipping extra data");
    String name = new String(filename, "UTF-8");

    if (!name.equals(entry.name)) {
      throw new IOException(entryDesc + ": file name in central directory does not match original "
          + "name");
    }
    if (offset != entry.pos) {
      throw new IOException(entryDesc);
    }
    if (flags != entry.flags) {
      throw new IOException(entryDesc);
    }
    if (method != entry.method) {
      throw new IOException(entryDesc);
    }
    if (dosTime != entry.dosTime) {
      throw new IOException(entryDesc);
    }
  }

  private void validateCentralDirectory() throws IOException {
    boolean first = true;
    for (Entry entry : entries) {
      if (first) {
        first = false;
      } else {
        long id = getUnsignedInt("reading marker");
        if (id != CENTRAL_DIRECTORY_MARKER) {
          throw new IOException();
        }
      }
      validateCentralDirectoryEntry(entry);
    }
  }

  @SuppressWarnings("unused") // A couple of unused local variables.
  private void validateEndOfCentralDirectory() throws IOException {
    long id = getUnsignedInt("expecting end of central directory");
    byte[] entryBuffer = new byte[END_OF_CENTRAL_DIRECTORY_BUFFER_SIZE];
    readFully(entryBuffer, "reading end of central directory");
    int diskNumber = getUnsignedShort(entryBuffer, 0);
    int startDiskNumber = getUnsignedShort(entryBuffer, 2);
    int numEntries = getUnsignedShort(entryBuffer, 4);
    int numTotalEntries = getUnsignedShort(entryBuffer, 6);
    long centralDirectorySize = getUnsignedInt(entryBuffer, 8);
    long centralDirectoryOffset = getUnsignedInt(entryBuffer, 12);
    int commentLength = getUnsignedShort(entryBuffer, 16);
    if (diskNumber != 0) {
      throw new IOException(String.format("diskNumber=%d", diskNumber));
    }
    if (startDiskNumber != 0) {
      throw new IOException(String.format("startDiskNumber=%d", diskNumber));
    }
    if (numEntries != numTotalEntries) {
      throw new IOException(String.format("numEntries=%d numTotalEntries=%d",
                                          numEntries, numTotalEntries));
    }
    if (numEntries != (entries.size() % 0x10000)) {
      throw new IOException("bad number of entries in central directory footer");
    }
    if (numTotalEntries != (entries.size() % 0x10000)) {
      throw new IOException("bad number of entries in central directory footer");
    }
    if (commentLength != 0) {
      throw new IOException("Zip file comment is unexpected");
    }
    if (id != END_OF_CENTRAL_DIRECTORY_MARKER) {
      throw new IOException("Expected end of central directory marker");
    }
  }

  public void validate() throws IOException {
    while (true) {
      long id = getUnsignedInt("reading marker");
      if (id == LOCAL_FILE_HEADER_MARKER) {
        readEntry();
      } else if (id == CENTRAL_DIRECTORY_MARKER) {
        validateCentralDirectory();
        validateEndOfCentralDirectory();
        return;
      } else {
        throw new IOException("unexpected result for marker: "
            + Long.toHexString(id) + " at position " + Long.toHexString(pos - 4));
      }
    }
  }
}
