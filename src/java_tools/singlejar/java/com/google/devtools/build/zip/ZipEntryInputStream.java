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

import java.io.IOException;
import java.io.InputStream;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;
import java.util.zip.ZipException;

/** An input stream for reading the file data of a ZIP file entry. */
class ZipEntryInputStream extends InputStream {
  private InputStream stream;
  private long rem;

  /**
   * Opens an input stream for reading at the beginning of the ZIP file entry's content.
   *
   * @param zipReader the backing ZIP reader for this InputStream
   * @param zipEntry the ZIP file entry to open the input stream for
   * @param raw if the entry should be opened for raw read mode
   * @throws ZipException if a ZIP format error has occurred
   * @throws IOException if an I/O error has occurred
   */
  ZipEntryInputStream(ZipReader zipReader, ZipFileEntry zipEntry, boolean raw)
      throws IOException {
    stream = zipReader.getStreamAt(zipEntry.getLocalHeaderOffset());

    byte[] fileHeader = new byte[LocalFileHeader.FIXED_DATA_SIZE];
    ZipUtil.readFully(stream, fileHeader);

    if (!ZipUtil.arrayStartsWith(fileHeader,
        ZipUtil.intToLittleEndian(LocalFileHeader.SIGNATURE))) {
      throw new ZipException(String.format("The file '%s' is not a correctly formatted zip file: "
          + "Expected a File Header at file offset %d, but was not present.",
          zipReader.getFilename(), zipEntry.getLocalHeaderOffset()));
    }

    int nameLength = ZipUtil.getUnsignedShort(fileHeader,
        LocalFileHeader.FILENAME_LENGTH_OFFSET);
    int extraFieldLength = ZipUtil.getUnsignedShort(fileHeader,
        LocalFileHeader.EXTRA_FIELD_LENGTH_OFFSET);
    ZipUtil.readFully(stream, new byte[nameLength + extraFieldLength]);
    if (raw) {
      rem = zipEntry.getCompressedSize();
    } else {
      rem = zipEntry.getSize();
    }
    if (!raw && zipEntry.getMethod() == Compression.DEFLATED) {
      stream = new InflaterInputStream(stream, new Inflater(true));
    }
  }

  @Override public int available() throws IOException {
    return (int) Math.min(rem, Integer.MAX_VALUE);
  }

  @Override public void close() throws IOException {
  }

  @Override public synchronized void mark(int readlimit) {
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

  @Override public synchronized void reset() throws IOException {
    throw new IOException("Reset is not supported on this type of stream.");
  }
}
