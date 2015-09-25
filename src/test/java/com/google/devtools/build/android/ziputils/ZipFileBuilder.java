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
package com.google.devtools.build.android.ziputils;

import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTLEN;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENFLG;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENOFF;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENTIM;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCFLG;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCLEN;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCSIZ;
import static com.google.devtools.build.android.ziputils.LocalFileHeader.LOCTIM;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * Zip file builder for testing, For now it only supports building
 * valid zip files.
 */
class ZipFileBuilder {
  private final List<FileInfo> input;
  private static final Charset CHARSET = Charset.forName("UTF-8");
  private static final byte[] EMPTY = {};

  public ZipFileBuilder() {
    input = new ArrayList<>();
  }

  public ZipFileBuilder add(String filename, String content) {
    input.add(new FileInfo(filename, content.getBytes(Charset.defaultCharset())));
    return this;
  }

  public ZipFileBuilder add(FileInfo fileInfo) {
    input.add(fileInfo);
    return this;
  }

  public void create(String filename) throws IOException {
    ZipOut out = new ZipOut(FileSystem.fileSystem().getOutputChannel(filename, false), filename);
    for (FileInfo info : input) {
      int compressed = info.compressedSize();
      int uncompressed = info.uncompressedSize();
      int dirCompressed = info.dirCompressedSize();
      int dirUncompressed = info.dirUncompressedSize();
      short flags = info.flags();
      DirectoryEntry entry = DirectoryEntry.allocate(info.name, info.extra, info.comment);
      out.nextEntry(entry)
          .set(CENOFF, out.fileOffset())
          .set(CENFLG, flags)
          .set(CENTIM, info.date)
          .set(CENLEN, dirUncompressed)
          .set(CENSIZ, dirCompressed);
      LocalFileHeader header = LocalFileHeader.allocate(info.name, null)
          .set(LOCFLG, flags)
          .set(LOCTIM, info.date)
          .set(LOCLEN, uncompressed)
          .set(LOCSIZ, compressed);
      out.write(header);
      out.write(ByteBuffer.wrap(info.data));
      if (flags != 0) {
        DataDescriptor desc = DataDescriptor.allocate()
            .set(EXTLEN, dirUncompressed)
            .set(EXTSIZ, dirCompressed);
        out.write(desc);
      }
    }
    out.close();
  }

  public static class FileInfo {
    private final String name;
    private final short method;
    private final int date;
    private final int uncompressed;
    private final byte[] data;
    private final byte[] extra;
    private final String comment;
    boolean maskSize;

    static final short STORED = 0;
    static final short DEFLATED = 8;

    public FileInfo(String filename, String content) {
      this(filename, DosTime.EPOCH.time, STORED, 0,
          (content == null ? EMPTY : content.getBytes(CHARSET)), null, null);
    }

    public FileInfo(String filename, byte[] data) {
      this(filename, DosTime.EPOCH.time, STORED, 0, data, null, null);
    }

    public FileInfo(String filename, byte[] data, int uncompressed) {
      this(filename, DosTime.EPOCH.time, DEFLATED, uncompressed, data, null, null);
    }

    public FileInfo(String filename, int dosTime, String content) {
      this(filename, dosTime, STORED, 0,
          (content == null ? EMPTY : content.getBytes(CHARSET)), null, null);
    }

    public FileInfo(String filename, int dosTime, byte[] data) {
      this(filename, dosTime, STORED, 0, data, null, null);
    }

    public FileInfo(String filename, int dosTime, byte[] data, int uncompressed) {
      this(filename, dosTime, DEFLATED, uncompressed, data, null, null);
    }

    public FileInfo(String filename, int dosTime, short method, int uncompressed,
        byte[] content, byte[] extra, String comment) {
      this.name = filename;
      this.date = dosTime;
      this.method = method;
      this.uncompressed = uncompressed;
      this.data = content;
      this.extra = extra;
      this.comment = comment;
      maskSize = false;
    }

    public void setMaskSize(boolean ignore) {
      maskSize = ignore;
    }

    int compressedSize() {
      return method != 0 && !maskSize ? data.length : 0;
    }

    int uncompressedSize() {
      return method == 0 ? data.length : maskSize ? 0 : uncompressed;
    }

    int dirCompressedSize() {
      return method == 0 ? 0 : data.length;
    }

    int dirUncompressedSize() {
      return method == 0 ? data.length : uncompressed;
    }

    short flags() {
      return method != 0 && uncompressed == 0 ? LocalFileHeader.SIZE_MASKED_FLAG : 0;
    }
  }
}
