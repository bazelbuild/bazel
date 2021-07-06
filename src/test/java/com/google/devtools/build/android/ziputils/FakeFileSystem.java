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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.util.HashMap;
import java.util.Map;

/**
 * Simple in-memory test file system.
 */
class FakeFileSystem extends FileSystem  {

  private final Map<String, byte[]> files = new HashMap<>();

  public FakeFileSystem() {
    FileSystem.fileSystem = this;
  }

  public void addFile(String name, byte[] content) {
    files.put(name, content);
  }

  public void addFile(String name, String content) {
    files.put(name, content.getBytes(UTF_8));
  }

  public String content(String filename) throws IOException {
    byte[] data = files.get(filename);
    if (data == null) {
      throw new FileNotFoundException();
    }
    return new String(data, UTF_8);
  }

  public byte[] toByteArray(String filename) throws IOException {
    byte[] data = files.get(filename);
    if (data == null) {
      throw new FileNotFoundException();
    }
    return data;
  }

  @Override
  public FileChannel getInputChannel(String filename) throws IOException {
    return new FakeReadChannel(filename);
  }

  @Override
  public FileChannel getOutputChannel(String filename, boolean append) throws IOException {
    return new FakeWriteChannel(filename);
  }

  @Override
  public InputStream getInputStream(String filename) throws IOException {
    byte[] data = files.get(filename);
    if (data == null) {
      throw new FileNotFoundException();
    }
    return new ByteArrayInputStream(data);
  }

  class FakeReadChannel extends FileChannel {

    final String name;
    byte[] data;
    int position;

    public FakeReadChannel(String filename) throws IOException {
      this.name = filename;
      this.data = toByteArray(filename);
      this.position = 0;
    }

    @Override
    public int read(ByteBuffer dst) throws IOException {
      if (position >= data.length) {
        return -1;
      }
      int remaining = data.length - position;
      if (dst.remaining() < remaining) {
        remaining = dst.remaining();
      }
      dst.put(data, position, remaining);
      position += remaining;
      return remaining;
    }

    @Override
    public long read(ByteBuffer[] dsts, int offset, int length) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int read(ByteBuffer dst, long position) throws IOException {
      if (position < 0 || position >= data.length) {
        throw new IOException("out of bounds");
      }
      int remaining = data.length - (int) position;
      if (dst.remaining() < remaining) {
        remaining = dst.remaining();
      }
      dst.put(data, (int) position, remaining);
      return remaining;
    }

    @Override
    public int write(ByteBuffer src) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long write(ByteBuffer[] srcs, int offset, int length) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int write(ByteBuffer src, long position) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long position() throws IOException {
      return position;
    }

    @Override
    public FileChannel position(long newPosition) throws IOException {
      position = (int) newPosition;
      return this;
    }

    @Override
    public long size() throws IOException {
      return data.length;
    }

    @Override
    public FileChannel truncate(long size) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void force(boolean metaData) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long transferTo(long position, long count, WritableByteChannel target)
        throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long transferFrom(ReadableByteChannel src, long position, long count)
        throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public MappedByteBuffer map(FileChannel.MapMode mode, long position, long size)
        throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    public ByteBuffer map(long position, long size) {
      return ByteBuffer.wrap(data, (int) position, (int) size).slice();
    }

    @Override
    public FileLock lock(long position, long size, boolean shared) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public FileLock tryLock(long position, long size, boolean shared) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected void implCloseChannel() throws IOException {
    }
  }

  class FakeWriteChannel extends FileChannel {

    final String name;
    final ByteArrayOutputStream buf;
    int position;

    public FakeWriteChannel(String filename) {
      this.name = filename;
      this.buf = new ByteArrayOutputStream();
      this.position = 0;
    }

    @Override
    public int read(ByteBuffer dst) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long read(ByteBuffer[] dsts, int offset, int length) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int read(ByteBuffer dst, long position) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int write(ByteBuffer src) throws IOException {
      byte[] bytes = new byte[src.remaining()];
      src.get(bytes);
      buf.write(bytes);
      position += bytes.length;
      return bytes.length;
    }

    @Override
    public long write(ByteBuffer[] srcs, int offset, int length) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int write(ByteBuffer src, long position) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long position() throws IOException {
      return position;
    }

    @Override
    public FileChannel position(long newPosition) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long size() throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public FileChannel truncate(long size) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void force(boolean metaData) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long transferTo(long position, long count, WritableByteChannel target)
        throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public long transferFrom(ReadableByteChannel src, long position, long count)
        throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public MappedByteBuffer map(FileChannel.MapMode mode, long position, long size)
        throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public FileLock lock(long position, long size, boolean shared) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public FileLock tryLock(long position, long size, boolean shared) throws IOException {
      throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected void implCloseChannel() throws IOException {
      files.put(name, buf.toByteArray());
    }
  }
}
