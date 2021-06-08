// Copyright 2019 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.vfs.inmemoryfs;

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.util.function.Consumer;

/**
 * InMemoryFileInfo manages file contents by storing them entirely in memory.
 */
@ThreadSafe
public class InMemoryFileInfo extends FileInfo {

  /**
   * Updates to the content must atomically update the lastModifiedTime. So all
   * accesses to this field must be synchronized.
   */
  protected byte[] content;

  InMemoryFileInfo(Clock clock) {
    super(clock);
    content = new byte[0]; // New files start out empty.
  }

  @Override
  public synchronized long getSize() {
    return content.length;
  }

  @Override
  public byte[] getxattr(String name) {
    return null;
  }

  @Override
  public byte[] getFastDigest() {
    return null;
  }

  private synchronized void setContent(byte[] newContent) {
    content = newContent;
    markModificationTime();
  }

  @Override
  public synchronized InputStream getInputStream() {
    return new ByteArrayInputStream(content);
  }

  @Override
  public ReadableByteChannel createReadableByteChannel() {
    return new ReadableByteChannel() {
      private int offset = 0;

      @Override
      public int read(ByteBuffer dst) {
        if (offset >= content.length) {
          return -1;
        }
        int length = Math.min(dst.remaining(), content.length - offset);
        dst.put(content, offset, length);
        offset += length;
        return length;
      }

      @Override
      public boolean isOpen() {
        return true;
      }

      @Override
      public void close() {}
    };
  }

  @Override
  public synchronized OutputStream getOutputStream(boolean append) {
    OutputStream out = new InMemoryOutputStream(this::setContent);
    if (append) {
      try (InputStream in = getInputStream()) {
        ByteStreams.copy(in, out);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
    return out;
  }

  /** A {@link ByteArrayOutputStream} which notifiers a callback when it has flushed its data. */
  public static class InMemoryOutputStream extends ByteArrayOutputStream {
    private final Consumer<byte[]> receiver;

    public InMemoryOutputStream(Consumer<byte[]> receiver) {
      this.receiver = receiver;
    }

    @Override
    public void close() {
      flush();
    }

    @Override
    public synchronized void flush() {
      receiver.accept(toByteArray());
    }
  }
}
