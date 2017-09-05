// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

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

  protected InMemoryFileInfo(Clock clock) {
    super(clock);
    content = new byte[0]; // New files start out empty.
  }

  @Override
  public synchronized long getSize() {
    return content.length;
  }

  @Override
  public synchronized byte[] readContent() {
    return content.clone();
  }

  private synchronized void setContent(byte[] newContent) {
    content = newContent;
    markModificationTime();
  }

  @Override
  protected synchronized OutputStream getOutputStream(boolean append)
      throws IOException {
    OutputStream out = new ByteArrayOutputStream() {
      private boolean closed = false;

      @Override
      public void write(byte[] data) throws IOException {
        Preconditions.checkState(!closed);
        super.write(data);
      }

      @Override
      public synchronized void write(int dataByte) {
        Preconditions.checkState(!closed);
        super.write(dataByte);
      }

      @Override
      public synchronized void write(byte[] data, int offset, int length) {
        Preconditions.checkState(!closed);
        super.write(data, offset, length);
      }

      @Override
      public void close() {
        flush();
        closed = true;
      }

      @Override
      public void flush() {
        setContent(toByteArray().clone());
      }
    };

    if (append) {
      out.write(readContent());
    }
    return out;
  }
}
