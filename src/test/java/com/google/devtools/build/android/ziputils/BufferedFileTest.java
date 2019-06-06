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

import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link BufferedFile}. */
@RunWith(JUnit4.class)
public class BufferedFileTest {

  private static final FakeFileSystem fileSystem = new FakeFileSystem();

  @Test
  public void testBufferedFile() throws Exception {
    int fileSize = 64;
    String filename = "bytes64";
    byte[] bytes = fileData(fileSize);
    fileSystem.addFile(filename, bytes);
    FileChannel file = fileSystem.getInputChannel(filename);
    int maxAlloc = 16;
    int regionOff = 0;
    assertException("channel null",
        null, regionOff, fileSize, maxAlloc, NullPointerException.class);
    assertException("region offset negative",
        file, -1, fileSize, maxAlloc, IllegalArgumentException.class);
    assertException("region size negative",
        file, regionOff, -1, maxAlloc, IllegalArgumentException.class);
    assertException("maxAlloc negative",
        file, regionOff, fileSize, -1, IllegalArgumentException.class);
    assertException("region exceeds file",
        file, regionOff, fileSize + 1, maxAlloc, IllegalArgumentException.class);
    assertException("region too long from offset",
        file, regionOff + 1, fileSize, maxAlloc, IllegalArgumentException.class);
    assertException("region short, still too long",
        file, regionOff + 50, fileSize - 49, maxAlloc, IllegalArgumentException.class);

    new BufferedFile(file, regionOff, fileSize, 0); // alloc minimal buffers for request
    new BufferedFile(file, regionOff, fileSize, fileSize); // alloc for full region
  }

  @Test
  public void testGetBufferThrows() throws Exception {
    BufferedFile instance;

    int fileSize = 64;
    String filename = "bytes64";
    byte[] bytes = fileData(fileSize);
    fileSystem.addFile(filename, bytes);
    FileChannel file = fileSystem.getInputChannel(filename);
    int regionOff = 4;
    int regionSize = 50;
    int maxAlloc = 16;
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertException("buffer negative size",
        instance, regionOff, -1, IllegalArgumentException.class);
    assertException("buffer lower bound",
        instance, regionOff - 1, regionSize, IllegalArgumentException.class);
    assertException("buffer upper bound",
        instance, regionOff + regionSize + 1, 1, IllegalArgumentException.class);
    assertException("buffer upper bound zero read",
        instance, regionOff + regionSize + 1, 0, IllegalArgumentException.class);
    assertException("buffer beyond region non zero read",
        instance, regionOff + regionSize, 1, IllegalArgumentException.class);
  }

  @Test
  public void testGetBufferAllocationLimits() throws Exception {
    BufferedFile instance;
    int fileSize = 64;
    String filename = "bytes64";
    byte[] bytes = fileData(fileSize);
    fileSystem.addFile(filename, bytes);
    FileChannel file = fileSystem.getInputChannel(filename);
    int regionOff = 4;
    int regionSize = 50;
    int maxAlloc = 16;
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertCase("buffer, empty, start", instance, regionOff, 0, 0, maxAlloc);
    assertWithMessage("buffer, empty, start").that(regionOff + regionSize)
        .isEqualTo(instance.limit());
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertCase("buffer, empty, end", instance, regionOff + regionSize, 0, 0, 0);
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertCase("buffer, one, end", instance, regionOff + regionSize - 1, 1, 1, 1);
    assertWithMessage("buffer, one, end").that(regionOff + regionSize)
        .isEqualTo(instance.limit());
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertCase("buffer, small, end", instance, regionOff + regionSize - 2, 2, 2, 2);
    assertWithMessage("buffer, small, end").that(regionOff + regionSize)
        .isEqualTo(instance.limit());
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertCase("buffer, small, start", instance, regionOff, 2, 2, maxAlloc);
    assertWithMessage("buffer, small, start").that(regionOff + regionSize)
        .isEqualTo(instance.limit());
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertCase("buffer, all region", instance, regionOff, regionSize, regionSize, regionSize);
    assertWithMessage("buffer, all region").that(regionOff + regionSize)
        .isEqualTo(instance.limit());
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    assertCase("buffer, request more",
        instance, regionOff + 5, regionSize, regionSize - 5, regionSize - 5);
    assertWithMessage("buffer, request more").that(regionOff + regionSize)
        .isEqualTo(instance.limit());
  }

  @Test
  public void testGetBufferInCache() throws Exception {
    BufferedFile instance;
    int fileSize = 64;
    String filename = "bytes64";
    byte[] bytes = fileData(fileSize);
    fileSystem.addFile(filename, bytes);
    FileChannel file = fileSystem.getInputChannel(filename);
    int regionOff = 5;
    int regionSize = 50;
    int maxAlloc = 20;
    int cacheOff = regionOff + 5;
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    instance.getBuffer(cacheOff, maxAlloc);
    assertCase("Cached zero buf", instance, cacheOff, 0, 0, maxAlloc);
    assertCase("Cached at front", instance, cacheOff, 5, 5, maxAlloc);
    assertCase("Cached at end", instance, cacheOff + 2, 5, 5, maxAlloc - 2);
    assertCase("Cached", instance, cacheOff, maxAlloc, maxAlloc, maxAlloc);
  }

  @Test
  public void testGetBufferReadMore() throws Exception {
    BufferedFile instance;
    int fileSize = 64;
    String filename = "bytes64";
    byte[] bytes = fileData(fileSize);
    fileSystem.addFile(filename, bytes);
    FileChannel file = fileSystem.getInputChannel(filename);
    int regionOff = 5;
    int regionSize = 50;
    int maxAlloc = 20;
    int cacheOff = regionOff + 5;
    int initialRead = maxAlloc / 2;
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    instance.getBuffer(cacheOff, initialRead);
    assertCase("Read more overlap",
        instance, cacheOff + 1, initialRead, initialRead, maxAlloc - 1);
    assertCase("Read more jump",
        instance, cacheOff + initialRead + 5, 5, 5, maxAlloc - initialRead - 5);
  }

  @Test
  public void testGetBufferReallocate() throws Exception {
    BufferedFile instance;
    int fileSize = 64;
    String filename = "bytes64";
    byte[] bytes = fileData(fileSize);
    fileSystem.addFile(filename, bytes);
    FileChannel file = fileSystem.getInputChannel(filename);
    int regionOff = 5;
    int regionSize = 50;
    int maxAlloc = 20;
    int cacheOff = regionOff + 5;
    instance = new BufferedFile(file, regionOff, regionSize, maxAlloc);
    instance.getBuffer(cacheOff, maxAlloc);
    assertCase("Realloc after", instance, cacheOff + maxAlloc, maxAlloc, maxAlloc, maxAlloc);
    assertCase("Realloc before", instance, cacheOff, maxAlloc, maxAlloc, maxAlloc);
    assertCase("Realloc just after", instance, cacheOff + 5, maxAlloc, maxAlloc, maxAlloc);
    assertCase("Realloc just before", instance, cacheOff, maxAlloc, maxAlloc, maxAlloc);
    assertCase("Realloc supersize", instance, cacheOff, maxAlloc + 5, maxAlloc + 5, maxAlloc + 5);
  }

  void assertException(String msg, FileChannel file, long off, long len, int maxAlloc,
      Class<?> expect) {
    Exception ex =
        assertThrows(
            msg + " - no exception",
            Exception.class,
            () -> new BufferedFile(file, off, len, maxAlloc));
    assertWithMessage(msg + " - exception, ").that(expect).isSameInstanceAs(ex.getClass());
  }

  void assertException(String msg, BufferedFile instance, long off, int len, Class<?> expect) {
    Exception ex =
        assertThrows(msg + " - no exception", Exception.class, () -> instance.getBuffer(off, len));
    assertWithMessage(msg + " - exception, ").that(expect).isSameInstanceAs(ex.getClass());
  }

  void assertCase(String msg, BufferedFile instance, long off, int len, int expectLimit,
      int capacityBound) throws IOException {
    ByteBuffer buf = instance.getBuffer(off, len);
    assertWithMessage(msg + " - position, ").that(0).isEqualTo(buf.position());
    assertWithMessage(msg + " - limit, ").that(expectLimit).isEqualTo(buf.limit());
    assertWithMessage(msg + " - capacity, ").that(buf.capacity()).isAtLeast(expectLimit);
    assertWithMessage(msg + " - capacity, ").that(buf.capacity()).isAtMost(capacityBound);
    if (len > 0 && expectLimit > 0) {
      assertWithMessage(msg + " - value, ").that(buf.get(0)).isEqualTo((byte) off);
    }
  }

  byte[] fileData(int count) {
    byte[] bytes = new byte[count];
    for (int i = 0; i < count; i++) {
      bytes[i] = (byte) i;
    }
    return bytes;
  }
}
