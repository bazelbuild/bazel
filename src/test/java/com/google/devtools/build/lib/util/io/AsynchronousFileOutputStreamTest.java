// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.io;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.runtime.commands.proto.BazelFlagsProto.FlagInfo;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadLocalRandom;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests {@link AsynchronousFileOutputStream}. */
@RunWith(JUnit4.class)
public class AsynchronousFileOutputStreamTest {
  private final Random random = ThreadLocalRandom.current();
  private static final char[] RAND_CHARS = "abcdefghijklmnopqrstuvwxzy0123456789-".toCharArray();
  private static final int RAND_STRING_LENGTH = 10;

  @Before
  public void initMocks() {
    MockitoAnnotations.initMocks(this);
  }

  @After
  public void validateMocks() {
    Mockito.validateMockitoUsage();
  }

  private FlagInfo generateRandomMessage() {
    FlagInfo.Builder b = FlagInfo.newBuilder();
    b.setName(generateRandomString() + "a");  // Name is required, cannot be empty.
    b.setHasNegativeFlag(random.nextBoolean());
    b.setDocumentation(generateRandomString());
    int commandsSize = random.nextInt(5);
    for (int i = 0; i < commandsSize; ++i) {
      b.addCommands(generateRandomString());
    }
    return b.build();
  }

  private String generateRandomString() {
    int len = random.nextInt(RAND_STRING_LENGTH + 1);
    char[] data = new char[len];
    for (int i = 0; i < len; ++i) {
      data[i] = RAND_CHARS[random.nextInt(RAND_CHARS.length)];
    }
    return new String(data);
  }

  @Test
  public void testConcurrentWrites() throws Exception {
    FileSystem fileSystem = new InMemoryFileSystem();
    Path logPath = fileSystem.getPath("/logFile");
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(logPath);
    Thread[] writers = new Thread[10];
    final CountDownLatch start = new CountDownLatch(writers.length);
    for (int i = 0; i < writers.length; ++i) {
      String name = "Thread # " + i;
      Thread thread = new Thread() {
        @Override
        public void run() {
          try {
            start.countDown();
            start.await();
          } catch (InterruptedException e) {
            return;
          }
          for (int j = 0; j < 10; ++j) {
            out.write(name + " time # " + j + "\n");
          }
        }
      };
      writers[i] = thread;
      thread.start();
    }
    for (int i = 0; i < writers.length; ++i) {
      writers[i].join();
    }
    out.close();
    String contents =
        new String(ByteStreams.toByteArray(logPath.getInputStream()), StandardCharsets.UTF_8);
    for (int i = 0; i < writers.length; ++i) {
      for (int j = 0; j < 10; ++j) {
        assertThat(contents).contains("Thread # " + i + " time # " + j + "\n");
      }
    }
  }

  @Test
  public void testConcurrentProtoWrites() throws Exception {
    final String filename = "/logFile";
    FileSystem fileSystem = new InMemoryFileSystem();
    Path logPath = fileSystem.getPath(filename);
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(logPath);
    ArrayList<FlagInfo> messages = new ArrayList<>();
    for (int i = 0; i < 100; ++i) {
      messages.add(generateRandomMessage());
    }
    Thread[] writers = new Thread[messages.size() / 10];
    final CountDownLatch start = new CountDownLatch(writers.length);
    for (int i = 0; i < writers.length; ++i) {
      int startIndex = i * 10;
      Thread thread = new Thread() {
        @Override
        public void run() {
          try {
            start.countDown();
            start.await();
          } catch (InterruptedException e) {
            return;
          }
          for (int j = startIndex; j < startIndex + 10; ++j) {
            out.write(messages.get(j));
          }
        }
      };
      writers[i] = thread;
      thread.start();
    }
    for (int i = 0; i < writers.length; ++i) {
      writers[i].join();
    }
    out.close();
    ArrayList<FlagInfo> readMessages = new ArrayList<>();
    try (InputStream in = fileSystem.getPath(filename).getInputStream()) {
      for (int i = 0; i < messages.size(); ++i) {
        readMessages.add(FlagInfo.parseDelimitedFrom(in));
      }
    }
    assertThat(readMessages).containsExactlyElementsIn(messages);
  }

  @Test
  public void testFailedClosePropagatesIOException() throws Exception {
    OutputStream failingOutputStream = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
      }
      @Override
      public void close() throws IOException {
        throw new IOException("foo");
      }
    };
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream("", failingOutputStream);
    out.write("bla");
    try {
      out.close();
      fail("Expected an IOException");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("foo");
    }
  }

  @Test
  public void testFailedClosePropagatesUncheckedException() throws Exception {
    OutputStream failingOutputStream = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
      }
      @Override
      public void close() throws IOException {
        throw new RuntimeException("foo");
      }
    };
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream("", failingOutputStream);
    out.write("bla");
    try {
      out.close();
      fail("Expected a RuntimeException");
    } catch (RuntimeException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("foo");
    }
  }

  @Test
  public void testFailedWritePropagatesIOException() throws Exception {
    OutputStream failingOutputStream = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new IOException("foo");
      }
      @Override
      public void close() throws IOException {
      }
    };
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream("", failingOutputStream);
    out.write("bla");
    out.write("blo");
    try {
      out.close();
      fail("Expected an IOException");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("foo");
    }
  }

  @Test
  public void testFailedWritePropagatesUncheckedException() throws Exception {
    OutputStream failingOutputStream = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new RuntimeException("foo");
      }
      @Override
      public void close() throws IOException {
      }
    };
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream("", failingOutputStream);
    out.write("bla");
    out.write("blo");
    try {
      out.close();
      fail("Expected a RuntimeException");
    } catch (RuntimeException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("foo");
    }
  }

  @Test
  public void testWriteAfterCloseThrowsException() throws Exception {
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(
        "", new ByteArrayOutputStream());
    out.write("bla");
    out.close();

    try {
      out.write("blo");
      fail("Expected an IllegalStateException");
    } catch (IllegalStateException expected) {
      // Expected.
    }
  }
}
