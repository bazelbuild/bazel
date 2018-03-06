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
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.when;

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.runtime.commands.proto.BazelFlagsProto.FlagInfo;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.CompletionHandler;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadLocalRandom;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests {@link AsynchronousFileOutputStream}. */
@RunWith(JUnit4.class)
public class AsynchronousFileOutputStreamTest {
  @Rule public TemporaryFolder tmp = new TemporaryFolder();
  @Mock AsynchronousFileChannel mockChannel;
  Random random = ThreadLocalRandom.current();
  static final char[] RAND_CHARS = "abcdefghijklmnopqrstuvwxzy0123456789-".toCharArray();
  static final int RAND_STRING_LENGTH = 10;

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
    Path logPath = tmp.newFile().toPath();
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(logPath.toString());
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
        new String(ByteStreams.toByteArray(Files.newInputStream(logPath)), StandardCharsets.UTF_8);
    for (int i = 0; i < writers.length; ++i) {
      for (int j = 0; j < 10; ++j) {
        assertThat(contents).contains("Thread # " + i + " time # " + j + "\n");
      }
    }
  }

  @Test
  public void testConcurrentProtoWrites() throws Exception {
    Path logPath = tmp.newFile().toPath();
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(logPath.toString());
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
    try (InputStream in = Files.newInputStream(logPath)) {
      for (int i = 0; i < messages.size(); ++i) {
        readMessages.add(FlagInfo.parseDelimitedFrom(in));
      }
    }
    assertThat(readMessages).containsExactlyElementsIn(messages);
  }

  @Test
  public void testFailedClosePropagatesIOException() throws Exception {
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(mockChannel);
    when(mockChannel.isOpen()).thenReturn(true);
    IOException ex = new IOException("foo");
    Mockito.doThrow(ex).when(mockChannel).close();
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              CompletionHandler<Integer, Void> handler =
                  (CompletionHandler<Integer, Void>) invocationOnMock.getArguments()[3];
              handler.completed(0, null); // We ignore the arguments.
              return null;
            })
        .when(mockChannel)
        .write(
            any(ByteBuffer.class),
            any(Integer.class),
            eq(null),
            Mockito.<CompletionHandler<Integer, Void>>anyObject());
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
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(mockChannel);
    when(mockChannel.isOpen()).thenReturn(true);
    RuntimeException ex = new RuntimeException("foo");
    Mockito.doThrow(ex).when(mockChannel).close();
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              CompletionHandler<Integer, Void> handler =
                  (CompletionHandler<Integer, Void>) invocationOnMock.getArguments()[3];
              handler.completed(0, null); // We ignore the arguments.
              return null;
            })
        .when(mockChannel)
        .write(
            any(ByteBuffer.class),
            any(Integer.class),
            eq(null),
            Mockito.<CompletionHandler<Integer, Void>>anyObject());
    out.write("bla");

    try {
      out.close();
      fail("Expected a RuntimeException");
    } catch (RuntimeException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("foo");
    }
  }

  @Test
  public void testFailedForcePropagatesIOException() throws Exception {
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(mockChannel);
    when(mockChannel.isOpen()).thenReturn(true);
    IOException ex = new IOException("foo");
    Mockito.doThrow(ex).when(mockChannel).force(eq(true));
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              CompletionHandler<Integer, Void> handler =
                  (CompletionHandler<Integer, Void>) invocationOnMock.getArguments()[3];
              handler.completed(0, null); // We ignore the arguments.
              return null;
            })
        .when(mockChannel)
        .write(
            any(ByteBuffer.class),
            any(Integer.class),
            eq(null),
            Mockito.<CompletionHandler<Integer, Void>>anyObject());
    out.write("bla");

    try {
      out.close();
      fail("Expected an IOException");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("foo");
    }
  }

  @Test
  public void testFailedForcePropagatesUncheckedException() throws Exception {
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(mockChannel);
    when(mockChannel.isOpen()).thenReturn(true);
    RuntimeException ex = new RuntimeException("foo");
    Mockito.doThrow(ex).when(mockChannel).force(eq(true));
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              CompletionHandler<Integer, Void> handler =
                  (CompletionHandler<Integer, Void>) invocationOnMock.getArguments()[3];
              handler.completed(0, null); // We ignore the arguments.
              return null;
            })
        .when(mockChannel)
        .write(
            any(ByteBuffer.class),
            any(Integer.class),
            eq(null),
            Mockito.<CompletionHandler<Integer, Void>>anyObject());
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
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(mockChannel);
    when(mockChannel.isOpen()).thenReturn(true);
    IOException ex = new IOException("foo");
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              CompletionHandler<Integer, Void> handler =
                  (CompletionHandler<Integer, Void>) invocationOnMock.getArguments()[3];
              handler.failed(ex, null);
              return null;
            })
        .when(mockChannel)
        .write(
            any(ByteBuffer.class),
            any(Integer.class),
            eq(null),
            Mockito.<CompletionHandler<Integer, Void>>anyObject());
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
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(mockChannel);
    when(mockChannel.isOpen()).thenReturn(true);
    RuntimeException ex = new RuntimeException("foo");
    Mockito.doAnswer(
            invocationOnMock -> {
              @SuppressWarnings("unchecked")
              CompletionHandler<Integer, Void> handler =
                  (CompletionHandler<Integer, Void>) invocationOnMock.getArguments()[3];
              handler.failed(ex, null);
              return null;
            })
        .when(mockChannel)
        .write(
            any(ByteBuffer.class),
            any(Integer.class),
            eq(null),
            Mockito.<CompletionHandler<Integer, Void>>anyObject());
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
    Path logPath = tmp.newFile().toPath();
    AsynchronousFileChannel ch = AsynchronousFileChannel.open(
            logPath,
            StandardOpenOption.WRITE,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING);
    AsynchronousFileOutputStream out = new AsynchronousFileOutputStream(ch);
    out.write("bla");
    ch.close();

    try {
      out.write("blo");
      fail("Expected an IllegalStateException");
    } catch (IllegalStateException expected) {
      // Expected.
    }
  }
}
