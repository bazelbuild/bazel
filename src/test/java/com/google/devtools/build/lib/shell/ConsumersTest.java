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
package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.shell.Consumers.OutErrConsumers;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ConsumersTest {

  @Before
  public final void configureLogger() throws Exception  {
    // enable all log statements to ensure there are no problems with
    // logging code
    Logger.getLogger("com.google.devtools.build.lib.shell.Command").setLevel(Level.FINEST);
  }

  private static final String SECRET_MESSAGE = "This is a secret message.";

  /**
   * Tests that if an IOException occurs in an output stream, the exception
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousIOExceptionInConsumerOutputStream() {

    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new IOException(SECRET_MESSAGE);
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    IOException e = assertThrows(IOException.class, () -> outErr.waitForCompletion());
    assertThat(e).hasMessageThat().isEqualTo(SECRET_MESSAGE);
  }

  /**
   * Tests that if an OutOfMemeoryError occurs in an output stream, it
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousOutOfMemoryErrorInConsumerOutputStream() {
    final OutOfMemoryError error = new OutOfMemoryError(SECRET_MESSAGE);
    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw error;
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    try {
      outErr.waitForCompletion();
      fail();
    } catch (IOException e) {
      fail();
    } catch (OutOfMemoryError e) {
      assertWithMessage("OutOfMemoryError is not masked").that(e).isSameInstanceAs(error);
    }
  }

  /**
   * Tests that if an Error occurs in an output stream, the error
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousErrorInConsumerOutputStream() {
    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new OutOfMemoryError(SECRET_MESSAGE);
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    Error error = assertThrows(Error.class, () -> outErr.waitForCompletion());
    assertThat(error).isNotInstanceOf(IOException.class);
    assertThat(error).hasMessageThat().isEqualTo(SECRET_MESSAGE);
  }

  /**
   * Tests that once cancel() returns, the sink no longer writes to the accumulated output, so that
   * callers can safely read it. The sink is blocked in a read that ignores the interrupt from the
   * cancellation, as reads from a subprocess's output pipe do.
   */
  @Test
  public void testCancelWaitsForSinkToStopWriting() throws Exception {
    Semaphore sinkBlocked = new Semaphore(0);
    Semaphore sinkMayContinue = new Semaphore(0);
    InputStream blockingIn =
        new InputStream() {
          private int reads = 0;

          @Override
          public int read() {
            throw new UnsupportedOperationException("expected only array reads");
          }

          @Override
          public int read(byte[] b, int off, int len) {
            return switch (reads++) {
              case 0 -> {
                b[off] = 'A';
                yield 1;
              }
              case 1 -> {
                sinkBlocked.release();
                sinkMayContinue.acquireUninterruptibly();
                b[off] = 'B';
                yield 1;
              }
              default -> -1;
            };
          }
        };
    OutErrConsumers outErr = Consumers.createAccumulatingConsumers();
    outErr.registerInputs(blockingIn, new ByteArrayInputStream(new byte[0]), false);
    sinkBlocked.acquire();

    AtomicInteger sizeWhenCancelReturned = new AtomicInteger(-1);
    CountDownLatch cancelReturned = new CountDownLatch(1);
    TestThread canceller =
        new TestThread(
            () -> {
              outErr.cancel();
              sizeWhenCancelReturned.set(outErr.getAccumulatedOut().size());
              cancelReturned.countDown();
            });
    canceller.start();

    // Produce the sink's last byte only after cancel() had ample time to return early: a
    // cancellation that does not wait for the sink then observes the write that follows it.
    boolean unused = Uninterruptibles.awaitUninterruptibly(cancelReturned, 200, MILLISECONDS);
    sinkMayContinue.release();
    canceller.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);

    assertThat(outErr.getAccumulatedOut().size()).isEqualTo(sizeWhenCancelReturned.get());
  }

  @Test
  public void testCancelUsesOneTimeoutForBothSinks() throws Exception {
    Semaphore sinksBlocked = new Semaphore(0);
    Semaphore sinksMayFinish = new Semaphore(0);
    InputStream blockingOut = blockingInput(sinksBlocked, sinksMayFinish);
    InputStream blockingErr = blockingInput(sinksBlocked, sinksMayFinish);
    OutErrConsumers outErr = Consumers.createAccumulatingConsumers();
    outErr.registerInputs(blockingOut, blockingErr, false);
    sinksBlocked.acquire(2);

    Stopwatch cancellationTime = Stopwatch.createStarted();
    try {
      outErr.cancel(Duration.ofSeconds(1));
    } finally {
      sinksMayFinish.release(2);
    }

    // Each sink ignores its interrupt and stays blocked, but they share one cancellation deadline
    // rather than each consuming the full timeout sequentially.
    assertThat(cancellationTime.elapsed().toMillis()).isLessThan(1500);
    outErr.waitForCompletion();
  }

  private static InputStream blockingInput(Semaphore blocked, Semaphore mayFinish) {
    return new InputStream() {
      @Override
      public int read() {
        throw new UnsupportedOperationException("expected only array reads");
      }

      @Override
      public int read(byte[] b, int off, int len) {
        blocked.release();
        mayFinish.acquireUninterruptibly();
        return -1;
      }
    };
  }

  /**
   * Tests that if an RuntimeException occurs in an output stream, the exception
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousRuntimeExceptionInConsumerOutputStream()
  throws Exception {
    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) {
        throw new RuntimeException(SECRET_MESSAGE);
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    RuntimeException e = assertThrows(RuntimeException.class, () -> outErr.waitForCompletion());
    assertThat(e).hasMessageThat().isEqualTo(SECRET_MESSAGE);
  }
}
