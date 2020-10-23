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
package com.google.devtools.build.lib.shell;

import com.google.common.base.Preconditions;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Uninterruptibles;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class provides convenience methods for consuming (actively reading)
 * output and error streams with different consumption policies:
 * accumulating ({@link #createAccumulatingConsumers()},
 * and streaming ({@link #createStreamingConsumers(OutputStream, OutputStream)}).
 */
final class Consumers {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private Consumers() {}

  private static final ExecutorService pool =
    Executors.newCachedThreadPool(new AccumulatorThreadFactory());

  static OutErrConsumers createAccumulatingConsumers() {
    return new OutErrConsumers(new AccumulatingConsumer(), new AccumulatingConsumer());
  }

  static OutErrConsumers createStreamingConsumers(OutputStream out, OutputStream err) {
    Preconditions.checkNotNull(out);
    Preconditions.checkNotNull(err);
    return new OutErrConsumers(new StreamingConsumer(out), new StreamingConsumer(err));
  }

  static class OutErrConsumers {
    private final OutputConsumer out;
    private final OutputConsumer err;

    private OutErrConsumers(final OutputConsumer out, final OutputConsumer err) {
      this.out = out;
      this.err = err;
    }

    void registerInputs(InputStream outInput, InputStream errInput, boolean closeStreams) {
      out.registerInput(outInput, closeStreams);
      err.registerInput(errInput, closeStreams);
    }

    void cancel() {
      out.cancel();
      err.cancel();
    }

    void waitForCompletion() throws IOException {
      out.waitForCompletion();
      err.waitForCompletion();
    }

    ByteArrayOutputStream getAccumulatedOut(){
      return out.getAccumulatedOut();
    }

    ByteArrayOutputStream getAccumulatedErr() {
      return err.getAccumulatedOut();
    }

    void logConsumptionStrategy() {
      // The creation methods guarantee that the consumption strategy is
      // the same for out and err - doesn't matter whether we call out or err,
      // let's pick out.
      out.logConsumptionStrategy();
    }

  }

  /**
   * This interface describes just one consumer, which consumes the
   * InputStream provided by {@link #registerInput(InputStream, boolean)}.
   * Implementations implement different consumption strategies.
   */
  private static interface OutputConsumer {
    /**
     * Returns whatever the consumer accumulated internally, or
     * {@link CommandResult#NO_OUTPUT_COLLECTED} if it doesn't accumulate
     * any output.
     *
     * @see AccumulatingConsumer
     */
    ByteArrayOutputStream getAccumulatedOut();

    void logConsumptionStrategy();

    void registerInput(InputStream in, boolean closeConsumer);

    void cancel();

    void waitForCompletion() throws IOException;
  }

  /** This consumer sends the input to a stream while consuming it. */
  private static class StreamingConsumer extends FutureConsumption {
    private final OutputStream out;

    StreamingConsumer(OutputStream out) {
      this.out = out;
    }

    @Override
    public ByteArrayOutputStream getAccumulatedOut() {
      return CommandResult.NO_OUTPUT_COLLECTED;
    }

    @Override
    public void logConsumptionStrategy() {
      logger.atFiner().log("Output will be sent to streams provided by client");
    }

    @Override protected Runnable createConsumingAndClosingSink(InputStream in,
                                                               boolean closeConsumer) {
      return new ClosingSink(in, out, closeConsumer);
    }
  }

  /**
   * This consumer sends the input to a {@link ByteArrayOutputStream}
   * while consuming it. This accumulated stream can be obtained by
   * calling {@link #getAccumulatedOut()}.
   */
  private static class AccumulatingConsumer extends FutureConsumption {
    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    @Override
    public ByteArrayOutputStream getAccumulatedOut() {
      return out;
    }

    @Override
    public void logConsumptionStrategy() {
      logger.atFiner().log("Output will be accumulated (promptly read off) and returned");
    }

    @Override public Runnable createConsumingAndClosingSink(InputStream in, boolean closeConsumer) {
      return new ClosingSink(in, out);
    }
  }

  /**
   * A mixin that makes consumers active - this is where we kick of
   * multithreading ({@link #registerInput(InputStream, boolean)}), cancel actions
   * and wait for the consumers to complete.
   */
  private abstract static class FutureConsumption implements OutputConsumer {

    private Future<?> future;

    @Override
    public void registerInput(InputStream in, boolean closeConsumer){
      Runnable sink = createConsumingAndClosingSink(in, closeConsumer);
      future = pool.submit(sink);
    }

    protected abstract Runnable createConsumingAndClosingSink(InputStream in, boolean close);

    @Override
    public void cancel() {
      future.cancel(true);
    }

    @Override
    public void waitForCompletion() throws IOException {
      try {
        Uninterruptibles.getUninterruptibly(future);
      } catch (ExecutionException ee) {
        // Runnable threw a RuntimeException
        Throwable nested = ee.getCause();
        if (nested instanceof RuntimeException) {
          final RuntimeException re = (RuntimeException) nested;
          // The stream sink classes, unfortunately, tunnel IOExceptions
          // out of run() in a RuntimeException. If that's the case,
          // unpack and re-throw the IOException. Otherwise, re-throw
          // this unexpected RuntimeException
          final Throwable cause = re.getCause();
          if (cause instanceof IOException) {
            throw (IOException) cause;
          } else {
            throw re;
          }
        } else if (nested instanceof OutOfMemoryError) {
          // OutOfMemoryError does not support exception chaining.
          throw (OutOfMemoryError) nested;
        } else if (nested instanceof Error) {
          throw new Error("unhandled Error in worker thread", ee);
        } else {
          throw new RuntimeException("unknown execution problem", ee);
        }
      }
    }
  }

  private static class AccumulatorThreadFactory implements ThreadFactory {
    private static final AtomicInteger threadInitNumber = new AtomicInteger(0);

    @Override
    public Thread newThread(final Runnable runnable) {
      final Thread t =
        new Thread(null,
                   runnable,
                   "Command-Accumulator-Thread-" + threadInitNumber.getAndIncrement());
      // Don't let this thread hold up JVM exit
      t.setDaemon(true);
      return t;
    }
  }

  /**
   * A sink that closes its input stream once its done.
   */
  private static class ClosingSink implements Runnable {
    private final InputStream in;
    private final OutputStream out;
    private final Runnable sink;
    private final boolean close;

    /**
     * Creates a sink that will pump InputStream <code>in</code>
     * into OutputStream <code>out</code>.
     */
    ClosingSink(final InputStream in, OutputStream out) {
      this(in, out, false);
    }

    /**
     * Creates a sink that will read <code>in</code> and discard it.
     */
    ClosingSink(final InputStream in) {
      this.sink = InputStreamSink.newRunnableSink(in);
      this.in = in;
      this.close = false;
      this.out = null;
    }

    ClosingSink(final InputStream in, OutputStream out, boolean close){
      this.sink = InputStreamSink.newRunnableSink(in, out);
      this.in = in;
      this.out = out;
      this.close = close;
    }


    @Override
    public void run() {
      try {
        sink.run();
      } finally {
        silentClose(in);
        if (close && out != null) {
          silentClose(out);
        }
      }
    }

  }

  /**
   * Close the <code>in</code> stream and log a warning if anything happens.
   */
  private static void silentClose(final Closeable closeable) {
    try {
      closeable.close();
    } catch (IOException ioe) {
      logger.atWarning().withCause(ioe).log("Unexpected exception while closing input stream");
    }
  }

}
