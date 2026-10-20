// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.ExecuteOperationMetadata;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import com.google.longrunning.Operation;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.CancellationException;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;

/**
 * Writes the stdout and stderr of a remotely executing action to its local stdout and stderr files
 * while the action runs, by reading the live output streams the server advertises in {@link
 * ExecuteOperationMetadata#getStdoutStreamName()} and {@link
 * ExecuteOperationMetadata#getStderrStreamName()}.
 *
 * <p>This is what makes {@code --test_output=streamed} work with remote execution: streamed test
 * output is displayed by tailing the test log, which is the test spawn's stdout file.
 *
 * <p>Live output is best effort and never affects the result of the action. Once the action
 * finishes, {@link #close} stops streaming. If the action produced a result, the live output is
 * discarded by truncating the files back to empty in place, so that the regular download of the
 * {@code ActionResult} writes the authoritative output. A reader tailing a file by offset then sees
 * the final output continue where the live output left off, as long as the live output was a prefix
 * of it. If the action produced no result, e.g. because the build was interrupted, the live output
 * is kept, since it is all the output there is.
 */
final class LiveOutputStreamer implements OperationObserver {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Reads the live output stream with the given resource name into the given output stream. */
  @FunctionalInterface
  interface StreamReader {
    ListenableFuture<Void> read(String resourceName, OutputStream out);
  }

  private final LiveOutput stdout;
  private final LiveOutput stderr;

  /**
   * Returns a streamer that writes to the files of the given {@link FileOutErr}, or null if either
   * file already has content: live output can only be reconciled with the final output in an
   * initially empty file.
   *
   * @param readFailureHandler called with the stream name and the cause if reading a stream fails
   *     for any reason other than the streamer being closed
   */
  @Nullable
  static LiveOutputStreamer create(
      StreamReader reader, FileOutErr outErr, BiConsumer<String, Throwable> readFailureHandler) {
    Path stdoutPath = outErr.getOutputPath();
    Path stderrPath = outErr.getErrorPath();
    if (stdoutPath == null || stderrPath == null) {
      return null;
    }
    try {
      if (hasContent(stdoutPath) || hasContent(stderrPath)) {
        return null;
      }
    } catch (IOException e) {
      logger.atInfo().withCause(e).log("Not streaming live output to %s", stdoutPath);
      return null;
    }
    return new LiveOutputStreamer(
        new LiveOutput(reader, stdoutPath, readFailureHandler),
        new LiveOutput(reader, stderrPath, readFailureHandler));
  }

  private static boolean hasContent(Path path) throws IOException {
    FileStatus status = path.statIfFound();
    return status != null && status.getSize() > 0;
  }

  private LiveOutputStreamer(LiveOutput stdout, LiveOutput stderr) {
    this.stdout = stdout;
    this.stderr = stderr;
  }

  @Override
  public void onNext(Operation operation) {
    ExecuteOperationMetadata metadata = getMetadata(operation);
    if (metadata == null) {
      return;
    }
    stdout.follow(metadata.getStdoutStreamName());
    stderr.follow(metadata.getStderrStreamName());
  }

  /**
   * Stops streaming.
   *
   * @param discardOutput whether to truncate the files back to empty because the final output of
   *     the action is about to replace the live output, rather than keeping the live output as the
   *     only output there is
   */
  public void close(boolean discardOutput) {
    stdout.stop(discardOutput);
    stderr.stop(discardOutput);
  }

  @Nullable
  private static ExecuteOperationMetadata getMetadata(Operation operation) {
    if (!operation.getMetadata().is(ExecuteOperationMetadata.class)) {
      return null;
    }
    try {
      return operation.getMetadata().unpack(ExecuteOperationMetadata.class);
    } catch (InvalidProtocolBufferException e) {
      return null;
    }
  }

  /** One of the two outputs of the action, written to one file. */
  private static final class LiveOutput {
    private final StreamReader reader;
    private final Path path;
    private final BiConsumer<String, Throwable> readFailureHandler;

    /** Whether no more live output will be written, because the streamer was closed. */
    @GuardedBy("this")
    private boolean stopped;

    /** The stream currently being read, which belongs to the latest execution of the action. */
    @GuardedBy("this")
    @Nullable
    private String streamName;

    @GuardedBy("this")
    @Nullable
    private ListenableFuture<Void> read;

    /** Appends to {@link #path}. Opened, creating the file, when the first output arrives. */
    @GuardedBy("this")
    @Nullable
    private OutputStream out;

    LiveOutput(StreamReader reader, Path path, BiConsumer<String, Throwable> readFailureHandler) {
      this.reader = reader;
      this.path = path;
      this.readFailureHandler = readFailureHandler;
    }

    /** Starts reading the given stream, unless it is empty or already being read. */
    void follow(String name) {
      if (name.isEmpty()) {
        return;
      }
      ListenableFuture<Void> previousRead;
      synchronized (this) {
        if (stopped || name.equals(streamName)) {
          return;
        }
        // A different stream means that the server started a new execution, e.g. after losing the
        // previous one. Its output is appended to that of the previous one rather than replacing
        // it, since readers following the file can't go back. Discarding the output drops both.
        previousRead = read;
        read = null;
        streamName = name;
      }
      if (previousRead != null) {
        previousRead.cancel(true);
      }
      StreamOutput streamOutput = new StreamOutput(name);
      ListenableFuture<Void> newRead = reader.read(name, streamOutput);
      synchronized (this) {
        if (stopped || !name.equals(streamName)) {
          newRead.cancel(true);
          return;
        }
        read = newRead;
      }
      Futures.addCallback(
          newRead,
          new FutureCallback<Void>() {
            @Override
            public void onSuccess(Void unused) {}

            @Override
            public void onFailure(Throwable t) {
              if (t instanceof CancellationException || streamOutput.isStale()) {
                return;
              }
              readFailureHandler.accept(name, t);
            }
          },
          directExecutor());
    }

    void stop(boolean discardOutput) {
      ListenableFuture<Void> read;
      synchronized (this) {
        stopped = true;
        read = this.read;
        this.read = null;
      }
      if (read != null) {
        read.cancel(true);
      }
      synchronized (this) {
        if (out == null) {
          return;
        }
        try {
          out.close();
          if (discardOutput) {
            // Truncate in place rather than deleting the file, so that readers that already opened
            // it, like the streamed test output, keep following it.
            path.getOutputStream(/* append= */ false).close();
          }
        } catch (IOException e) {
          // The download of the final output, which writes the same file, will report the problem.
          logger.atWarning().withCause(e).log("Failed to finish live output in %s", path);
        } finally {
          out = null;
        }
      }
    }

    /**
     * Appends the data read from one stream to the file, as long as that stream is still the
     * current one and the output hasn't been stopped.
     */
    private final class StreamOutput extends OutputStream {
      private final String name;

      StreamOutput(String name) {
        this.name = name;
      }

      boolean isStale() {
        synchronized (LiveOutput.this) {
          return stopped || !name.equals(streamName);
        }
      }

      @Override
      public void write(int b) throws IOException {
        write(new byte[] {(byte) b}, 0, 1);
      }

      @Override
      public void write(byte[] b, int off, int len) throws IOException {
        synchronized (LiveOutput.this) {
          if (stopped || !name.equals(streamName)) {
            throw new IOException("Live output stream " + name + " is no longer needed");
          }
          if (len == 0) {
            return;
          }
          if (out == null) {
            out = path.getOutputStream(/* append= */ true);
          }
          out.write(b, off, len);
          out.flush();
        }
      }
    }
  }
}
