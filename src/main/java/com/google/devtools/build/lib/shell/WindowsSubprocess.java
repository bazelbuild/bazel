// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Throwables;
import com.google.devtools.build.lib.windows.WindowsProcesses;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/** A Windows subprocess backed by a native object. */
public class WindowsSubprocess implements Subprocess {
  // For debugging purposes.
  private String commandLine;

  private static enum WaitResult {
    SUCCESS,
    TIMEOUT
  }

  /** Output stream for writing to the stdin of a Windows process. */
  private class ProcessOutputStream extends OutputStream {
    private ProcessOutputStream() {}

    @Override
    public void write(int b) throws IOException {
      byte[] buf = new byte[] {(byte) b};
      write(buf, 0, 1);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      writeStream(b, off, len);
    }
  }

  /**
   * Input stream for reading the stdout or stderr of a Windows process.
   *
   * <p>This class is non-static for debugging purposes.
   */
  private static final class ProcessInputStream extends InputStream {
    private long nativeStream;

    ProcessInputStream(long nativeStream) {
      this.nativeStream = nativeStream;
    }

    @Override
    public int available() throws IOException {
      if (nativeStream == WindowsProcesses.INVALID) {
        throw new IllegalStateException("Stream already closed");
      }

      int result = WindowsProcesses.streamBytesAvailable(nativeStream);
      if (result == -1) {
        throw new IOException(WindowsProcesses.streamGetLastError(nativeStream));
      }

      return result;
    }

    @Override
    public int read() throws IOException {
      byte[] buf = new byte[1];
      if (read(buf, 0, 1) != 1) {
        return -1;
      } else {
        return buf[0] & 0xff;
      }
    }

    @Override
    public synchronized int read(byte[] b, int off, int len) throws IOException {
      if (nativeStream == WindowsProcesses.INVALID) {
        throw new IllegalStateException();
      }

      int result = WindowsProcesses.readStream(nativeStream, b, off, len);

      if (result == 0) {
        return -1; // EOF
      }
      if (result == -1) {
        throw new IOException(WindowsProcesses.streamGetLastError(nativeStream));
      }

      return result;
    }

    @Override
    public synchronized void close() {
      if (nativeStream != WindowsProcesses.INVALID) {
        WindowsProcesses.closeStream(nativeStream);
        nativeStream = WindowsProcesses.INVALID;
      }
    }

    @Override
    protected void finalize() throws Throwable {
      close();
      super.finalize();
    }
  }

  private static final AtomicInteger THREAD_SEQUENCE_NUMBER = new AtomicInteger(1);
  private static final ExecutorService WAITER_POOL =
      Executors.newCachedThreadPool(
          new ThreadFactory() {
            @Override
            public Thread newThread(Runnable runnable) {
              Thread thread =
                  new Thread(
                      null,
                      runnable,
                      "Windows-Process-Waiter-Thread-" + THREAD_SEQUENCE_NUMBER.getAndIncrement());
              thread.setDaemon(true);
              return thread;
            }
          });

  private volatile long nativeProcess;
  private final OutputStream stdinStream;
  private final ProcessInputStream stdoutStream;
  private final ProcessInputStream stderrStream;
  private final Future<WaitResult> processFuture;
  private final long timeoutMillis;
  private boolean timedout = false;

  WindowsSubprocess(
      long nativeProcess,
      String commandLine,
      boolean stdoutRedirected,
      boolean stderrRedirected,
      long timeoutMillis) {
    this.commandLine = commandLine;
    this.nativeProcess = nativeProcess;
    // As per the spec of Command, we should only apply timeouts that are > 0.
    this.timeoutMillis = timeoutMillis <= 0 ? -1 : timeoutMillis;
    stdoutStream =
        stdoutRedirected ? null : new ProcessInputStream(WindowsProcesses.getStdout(nativeProcess));
    stderrStream =
        stderrRedirected ? null : new ProcessInputStream(WindowsProcesses.getStderr(nativeProcess));
    stdinStream = new ProcessOutputStream();
    // Every Windows process we start consumes a thread here. This is suboptimal, but seems to be
    // the sanest way to reconcile WaitForMultipleObjects() and Java-style interruption.
    processFuture = WAITER_POOL.submit(this::waiterThreadFunc);
  }

  // Waits for the process to finish.
  private WaitResult waiterThreadFunc() {
    switch (WindowsProcesses.waitFor(nativeProcess, timeoutMillis)) {
      case 0:
        // Excellent, process finished in time.
        return WaitResult.SUCCESS;

      case 1:
        // Timeout. We don't need to call `terminate` here, because waitFor
        // automatically terminates the process in case of a timeout.
        return WaitResult.TIMEOUT;

      default:
        // Error. There isn't a lot we can do -- the process is still alive but
        // WaitForSingleObject() failed for some odd reason. This should
        // basically never happen, but if it does... let's get a stack trace.
        String errorMessage = WindowsProcesses.processGetLastError(nativeProcess);
        throw new IllegalStateException(
            "Waiting for process "
                + WindowsProcesses.getProcessPid(nativeProcess)
                + " failed: "
                + errorMessage);
    }
  }

  @Override
  public synchronized void finalize() throws Throwable {
    if (nativeProcess != WindowsProcesses.INVALID) {
      close();
    }
    super.finalize();
  }

  @Override
  public synchronized boolean destroy() {
    checkLiveness();
    return WindowsProcesses.terminate(nativeProcess);
  }

  @Override
  public synchronized int exitValue() {
    checkLiveness();

    int result = WindowsProcesses.getExitCode(nativeProcess);
    String error = WindowsProcesses.processGetLastError(nativeProcess);
    if (!error.isEmpty()) {
      throw new IllegalStateException(error);
    }

    return result;
  }

  @Override
  public boolean finished() {
    return processFuture.isDone();
  }

  @Override
  public boolean isAlive() {
    return !processFuture.isDone();
  }

  @Override
  public boolean timedout() {
    return timedout;
  }

  @Override
  public void waitFor() throws InterruptedException {
    try {
      timedout = processFuture.get() == WaitResult.TIMEOUT;
    } catch (ExecutionException e) {
      Throwables.throwIfUnchecked(e.getCause());
      // This should never happen, because waiterThreadFunc does not throw any
      // checked exceptions.
      throw new IllegalStateException("Unexpected exception", e);
    }
  }

  @Override
  public synchronized void close() {
    if (nativeProcess != WindowsProcesses.INVALID) {
      // stdoutStream and stderrStream are null if they are redirected to files.
      if (stdoutStream != null) {
        stdoutStream.close();
      }
      if (stderrStream != null) {
        stderrStream.close();
      }
      long process = nativeProcess;
      nativeProcess = WindowsProcesses.INVALID;
      WindowsProcesses.deleteProcess(process);
    }
  }

  @Override
  public OutputStream getOutputStream() {
    return stdinStream;
  }

  @Override
  public InputStream getInputStream() {
    return stdoutStream;
  }

  @Override
  public InputStream getErrorStream() {
    return stderrStream;
  }

  private synchronized void writeStream(byte[] b, int off, int len) throws IOException {
    checkLiveness();

    int remaining = len;
    int currentOffset = off;
    while (remaining != 0) {
      int written = WindowsProcesses.writeStdin(nativeProcess, b, currentOffset, remaining);
      // I think the Windows API never returns 0 in dwNumberOfBytesWritten
      // Verify.verify(written != 0);
      if (written == -1) {
        throw new IOException(WindowsProcesses.processGetLastError(nativeProcess));
      }

      remaining -= written;
      currentOffset += written;
    }
  }

  private void checkLiveness() {
    if (nativeProcess == WindowsProcesses.INVALID) {
      throw new IllegalStateException();
    }
  }

  @Override
  public String toString() {
    return String.format("%s:[%s]", super.toString(), commandLine);
  }

  @Override
  public long getProcessId() {
    return this.nativeProcess;
  }
}
