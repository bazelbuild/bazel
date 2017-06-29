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

package com.google.devtools.build.lib.windows;

import com.google.devtools.build.lib.shell.Subprocess;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A Windows subprocess backed by a native object.
 */
public class WindowsSubprocess implements Subprocess {
  // For debugging purposes.
  private String commandLine;

  /**
   * Output stream for writing to the stdin of a Windows process.
   */
  private class ProcessOutputStream extends OutputStream {
    private ProcessOutputStream() {
    }

    @Override
    public void write(int b) throws IOException {
      byte[] buf = new byte[]{ (byte) b };
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
  private class ProcessInputStream extends InputStream {
    private long nativeStream;

    ProcessInputStream(long nativeStream) {
      this.nativeStream = nativeStream;
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
    public synchronized int read(byte b[], int off, int len) throws IOException {
      if (nativeStream == WindowsProcesses.INVALID) {
        throw new IllegalStateException();
      }

      int result = WindowsProcesses.nativeReadStream(nativeStream, b, off, len);

      if (result == 0) {
        return -1; // EOF
      }
      if (result == -1) {
        throw new IOException(WindowsProcesses.nativeStreamGetLastError(nativeStream));
      }

      return result;
    }

    @Override
    public synchronized void close() {
      if (nativeStream != WindowsProcesses.INVALID) {
        WindowsProcesses.nativeCloseStream(nativeStream);
        nativeStream = WindowsProcesses.INVALID;
      }
    }

    @Override
    protected void finalize() throws Throwable {
      close();
      super.finalize();
    }
  }

  private static AtomicInteger THREAD_SEQUENCE_NUMBER = new AtomicInteger(1);
  private static final ExecutorService WAITER_POOL = Executors.newCachedThreadPool(
      new ThreadFactory() {
        @Override
        public Thread newThread(Runnable runnable) {
          Thread thread = new Thread(null, runnable,
              "Windows-Process-Waiter-Thread-" + THREAD_SEQUENCE_NUMBER.getAndIncrement(),
              16 * 1024);
          thread.setDaemon(true);
          return thread;
        }
      });

  private volatile long nativeProcess;
  private final OutputStream stdinStream;
  private final ProcessInputStream stdoutStream;
  private final ProcessInputStream stderrStream;
  private final CountDownLatch waitLatch;
  private final long timeoutMillis;
  private final AtomicBoolean timedout = new AtomicBoolean(false);

  WindowsSubprocess(long nativeProcess, String commandLine, boolean stdoutRedirected,
      boolean stderrRedirected, long timeoutMillis) {
    this.commandLine = commandLine;
    this.nativeProcess = nativeProcess;
    this.timeoutMillis = timeoutMillis;
    stdoutStream =
        stdoutRedirected
            ? null
            : new ProcessInputStream(WindowsProcesses.nativeGetStdout(nativeProcess));
    stderrStream =
        stderrRedirected
            ? null
            : new ProcessInputStream(WindowsProcesses.nativeGetStderr(nativeProcess));
    stdinStream = new ProcessOutputStream();
    waitLatch = new CountDownLatch(1);
    // Every Windows process we start consumes a thread here. This is suboptimal, but seems to be
    // the sanest way to reconcile WaitForMultipleObjects() and Java-style interruption.
    @SuppressWarnings("unused")
    Future<?> possiblyIgnoredError = WAITER_POOL.submit(this::waiterThreadFunc);
  }

  private void waiterThreadFunc() {
    switch (WindowsProcesses.nativeWaitFor(nativeProcess, timeoutMillis)) {
      case 0:
        // Excellent, process finished in time.
        break;

      case 1:
        // Timeout. Terminate the process if we can.
        timedout.set(true);
        WindowsProcesses.nativeTerminate(nativeProcess);
        break;

      case 2:
        // Error. There isn't a lot we can do -- the process is still alive but
        // WaitForMultipleObjects() failed for some odd reason. We'll pretend it terminated and
        // log a message to jvm.out .
        System.err.println("Waiting for process "
            + WindowsProcesses.nativeGetProcessPid(nativeProcess) + " failed");
        break;
    }

    waitLatch.countDown();
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

    if (!WindowsProcesses.nativeTerminate(nativeProcess)) {
      return false;
    }

    return true;
  }

  @Override
  public synchronized int exitValue() {
    checkLiveness();

    int result = WindowsProcesses.nativeGetExitCode(nativeProcess);
    String error = WindowsProcesses.nativeProcessGetLastError(nativeProcess);
    if (!error.isEmpty()) {
      throw new IllegalStateException(error);
    }

    return result;
  }

  @Override
  public boolean finished() {
    return waitLatch.getCount() == 0;
  }

  @Override
  public boolean timedout() {
    return timedout.get();
  }

  @Override
  public void waitFor() throws InterruptedException {
    waitLatch.await();
  }

  @Override
  public synchronized void close() {
    if (nativeProcess != WindowsProcesses.INVALID) {
      stdoutStream.close();
      stderrStream.close();
      long process = nativeProcess;
      nativeProcess = WindowsProcesses.INVALID;
      WindowsProcesses.nativeDeleteProcess(process);
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
      int written = WindowsProcesses.nativeWriteStdin(
          nativeProcess, b, currentOffset, remaining);
      // I think the Windows API never returns 0 in dwNumberOfBytesWritten
      // Verify.verify(written != 0);
      if (written == -1) {
        throw new IOException(WindowsProcesses.nativeProcessGetLastError(nativeProcess));
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
}
