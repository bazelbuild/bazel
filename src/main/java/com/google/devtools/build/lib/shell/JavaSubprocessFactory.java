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

import com.google.devtools.build.lib.shell.SubprocessBuilder.StreamAction;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ProcessBuilder.Redirect;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A subprocess factory that uses {@link java.lang.ProcessBuilder}.
 */
public class JavaSubprocessFactory implements Subprocess.Factory {

  /**
   * A subprocess backed by a {@link java.lang.Process}.
   */
  private static class JavaSubprocess implements Subprocess {
    private final Process process;
    private final long deadlineMillis;
    private final AtomicBoolean deadlineExceeded = new AtomicBoolean();

    private JavaSubprocess(Process process, long deadlineMillis) {
      this.process = process;
      this.deadlineMillis = deadlineMillis;
    }

    @Override
    public boolean destroy() {
      process.destroy();
      return true;
    }

    @Override
    public int exitValue() {
      return process.exitValue();
    }

    @Override
    public boolean finished() {
      try {
        if (deadlineMillis > 0
            && System.currentTimeMillis() > deadlineMillis
            && deadlineExceeded.compareAndSet(false, true)) {
          // We use compareAndSet here to avoid calling destroy multiple times. Note that destroy
          // returns immediately, and we don't want to wait in this method.
          process.destroy();
        }
        // this seems to be the only non-blocking call for checking liveness
        process.exitValue();
        return true;
      } catch (IllegalThreadStateException e) {
        return false;
      }
    }

    @Override
    public boolean timedout() {
      return deadlineExceeded.get();
    }

    @Override
    public void waitFor() throws InterruptedException {
      if (deadlineMillis > 0) {
        // Careful: I originally used Long.MAX_VALUE if there's no timeout. This is safe with
        // Process, but not for the UNIXProcess subclass, which has an integer overflow for very
        // large timeouts. As of this writing, it converts the passed in value to nanos (which
        // saturates at Long.MAX_VALUE), then adds 999999 to round up (which overflows), converts
        // back to millis, and then calls Object.wait with a negative timeout, which throws.
        long waitTimeMillis = deadlineMillis - System.currentTimeMillis();
        boolean exitedInTime = process.waitFor(waitTimeMillis, TimeUnit.MILLISECONDS);
        if (!exitedInTime && deadlineExceeded.compareAndSet(false, true)) {
          process.destroy();
          // The destroy call returns immediately, so we still need to wait for the actual exit. The
          // sole caller assumes that waitFor only exits when the process is gone (or throws).
          process.waitFor();
        }
      } else {
        process.waitFor();
      }
    }

    @Override
    public OutputStream getOutputStream() {
      return process.getOutputStream();
    }

    @Override
    public InputStream getErrorStream() {
      return process.getErrorStream();
    }

    @Override
    public InputStream getInputStream() {
      return process.getInputStream();
    }

    @Override
    public void close() {
      // java.lang.Process doesn't give us a way to clean things up other than #destroy(), which was
      // already called by this point.
    }
  }

  public static final JavaSubprocessFactory INSTANCE = new JavaSubprocessFactory();

  private JavaSubprocessFactory() {
    // We are a singleton
  }

  @Override
  public Subprocess create(SubprocessBuilder params) throws IOException {
    ProcessBuilder builder = new ProcessBuilder();
    builder.command(params.getArgv());
    if (params.getEnv() != null) {
      builder.environment().clear();
      builder.environment().putAll(params.getEnv());
    }

    builder.redirectOutput(getRedirect(params.getStdout(), params.getStdoutFile()));
    builder.redirectError(getRedirect(params.getStderr(), params.getStderrFile()));
    builder.directory(params.getWorkingDirectory());

    // Deadline is now + given timeout.
    long deadlineMillis = params.getTimeoutMillis() > 0
        ? Math.addExact(System.currentTimeMillis(), params.getTimeoutMillis())
        : 0;
    return new JavaSubprocess(builder.start(), deadlineMillis);
  }

  /**
   * Returns a {@link java.lang.ProcessBuilder.Redirect} appropriate for the parameters. If a file
   * redirected to exists, deletes the file before redirecting to it.
   */
  private Redirect getRedirect(StreamAction action, File file) {
    switch (action) {
      case DISCARD:
        return Redirect.to(new File("/dev/null"));

      case REDIRECT:
        // We need to use Redirect.appendTo() here, because on older Linux kernels writes are
        // otherwise not atomic and might result in lost log messages:
        // https://lkml.org/lkml/2014/3/3/308
        if (file.exists()) {
          file.delete();
        }
        return Redirect.appendTo(file);

      case STREAM:
        return Redirect.PIPE;

      default:
        throw new IllegalStateException();
    }
  }
}
