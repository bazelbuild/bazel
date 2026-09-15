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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import javax.annotation.WillCloseWhenClosed;

/**
 * Input stream that guarantees {@link InterruptedIOException}.
 *
 * <p>This class exists to hedge against the possibility that the JVM might not implement this
 * functionality. See <a href="http://bugs.java.com/view_bug.do?bug_id=4385444">bug 4385444</a>.
 */
@ConditionallyThreadSafe
final class InterruptibleInputStream extends InputStream {

  private final InputStream delegate;

  InterruptibleInputStream(@WillCloseWhenClosed InputStream delegate) {
    this.delegate = delegate;
  }

  @Override
  public int read() throws IOException {
    check();
    return delegate.read();
  }

  @Override
  public int read(byte[] buffer) throws IOException {
    check();
    return delegate.read(buffer);
  }

  @Override
  public int read(byte[] buffer, int offset, int length) throws IOException {
    check();
    return delegate.read(buffer, offset, length);
  }

  @Override
  public int available() throws IOException {
    return delegate.available();
  }

  @Override
  public boolean markSupported() {
    return delegate.markSupported();
  }

  @Override
  @SuppressWarnings("sync-override")
  public void mark(int readlimit) {
    delegate.mark(readlimit);
  }

  @Override
  @SuppressWarnings("sync-override")
  public void reset() throws IOException {
    delegate.reset();
  }

  @Override
  public void close() throws IOException {
    delegate.close();
  }

  private static void check() throws InterruptedIOException {
    if (Thread.interrupted()) {
      throw new InterruptedIOException();
    }
  }
}
