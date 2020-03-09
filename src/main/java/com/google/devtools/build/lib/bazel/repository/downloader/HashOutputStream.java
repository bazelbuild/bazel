// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;
import javax.annotation.WillCloseWhenClosed;

/**
 * Output stream that guarantees its contents matches a hash code.
 *
 * <p>The actual checksum is computed gradually as the output is written. If it doesn't match, then
 * an {@link IOException} will be thrown when {@link #close()} is called. This error will be thrown
 * multiple times if these methods are called again for some reason.
 *
 * <p>Note that as the checksum can only be computed once the stream is closed, data will be written
 * to the underlying stream regardless of whether it matches the expected checksum.
 *
 * <p>This class is not thread safe, but it is safe to message pass this object between threads.
 */
@ThreadCompatible
public final class HashOutputStream extends OutputStream {

  private final OutputStream delegate;
  private final Hasher hasher;
  private final HashCode code;
  @Nullable private volatile HashCode actual;

  public HashOutputStream(@WillCloseWhenClosed OutputStream delegate, Checksum checksum) {
    this.delegate = delegate;
    this.hasher = checksum.getKeyType().newHasher();
    this.code = checksum.getHashCode();
  }

  @Override
  public void write(int buffer) throws IOException {
    hasher.putByte((byte) buffer);
    delegate.write(buffer);
  }

  @Override
  public void write(byte[] buffer) throws IOException {
    hasher.putBytes(buffer);
    delegate.write(buffer);
  }

  @Override
  public void write(byte[] buffer, int offset, int length) throws IOException {
    hasher.putBytes(buffer, offset, length);
    delegate.write(buffer, offset, length);
  }

  @Override
  public void flush() throws IOException {
    delegate.flush();
  }

  @Override
  public void close() throws IOException {
    delegate.close();
    check();
  }

  private void check() throws IOException {
    if (actual == null) {
      actual = hasher.hash();
    }
    if (!code.equals(actual)) {
      throw new UnrecoverableHttpException(
          String.format("Checksum was %s but wanted %s", actual, code));
    }
  }
}
