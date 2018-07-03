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

import com.google.common.hash.HashCode;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hasher;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;
import javax.annotation.WillCloseWhenClosed;

/**
 * Input stream that guarantees its contents matches a hash code.
 *
 * <p>The actual checksum is computed gradually as the input is read. If it doesn't match, then an
 * {@link IOException} will be thrown when {@link #close()} is called, or when any read method is
 * called that detects the end of stream. This error will be thrown multiple times if these methods
 * are called again for some reason.
 *
 * <p>This class is not thread safe, but it is safe to message pass this object between threads.
 */
@ThreadCompatible
final class HashInputStream extends InputStream {

  private final InputStream delegate;
  private final Hasher hasher;
  private final HashCode code;
  @Nullable private volatile HashCode actual;

  HashInputStream(
      @WillCloseWhenClosed InputStream delegate, HashFunction function, HashCode code) {
    this.delegate = delegate;
    this.hasher = function.newHasher();
    this.code = code;
  }

  @Override
  public int read() throws IOException {
    int result = delegate.read();
    if (result == -1) {
      check();
    } else {
      hasher.putByte((byte) result);
    }
    return result;
  }

  @Override
  public int read(byte[] buffer, int offset, int length) throws IOException {
    int amount = delegate.read(buffer, offset, length);
    if (amount == -1) {
      check();
    } else {
      hasher.putBytes(buffer, offset, amount);
    }
    return amount;
  }

  @Override
  public int available() throws IOException {
    return delegate.available();
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
