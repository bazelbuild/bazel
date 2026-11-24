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
package com.google.devtools.build.lib.util.io;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.IntMath.ceilingPowerOfTwo;

import java.io.OutputStream;
import java.util.Arrays;
import java.util.Objects;

/** A {@link java.io.ByteArrayOutputStream} variant that doesn't synchronize. */
public final class UnsynchronizedByteArrayOutputStream extends OutputStream {

  private static final int DEFAULT_INITIAL_SIZE = 32;

  private byte[] buf;
  private int count;

  /**
   * Creates a new {@link UnsynchronizedByteArrayOutputStream} with a default initial buffer size.
   */
  public UnsynchronizedByteArrayOutputStream() {
    this(DEFAULT_INITIAL_SIZE);
  }

  /**
   * Creates a new {@link UnsynchronizedByteArrayOutputStream} with the given initial buffer size.
   *
   * @param size the initial size.
   * @throws IllegalArgumentException if size is negative
   */
  public UnsynchronizedByteArrayOutputStream(int size) {
    checkArgument(size >= 0, "size must be non-negative");
    this.buf = new byte[size];
    this.count = 0;
  }

  @Override
  public void write(int b) {
    ensureAvailable(1);
    buf[count++] = (byte) b;
  }

  @Override
  public void write(byte[] b, int off, int len) {
    Objects.checkFromIndexSize(off, len, b.length);
    ensureAvailable(len);
    System.arraycopy(b, off, buf, count, len);
    count += len;
  }

  /** Returns the size of the current contents of this output stream. */
  public int size() {
    return count;
  }

  /** Discards the current contents of this output stream. */
  public void reset() {
    count = 0;
  }

  /** Returns the current contents of this output stream as a byte array. */
  public byte[] toByteArray() {
    return Arrays.copyOf(buf, count);
  }

  private void ensureAvailable(int len) {
    if (count + len > buf.length) {
      buf = Arrays.copyOf(buf, ceilingPowerOfTwo(count + len));
    }
  }
}
