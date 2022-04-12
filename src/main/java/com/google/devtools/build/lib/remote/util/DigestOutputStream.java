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

package com.google.devtools.build.lib.remote.util;

import static com.google.common.base.Preconditions.checkNotNull;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hasher;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An {@link OutputStream} that maintains a {@link Digest} of the data written to it.
 *
 * <p>Similar to Guava's {@link com.google.common.hash.HashingOutputStream}, but for computing the
 * {@link Digest} of the written data as specified by the remote execution protocol.
 */
public final class DigestOutputStream extends FilterOutputStream {
  private final Hasher hasher;
  private long size = 0;

  /**
   * Creates an output stream that creates an {@link Digest} using the given {@link HashFunction},
   * and forwards all data written to it to the underlying {@link OutputStream}.
   *
   * <p>The {@link OutputStream} should not be written to before or after the hand-off.
   */
  public DigestOutputStream(HashFunction hashFunction, OutputStream out) {
    super(checkNotNull(out));
    this.hasher = checkNotNull(hashFunction.newHasher());
  }

  @Override
  public void write(int b) throws IOException {
    size++;
    hasher.putByte((byte) b);
    out.write(b);
  }

  @Override
  public void write(byte[] bytes, int off, int len) throws IOException {
    size += len;
    hasher.putBytes(bytes, off, len);
    out.write(bytes, off, len);
  }

  /**
   * Returns the {@link Digest} of the data written to this stream. The result is unspecified if
   * this method is called more than once on the same instance.
   */
  public Digest digest() {
    return Digest.newBuilder().setHash(hasher.hash().toString()).setSizeBytes(size).build();
  }

  @Override
  public void close() throws IOException {
    out.close();
  }
}
