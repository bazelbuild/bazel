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

package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.ByteString;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Splits an {@link InputStream} into one or more {@link Chunk}s of at most {@code chunkSize} bytes.
 */
public final class Chunker {

  private static final Chunk EMPTY_CHUNK =
      new Chunk(Digests.computeDigest(new byte[0]), ByteString.EMPTY, 0);

  private static int defaultChunkSize = 1024 * 16;

  /** This method must only be called in tests! */
  @VisibleForTesting
  static void setDefaultChunkSizeForTesting(int value) {
    defaultChunkSize = value;
  }

  static int getDefaultChunkSize() {
    return defaultChunkSize;
  }

  /** A piece of a byte[] blob. */
  public static final class Chunk {

    private final Digest digest;
    private final long offset;
    private final ByteString data;

    private Chunk(Digest digest, ByteString data, long offset) {
      this.digest = digest;
      this.data = data;
      this.offset = offset;
    }

    public Digest getDigest() {
      return digest;
    }

    public long getOffset() {
      return offset;
    }

    public ByteString getData() {
      return data;
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof Chunk)) {
        return false;
      }
      Chunk other = (Chunk) o;
      return other.offset == offset
          && other.digest.equals(digest)
          && other.data.equals(data);
    }

    @Override
    public int hashCode() {
      return Objects.hash(digest, offset, data);
    }
  }

  private final Supplier<InputStream> dataSupplier;
  private final Digest digest;
  private final int chunkSize;

  private InputStream data;
  private long offset;
  private byte[] chunkCache;

  // Set to true on the first call to next(). This is so that the Chunker can open its data source
  // lazily on the first call to next(), as opposed to opening it in the constructor.
  private boolean initialized;

  public Chunker(byte[] data) throws IOException {
    this(data, getDefaultChunkSize());
  }

  public Chunker(byte[] data, int chunkSize) throws IOException {
    this(() -> new ByteArrayInputStream(data), Digests.computeDigest(data), chunkSize);
  }

  public Chunker(Path file) throws IOException {
    this(file, getDefaultChunkSize());
  }

  public Chunker(Path file, int chunkSize) throws IOException {
    this(() -> {
      try {
        return file.getInputStream();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }, Digests.computeDigest(file), chunkSize);
  }

  public Chunker(ActionInput actionInput, ActionInputFileCache inputCache, Path execRoot) throws
      IOException{
    this(actionInput, inputCache, execRoot, getDefaultChunkSize());
  }

  public Chunker(ActionInput actionInput, ActionInputFileCache inputCache, Path execRoot,
      int chunkSize)
      throws IOException {
    this(() -> {
      try {
        return execRoot.getRelative(actionInput.getExecPathString()).getInputStream();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }, Digests.getDigestFromInputCache(actionInput, inputCache), chunkSize);
  }

  private Chunker(Supplier<InputStream> dataSupplier, Digest digest, int chunkSize)
      throws IOException {
    this.dataSupplier = checkNotNull(dataSupplier);
    this.digest = checkNotNull(digest);
    this.chunkSize = chunkSize;
  }

  public Digest digest() {
    return digest;
  }

  /**
   * Reset the {@link Chunker} state to when it was newly constructed.
   */
  public void reset() throws IOException {
    if (data != null) {
      data.close();
    }
    try {
      data = dataSupplier.get();
    } catch (RuntimeException e) {
      Throwables.propagateIfPossible(e.getCause(), IOException.class);
      throw e;
    }
    offset = 0;
    chunkCache = null;
  }

  /**
   * Returns {@code true} if a subsequent call to {@link #next()} returns a {@link Chunk} object;
   */
  public boolean hasNext() {
    return data != null || !initialized;
  }

  /**
   * Returns the next {@link Chunk} or throws a {@link NoSuchElementException} if no data is left.
   *
   * <p>Always call {@link #hasNext()} before calling this method.
   *
   * <p>Zero byte inputs are treated special. Instead of throwing a {@link NoSuchElementException}
   * on the first call to {@link #next()}, a {@link Chunk} with an empty {@link ByteString} is
   * returned.
   */
  public Chunk next() throws IOException {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }

    if (!initialized) {
      reset();
      initialized = true;
    }

    if (digest.getSizeBytes() == 0) {
      data = null;
      return EMPTY_CHUNK;
    }

    // The cast to int is safe, because the return value is capped at chunkSize.
    int bytesToRead = (int) Math.min(bytesLeft(), chunkSize);
    if (bytesToRead == 0) {
      chunkCache = null;
      data = null;
      throw new NoSuchElementException();
    }

    if (chunkCache == null) {
      // Lazily allocate it in order to save memory on small data.
      // 1) bytesToRead < chunkSize: There will only ever be one next() call.
      // 2) bytesToRead == chunkSize: chunkCache will be set to its biggest possible value.
      // 3) bytestoRead > chunkSize: Not possible, due to Math.min above.
      chunkCache = new byte[bytesToRead];
    }

    long offsetBefore = offset;
    try {
      ByteStreams.readFully(data, chunkCache, 0, bytesToRead);
    } catch (EOFException e) {
      throw new IllegalStateException("Reached EOF, but expected "
          + bytesToRead + " bytes.", e);
    }
    offset += bytesToRead;

    ByteString blob = ByteString.copyFrom(chunkCache, 0, bytesToRead);

    if (bytesLeft() == 0) {
      data.close();
      data = null;
      chunkCache = null;
    }

    return new Chunk(digest, blob, offsetBefore);
  }

  private long bytesLeft() {
    return digest.getSizeBytes() - offset;
  }
}
