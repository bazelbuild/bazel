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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.Digest;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;

/** An iterator-type object that transforms byte sources into a stream of Chunks. */
public final class Chunker {
  // This is effectively final, should be changed only in unit-tests!
  private static int defaultChunkSize = 1024 * 16;
  private static final byte[] EMPTY_BLOB = new byte[0];

  @VisibleForTesting
  static void setDefaultChunkSizeForTesting(int value) {
    defaultChunkSize = value;
  }

  public static int getDefaultChunkSize() {
    return defaultChunkSize;
  }

  /** A piece of a byte[] blob. */
  public static final class Chunk {

    private final Digest digest;
    private final long offset;
    // TODO(olaola): consider saving data in a different format that byte[].
    private final byte[] data;

    @VisibleForTesting
    Chunk(Digest digest, byte[] data, long offset) {
      this.digest = digest;
      this.data = data;
      this.offset = offset;
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
          && Arrays.equals(other.data, data);
    }

    @Override
    public int hashCode() {
      return Objects.hash(digest, offset, Arrays.hashCode(data));
    }

    public Digest getDigest() {
      return digest;
    }

    public long getOffset() {
      return offset;
    }

    // This returns a mutable copy, for efficiency.
    public byte[] getData() {
      return data;
    }
  }

  /** An Item is an opaque digestable source of bytes. */
  interface Item {
    Digest getDigest() throws IOException;

    InputStream getInputStream() throws IOException;
  }

  private final Iterator<Item> inputIterator;
  private InputStream currentStream;
  private Digest digest;
  private long bytesLeft;
  private final int chunkSize;

  Chunker(Iterator<Item> inputIterator, int chunkSize) throws IOException {
    Preconditions.checkArgument(chunkSize > 0, "Chunk size must be greater than 0");
    this.inputIterator = inputIterator;
    this.chunkSize = chunkSize;
    advanceInput();
  }

  Chunker(Item input, int chunkSize) throws IOException {
    this(Iterators.singletonIterator(input), chunkSize);
  }

  public void advanceInput() throws IOException {
    if (inputIterator.hasNext()) {
      Item input = inputIterator.next();
      digest = input.getDigest();
      currentStream = input.getInputStream();
      bytesLeft = digest.getSizeBytes();
    } else {
      digest = null;
      currentStream = null;
      bytesLeft = 0;
    }
  }

  /** True if the object has more {@link Chunk} elements. */
  public boolean hasNext() {
    return currentStream != null;
  }

  /** Consume the next Chunk element. */
  public Chunk next() throws IOException {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    long offset = digest.getSizeBytes() - bytesLeft;
    byte[] blob = EMPTY_BLOB;
    if (bytesLeft > 0) {
      blob = new byte[(int) Math.min(bytesLeft, chunkSize)];
      currentStream.read(blob);
      bytesLeft -= blob.length;
    }
    Chunk result = new Chunk(digest, blob, offset);
    if (bytesLeft == 0) {
      currentStream.close();
      advanceInput(); // Sets the current stream to null, if it was the last.
    }
    return result;
  }

  private static Item toItem(final byte[] blob) {
    return new Item() {
      Digest digest = null;

      @Override
      public Digest getDigest() throws IOException {
        if (digest == null) {
          digest = Digests.computeDigest(blob);
        }
        return digest;
      }

      @Override
      public InputStream getInputStream() throws IOException {
        return new ByteArrayInputStream(blob);
      }
    };
  }

  private static Item toItem(final Path file) {
    return new Item() {
      Digest digest = null;

      @Override
      public Digest getDigest() throws IOException {
        if (digest == null) {
          digest = Digests.computeDigest(file);
        }
        return digest;
      }

      @Override
      public InputStream getInputStream() throws IOException {
        return file.getInputStream();
      }
    };
  }

  private static Item toItem(
      final ActionInput input, final ActionInputFileCache inputCache, final Path execRoot) {
    if (input instanceof VirtualActionInput) {
      return toItem((VirtualActionInput) input);
    }
    return new Item() {
      @Override
      public Digest getDigest() throws IOException {
        return Digests.getDigestFromInputCache(input, inputCache);
      }

      @Override
      public InputStream getInputStream() throws IOException {
        return execRoot.getRelative(input.getExecPathString()).getInputStream();
      }
    };
  }

  private static Item toItem(final VirtualActionInput input) {
    return new Item() {
      Digest digest = null;

      @Override
      public Digest getDigest() throws IOException {
        if (digest == null) {
          digest = Digests.computeDigest(input);
        }
        return digest;
      }

      @Override
      public InputStream getInputStream() throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        input.writeTo(buffer);
        return new ByteArrayInputStream(buffer.toByteArray());
      }
    };
  }

  private static class MemberOf implements Predicate<Item> {
    private final Set<Digest> digests;

    public MemberOf(Set<Digest> digests) {
      this.digests = digests;
    }

    @Override
    public boolean apply(Item item) {
      try {
        return digests.contains(item.getDigest());
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /**
   * Create a Chunker from multiple input sources. The order of the sources provided to the Builder
   * will be the same order they will be chunked by.
   */
  public static final class Builder {
    private final ImmutableList.Builder<Item> items = ImmutableList.builder();
    private Set<Digest> digests = null;
    private int chunkSize = getDefaultChunkSize();

    public Chunker build() throws IOException {
      return new Chunker(
          digests == null
              ? items.build().iterator()
              : Iterators.filter(items.build().iterator(), new MemberOf(digests)),
          chunkSize);
    }

    public Builder chunkSize(int chunkSize) {
      this.chunkSize = chunkSize;
      return this;
    }

    /**
     * Restricts the Chunker to use only inputs with these digests. This is an optimization for CAS
     * uploads where a list of digests missing from the CAS is known.
     */
    public Builder onlyUseDigests(Set<Digest> digests) {
      this.digests = digests;
      return this;
    }

    public Builder addInput(byte[] blob) {
      items.add(toItem(blob));
      return this;
    }

    public Builder addInput(Path file) {
      items.add(toItem(file));
      return this;
    }

    public Builder addInput(ActionInput input, ActionInputFileCache inputCache, Path execRoot) {
      items.add(toItem(input, inputCache, execRoot));
      return this;
    }

    public Builder addAllInputs(
        Collection<? extends ActionInput> inputs, ActionInputFileCache inputCache, Path execRoot) {
      for (ActionInput input : inputs) {
        items.add(toItem(input, inputCache, execRoot));
      }
      return this;
    }
  }
}
