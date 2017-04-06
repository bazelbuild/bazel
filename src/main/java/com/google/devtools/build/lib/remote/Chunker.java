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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.remote.RemoteProtocol.BlobChunk;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Set;
import javax.annotation.Nullable;

/** An iterator-type object that transforms byte sources into a stream of BlobChunk messages. */
public final class Chunker {
  /** An Item is an opaque digestable source of bytes. */
  interface Item {
    ContentDigest getDigest() throws IOException;

    InputStream getInputStream() throws IOException;
  }

  private final Iterator<Item> inputIterator;
  private InputStream currentStream;
  private final Set<ContentDigest> digests;
  private ContentDigest digest;
  private long bytesLeft;
  private final int chunkSize;

  Chunker(
      Iterator<Item> inputIterator,
      int chunkSize,
      // If present, specifies which digests to output out of the whole input.
      @Nullable Set<ContentDigest> digests)
      throws IOException {
    Preconditions.checkArgument(chunkSize > 0, "Chunk size must be greater than 0");
    this.digests = digests;
    this.inputIterator = inputIterator;
    this.chunkSize = chunkSize;
    advanceInput();
  }

  Chunker(Iterator<Item> inputIterator, int chunkSize) throws IOException {
    this(inputIterator, chunkSize, null);
  }

  Chunker(Item input, int chunkSize) throws IOException {
    this(Iterators.singletonIterator(input), chunkSize, ImmutableSet.of(input.getDigest()));
  }

  private void advanceInput() throws IOException {
    do {
      if (inputIterator != null && inputIterator.hasNext()) {
        Item input = inputIterator.next();
        digest = input.getDigest();
        currentStream = input.getInputStream();
        bytesLeft = digest.getSizeBytes();
      } else {
        digest = null;
        currentStream = null;
        bytesLeft = 0;
      }
    } while (digest != null && digests != null && !digests.contains(digest));
  }

  /** True if the object has more BlobChunk elements. */
  public boolean hasNext() {
    return currentStream != null;
  }

  /** Consume the next BlobChunk element. */
  public BlobChunk next() throws IOException {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    BlobChunk.Builder chunk = BlobChunk.newBuilder();
    long offset = digest.getSizeBytes() - bytesLeft;
    if (offset == 0) {
      chunk.setDigest(digest);
    } else {
      chunk.setOffset(offset);
    }
    if (bytesLeft > 0) {
      byte[] blob = new byte[(int) Math.min(bytesLeft, (long) chunkSize)];
      currentStream.read(blob);
      chunk.setData(ByteString.copyFrom(blob));
      bytesLeft -= blob.length;
    }
    if (bytesLeft == 0) {
      currentStream.close();
      advanceInput(); // Sets the current stream to null, if it was the last.
    }
    return chunk.build();
  }

  static Item toItem(final byte[] blob) {
    return new Item() {
      @Override
      public ContentDigest getDigest() throws IOException {
        return ContentDigests.computeDigest(blob);
      }

      @Override
      public InputStream getInputStream() throws IOException {
        return new ByteArrayInputStream(blob);
      }
    };
  }

  static Item toItem(final Path file) {
    return new Item() {
      @Override
      public ContentDigest getDigest() throws IOException {
        return ContentDigests.computeDigest(file);
      }

      @Override
      public InputStream getInputStream() throws IOException {
        return file.getInputStream();
      }
    };
  }

  static Item toItem(
      final ActionInput input, final ActionInputFileCache inputCache, final Path execRoot) {
    return new Item() {
      @Override
      public ContentDigest getDigest() throws IOException {
        return ContentDigests.getDigestFromInputCache(input, inputCache);
      }

      @Override
      public InputStream getInputStream() throws IOException {
        return execRoot.getRelative(input.getExecPathString()).getInputStream();
      }
    };
  }

  /**
   * Create a Chunker from a given ActionInput, taking its digest from the provided
   * ActionInputFileCache.
   */
  public static Chunker from(
      ActionInput input, int chunkSize, ActionInputFileCache inputCache, Path execRoot)
      throws IOException {
    return new Chunker(toItem(input, inputCache, execRoot), chunkSize);
  }

  /** Create a Chunker from a given blob and chunkSize. */
  public static Chunker from(byte[] blob, int chunkSize) throws IOException {
    return new Chunker(toItem(blob), chunkSize);
  }

  /** Create a Chunker from a given Path and chunkSize. */
  public static Chunker from(Path file, int chunkSize) throws IOException {
    return new Chunker(toItem(file), chunkSize);
  }

  /**
   * Create a Chunker from multiple input sources. The order of the sources provided to the Builder
   * will be the same order they will be chunked by.
   */
  public static final class Builder {
    private final ArrayList<Item> items = new ArrayList<>();
    private Set<ContentDigest> digests = null;
    private int chunkSize = 0;

    public Chunker build() throws IOException {
      return new Chunker(items.iterator(), chunkSize, digests);
    }

    public Builder chunkSize(int chunkSize) {
      this.chunkSize = chunkSize;
      return this;
    }

    /**
     * Restricts the Chunker to use only inputs with these digests. This is an optimization for CAS
     * uploads where a list of digests missing from the CAS is known.
     */
    public Builder onlyUseDigests(Set<ContentDigest> digests) {
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
