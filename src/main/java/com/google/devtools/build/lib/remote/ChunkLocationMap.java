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

package com.google.devtools.build.lib.remote;

import build.bazel.remote.execution.v2.Digest;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A bounded map from a chunk digest to a place on local disk where that chunk was recently seen: a
 * whole file and a byte offset into it.
 *
 * <p>A location is only a hint. It points into a build output, which may be deleted or rewritten at
 * any time, including while it is being read. Chunks read from a location are therefore always
 * verified against the requested digest, and anything unexpected is reported as a miss.
 *
 * <p>The map stores no content of its own and is never written to disk. Losing it costs nothing
 * beyond the downloads it would have saved, so it lives in memory for as long as the server does.
 */
@ThreadSafe
public final class ChunkLocationMap {
  private static final int MAX_ENTRIES = 100_000; // ~20MB of heap.

  private final Cache<Digest, ChunkLocation> locations =
      Caffeine.newBuilder().maximumSize(MAX_ENTRIES).build();

  /**
   * Records where each chunk of {@code path} can be found, given that the contents of the file are
   * the concatenation of {@code chunkDigests}.
   */
  void addFile(Path path, List<Digest> chunkDigests) {
    Path hostPath = path.forHostFileSystem();
    long offset = 0;
    for (Digest chunkDigest : chunkDigests) {
      locations.put(chunkDigest, new ChunkLocation(hostPath, offset));
      offset += chunkDigest.getSizeBytes();
    }
  }

  /**
   * Reads a chunk from the location it was most recently seen at, verifying it against {@code
   * digest}.
   *
   * @param destination the file the chunk is about to be written to, if it is known. Locations
   *     inside that file are ignored, as the download in progress is overwriting it.
   * @return the contents of the chunk, or {@code null} if no usable location is known
   */
  @Nullable
  byte[] read(Digest digest, @Nullable Path destination, DigestUtil digestUtil) {
    ChunkLocation location = locations.getIfPresent(digest);
    if (location == null) {
      return null;
    }
    // Skipping the destination is an optimization, not a correctness requirement: the location
    // could still alias the destination through a symlink or hard link, in which case the read
    // below races with the download overwriting it. Since every chunk is verified, bytes that
    // match the digest are the correct chunk content no matter which file they were read from,
    // and anything else is reported as a miss.
    if (destination != null && location.path().equals(destination.forHostFileSystem())) {
      return null;
    }

    byte[] chunk = location.read(digest, digestUtil);
    if (chunk == null) {
      locations.asMap().remove(digest, location);
    }
    return chunk;
  }

  /** Removes all entries. */
  void clear() {
    locations.invalidateAll();
  }

  /** The byte offset of a chunk within a file that contained it when it was last seen. */
  private record ChunkLocation(Path path, long offset) {

    /** Returns the chunk at this location, or {@code null} if it is no longer there. */
    @Nullable
    byte[] read(Digest digest, DigestUtil digestUtil) {
      try (InputStream in = path.getInputStream()) {
        int size = Math.toIntExact(digest.getSizeBytes());
        in.skipNBytes(offset);
        byte[] chunk = in.readNBytes(size);
        return chunk.length == size && digest.equals(digestUtil.compute(chunk)) ? chunk : null;
      } catch (IOException | ArithmeticException e) {
        return null;
      }
    }
  }
}
