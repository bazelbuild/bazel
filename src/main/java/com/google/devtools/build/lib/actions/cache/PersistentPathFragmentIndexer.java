// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions.cache;

import static java.lang.Math.max;

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.devtools.build.lib.util.MapCodec;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReentrantLock;
import javax.annotation.Nullable;

/**
 * Provides persistent bidirectional {@link PathFragment} ⇔ unique {@link Integer} mapping.
 *
 * <p>{@link Integer} is used in place of primitive {@code int} as it is assumed that indices will
 * be stored in maps. Passing {@link Integer} across API boundaries makes it less likely to store
 * duplicate instances and create garbage due to autoboxing.
 *
 * <p>This class is backed by a {@link PersistentMap} that holds one direction of the
 * canonicalization mapping. The other direction is handled purely in memory and reconstituted at
 * load-time.
 *
 * <p>Thread-safety is ensured by locking on all mutating operations. Read-only operations are not
 * locked.
 */
@ConditionallyThreadSafe // Each instance must be instantiated with a different dataPath.
final class PersistentPathFragmentIndexer {

  private static final int INITIAL_CAPACITY = 8192;

  /** Instantiates and loads instance of the persistent path fragment indexer. */
  static PersistentPathFragmentIndexer create(Path dataPath, Path journalPath, Clock clock)
      throws IOException {
    PersistentIndexMap pathToInt = new PersistentIndexMap(dataPath, journalPath, clock);

    // INITIAL_CAPACITY or the next power of two greater than the size.
    int capacity = max(INITIAL_CAPACITY, Integer.highestOneBit(pathToInt.size()) << 1);

    var intToPath = new AtomicReferenceArray<PathFragment>(capacity);
    for (Map.Entry<PathFragment, Integer> entry : pathToInt.entrySet()) {
      int index = entry.getValue();
      if (index < 0 || index >= capacity) {
        throw new IOException(
            String.format(
                "Corrupted filename index %d out of bounds for length %d (map size %d)",
                index, capacity, pathToInt.size()));
      }
      if (intToPath.getAndSet(index, entry.getKey()) != null) {
        throw new IOException("Corrupted filename index has duplicate entry: " + entry.getKey());
      }
    }
    return new PersistentPathFragmentIndexer(pathToInt, intToPath);
  }

  private final ReentrantLock lock = new ReentrantLock();

  // These two fields act similarly to a (synchronized) BiMap. Mutating operations are performed in
  // synchronized blocks. Reads are done lock-free.
  private final PersistentIndexMap pathToInt;
  private volatile AtomicReferenceArray<PathFragment> intToPath;

  private PersistentPathFragmentIndexer(
      PersistentIndexMap pathToInt, AtomicReferenceArray<PathFragment> intToPath) {
    this.pathToInt = pathToInt;
    this.intToPath = intToPath;
  }

  /** Removes all mappings. */
  public void clear() {
    lock.lock();
    try {
      pathToInt.clear();
      intToPath = new AtomicReferenceArray<>(INITIAL_CAPACITY);
    } finally {
      lock.unlock();
    }
  }

  /** Returns the number of path fragments in the index. */
  public int size() {
    return pathToInt.size();
  }

  /**
   * Creates new mapping for the given path fragment if necessary and returns path fragment index.
   */
  public Integer getOrCreateIndex(PathFragment path) {
    Integer i = pathToInt.get(path);
    if (i != null) {
      return i;
    }
    lock.lock();
    try {
      i = pathToInt.size();
      Integer existing = pathToInt.putIfAbsent(path, i);
      if (existing != null) {
        return existing; // Another thread won the race.
      }
      int capacity = intToPath.length();
      if (i == capacity) {
        intToPath = copyOf(intToPath, capacity * 2);
      }
      intToPath.set(i, path);
      return i;
    } finally {
      lock.unlock();
    }
  }

  private static AtomicReferenceArray<PathFragment> copyOf(
      AtomicReferenceArray<PathFragment> oldArray, int newCapacity) {
    var newArray = new AtomicReferenceArray<PathFragment>(newCapacity);
    for (int j = 0; j < oldArray.length(); j++) {
      newArray.setPlain(j, oldArray.getPlain(j));
    }
    return newArray;
  }

  /**
   * Returns the unique index for the given path fragment if one was created via {@link
   * #getOrCreateIndex}, or else {@code null}.
   */
  @Nullable
  public Integer getIndex(PathFragment path) {
    return pathToInt.get(path);
  }

  /** Returns the indexed path fragment equal to {@code path}, or {@code path} if there is none. */
  public PathFragment canonicalize(PathFragment path) {
    Integer i = pathToInt.get(path);
    if (i == null) {
      return path;
    }
    // Null if a concurrent getOrCreateIndex hasn't set it yet.
    PathFragment indexed = getPathForIndex(i);
    return indexed != null ? indexed : path;
  }

  /**
   * Returns the path fragment associated with the given index or {@code null} if it is not in the
   * index.
   */
  @Nullable
  public PathFragment getPathForIndex(int i) {
    if (i < 0) {
      return null;
    }
    var snapshot = intToPath;
    return i < snapshot.length() ? snapshot.get(i) : null;
  }

  /** Saves index data to the file. */
  long save() throws IOException {
    lock.lock();
    try {
      return pathToInt.save();
    } finally {
      lock.unlock();
    }
  }

  /** Flushes the journal. */
  void flush() {
    lock.lock();
    try {
      pathToInt.flush();
    } finally {
      lock.unlock();
    }
  }

  public void dump(PrintStream out) {
    lock.lock();
    try {
      out.format("Path indexer (%d records):\n", size());
      for (int i = 0; i < size(); i++) {
        out.format("  %s <=> %s\n", i, getPathForIndex(i));
      }
    } finally {
      lock.unlock();
    }
  }

  private static final MapCodec<PathFragment, Integer> CODEC =
      new MapCodec<>() {
        @Override
        protected PathFragment readKey(DataInput in) throws IOException {
          int length = in.readInt();
          if (length < 0) {
            throw new IOException("corrupt key length: " + length);
          }
          byte[] content = new byte[length];
          in.readFully(content);
          // Only normalized path fragments are written.
          return PathFragment.createAlreadyNormalized(
              StringUnsafe.newInstance(content, StringUnsafe.LATIN1));
        }

        @Override
        protected Integer readValue(DataInput in) throws IOException {
          return in.readInt();
        }

        @Override
        protected void writeKey(PathFragment key, DataOutput out) throws IOException {
          byte[] content = StringUnsafe.getInternalStringBytes(key.getPathString());
          out.writeInt(content.length);
          out.write(content);
        }

        @Override
        protected void writeValue(Integer value, DataOutput out) throws IOException {
          out.writeInt(value);
        }
      };

  /**
   * Persistent metadata map. Used as a backing map to provide a persistent implementation of the
   * metadata cache.
   */
  private static final class PersistentIndexMap extends PersistentMap<PathFragment, Integer> {
    private static final int VERSION = 0x02;
    private static final long SAVE_INTERVAL_NS = 3L * 1000 * 1000 * 1000;

    private final Clock clock;
    private long nextUpdate;

    PersistentIndexMap(Path mapFile, Path journalFile, Clock clock) throws IOException {
      super(VERSION, CODEC, new ConcurrentHashMap<>(INITIAL_CAPACITY), mapFile, journalFile);
      this.clock = clock;
      nextUpdate = clock.nanoTime();
      load();
    }

    @Override
    protected boolean shouldFlushJournal() {
      long time = clock.nanoTime();
      if (SAVE_INTERVAL_NS == 0L || time > nextUpdate) {
        nextUpdate = time + SAVE_INTERVAL_NS;
        return true;
      }
      return false;
    }

    @Override
    public Integer remove(Object object) {
      throw new UnsupportedOperationException();
    }

    void flush() {
      flushJournal();
    }
  }
}
