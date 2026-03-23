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

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.devtools.build.lib.util.MapCodec;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.util.StringIndexer;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import javax.annotation.Nullable;

/**
 * Persistent implementation of {@link StringIndexer}.
 *
 * <p>This class is backed by a {@link PersistentMap} that holds one direction of the
 * canonicalization mapping. The other direction is handled purely in memory and reconstituted at
 * load-time.
 *
 * <p>Thread-safety is ensured by locking on all mutating operations. Read-only operations are not
 * locked.
 */
@ConditionallyThreadSafe // Each instance must be instantiated with a different dataPath.
final class PersistentStringIndexer implements StringIndexer {

  private static final int INITIAL_CAPACITY = 8192;

  /** Instantiates and loads instance of the persistent string indexer. */
  static PersistentStringIndexer create(Path dataPath, Path journalPath, Clock clock)
      throws IOException {
    PersistentIndexMap stringToInt = new PersistentIndexMap(dataPath, journalPath, clock);
    PagedStringList intToString = new PagedStringList();
    int mapSize = stringToInt.size();
    for (Map.Entry<String, Integer> entry : stringToInt.entrySet()) {
      int index = entry.getValue();
      if (index < 0 || index >= mapSize) {
        throw new IOException(
            String.format(
                "Corrupted filename index %d out of bounds (map size %d)",
                index, stringToInt.size()));
      }
      if (!intToString.set(index, entry.getKey())) {
        throw new IOException("Corrupted filename index has duplicate entry: " + entry.getKey());
      }
    }
    var nextIndex = new AtomicInteger(mapSize);
    return new PersistentStringIndexer(stringToInt, intToString, nextIndex);
  }

  private final PersistentIndexMap stringToInt;
  private final PagedStringList intToString;
  private final AtomicInteger nextIndex;

  private PersistentStringIndexer(
      PersistentIndexMap stringToInt, PagedStringList intToString, AtomicInteger nextIndex) {
    this.stringToInt = stringToInt;
    this.intToString = intToString;
    this.nextIndex = nextIndex;
  }

  @Override
  public void clear() {
    stringToInt.clear();
    intToString.clear();
    nextIndex.set(0);
  }

  @Override
  public int size() {
    return stringToInt.size();
  }

  @Override
  public Integer getOrCreateIndex(String s) {
    Integer i = stringToInt.get(s);
    if (i != null) {
      return i;
    }
    s = s.intern();
    return stringToInt.computeIfAbsent(
        s,
        k -> {
          int index = nextIndex.getAndIncrement();
          intToString.set(index, k);
          return index;
        });
  }

  @Override
  @Nullable
  public Integer getIndex(String s) {
    return stringToInt.get(s);
  }

  @Override
  @Nullable
  public String getStringForIndex(Integer i) {
    if (i < 0) {
      return null;
    }
    return intToString.get(i);
  }

  /** Saves index data to the file. */
  long save() throws IOException {
    return stringToInt.save();
  }

  /** Flushes the journal. */
  void flush() {
    stringToInt.flush();
  }

  public void dump(PrintStream out) {
    out.format("String indexer (%d records):\n", size());
    for (int i = 0; i < size(); i++) {
      out.format("  %s <=> %s\n", i, getStringForIndex(i));
    }
  }

  private static final MapCodec<String, Integer> CODEC =
      new MapCodec<>() {
        @Override
        protected String readKey(DataInput in) throws IOException {
          int length = in.readInt();
          if (length < 0) {
            throw new IOException("corrupt key length: " + length);
          }
          byte[] content = new byte[length];
          in.readFully(content);
          return StringUnsafe.newInstance(content, StringUnsafe.LATIN1);
        }

        @Override
        protected Integer readValue(DataInput in) throws IOException {
          return in.readInt();
        }

        @Override
        protected void writeKey(String key, DataOutput out) throws IOException {
          byte[] content = StringUnsafe.getInternalStringBytes(key);
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
  private static final class PersistentIndexMap extends PersistentMap<String, Integer> {
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

  /**
   * A thread-safe paged list of strings.
   *
   * <p>This is efficient in avoiding contention because initializing a new page doesn't require
   * copying data.
   */
  private static final class PagedStringList {
    private static final int PAGE_SIZE = 8192;
    private static final int MAX_PAGES = 65536;

    // Over 536 million. There is at most one string per output artifact, so builds should never
    // approach this number.
    private static final int MAX_CAPACITY = PAGE_SIZE * MAX_PAGES;

    private AtomicReferenceArray<AtomicReferenceArray<String>> pages;

    PagedStringList() {
      init();
    }

    private void init() {
      pages = new AtomicReferenceArray<>(MAX_PAGES);
      // Eagerly initialize the first page to reduce contention.
      pages.set(0, new AtomicReferenceArray<>(PAGE_SIZE));
    }

    void clear() {
      init();
    }

    // Returns null so that any out of bounds access is treated as a corrupt index. This can happen
    // with arbitrary modifications to the action cache files.
    @Nullable
    String get(int index) {
      int pageIndex = index / PAGE_SIZE;
      int elementIndex = index % PAGE_SIZE;
      if (pageIndex >= pages.length()) {
        return null;
      }
      AtomicReferenceArray<String> page = pages.get(pageIndex);
      return page != null ? page.get(elementIndex) : null;
    }

    @CanIgnoreReturnValue // Used for corruption check when loading from a file.
    boolean set(int index, String value) {
      checkState(index < MAX_CAPACITY, "Max capacity exceeded");
      int pageIndex = index / PAGE_SIZE;
      int elementIndex = index % PAGE_SIZE;
      AtomicReferenceArray<String> page = getOrCreatePage(pageIndex);
      return page.getAndSet(elementIndex, value) == null;
    }

    private AtomicReferenceArray<String> getOrCreatePage(int pageIndex) {
      AtomicReferenceArray<String> page = pages.get(pageIndex);
      if (page == null) {
        synchronized (this) {
          page = pages.get(pageIndex);
          if (page == null) {
            page = new AtomicReferenceArray<>(PAGE_SIZE);
            pages.set(pageIndex, page);
          }
        }
      }
      return page;
    }
  }
}
