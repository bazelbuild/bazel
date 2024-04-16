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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.util.StringIndexer;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Persistent implementation of {@link StringIndexer}.
 *
 * <p>This class is backed by a {@link PersistentMap} that holds one direction of the
 * canonicalization mapping. The other direction is handled purely in memory and reconstituted at
 * load-time.
 *
 * <p>Thread-safety is ensured by locking on all mutating operations from the superclass. Read-only
 * operations are not locked, but rather backed by ConcurrentMaps.
 */
@ConditionallyThreadSafe // Each instance must be instantiated with a different dataPath.
final class PersistentStringIndexer implements StringIndexer {

  private static final int NOT_FOUND = -1;
  private static final int INITIAL_ENTRIES = 10000;

  /** Instantiates and loads instance of the persistent string indexer. */
  static PersistentStringIndexer create(Path dataPath, Clock clock) throws IOException {
    PersistentIndexMap persistentIndexMap =
        new PersistentIndexMap(
            dataPath, FileSystemUtils.replaceExtension(dataPath, ".journal"), clock);
    Map<Integer, String> reverseMapping = new ConcurrentHashMap<>(INITIAL_ENTRIES);
    for (Map.Entry<String, Integer> entry : persistentIndexMap.entrySet()) {
      if (reverseMapping.put(entry.getValue(), entry.getKey()) != null) {
        throw new IOException("Corrupted filename index has duplicate entry: " + entry.getKey());
      }
    }
    return new PersistentStringIndexer(persistentIndexMap, reverseMapping);
  }

  // This is similar to (Synchronized) BiMap.
  // These maps *must* be weakly threadsafe to ensure thread safety for string indexer as a whole.
  // Specifically, mutating operations are serialized, but read-only operations may be executed
  // concurrently with mutators.
  private final PersistentIndexMap stringToInt;
  private final Map<Integer, String> intToString;

  private PersistentStringIndexer(
      PersistentIndexMap stringToInt, Map<Integer, String> intToString) {
    this.stringToInt = stringToInt;
    this.intToString = intToString;
  }

  @Override
  public synchronized void clear() {
    stringToInt.clear();
    intToString.clear();
  }

  @Override
  public int size() {
    return intToString.size();
  }

  @Override
  public int getOrCreateIndex(String s) {
    Integer i = stringToInt.get(s);
    if (i == null) {
      s = StringCanonicalizer.intern(s);
      synchronized (this) {
        // First, make sure another thread hasn't just added the entry:
        i = stringToInt.get(s);
        if (i != null) {
          return i;
        }

        int ind = intToString.size();
        stringToInt.put(s, ind);
        intToString.put(ind, s);
        return ind;
      }
    } else {
      return i;
    }
  }

  @Override
  public int getIndex(String s) {
    return stringToInt.getOrDefault(s, NOT_FOUND);
  }

  @Override
  @Nullable
  public String getStringForIndex(int i) {
    return intToString.get(i);
  }

  /** Saves index data to the file. */
  synchronized long save() throws IOException {
    return stringToInt.save();
  }

  /** Flushes the journal. */
  synchronized void flush() {
    stringToInt.flush();
  }

  @Override
  public synchronized String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("size = ").append(size()).append("\n");
    for (Map.Entry<String, Integer> entry : stringToInt.entrySet()) {
      builder.append(entry.getKey()).append(" <==> ").append(entry.getValue()).append("\n");
    }
    return builder.toString();
  }

  /**
   * Persistent metadata map. Used as a backing map to provide a persistent implementation of the
   * metadata cache.
   */
  private static final class PersistentIndexMap extends PersistentMap<String, Integer> {
    private static final int VERSION = 0x01;
    private static final long SAVE_INTERVAL_NS = 3L * 1000 * 1000 * 1000;

    private final Clock clock;
    private long nextUpdate;

    PersistentIndexMap(Path mapFile, Path journalFile, Clock clock) throws IOException {
      super(VERSION, new ConcurrentHashMap<>(INITIAL_ENTRIES), mapFile, journalFile);
      this.clock = clock;
      nextUpdate = clock.nanoTime();
      load(/* failFast= */ true);
    }

    @Override
    protected boolean updateJournal() {
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
      forceFlush();
    }

    @Override
    protected String readKey(DataInputStream in) throws IOException {
      int length = in.readInt();
      if (length < 0) {
        throw new IOException("corrupt key length: " + length);
      }
      byte[] content = new byte[length];
      in.readFully(content);
      return new String(content, UTF_8);
    }

    @Override
    protected Integer readValue(DataInputStream in) throws IOException {
      return in.readInt();
    }

    @Override
    protected void writeKey(String key, DataOutputStream out) throws IOException {
      byte[] content = key.getBytes(UTF_8);
      out.writeInt(content.length);
      out.write(content);
    }

    @Override
    protected void writeValue(Integer value, DataOutputStream out) throws IOException {
      out.writeInt(value);
    }
  }
}
