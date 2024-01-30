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

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.util.CanonicalStringIndexer;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * Persistent version of the CanonicalStringIndexer.
 *
 * <p>This class is backed by a PersistentMap that holds one direction of the canonicalization
 * mapping. The other direction is handled purely in memory and reconstituted at load-time.
 *
 * <p>Thread-safety is ensured by locking on all mutating operations from the superclass. Read-only
 * operations are not locked, but rather backed by ConcurrentMaps.
 */
@ConditionallyThreadSafe // condition: each instance must instantiated with
// different dataFile.
final class PersistentStringIndexer extends CanonicalStringIndexer {

  /**
   * Persistent metadata map. Used as a backing map to provide a persistent implementation of the
   * metadata cache.
   */
  private static final class PersistentIndexMap extends PersistentMap<String, Integer> {
    private static final int VERSION = 0x01;
    private static final long SAVE_INTERVAL_NS = 3L * 1000 * 1000 * 1000;

    private final Clock clock;
    private long nextUpdate;

    public PersistentIndexMap(Path mapFile, Path journalFile, Clock clock) throws IOException {
      super(
          VERSION,
          PersistentStringIndexer.<String, Integer>newConcurrentMap(INITIAL_ENTRIES),
          mapFile,
          journalFile);
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
    @Nullable
    public Integer remove(Object object) {
      throw new UnsupportedOperationException();
    }

    public void flush() {
      super.forceFlush();
    }

    @Override
    protected String readKey(DataInputStream in) throws IOException {
      int length = in.readInt();
      if (length < 0) {
        throw new IOException("corrupt key length: " + length);
      }
      byte[] content = new byte[length];
      in.readFully(content);
      return bytes2string(content);
    }

    @Override
    protected Integer readValue(DataInputStream in) throws IOException {
      return in.readInt();
    }

    @Override
    protected void writeKey(String key, DataOutputStream out) throws IOException {
      byte[] content = string2bytes(key);
      out.writeInt(content.length);
      out.write(content);
    }

    @Override
    protected void writeValue(Integer value, DataOutputStream out) throws IOException {
      out.writeInt(value);
    }
  }

  private final PersistentIndexMap persistentIndexMap;
  private static final int INITIAL_ENTRIES = 10000;

  /**
   * Instantiates and loads instance of the persistent string indexer.
   */
  static PersistentStringIndexer newPersistentStringIndexer(Path dataPath,
                                                            Clock clock) throws IOException {
    PersistentIndexMap persistentIndexMap = new PersistentIndexMap(dataPath,
        FileSystemUtils.replaceExtension(dataPath, ".journal"), clock);
    Map<Integer, String> reverseMapping = newConcurrentMap(INITIAL_ENTRIES);
    for (Map.Entry<String, Integer> entry : persistentIndexMap.entrySet()) {
      if (reverseMapping.put(entry.getValue(), entry.getKey()) != null) {
        throw new IOException("Corrupted filename index has duplicate entry: " + entry.getKey());
      }
    }
    return new PersistentStringIndexer(persistentIndexMap, reverseMapping);
  }

  private PersistentStringIndexer(PersistentIndexMap stringToInt,
                                  Map<Integer, String> intToString) {
    super(stringToInt, intToString);
    this.persistentIndexMap = stringToInt;
  }

  /**
   * Saves index data to the file.
   */
  synchronized long save() throws IOException {
    return persistentIndexMap.save();
  }

  /**
   * Flushes the journal.
   */
  synchronized void flush() {
    persistentIndexMap.flush();
  }

  private static <K, V> ConcurrentMap<K, V> newConcurrentMap(int expectedCapacity) {
    return new ConcurrentHashMap<>(expectedCapacity);
  }

}
