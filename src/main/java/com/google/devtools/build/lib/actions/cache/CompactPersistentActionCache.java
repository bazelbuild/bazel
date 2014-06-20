// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.CompactStringIndexer;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixGlob;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * An implementation of the ActionCache interface that uses
 * {@link CompactStringIndexer} to reduce memory footprint and saves
 * cached actions using the {@link PersistentMap}.
 */
@ConditionallyThreadSafe // condition: each instance must instantiated with
                         // different cache root
public class CompactPersistentActionCache implements ActionCache {
  private static final int SAVE_INTERVAL_SECONDS = 3;
  private static final long NANOS_PER_SECOND = 1000 * 1000 * 1000;

  // Key of the action cache record that holds information used to verify referential integrity
  // between action cache and string indexer. Must be < 0 to avoid conflict with real action
  // cache records.
  private static final int VALIDATION_KEY = -10;

  private static final int VERSION = 10;

  private final class ActionMap extends PersistentMap<Integer, byte[]> {
    private final Clock clock;
    private long nextUpdate;

    public ActionMap(Map<Integer, byte[]> map, Clock clock, Path mapFile, Path journalFile)
        throws IOException {
      super(VERSION, map, mapFile, journalFile);
      this.clock = clock;
      // Using nanoTime. currentTimeMillis may not provide enough granularity.
      nextUpdate = clock.nanoTime() / NANOS_PER_SECOND + SAVE_INTERVAL_SECONDS;
      load();
    }

    @Override
    protected boolean updateJournal() {
      // Using nanoTime. currentTimeMillis may not provide enough granularity.
      long time = clock.nanoTime() / NANOS_PER_SECOND;
      if (SAVE_INTERVAL_SECONDS == 0 || time > nextUpdate) {
        nextUpdate = time + SAVE_INTERVAL_SECONDS;
        // Force flushing of the PersistentStringIndexer instance. This is needed to ensure
        // that filename index data on disk is always up-to-date when we save action cache
        // data.
        indexer.flush();
        return true;
      }
      return false;
    }

    @Override
    protected boolean keepJournal() {
      // We must first flush the journal to get an accurate measure of its size.
      forceFlush();
      try {
        return journalSize() * 100 < cacheSize();
      } catch (IOException e) {
        return false;
      }
    }

    @Override
    protected Integer readKey(DataInputStream in) throws IOException {
      return in.readInt();
    }

    @Override
    protected byte[] readValue(DataInputStream in)
        throws IOException {
      byte[] data = new byte[in.readInt()];
      in.readFully(data);
      return data;
    }

    @Override
    protected void writeKey(Integer key, DataOutputStream out)
        throws IOException {
      out.writeInt(key);
    }

    @Override
    // TODO(bazel-team): (2010) This method, writeKey() and related Metadata methods
    // should really use protocol messages. Doing so would allow easy inspection
    // of the action cache content and, more importantly, would cut down on the
    // need to change VERSION to different number every time we touch those
    // methods. Especially when we'll start to add stuff like statistics for
    // each action.
    protected void writeValue(byte[] value, DataOutputStream out)
        throws IOException {
      out.writeInt(value.length);
      out.write(value);
    }
  }

  private final PersistentMap<Integer, byte[]> map;
  private final PersistentStringIndexer indexer;

  public CompactPersistentActionCache(Path cacheRoot, Clock clock) throws IOException {
    Path cacheFile = cacheFile(cacheRoot);
    Path journalFile = journalFile(cacheRoot);
    Path indexFile = cacheRoot.getChild("filename_index_v" + VERSION + ".blaze");
    // we can now use normal hash map as backing map, since dependency checker
    // will manually purge records from the action cache.
    Map<Integer, byte[]> backingMap = new HashMap<>();

    try {
      indexer = PersistentStringIndexer.newPersistentStringIndexer(indexFile, clock);
    } catch (IOException e) {
      renameCorruptedFiles(cacheRoot);
      throw new IOException("Failed to load filename index data", e);
    }

    try {
      map = new ActionMap(backingMap, clock, cacheFile, journalFile);
    } catch (IOException e) {
      renameCorruptedFiles(cacheRoot);
      throw new IOException("Failed to load action cache data", e);
    }

    // Validate referential integrity between two collections.
    if (!map.isEmpty()) {
      String integrityError = validateIntegrity(indexer.size(), map.get(VALIDATION_KEY));
      if (integrityError != null) {
        renameCorruptedFiles(cacheRoot);
        throw new IOException("Failed action cache referential integrity check: " + integrityError);
      }
    }
  }

  /**
   * Rename corrupted files so they could be analyzed later. This would also ensure
   * that next initialization attempt will create empty cache.
   */
  private static void renameCorruptedFiles(Path cacheRoot) {
    try {
      for (Path path : UnixGlob.forPath(cacheRoot).addPattern("action_*_v" + VERSION + ".*")
          .glob()) {
        path.renameTo(path.getParentDirectory().getChild(path.getBaseName() + ".bad"));
      }
      for (Path path : UnixGlob.forPath(cacheRoot).addPattern("filename_*_v" + VERSION + ".*")
          .glob()) {
        path.renameTo(path.getParentDirectory().getChild(path.getBaseName() + ".bad"));
      }
    } catch (IOException e) {
      // do nothing
    }
  }

  /**
   * @return false iff indexer contains no data or integrity check has failed.
   */
  private static String validateIntegrity(int indexerSize, byte[] validationRecord) {
    if (indexerSize == 0) {
      return "empty index";
    }
    if (validationRecord == null) {
      return "no validation record";
    }
    try {
      int validationSize = ByteBuffer.wrap(validationRecord).asIntBuffer().get();
      if (validationSize <= indexerSize) {
        return null;
      } else {
        return String.format("Validation mismatch: validation entry %d is too large " +
                             "compared to index size %d", validationSize, indexerSize);
      }
    } catch (BufferUnderflowException e) {
      return e.getMessage();
    }

  }

  public static Path cacheFile(Path cacheRoot) {
    return cacheRoot.getChild("action_cache_v" + VERSION + ".blaze");
  }

  public static Path journalFile(Path cacheRoot) {
    return cacheRoot.getChild("action_journal_v" + VERSION + ".blaze");
  }

  @Override
  public ActionCache.Entry createEntry(String key) {
    return new CompactActionCacheEntry(key);
  }

  @Override
  public ActionCache.Entry get(String key) {
    int index = indexer.getIndex(key);
    if (index < 0) {
      return null;
    }
    byte[] data;
    synchronized (this) {
      data = map.get(index);
    }
    try {
      return data != null ? new CompactActionCacheEntry(indexer, data) : null;
    } catch (IOException e) {
      // return entry marked as corrupted.
      return CompactActionCacheEntry.CORRUPTED;
    }
  }

  @Override
  public void put(String key, ActionCache.Entry entry) {
    Preconditions.checkArgument(entry instanceof CompactActionCacheEntry);

    // Encode record. Note that both methods may create new mappings in the indexer.
    int index = indexer.getOrCreateIndex(key);
    byte[] content = ((CompactActionCacheEntry) entry).getData(indexer);

    // Update validation record.
    ByteBuffer buffer = ByteBuffer.allocate(4); // size of int in bytes
    int indexSize = indexer.size();
    buffer.asIntBuffer().put(indexSize);

    // Note the benign race condition here in which two threads might race on
    // updating the VALIDATION_KEY. If the most recent update loses the race,
    // a value lower than the indexer size will remain in the validation record.
    // This will still pass the integrity check.
    synchronized (this) {
      map.put(VALIDATION_KEY, buffer.array());
      // Now update record itself.
      map.put(index, content);
    }
  }

  @Override
  public synchronized void remove(String key) {
    map.remove(indexer.getIndex(key));
  }

  @Override
  public synchronized long save() throws IOException {
    long indexSize = indexer.save();
    long mapSize = map.save();
    return indexSize + mapSize;
  }

  @Override
  public synchronized String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("Action cache (" + map.size() + " records):\n");
    for (Map.Entry<Integer, byte[]> entry: map.entrySet()) {
      if (entry.getKey() == VALIDATION_KEY) { continue; }
      String content;
      try {
        content = new CompactActionCacheEntry(indexer, entry.getValue()).toString();
      } catch (IOException e) {
        content = e.toString() + "\n";
      }
      builder.append("-> ").append(indexer.getStringForIndex(entry.getKey())).append("\n")
          .append(content).append("  packed_len = ").append(entry.getValue().length).append("\n");
    }
    return builder.toString();
  }

  /**
   * Dumps action cache content.
   */
  @Override
  public synchronized void dump(PrintStream out) {
    out.println("String indexer content:\n");
    out.println(indexer.toString());
    out.println("Action cache (" + map.size() + " records):\n");
    for (Map.Entry<Integer, byte[]> entry: map.entrySet()) {
      if (entry.getKey() == VALIDATION_KEY) { continue; }
      String content;
      try {
        content = new CompactActionCacheEntry(indexer, entry.getValue()).toString();
      } catch (IOException e) {
        content = e.toString() + "\n";
      }
      out.println(entry.getKey() + ", " + indexer.getStringForIndex(entry.getKey()) + ":\n"
          +  content + "\n      packed_len = " + entry.getValue().length + "\n");
    }
  }
}
