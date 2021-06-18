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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics.MissReason;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.util.StringIndexer;
import com.google.devtools.build.lib.util.VarInt;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.Collection;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * An implementation of the ActionCache interface that uses a {@link StringIndexer} to reduce memory
 * footprint and saves cached actions using the {@link PersistentMap}.
 */
@ConditionallyThreadSafe // condition: each instance must instantiated with
// different cache root
public class CompactPersistentActionCache implements ActionCache {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final int SAVE_INTERVAL_SECONDS = 3;
  // Log if periodically saving the action cache incurs more than 5% overhead.
  private static final Duration MIN_TIME_FOR_LOGGING =
      Duration.ofSeconds(SAVE_INTERVAL_SECONDS).dividedBy(20);

  // Key of the action cache record that holds information used to verify referential integrity
  // between action cache and string indexer. Must be < 0 to avoid conflict with real action
  // cache records.
  private static final int VALIDATION_KEY = -10;

  private static final int NO_INPUT_DISCOVERY_COUNT = -1;

  private static final int VERSION = 12;

  private static final class ActionMap extends PersistentMap<Integer, byte[]> {
    private final Clock clock;
    private final PersistentStringIndexer indexer;
    private long nextUpdateSecs;

    public ActionMap(
        Map<Integer, byte[]> map,
        PersistentStringIndexer indexer,
        Clock clock,
        Path mapFile,
        Path journalFile)
        throws IOException {
      super(VERSION, map, mapFile, journalFile);
      this.indexer = indexer;
      this.clock = clock;
      // Using nanoTime. currentTimeMillis may not provide enough granularity.
      nextUpdateSecs = TimeUnit.NANOSECONDS.toSeconds(clock.nanoTime()) + SAVE_INTERVAL_SECONDS;
      load();
    }

    @Override
    protected boolean updateJournal() {
      // Using nanoTime. currentTimeMillis may not provide enough granularity.
      long timeSecs = TimeUnit.NANOSECONDS.toSeconds(clock.nanoTime());
      if (SAVE_INTERVAL_SECONDS == 0 || timeSecs > nextUpdateSecs) {
        nextUpdateSecs = timeSecs + SAVE_INTERVAL_SECONDS;
        // Force flushing of the PersistentStringIndexer instance. This is needed to ensure
        // that filename index data on disk is always up-to-date when we save action cache
        // data.
        indexer.flush();
        return true;
      }
      return false;
    }

    @Override
    protected void markAsDirty() {
      try (AutoProfiler p =
          GoogleAutoProfilerUtils.logged("slow write to journal", MIN_TIME_FOR_LOGGING)) {
        super.markAsDirty();
      }
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
      int size = in.readInt();
      if (size < 0) {
        throw new IOException("found negative array size: " + size);
      }
      byte[] data = new byte[size];
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

  private final PersistentStringIndexer indexer;
  private final PersistentMap<Integer, byte[]> map;
  private final ImmutableMap<MissReason, AtomicInteger> misses;
  private final AtomicInteger hits = new AtomicInteger();

  private CompactPersistentActionCache(
      PersistentStringIndexer indexer,
      PersistentMap<Integer, byte[]> map,
      ImmutableMap<MissReason, AtomicInteger> misses) {
    this.indexer = indexer;
    this.map = map;
    this.misses = misses;
  }

  public static CompactPersistentActionCache create(
      Path cacheRoot, Clock clock, EventHandler reporterForInitializationErrors)
      throws IOException {
    return create(
        cacheRoot, clock, reporterForInitializationErrors, /*alreadyFoundCorruption=*/ false);
  }

  private static CompactPersistentActionCache create(
      Path cacheRoot,
      Clock clock,
      EventHandler reporterForInitializationErrors,
      boolean alreadyFoundCorruption)
      throws IOException {
    PersistentMap<Integer, byte[]> map;
    Path cacheFile = cacheFile(cacheRoot);
    Path journalFile = journalFile(cacheRoot);
    Path indexFile = cacheRoot.getChild("filename_index_v" + VERSION + ".blaze");
    // we can now use normal hash map as backing map, since dependency checker
    // will manually purge records from the action cache.
    Map<Integer, byte[]> backingMap = new HashMap<>();

    PersistentStringIndexer indexer;
    try {
      indexer = PersistentStringIndexer.newPersistentStringIndexer(indexFile, clock);
    } catch (IOException e) {
      return logAndThrowOrRecurse(
          cacheRoot,
          clock,
          "Failed to load filename index data",
          e,
          reporterForInitializationErrors,
          alreadyFoundCorruption);
    }

    try {
      map = new ActionMap(backingMap, indexer, clock, cacheFile, journalFile);
    } catch (IOException e) {
      return logAndThrowOrRecurse(
          cacheRoot,
          clock,
          "Failed to load action cache data",
          e,
          reporterForInitializationErrors,
          alreadyFoundCorruption);
    }

    // Validate referential integrity between two collections.
    if (!map.isEmpty()) {
      try {
        validateIntegrity(indexer.size(), map.get(VALIDATION_KEY));
      } catch (IOException e) {
        return logAndThrowOrRecurse(
            cacheRoot, clock, null, e, reporterForInitializationErrors, alreadyFoundCorruption);
      }
    }

    // Populate the map now, so that concurrent updates to the values can happen safely.
    Map<MissReason, AtomicInteger> misses = new EnumMap<>(MissReason.class);
    for (MissReason reason : MissReason.values()) {
      if (reason == MissReason.UNRECOGNIZED) {
        // The presence of this enum value is a protobuf artifact and confuses our metrics
        // externalization code below. Just skip it.
        continue;
      }
      misses.put(reason, new AtomicInteger(0));
    }
    return new CompactPersistentActionCache(indexer, map, Maps.immutableEnumMap(misses));
  }

  private static CompactPersistentActionCache logAndThrowOrRecurse(
      Path cacheRoot,
      Clock clock,
      String message,
      IOException e,
      EventHandler reporterForInitializationErrors,
      boolean alreadyFoundCorruption)
      throws IOException {
    renameCorruptedFiles(cacheRoot);
    if (message != null) {
      e = new IOException(message, e);
    }
    logger.atWarning().withCause(e).log("Failed to load action cache");
    reporterForInitializationErrors.handle(
        Event.error(
            "Error during action cache initialization: "
                + e.getMessage()
                + ". Corrupted files were renamed to '"
                + cacheRoot
                + "/*.bad'. "
                + "Bazel will now reset action cache data, potentially causing rebuilds"));
    if (alreadyFoundCorruption) {
      throw e;
    }
    return create(
        cacheRoot, clock, reporterForInitializationErrors, /*alreadyFoundCorruption=*/ true);
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
    } catch (UnixGlob.BadPattern ex) {
      throw new IllegalStateException(ex); // can't happen
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Unable to rename corrupted action cache files");
    }
  }

  private static final String FAILURE_PREFIX = "Failed action cache referential integrity check: ";
  /** Throws IOException if indexer contains no data or integrity check has failed. */
  private static void validateIntegrity(int indexerSize, byte[] validationRecord)
      throws IOException {
    if (indexerSize == 0) {
      throw new IOException(FAILURE_PREFIX + "empty index");
    }
    if (validationRecord == null) {
      throw new IOException(FAILURE_PREFIX + "no validation record");
    }
    try {
      int validationSize = ByteBuffer.wrap(validationRecord).asIntBuffer().get();
      if (validationSize > indexerSize) {
        throw new IOException(
            String.format(
                FAILURE_PREFIX
                    + "Validation mismatch: validation entry %d is too large "
                    + "compared to index size %d",
                validationSize,
                indexerSize));
      }
    } catch (BufferUnderflowException e) {
      throw new IOException(FAILURE_PREFIX + e.getMessage(), e);
    }
  }

  public static Path cacheFile(Path cacheRoot) {
    return cacheRoot.getChild("action_cache_v" + VERSION + ".blaze");
  }

  public static Path journalFile(Path cacheRoot) {
    return cacheRoot.getChild("action_journal_v" + VERSION + ".blaze");
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
      return data != null ? CompactPersistentActionCache.decode(indexer, data) : null;
    } catch (IOException e) {
      // return entry marked as corrupted.
      return ActionCache.Entry.CORRUPTED;
    }
  }

  @Override
  public void put(String key, ActionCache.Entry entry) {
    // Encode record. Note that both methods may create new mappings in the indexer.
    int index = indexer.getOrCreateIndex(key);
    byte[] content = encode(indexer, entry);

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
  public void clear() {
    indexer.clear();
    map.clear();
  }

  @Override
  public synchronized String toString() {
    StringBuilder builder = new StringBuilder();
    // map.size() - 1 to avoid counting the validation key.
    builder.append("Action cache (" + (map.size() - 1) + " records):\n");
    int size = map.size() > 1000 ? 10 : map.size();
    int ct = 0;
    for (Map.Entry<Integer, byte[]> entry: map.entrySet()) {
      if (entry.getKey() == VALIDATION_KEY) { continue; }
      String content;
      try {
        content = decode(indexer, entry.getValue()).toString();
      } catch (IOException e) {
        content = e + "\n";
      }
      builder.append("-> ").append(indexer.getStringForIndex(entry.getKey())).append("\n")
          .append(content).append("  packed_len = ").append(entry.getValue().length).append("\n");
      if (++ct > size) {
        builder.append("...");
        break;
      }
    }
    return builder.toString();
  }

  /**
   * Dumps action cache content.
   */
  @Override
  public synchronized void dump(PrintStream out) {
    out.println("String indexer content:\n");
    out.println(indexer);
    out.println("Action cache (" + map.size() + " records):\n");
    for (Map.Entry<Integer, byte[]> entry: map.entrySet()) {
      if (entry.getKey() == VALIDATION_KEY) { continue; }
      String content;
      try {
        content = CompactPersistentActionCache.decode(indexer, entry.getValue()).toString();
      } catch (IOException e) {
        content = e + "\n";
      }
      out.println(entry.getKey() + ", " + indexer.getStringForIndex(entry.getKey()) + ":\n"
          +  content + "\n      packed_len = " + entry.getValue().length + "\n");
    }
  }

  /**
   * @return action data encoded as a byte[] array.
   */
  private static byte[] encode(StringIndexer indexer, ActionCache.Entry entry) {
    Preconditions.checkState(!entry.isCorrupted());

    try {
      byte[] actionKeyBytes = entry.getActionKey().getBytes(ISO_8859_1);
      Collection<String> files = entry.getPaths();

      // Estimate the size of the buffer:
      //   5 bytes max for the actionKey length
      // + the actionKey itself
      // + 32 bytes for the digest
      // + 5 bytes max for the file list length
      // + 5 bytes max for each file id
      // + 32 bytes for the environment digest
      int maxSize =
          VarInt.MAX_VARINT_SIZE
              + actionKeyBytes.length
              + DigestUtils.ESTIMATED_SIZE
              + VarInt.MAX_VARINT_SIZE
              + files.size() * VarInt.MAX_VARINT_SIZE
              + DigestUtils.ESTIMATED_SIZE;
      ByteArrayOutputStream sink = new ByteArrayOutputStream(maxSize);

      VarInt.putVarInt(actionKeyBytes.length, sink);
      sink.write(actionKeyBytes);

      MetadataDigestUtils.write(entry.getFileDigest(), sink);

      VarInt.putVarInt(entry.discoversInputs() ? files.size() : NO_INPUT_DISCOVERY_COUNT, sink);
      for (String file : files) {
        VarInt.putVarInt(indexer.getOrCreateIndex(file), sink);
      }

      MetadataDigestUtils.write(entry.getUsedClientEnvDigest(), sink);

      return sink.toByteArray();
    } catch (IOException e) {
      // This Exception can never be thrown by ByteArrayOutputStream.
      throw new AssertionError(e);
    }
  }

  /**
   * Creates new action cache entry using given compressed entry data. Data
   * will stay in the compressed format until entry is actually used by the
   * dependency checker.
   */
  private static ActionCache.Entry decode(StringIndexer indexer, byte[] data) throws IOException {
    try {
      ByteBuffer source = ByteBuffer.wrap(data);

      byte[] actionKeyBytes = new byte[VarInt.getVarInt(source)];
      source.get(actionKeyBytes);
      String actionKey = new String(actionKeyBytes, ISO_8859_1);

      byte[] digest = MetadataDigestUtils.read(source);

      int count = VarInt.getVarInt(source);
      if (count != NO_INPUT_DISCOVERY_COUNT && count < 0) {
        throw new IOException("Negative discovered file count: " + count);
      }
      ImmutableList<String> files = null;
      if (count != NO_INPUT_DISCOVERY_COUNT) {
        ImmutableList.Builder<String> builder = ImmutableList.builderWithExpectedSize(count);
        for (int i = 0; i < count; i++) {
          int id = VarInt.getVarInt(source);
          String filename = (id >= 0 ? indexer.getStringForIndex(id) : null);
          if (filename == null) {
            throw new IOException("Corrupted file index");
          }
          builder.add(filename);
        }
        files = builder.build();
      }

      byte[] usedClientEnvDigest = MetadataDigestUtils.read(source);

      if (source.remaining() > 0) {
        throw new IOException("serialized entry data has not been fully decoded");
      }
      return new ActionCache.Entry(actionKey, usedClientEnvDigest, files, digest);
    } catch (BufferUnderflowException e) {
      throw new IOException("encoded entry data is incomplete", e);
    }
  }

  @Override
  public void accountHit() {
    hits.incrementAndGet();
  }

  @Override
  public void accountMiss(MissReason reason) {
    AtomicInteger counter = misses.get(reason);
    Preconditions.checkNotNull(counter, "Miss reason %s was not registered in the misses map "
        + "during cache construction", reason);
    counter.incrementAndGet();
  }

  @Override
  public void mergeIntoActionCacheStatistics(ActionCacheStatistics.Builder builder) {
    builder.setHits(hits.get());

    int totalMisses = 0;
    for (Map.Entry<MissReason, AtomicInteger> entry : misses.entrySet()) {
      int count = entry.getValue().get();
      builder.addMissDetailsBuilder().setReason(entry.getKey()).setCount(count);
      totalMisses += count;
    }
    builder.setMisses(totalMisses);
  }

  @Override
  public void resetStatistics() {
    hits.set(0);
    for (Map.Entry<MissReason, AtomicInteger> entry : misses.entrySet()) {
      entry.getValue().set(0);
    }
  }
}
