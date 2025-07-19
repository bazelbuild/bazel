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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.ActionCache.Entry.SerializableTreeArtifactValue;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics.MissReason;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.util.MapCodec;
import com.google.devtools.build.lib.util.MapCodec.IncompatibleFormatException;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.util.StringIndexer;
import com.google.devtools.build.lib.util.VarInt;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.time.Instant;
import java.util.EnumMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * An implementation of the ActionCache interface that uses a {@link StringIndexer} to reduce memory
 * footprint and saves cached actions using the {@link PersistentMap}.
 */
@ConditionallyThreadSafe // condition: each instance must be instantiated with different cache root
public class CompactPersistentActionCache implements ActionCache {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Duration SAVE_INTERVAL = Duration.ofSeconds(3);

  // Key of the action cache record that holds information used to verify referential integrity
  // between action cache and string indexer. Must be < 0 to avoid conflict with real action
  // cache records.
  private static final int VALIDATION_KEY = -10;

  private static final int VERSION = 21;

  /**
   * A timestamp, represented as the number of minutes since the Unix epoch.
   *
   * <p>This provides adequate accuracy for garbage collection purposes while reducing storage
   * requirements.
   */
  private static final class Timestamp {
    // Expect many recurring values and deduplicate them.
    private static final Interner<Timestamp> INTERNER = BlazeInterners.newWeakInterner();

    private static final long MINUTE_IN_MILLIS = Duration.ofMinutes(1).toMillis();

    private final int epochMinutes;

    private Timestamp(int epochMinutes) {
      this.epochMinutes = epochMinutes;
    }

    static Timestamp fromEpochMinutes(int epochMinutes) {
      return INTERNER.intern(new Timestamp(epochMinutes));
    }

    static Timestamp fromInstant(Instant instant) {
      return fromEpochMinutes((int) (instant.toEpochMilli() / MINUTE_IN_MILLIS));
    }

    int toEpochMinutes() {
      return epochMinutes;
    }

    Instant toInstant() {
      return Instant.ofEpochMilli(epochMinutes * MINUTE_IN_MILLIS);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof Timestamp that)) {
        return false;
      }
      return epochMinutes == that.epochMinutes;
    }

    @Override
    public int hashCode() {
      return Integer.hashCode(epochMinutes);
    }

    @Override
    public String toString() {
      return toInstant().toString();
    }
  }

  private static final MapCodec<Integer, Timestamp> TIMESTAMP_CODEC =
      new MapCodec<Integer, Timestamp>() {
        @Override
        protected Integer readKey(DataInput in) throws IOException {
          return in.readInt();
        }

        @Override
        protected Timestamp readValue(DataInput in) throws IOException {
          return Timestamp.fromEpochMinutes(in.readInt());
        }

        @Override
        protected void writeKey(Integer key, DataOutput out) throws IOException {
          out.writeInt(key);
        }

        @Override
        protected void writeValue(Timestamp value, DataOutput out) throws IOException {
          out.writeInt(value.toEpochMinutes());
        }
      };

  /**
   * A {@link PersistentMap} mapping the string index of the action's primary output path to that
   * entry's last access time.
   */
  private static final class TimestampMap extends PersistentMap<Integer, Timestamp> {
    private final Clock clock;
    private long nextUpdateNanos;

    TimestampMap(Clock clock, Path timestampFile, Path timestampJournalFile) throws IOException {
      super(
          VERSION, TIMESTAMP_CODEC, new ConcurrentHashMap<>(), timestampFile, timestampJournalFile);
      this.clock = clock;
      this.nextUpdateNanos = clock.nanoTime() + SAVE_INTERVAL.toNanos();
      load();
    }

    @Override
    protected boolean shouldFlushJournal() {
      // Use nanoTime() instead of currentTimeMillis() to get monotonic time, not wall time.
      long currentTimeNanos = clock.nanoTime();
      if (currentTimeNanos > nextUpdateNanos) {
        nextUpdateNanos = currentTimeNanos + SAVE_INTERVAL.toNanos();
        return true;
      }
      return false;
    }

    @Override
    protected boolean shouldKeepJournal() {
      // We must first flush the journal to get an accurate measure of its size.
      flushJournal();
      try {
        return journalSize() * 100 < cacheSize();
      } catch (IOException e) {
        return false;
      }
    }

    public void flush() {
      flushJournal();
    }
  }

  private static final MapCodec<Integer, byte[]> ACTION_CODEC =
      new MapCodec<Integer, byte[]>() {
        @Override
        protected Integer readKey(DataInput in) throws IOException {
          return in.readInt();
        }

        @Override
        protected byte[] readValue(DataInput in) throws IOException {
          int size = in.readInt();
          if (size < 0) {
            throw new IOException("found negative array size: " + size);
          }
          byte[] data = new byte[size];
          in.readFully(data);
          return data;
        }

        @Override
        protected void writeKey(Integer key, DataOutput out) throws IOException {
          out.writeInt(key);
        }

        @Override
        protected void writeValue(byte[] value, DataOutput out) throws IOException {
          out.writeInt(value.length);
          out.write(value);
        }
      };

  /**
   * A {@link PersistentMap} mapping the string index of the action's primary output path to the
   * serialized {@link ActionCache.Entry}.
   */
  private static final class ActionMap extends PersistentMap<Integer, byte[]> {
    private final Clock clock;
    private final PersistentStringIndexer indexer;
    private final TimestampMap timestampMap;
    private long nextUpdateNanos;

    ActionMap(
        PersistentStringIndexer indexer,
        TimestampMap timestampMap,
        Clock clock,
        Path mapFile,
        Path journalFile)
        throws IOException {
      super(VERSION, ACTION_CODEC, new ConcurrentHashMap<>(), mapFile, journalFile);
      this.indexer = indexer;
      this.timestampMap = timestampMap;
      this.clock = clock;
      // Use nanoTime() instead of currentTimeMillis() to get monotonic time, not wall time.
      nextUpdateNanos = clock.nanoTime() + SAVE_INTERVAL.toNanos();
      load();
    }

    @Override
    protected boolean shouldFlushJournal() {
      // Use nanoTime() instead of currentTimeMillis() to get monotonic time, not wall time.
      long currentTimeNanos = clock.nanoTime();
      if (currentTimeNanos > nextUpdateNanos) {
        nextUpdateNanos = currentTimeNanos + SAVE_INTERVAL.toNanos();
        // Flush the PersistentStringIndexer and TimestampMap.
        // This ensures an action isn't saved to disk before its timestamp or referenced strings.
        indexer.flush();
        timestampMap.flush();
        return true;
      }
      return false;
    }

    @Override
    protected boolean shouldKeepJournal() {
      // We must first flush the journal to get an accurate measure of its size.
      flushJournal();
      try {
        return journalSize() * 100 < cacheSize();
      } catch (IOException e) {
        return false;
      }
    }
  }

  private final Path cacheRoot;
  private final Path corruptedCacheRoot;
  private final Path tmpDir;
  private final Clock clock;
  private final PersistentStringIndexer indexer;
  private final ActionMap actionMap;
  private final TimestampMap timestampMap;
  private final ImmutableMap<MissReason, AtomicInteger> misses;
  private final AtomicInteger hits = new AtomicInteger();
  private Duration loadTime;

  private CompactPersistentActionCache(
      Path cacheRoot,
      Path corruptedCacheRoot,
      Path tmpDir,
      Clock clock,
      PersistentStringIndexer indexer,
      ActionMap actionMap,
      TimestampMap timestampMap,
      ImmutableMap<MissReason, AtomicInteger> misses) {
    this.cacheRoot = cacheRoot;
    this.corruptedCacheRoot = corruptedCacheRoot;
    this.tmpDir = tmpDir;
    this.clock = clock;
    this.indexer = indexer;
    this.actionMap = actionMap;
    this.timestampMap = timestampMap;
    this.misses = misses;
  }

  public static CompactPersistentActionCache create(
      Path cacheRoot,
      Path corruptedCacheRoot,
      Path tmpDir,
      Clock clock,
      EventHandler reporterForInitializationErrors)
      throws IOException {
    Instant before = clock.now();
    CompactPersistentActionCache compactPersistentActionCache =
        create(
            cacheRoot,
            corruptedCacheRoot,
            tmpDir,
            clock,
            reporterForInitializationErrors,
            /* retrying= */ false);
    Instant after = clock.now();
    compactPersistentActionCache.loadTime = Duration.between(before, after);

    return compactPersistentActionCache;
  }

  private static CompactPersistentActionCache create(
      Path cacheRoot,
      Path corruptedCacheRoot,
      Path tmpDir,
      Clock clock,
      EventHandler reporterForInitializationErrors,
      boolean retrying)
      throws IOException {
    cacheRoot.createDirectoryAndParents();

    Path cacheFile = cacheFile(cacheRoot);
    Path journalFile = journalFile(cacheRoot);
    Path indexFile = indexFile(cacheRoot);
    Path indexJournalFile = indexJournalFile(cacheRoot);
    Path timestampFile = timestampFile(cacheRoot);
    Path timestampJournalFile = timestampJournalFile(cacheRoot);

    PersistentStringIndexer indexer;
    try {
      indexer = PersistentStringIndexer.create(indexFile, indexJournalFile, clock);
    } catch (IOException e) {
      return logAndThrowOrRecurse(
          cacheRoot,
          corruptedCacheRoot,
          tmpDir,
          clock,
          "Failed to load action cache index data",
          e,
          reporterForInitializationErrors,
          retrying);
    }

    TimestampMap timestampMap;
    try {
      timestampMap = new TimestampMap(clock, timestampFile, timestampJournalFile);
    } catch (IOException e) {
      return logAndThrowOrRecurse(
          cacheRoot,
          corruptedCacheRoot,
          tmpDir,
          clock,
          "Failed to load action cache timestamp data",
          e,
          reporterForInitializationErrors,
          retrying);
    }

    ActionMap actionMap;
    try {
      actionMap = new ActionMap(indexer, timestampMap, clock, cacheFile, journalFile);
    } catch (IOException e) {
      return logAndThrowOrRecurse(
          cacheRoot,
          corruptedCacheRoot,
          tmpDir,
          clock,
          "Failed to load action cache data",
          e,
          reporterForInitializationErrors,
          retrying);
    }

    // Validate referential integrity between action map and indexer.
    if (!actionMap.isEmpty()) {
      try {
        validateIntegrity(indexer.size(), actionMap.get(VALIDATION_KEY));
      } catch (IOException e) {
        return logAndThrowOrRecurse(
            cacheRoot,
            corruptedCacheRoot,
            tmpDir,
            clock,
            "Failed action cache referential integrity check",
            e,
            reporterForInitializationErrors,
            retrying);
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
    return new CompactPersistentActionCache(
        cacheRoot,
        corruptedCacheRoot,
        tmpDir,
        clock,
        indexer,
        actionMap,
        timestampMap,
        Maps.immutableEnumMap(misses));
  }

  private static CompactPersistentActionCache logAndThrowOrRecurse(
      Path cacheRoot,
      Path corruptedCacheRoot,
      Path tmpDir,
      Clock clock,
      String message,
      IOException e,
      EventHandler reporterForInitializationErrors,
      boolean retrying)
      throws IOException {
    if (retrying) {
      // Prevent a retry loop.
      throw new IOException("Action cache initialization is stuck in a retry loop", e);
    }

    if (e instanceof IncompatibleFormatException) {
      // Format incompatibility is expected when switching between Bazel versions, so we don't treat
      // it as corruption; we simply delete the cache directory and start fresh.
      cacheRoot.deleteTree();
    } else {
      // Move the corrupted cache to a separate location so it can be analyzed later.
      // This also ensures that the next initialization attempt will create an empty cache.
      // To avoid using too much disk space, only keep the most recent corrupted cache around.
      corruptedCacheRoot.deleteTree();
      cacheRoot.renameTo(corruptedCacheRoot);

      e = new IOException("%s: %s".formatted(message, e.getMessage()), e);

      logger.atWarning().withCause(e).log(
          "Failed to load action cache, preexisting files kept in %s", corruptedCacheRoot);

      reporterForInitializationErrors.handle(
          Event.error(
              "Error during action cache initialization: "
                  + e.getMessage()
                  + ". Data may be incomplete, potentially causing rebuilds"));
    }

    return create(
        cacheRoot,
        corruptedCacheRoot,
        tmpDir,
        clock,
        reporterForInitializationErrors,
        /* retrying= */ true);
  }

  /** Throws IOException if indexer contains no data or integrity check has failed. */
  private static void validateIntegrity(int indexerSize, byte[] validationRecord)
      throws IOException {
    if (indexerSize == 0) {
      throw new IOException("empty index");
    }
    if (validationRecord == null) {
      throw new IOException("missing validation record");
    }
    try {
      int validationSize = ByteBuffer.wrap(validationRecord).asIntBuffer().get();
      if (validationSize > indexerSize) {
        throw new IOException(
            String.format(
                "validation record %d is too large compared to index size %d",
                validationSize, indexerSize));
      }
    } catch (BufferUnderflowException e) {
      throw new IOException("validation record is incomplete", e);
    }
  }

  public static Path cacheFile(Path cacheRoot) {
    return cacheRoot.getChild("action_cache.blaze");
  }

  public static Path journalFile(Path cacheRoot) {
    return cacheRoot.getChild("action_journal.blaze");
  }

  public static Path indexFile(Path cacheRoot) {
    return cacheRoot.getChild("filename_index.blaze");
  }

  public static Path indexJournalFile(Path cacheRoot) {
    return cacheRoot.getChild("filename_index_journal.blaze");
  }

  public static Path timestampFile(Path cacheRoot) {
    return cacheRoot.getChild("timestamp.blaze");
  }

  public static Path timestampJournalFile(Path cacheRoot) {
    return cacheRoot.getChild("timestamp_journal.blaze");
  }

  @Override
  @Nullable
  public ActionCache.Entry get(String key) {
    Integer index = indexer.getIndex(key);
    if (index == null) {
      return null;
    }
    byte[] data = actionMap.get(index);
    if (data == null) {
      return null;
    }
    ActionCache.Entry entry = decode(data);
    if (entry != null && !entry.isCorrupted()) {
      timestampMap.put(index, Timestamp.fromInstant(clock.now()));
    }
    return entry;
  }

  @Override
  public void put(String key, ActionCache.Entry entry) {
    put(key, entry, clock.now());
  }

  private void put(String key, ActionCache.Entry entry, Instant timestamp) {
    // Encode record. Note that both methods may create new mappings in the indexer.
    Integer index = indexer.getOrCreateIndex(key);
    byte[] content;
    try {
      content = encode(entry);
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Failed to save cache entry %s with key %s", entry, key);
      return;
    }

    // Update validation record.
    // Note the benign race condition in which two threads might race on updating the validation
    // record: if the most recent update loses the race, a value lower than the indexer size will
    // remain in the validation record, which will still pass the integrity check.
    ByteBuffer buffer = ByteBuffer.allocate(4); // size of int in bytes
    int indexSize = indexer.size();
    buffer.asIntBuffer().put(indexSize);
    actionMap.put(VALIDATION_KEY, buffer.array());

    // Update the timestamp map.
    timestampMap.put(index, Timestamp.fromInstant(timestamp));

    // Update the action map.
    // This is last so that, if a flush occurs, the index and timestamp also make it to disk.
    actionMap.put(index, content);
  }

  @Override
  public void remove(String key) {
    Integer index = indexer.getIndex(key);
    if (index != null) {
      actionMap.remove(index);
      timestampMap.remove(index);
    }
  }

  @Override
  public void removeIf(Predicate<Entry> predicate) {
    // Be careful not to cause the timestamp to be updated on kept entries (i.e., don't use get()).
    for (Map.Entry<Integer, byte[]> entry : actionMap.entrySet()) {
      if (entry.getKey() == VALIDATION_KEY) {
        // Skip the validation record.
        continue;
      }
      ActionCache.Entry decodedEntry = decode(entry.getValue());
      if (decodedEntry.isCorrupted()) {
        // Skip corrupted entries.
        continue;
      }
      if (predicate.test(decodedEntry)) {
        // Although this is racy (the key might be concurrently set to a different value), we don't
        // care because it's a very small window and it only impacts performance, not correctness.
        actionMap.remove(entry.getKey());
        timestampMap.remove(entry.getKey());
      }
    }
  }

  @ThreadSafety.ThreadHostile
  @Override
  public long save() throws IOException {
    // TODO(b/314086729): Remove after we understand the bug.
    try {
      validateIntegrity(indexer.size(), actionMap.get(VALIDATION_KEY));
    } catch (IOException e) {
      logger.atInfo().withCause(e).log(
          "Integrity check failed on the inmemory objects right before save");
    }

    long indexSize = indexer.save();
    long actionMapSize = actionMap.save();
    long timestampMapSize = timestampMap.save();
    return indexSize + actionMapSize + timestampMapSize;
  }

  @ThreadSafety.ThreadHostile
  @Override
  public void clear() {
    indexer.clear();
    actionMap.clear();
    timestampMap.clear();
  }

  /** Returns a map from action key to last access time. */
  ImmutableMap<String, Instant> getActionTimestampMap() throws IOException {
    // Iterate the timestamp map, not the action map, so that the result may be used for testing
    // that an entry is removed from the timestamp map when removed from the action map. Note that
    // the indexer does not support removing entries.
    ImmutableMap.Builder<String, Instant> builder =
        ImmutableMap.builderWithExpectedSize(timestampMap.size());
    for (Map.Entry<Integer, Timestamp> entry : timestampMap.entrySet()) {
      String actionKey = indexer.getStringForIndex(entry.getKey());
      if (actionKey != null) {
        builder.put(actionKey, entry.getValue().toInstant());
      }
    }
    return builder.buildKeepingLast();
  }

  @ThreadSafety.ThreadHostile
  @Override
  public CompactPersistentActionCache trim(float threshold, Duration maxAge)
      throws IOException, InterruptedException {
    Instant cutoffTime = clock.now().minus(maxAge);

    ImmutableMap<String, Instant> accessTimeMap = getActionTimestampMap();

    // Count the number of stale entries.
    int numStale = 0;
    for (Map.Entry<String, Instant> entry : accessTimeMap.entrySet()) {
      if (Thread.interrupted()) {
        // If interrupted, return promptly.
        throw new InterruptedException();
      }
      if (entry.getValue().isBefore(cutoffTime)) {
        numStale++;
      }
    }

    // Skip garbage collection if below the threshold.
    if (numStale == 0 || numStale < threshold * actionMap.size()) {
      return this;
    }

    // Clear preexisting temporary directory contents.
    tmpDir.deleteTree();

    Path newRoot = tmpDir.getChild("new");
    Path oldRoot = tmpDir.getChild("old");

    // Create a new cache backed by a temporary directory.
    var newCache =
        CompactPersistentActionCache.create(
            newRoot, corruptedCacheRoot, tmpDir, clock, NullEventHandler.INSTANCE);

    // Copy sufficiently recent entries into the new cache.
    for (Map.Entry<Integer, byte[]> entry : actionMap.entrySet()) {
      if (Thread.interrupted()) {
        // If interrupted, return promptly but avoid leaving the temporary directory behind.
        tmpDir.deleteTree();
        throw new InterruptedException();
      }
      if (entry.getKey() == VALIDATION_KEY) {
        // Skip the validation record.
        continue;
      }
      String actionKey = checkNotNull(indexer.getStringForIndex(entry.getKey()), entry.getKey());
      // If the timestamp is missing, assume the entry was recently added but its timestamp update
      // was lost.
      Instant timestamp = accessTimeMap.getOrDefault(actionKey, clock.now());
      if (timestamp.isBefore(cutoffTime)) {
        continue;
      }
      // The entry must be reencoded so that strings it references are inserted into the indexer.
      newCache.put(actionKey, decode(entry.getValue()), timestamp);
    }

    // Save the new cache to disk.
    newCache.save();

    // Replace the on-disk representation.
    cacheRoot.renameTo(oldRoot);
    newRoot.renameTo(cacheRoot);

    // Delete the temporary directory.
    tmpDir.deleteTree();

    // Reload the cache from disk and return it.
    return CompactPersistentActionCache.create(
        cacheRoot, corruptedCacheRoot, tmpDir, clock, NullEventHandler.INSTANCE);
  }

  /** Dumps the action cache into a human-readable format. */
  @Override
  public void dump(PrintStream out) {
    ImmutableList<Integer> sortedKeys =
        actionMap.keySet().stream()
            .filter(k -> !k.equals(VALIDATION_KEY))
            .sorted()
            .collect(toImmutableList());
    out.format("Action cache (%d records):\n", sortedKeys.size());
    for (Integer key : sortedKeys) {
      byte[] encodedEntry = actionMap.get(key);
      ActionCache.Entry decodedEntry = decode(encodedEntry);
      Timestamp timestamp = timestampMap.get(key);
      out.format("  %s -> %s\n", key, indexer.getStringForIndex(key));
      out.format("  packed_len = %s\n", encodedEntry.length);
      out.format("  timestamp = %s\n", timestamp != null ? timestamp.toString() : "unknown");
      decodedEntry.dump(out);
    }
    indexer.dump(out);
  }

  /**
   * Returns the number of entries in the action map. If non-zero, it means that the map has been
   * initialized and contains the validation record.
   */
  @Override
  public int size() {
    return actionMap.size();
  }

  private void encodeRemoteMetadata(FileArtifactValue value, ByteArrayOutputStream sink)
      throws IOException {
    checkArgument(value.isRemote(), "metadata is not remote: %s", value);

    MetadataDigestUtils.write(value.getDigest(), sink);

    VarInt.putVarLong(value.getSize(), sink);

    VarInt.putVarInt(value.getLocationIndex(), sink);

    VarInt.putVarLong(
        value.getExpirationTime() != null ? value.getExpirationTime().toEpochMilli() : -1, sink);

    PathFragment resolvedPath = value.getResolvedPath();
    if (resolvedPath != null) {
      VarInt.putVarInt(1, sink);
      VarInt.putVarInt(indexer.getOrCreateIndex(resolvedPath.toString()), sink);
    } else {
      VarInt.putVarInt(0, sink);
    }
  }

  private static final int MAX_REMOTE_METADATA_SIZE =
      (1 + DigestUtils.ESTIMATED_SIZE) // digest length + digest
          + VarInt.MAX_VARLONG_SIZE // size
          + VarInt.MAX_VARINT_SIZE // locationIndex
          + VarInt.MAX_VARLONG_SIZE // expirationTime
          + (1 + VarInt.MAX_VARINT_SIZE); // resolvedPath

  private FileArtifactValue decodeRemoteMetadata(ByteBuffer source) throws IOException {
    byte[] digest = MetadataDigestUtils.read(source);

    long size = VarInt.getVarLong(source);

    int locationIndex = VarInt.getVarInt(source);

    long expirationTimeEpochMilli = VarInt.getVarLong(source);

    PathFragment resolvedPath = null;
    int numResolvedPath = VarInt.getVarInt(source);
    if (numResolvedPath > 0) {
      if (numResolvedPath != 1) {
        throw new IOException("Invalid presence marker for resolved path");
      }
      resolvedPath = PathFragment.create(getStringForIndex(indexer, VarInt.getVarInt(source)));
    }

    FileArtifactValue metadata;
    if (expirationTimeEpochMilli < 0) {
      metadata = FileArtifactValue.createForRemoteFile(digest, size, locationIndex);
    } else {
      metadata =
          FileArtifactValue.createForRemoteFileWithMaterializationData(
              digest, size, locationIndex, Instant.ofEpochMilli(expirationTimeEpochMilli));
    }

    if (resolvedPath != null) {
      metadata = FileArtifactValue.createFromExistingWithResolvedPath(metadata, resolvedPath);
    }

    return metadata;
  }

  /**
   * @return action data encoded as a byte[] array.
   */
  private byte[] encode(ActionCache.Entry entry) throws IOException {
    Preconditions.checkState(!entry.isCorrupted());

    int maxDiscoveredInputsSize = 1; // presence marker
    if (entry.discoversInputs()) {
      maxDiscoveredInputsSize +=
          VarInt.MAX_VARINT_SIZE // length
              + (VarInt.MAX_VARINT_SIZE // execPath
                  * entry.getDiscoveredInputPaths().size());
    }

    int maxOutputFilesSize =
        VarInt.MAX_VARINT_SIZE // entry.getOutputFiles().size()
            + (VarInt.MAX_VARINT_SIZE // execPath
                    + MAX_REMOTE_METADATA_SIZE)
                * entry.getOutputFiles().size();

    int maxOutputTreesSize = VarInt.MAX_VARINT_SIZE; // entry.getOutputTrees().size()
    for (Map.Entry<String, SerializableTreeArtifactValue> tree :
        entry.getOutputTrees().entrySet()) {
      maxOutputTreesSize += VarInt.MAX_VARINT_SIZE; // execPath

      SerializableTreeArtifactValue value = tree.getValue();

      maxOutputTreesSize += VarInt.MAX_VARINT_SIZE; // value.childValues().size()
      maxOutputTreesSize +=
          (VarInt.MAX_VARINT_SIZE // parentRelativePath
                  + MAX_REMOTE_METADATA_SIZE)
              * value.childValues().size();

      maxOutputTreesSize +=
          // value.archivedFileValue() optional
          1 + value.archivedFileValue().map(ignored -> MAX_REMOTE_METADATA_SIZE).orElse(0);
      maxOutputTreesSize +=
          // value.resolvedPath() optional
          1 + value.resolvedPath().map(ignored -> VarInt.MAX_VARINT_SIZE).orElse(0);
    }

    // Estimate the size of the buffer.
    int maxSize =
        (1 + DigestUtils.ESTIMATED_SIZE) // digest length + digest
            + maxDiscoveredInputsSize
            + maxOutputFilesSize
            + maxOutputTreesSize;
    ByteArrayOutputStream sink = new ByteArrayOutputStream(maxSize);

    MetadataDigestUtils.write(entry.getDigest(), sink);

    VarInt.putVarInt(entry.discoversInputs() ? 1 : 0, sink);
    if (entry.discoversInputs()) {
      ImmutableList<String> discoveredInputPaths = entry.getDiscoveredInputPaths();
      VarInt.putVarInt(discoveredInputPaths.size(), sink);
      for (String discoveredInputPath : discoveredInputPaths) {
        VarInt.putVarInt(indexer.getOrCreateIndex(discoveredInputPath), sink);
      }
    }

    VarInt.putVarInt(entry.getOutputFiles().size(), sink);
    for (Map.Entry<String, FileArtifactValue> file : entry.getOutputFiles().entrySet()) {
      VarInt.putVarInt(indexer.getOrCreateIndex(file.getKey()), sink);
      encodeRemoteMetadata(file.getValue(), sink);
    }

    VarInt.putVarInt(entry.getOutputTrees().size(), sink);
    for (Map.Entry<String, SerializableTreeArtifactValue> tree :
        entry.getOutputTrees().entrySet()) {
      VarInt.putVarInt(indexer.getOrCreateIndex(tree.getKey()), sink);

      SerializableTreeArtifactValue serializableTreeArtifactValue = tree.getValue();

      VarInt.putVarInt(serializableTreeArtifactValue.childValues().size(), sink);
      for (Map.Entry<String, FileArtifactValue> child :
          serializableTreeArtifactValue.childValues().entrySet()) {
        VarInt.putVarInt(indexer.getOrCreateIndex(child.getKey()), sink);
        encodeRemoteMetadata(child.getValue(), sink);
      }

      Optional<FileArtifactValue> archivedFileValue =
          serializableTreeArtifactValue.archivedFileValue();
      if (archivedFileValue.isPresent()) {
        VarInt.putVarInt(1, sink);
        encodeRemoteMetadata(archivedFileValue.get(), sink);
      } else {
        VarInt.putVarInt(0, sink);
      }

      Optional<PathFragment> resolvedPath = serializableTreeArtifactValue.resolvedPath();
      if (resolvedPath.isPresent()) {
        VarInt.putVarInt(1, sink);
        VarInt.putVarInt(indexer.getOrCreateIndex(resolvedPath.get().toString()), sink);
      } else {
        VarInt.putVarInt(0, sink);
      }
    }

    return sink.toByteArray();
  }

  private static String getStringForIndex(StringIndexer indexer, int index) throws IOException {
    String path = index >= 0 ? indexer.getStringForIndex(index) : null;
    if (path == null) {
      throw new IOException("Corrupted string index");
    }
    return path;
  }

  /**
   * Creates a {@link ActionCache.Entry} from the given compressed data.
   *
   * @throws IOException if the compressed data is corrupted.
   */
  private ActionCache.Entry decodeInternal(byte[] data) throws IOException {
    try {
      ByteBuffer source = ByteBuffer.wrap(data);

      byte[] digest = MetadataDigestUtils.read(source);

      ImmutableList<String> discoveredInputPaths = null;
      int discoveredInputsPresenceMarker = VarInt.getVarInt(source);
      if (discoveredInputsPresenceMarker != 0) {
        if (discoveredInputsPresenceMarker != 1) {
          throw new IOException(
              "Invalid presence marker for discovered inputs: " + discoveredInputsPresenceMarker);
        }
        int numDiscoveredInputs = VarInt.getVarInt(source);
        if (numDiscoveredInputs < 0) {
          throw new IOException("Invalid discovered input count: " + numDiscoveredInputs);
        }
        ImmutableList.Builder<String> builder =
            ImmutableList.builderWithExpectedSize(numDiscoveredInputs);
        for (int i = 0; i < numDiscoveredInputs; i++) {
          int id = VarInt.getVarInt(source);
          String filename = getStringForIndex(indexer, id);
          builder.add(filename);
        }
        discoveredInputPaths = builder.build();
      }

      int numOutputFiles = VarInt.getVarInt(source);
      if (numOutputFiles < 0) {
        throw new IOException("Invalid output file count: " + numOutputFiles);
      }
      ImmutableMap.Builder<String, FileArtifactValue> outputFiles =
          ImmutableMap.builderWithExpectedSize(numOutputFiles);
      for (int i = 0; i < numOutputFiles; i++) {
        String execPath = getStringForIndex(indexer, VarInt.getVarInt(source));
        FileArtifactValue value = decodeRemoteMetadata(source);
        outputFiles.put(execPath, value);
      }

      int numOutputTrees = VarInt.getVarInt(source);
      if (numOutputTrees < 0) {
        throw new IOException("invalid output tree count: " + numOutputTrees);
      }
      ImmutableMap.Builder<String, SerializableTreeArtifactValue> outputTrees =
          ImmutableMap.builderWithExpectedSize(numOutputTrees);
      for (int i = 0; i < numOutputTrees; i++) {
        String treeKey = getStringForIndex(indexer, VarInt.getVarInt(source));

        ImmutableMap.Builder<String, FileArtifactValue> childValues = ImmutableMap.builder();
        int numChildValues = VarInt.getVarInt(source);
        for (int j = 0; j < numChildValues; ++j) {
          String childKey = getStringForIndex(indexer, VarInt.getVarInt(source));
          FileArtifactValue value = decodeRemoteMetadata(source);
          childValues.put(childKey, value);
        }

        Optional<FileArtifactValue> archivedFileValue = Optional.empty();
        int archivedFileValuePresenceMarker = VarInt.getVarInt(source);
        if (archivedFileValuePresenceMarker != 0) {
          if (archivedFileValuePresenceMarker != 1) {
            throw new IOException(
                "Invalid presence marker for archived representation: "
                    + archivedFileValuePresenceMarker);
          }
          archivedFileValue = Optional.of(decodeRemoteMetadata(source));
        }

        Optional<PathFragment> resolvedPath = Optional.empty();
        int resolvedPathPresenceMarker = VarInt.getVarInt(source);
        if (resolvedPathPresenceMarker != 0) {
          if (resolvedPathPresenceMarker != 1) {
            throw new IOException(
                "Invalid presence marker for resolved path: " + resolvedPathPresenceMarker);
          }
          resolvedPath =
              Optional.of(
                  PathFragment.create(getStringForIndex(indexer, VarInt.getVarInt(source))));
        }

        SerializableTreeArtifactValue value =
            SerializableTreeArtifactValue.create(
                childValues.buildOrThrow(), archivedFileValue, resolvedPath);
        outputTrees.put(treeKey, value);
      }

      if (source.remaining() > 0) {
        throw new IOException("serialized entry data has not been fully decoded");
      }
      return new ActionCache.Entry(
          digest, discoveredInputPaths, outputFiles.buildOrThrow(), outputTrees.buildOrThrow());
    } catch (BufferUnderflowException e) {
      throw new IOException("encoded entry data is incomplete", e);
    }
  }

  /**
   * Creates an {@link ActionCache.Entry} from the given compressed data, returning the special
   * value {@link ActionCache.Entry#CORRUPTED} if the compressed data is corrupted.
   */
  private ActionCache.Entry decode(byte[] data) {
    try {
      return decodeInternal(data);
    } catch (IOException e) {
      return ActionCache.Entry.CORRUPTED;
    }
  }

  @Override
  public void accountHit() {
    hits.incrementAndGet();
  }

  @Override
  public void accountMiss(MissReason reason) {
    AtomicInteger counter = misses.get(reason);
    Preconditions.checkNotNull(
        counter,
        "Miss reason %s was not registered in the misses map " + "during cache construction",
        reason);
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

  @Override
  @Nullable
  public Duration getLoadTime() {
    Duration ret = loadTime;
    // As a side effect, reset the load time, so it is only reported for the actual invocation that
    // loaded the action cache.
    loadTime = null;
    return ret;
  }
}
