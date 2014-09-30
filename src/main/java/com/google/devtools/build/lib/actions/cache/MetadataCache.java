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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.PersistentMap;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;

/**
 * Implements the metadata cache used by the dependency checker and (maybe) builder.
 * Class relies on the externally supplied Map<Long, Metadata> for the storage.
 */
@ConditionallyThreadSafe // Each instance must instantiated with different cache root.
public final class MetadataCache {

  private static final boolean DEBUG = false;

  // How many entries we want to keep in the metadata cache.
  private static final int OPTIMAL_CACHE_SIZE = 200000;

  // Maximum number of tracked generations. Any older generation will be automatically
  // rolled into the last one. Generation N represents entries accessed by the N-th
  // previous build cycle and not accessed since.
  private static final int MAX_TRACKED_GENERATION = 999;

  private static final class CacheEntry {
    final Metadata metadata;
    int generation;

    CacheEntry(Metadata metadata, int generation) throws InvalidObjectException {
      this.metadata = metadata;
      this.generation = generation;
      if (generation < 0 || generation > MAX_TRACKED_GENERATION) {
        throw new InvalidObjectException("Unexpected generation value " + generation
            + " for the metadata " + metadata);
      }
    }

    CacheEntry(Metadata metadata) throws InvalidObjectException { this(metadata, 0); }

    void resetGeneration() { generation = 0; }

    void incrementGeneration() {
      if (generation < MAX_TRACKED_GENERATION) { generation++; }
    }
  }

  /**
   * Defines backing metadata map interface for use by the MetadataCache.
   */
  private static interface MetadataMap extends Map<Object, CacheEntry> {

    /**
     * Save the cache.
     * @return the serialized size in bytes.
     * @throws IOException on failures.
     */
    long save() throws IOException;
  }

  private final MetadataMap metadataMap;

  private final TimestampGranularityMonitor timestampGranularityMonitor;

  /** The mtime of the most recently saved source file, used for statistics reporting. **/
  private volatile long lastFileSaveTime = 0;

  /**
   * The mtimes of all source files changed between the last build's start time and the
   * current build's start time. Only applies to builds running on existing Blaze servers.
   * Records up to {@link #MAX_FILE_SAVES_TO_TRACK} files, with no guarantee which subset of
   * files are recorded for builds that exceed this amount.
   **/
  private Map<Artifact, Long> changedFileSaveTimes = Maps.newConcurrentMap();

  /** Maximum number of file saves to track for a single build. **/
  private static final int MAX_FILE_SAVES_TO_TRACK = 20;

  /**
   * Switch for quickly checking whether or not we're still tracking file saves for
   * the current build. Initially true for each build, flips to false when
   * {@code changedFileSaveTimes.size() >= maxFileSaveToTrack}.
   **/
  private volatile boolean trackFileSavesForCurrentBuild;

  /** Start times of the last and current builds, respectively. **/
  private long lastInvocationStartTime = 0;
  private long currentInvocationStartTime = 0;

  // Statistic counters.
  private final AtomicInteger totalCount = new AtomicInteger();
  private final AtomicInteger hitCount = new AtomicInteger();

  private synchronized void resetHitCounts() {
    totalCount.set(0);
    hitCount.set(0);
  }

  /**
   * Metadata cache is not supposed to be instantiated directly but only through
   * the provided static helper methods.
   *
   * @param metadataMap backing metadata map
   */
  private MetadataCache(MetadataMap metadataMap, TimestampGranularityMonitor monitor) {
    this.metadataMap = metadataMap;
    this.timestampGranularityMonitor = monitor;
    resetHitCounts();
  }

  /**
   * Sets the time that the current invocation started.
   *
   * @param startTime - start time in ms since January 1, 1970 00:00:00 UTC.
   */
  public void setInvocationStartTime(long startTime) {
    lastInvocationStartTime = currentInvocationStartTime;
    currentInvocationStartTime = startTime;
    changedFileSaveTimes.clear();
    trackFileSavesForCurrentBuild = true;
  }

  /**
   * Returns metadata for the given artifact, or throws IOException if it cannot
   * be done for any reason (e.g. file is not accessible). This method should be
   * used instead of get() if caller is confident that operation should succeed
   * and wants to handle any exceptions by itself. This method will never remove
   * a cache entry since it assumes that metadata retrieval must succeed.
   *
   * <p>This method is PERFORMANCE CRITICAL.
   *
   * @param artifact path to retrieve metadata for
   * @param forceDigest if true, always make sure to attempt a digest lookup.
   * @return metadata object or null if it is not available
   * @throws IOException if I/O error occurred during metadata retrieval
   */
  public Metadata getOrFail(Artifact artifact, FileStatus stat, boolean forceDigest)
      throws IOException {
    Preconditions.checkNotNull(stat);
    Path path = artifact.getPath();

    long mtime = stat.getLastModifiedTime();

    // For source files (but not generated files), pass their timestamp
    // to the timestamp granularity monitor, which will check if the timestamp
    // exactly matches the current time; this is done so that we'll know if we
    // need to wait at the end of the build to avoid timestamp granularity issues.
    if (artifact.isSourceArtifact()) {
      timestampGranularityMonitor.notifyDependenceOnFileTime(mtime);

      // Also possibly update our "changed file save times" stats.
      reportInputFileSaveTime(artifact, mtime);
    }

    if (!forceDigest && artifact.forceConstantMetadata()) {
      return Metadata.CONSTANT_METADATA;
    }

    if (forceDigest || DigestUtils.useFileDigest(artifact, stat.isFile(), stat.getSize())) {
      // Synchronizing on the artifact object to avoid duplicate digest calculations
      synchronized (artifact) {
        if (DEBUG) { totalCount.incrementAndGet(); }
        CacheEntry entry = metadataMap.get(path);
        if (entry == null || entry.metadata.mtime != mtime || entry.metadata.digest == null) {
          Metadata metadata =
              new Metadata(mtime, DigestUtils.getDigestOrFail(artifact.getPath(), stat.getSize()));
          entry = new CacheEntry(metadata);
          metadataMap.put(path, entry);
        } else {
          if (DEBUG) { hitCount.incrementAndGet(); }
          entry.resetGeneration(); // Entry was used - reset generation number.
        }
        return entry.metadata;
      }
    } else {
      // We are not using file digest - skip cache lookup entirely.
      return new Metadata(mtime, null);
    }
  }

  /**
   * Injects metadata record into the cache.
   */
  void injectDigest(Path path, FileStatus stat, byte[] digest) throws IOException {
    metadataMap.put(path, new CacheEntry(new Metadata(stat.getLastModifiedTime(), digest)));
  }

  /**
   * Used only when DEBUG == true.
   */
  private void printStatistics() {
    int[] generationCounts = new int[MAX_TRACKED_GENERATION + 1];
    for (CacheEntry cacheEntry : metadataMap.values()) {
      generationCounts[cacheEntry.generation]++;
    }
    System.out.println("--- Metadata cache statistics ------------------------");
    for (int i = 0; i < MAX_TRACKED_GENERATION; i++) {
      if (generationCounts[i] > 0) {
        System.out.printf("Generation %-3d: %d\n", i, generationCounts[i]);
      }
    }
    System.out.println("Total: " + metadataMap.size());
    System.out.println("------------------------------------------------------");
    System.out.println("Total requests:       " + totalCount.get());
    System.out.println("Total hits:           " + hitCount.get());
    System.out.println("------------------------------------------------------");
  }

  public int getCacheSize() {
    return metadataMap.size();
  }

  /**
   * Reports the mtime of a source file. Tracks the time of the most recently saved
   * file as well as the full list of changed files since the last build up to the
   * limit of {@link #MAX_FILE_SAVES_TO_TRACK}.
   *
   * Note: Since this method is called by {@link #getOrFail}, it executes frequently.
   * So an efficient implementation is critical.
   **/
  private void reportInputFileSaveTime(Artifact inputFile, long mtime) {
    // Ignore save times after the start of the current build.
    if (mtime > currentInvocationStartTime) {
      return;
    }

    // If the file changed since the last build and we're still tracking file changes
    // in the current build, add it to our map.
    if (mtime > lastInvocationStartTime && trackFileSavesForCurrentBuild &&
        lastInvocationStartTime > 0) {
      // Efficiency note: all logic inside this conditional is executed at most
      // maxFileSavesToTrack times.
      changedFileSaveTimes.put(inputFile, mtime);
      // While the following is called concurrently, we don't need perfect accuracy here.
      if (changedFileSaveTimes.size() >= MAX_FILE_SAVES_TO_TRACK) {
        trackFileSavesForCurrentBuild = false;
      }
    }

    // If needed, update the "last file save time".
    if (mtime > lastFileSaveTime) {
      synchronized (this) {
        if (mtime > lastFileSaveTime) {
          lastFileSaveTime = mtime;
        }
      }
    }
  }

  public synchronized long getLastFileSaveTime() {
    return lastFileSaveTime;
  }

  public synchronized Map<String, Long> getChangedFileSaveTimes() {
    Map<String, Long> ans = new HashMap<>();
    for (Map.Entry<Artifact, Long> entry : changedFileSaveTimes.entrySet()) {
      ans.put(entry.getKey().prettyPrint(), entry.getValue());
    }
    return ans;
  }

  /**
   * Implements cache eviction policy. Every cache record has generation
   * number - how many builds ago it was accessed. Once cache size exceeds
   * given optimal cache size, we remove oldest generations to keep cache size
   * under control. We always keep entries accessed during last build
   * regardless of the cache size.
   * For the purposes of this policy separate generations are delimited by
   * calls to the MetadataCache.save() which is done at the end of every build.
   *
   * @param optimalCacheSize max cache size that cache should try to adhere to.
   */
  @VisibleForTesting
  public void evictCacheEntriesIfNeeded(int optimalCacheSize) {
    if (metadataMap.size() <= optimalCacheSize) {
      // No eviction - just increment generation number.
      for (CacheEntry cacheEntry : metadataMap.values()) {
        cacheEntry.incrementGeneration();
      }
    } else {
      // Increment generation number and gather generation counts.
      int[] generationCounts = new int[MAX_TRACKED_GENERATION + 1];
      for (CacheEntry cacheEntry : metadataMap.values()) {
        cacheEntry.incrementGeneration();
        generationCounts[cacheEntry.generation]++;
      }

      // Find minimum evictable generation.
      int entriesToEvict = metadataMap.size() - optimalCacheSize;
      int minEvictableGeneration = MAX_TRACKED_GENERATION;
      int willBeEvicted = generationCounts[MAX_TRACKED_GENERATION];
      // Always keep entries that were accessed during last build - even if that will result
      // in cache size exceeding OPTIMAL_CACHE_SIZE. Such entries will have generation == 1
      // since it was already incremented. Thus minEvictableGeneration should be at least 2.
      while (willBeEvicted < entriesToEvict && minEvictableGeneration > 2) {
        minEvictableGeneration--;
        willBeEvicted += generationCounts[minEvictableGeneration];
      }

      // Evict just enough entries from evictable generations to bring cache size down to
      // OPTIMAL_CACHE_SIZE.
      Iterator<MetadataMap.Entry<Object, CacheEntry>> it = metadataMap.entrySet().iterator();
      while (it.hasNext() && entriesToEvict > 0) {
        // Remove old generations to keep cache size under control.
        if (it.next().getValue().generation >= minEvictableGeneration) {
          it.remove();
          entriesToEvict--;
        }
      }
    }
  }

  public synchronized long save() throws IOException {
    if (DEBUG) {
      printStatistics();
    }
    evictCacheEntriesIfNeeded(OPTIMAL_CACHE_SIZE);
    long size = metadataMap.save();
    resetHitCounts();
    return size;
  }

  /**
   * Removes the specified cache entry.
   */
  public synchronized void discardCacheEntry(Artifact artifact) {
    metadataMap.remove(artifact.getPath());
  }

  public static Path cacheFile(Path cacherRoot) {
    return cacherRoot.getChild("metadata_cache_v" + PersistentMetadataMap.VERSION + ".blaze");
  }

  /**
   * Persistent metadata map. Used as a backing map to provide a persistent
   * implementation of the metadata cache.
   */
  private static final class PersistentMetadataMap
      extends PersistentMap<Object, CacheEntry> implements MetadataMap {
    private static final int VERSION = 0x04;
    private static final long SAVE_INTERVAL_NS = 5L * 1000 * 1000 * 1000;

    private final Clock clock;
    private long nextUpdate;
    private final FileSystem fileSystem;

    public PersistentMetadataMap(Path cacheRoot, Clock clock) {
      super(VERSION, new ConcurrentHashMap<Object, CacheEntry>(10000, 0.75f, 16),
          cacheFile(cacheRoot),
          cacheRoot.getChild("metadata_journal_v" + VERSION + ".blaze"));
      this.clock = clock;
      fileSystem = cacheRoot.getFileSystem();
      nextUpdate = clock.nanoTime() + SAVE_INTERVAL_NS;
      try {
        load(/*throwOnLoadFailure=*/true);
      } catch (InvalidObjectException e) {
        LoggingUtil.logToRemote(Level.WARNING, "Discarding metadata cache: " + e.getMessage(), e);
        clear();
      } catch (IOException e) {
        /* do not fail (it is just a cache), but log what happened */
        LoggingUtil.logToRemote(Level.WARNING, "Discarding metadata cache: " + e.getMessage(), e);
      }
    }

    @Override
    protected boolean updateJournal() {
      long time = clock.nanoTime();
      if (SAVE_INTERVAL_NS == 0 || time > nextUpdate) {
        nextUpdate = time + SAVE_INTERVAL_NS;
        return true;
      }
      return false;
    }

    @Override
    protected boolean keepJournal() {
      // We must first flush the journal to get an accurate measure of its size.
      forceFlush();
      try {
        // For very small journals, it's wasteful to fully serialize the cache.
        return journalSize() * 100 < cacheSize();
      } catch (IOException e) {
        return false;
      }
    }

    @Override
    public synchronized long save() throws IOException {
      return super.save();
    }

    @Override
    public CacheEntry get(Object key) {
      return super.get(key);
    }

    /**
     * This method needs to be synchronized even though we use ConcurrentHashMap,
     * since it can also perform journal write.
     */
    @Override
    public synchronized CacheEntry put(Object key, CacheEntry value) {
      return super.put(key, value);
    }

    @Override
    public synchronized CacheEntry remove(Object object) {
      return super.remove(object);
    }

    @Override
    protected Object readKey(DataInputStream in) throws IOException {
      return fileSystem.getPath(in.readUTF());
    }

    @Override
    protected CacheEntry readValue(DataInputStream in) throws IOException {
      Metadata metadata = Metadata.readValue(in);
      return new CacheEntry(metadata, in.readInt());
    }

    @Override
    protected void writeKey(Object key, DataOutputStream out) throws IOException {
      out.writeUTF(((Path) key).getPathString());
    }

    @Override
    protected void writeValue(CacheEntry value, DataOutputStream out)
        throws IOException {
      value.metadata.writeValue(out);
      out.writeInt(value.generation);
    }
  }

  public static MetadataCache getPersistentCache(Path cacheRoot,
                                                 Clock clock,
                                                 TimestampGranularityMonitor monitor) {
    return new MetadataCache(new PersistentMetadataMap(cacheRoot, clock), monitor);
  }

  /**
   * In-memory metadata map. Used as a backing map to provide an in-memory
   * implementation of the metadata cache (e.g. for testing purposes).
   */
  private static final class InMemoryMetadataMap
      extends ConcurrentHashMap<Object, CacheEntry> implements MetadataMap {
    @Override
    public long save() { return 0; }
  }

  /**
   * Creates in-memory version of metadata cache. Used only by tests.
   */
  @VisibleForTesting
  public static MetadataCache getInMemoryCache(TimestampGranularityMonitor monitor) {
    // We will never deserialize paths using in-memory cache, so we can use fake file system
    // instance here.
    return new MetadataCache(new InMemoryMetadataMap(), monitor);
  }
}
