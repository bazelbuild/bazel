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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ExecutorShutdownUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * A helper class to find dirty values by accessing the filesystem directly (contrast with
 * {@link DiffAwareness}).
 */
class FilesystemValueChecker {

  private static final int DIRTINESS_CHECK_THREADS = 50;
  private static final Logger LOG = Logger.getLogger(FilesystemValueChecker.class.getName());

  private static final Predicate<SkyKey> FILE_STATE_AND_DIRECTORY_LISTING_STATE_FILTER =
      SkyFunctionName.functionIsIn(ImmutableSet.of(SkyFunctions.FILE_STATE,
          SkyFunctions.DIRECTORY_LISTING_STATE));
  private static final Predicate<SkyKey> ACTION_FILTER =
      SkyFunctionName.functionIs(SkyFunctions.ACTION_EXECUTION);

  private final TimestampGranularityMonitor tsgm;
  private final Range<Long> lastExecutionTimeRange;
  private final Supplier<Map<SkyKey, SkyValue>> valuesSupplier;
  private AtomicInteger modifiedOutputFilesCounter = new AtomicInteger(0);
  private AtomicInteger modifiedOutputFilesIntraBuildCounter = new AtomicInteger(0);

  FilesystemValueChecker(final MemoizingEvaluator evaluator, TimestampGranularityMonitor tsgm,
      Range<Long> lastExecutionTimeRange) {
    this.tsgm = tsgm;
    this.lastExecutionTimeRange = lastExecutionTimeRange;

    // Construct the full map view of the entire graph at most once ("memoized"), lazily. If
    // getDirtyFilesystemValues(Iterable<SkyKey>) is called on an empty Iterable, we avoid having
    // to create the Map of value keys to values. This is useful in the case where the graph
    // getValues() method could be slow.
    this.valuesSupplier = Suppliers.memoize(new Supplier<Map<SkyKey, SkyValue>>() {
      @Override
      public Map<SkyKey, SkyValue> get() {
        return evaluator.getValues();
      }
    });
  }

  Iterable<SkyKey> getFilesystemSkyKeys() {
    return Iterables.filter(valuesSupplier.get().keySet(),
        FILE_STATE_AND_DIRECTORY_LISTING_STATE_FILTER);
  }

  Differencer.Diff getDirtyFilesystemSkyKeys() throws InterruptedException {
    return getDirtyFilesystemValues(getFilesystemSkyKeys());
  }

  /**
   * Check the given file and directory values for modifications. {@code values} is assumed to only
   * have {@link FileValue}s and {@link DirectoryListingStateValue}s.
   */
  Differencer.Diff getDirtyFilesystemValues(Iterable<SkyKey> values)
      throws InterruptedException {
    return getDirtyValues(values, FILE_STATE_AND_DIRECTORY_LISTING_STATE_FILTER,
        new DirtyChecker() {
      @Override
      public DirtyResult check(SkyKey key, SkyValue oldValue, TimestampGranularityMonitor tsgm) {
        if (key.functionName() == SkyFunctions.FILE_STATE) {
          return checkFileStateValue((RootedPath) key.argument(), (FileStateValue) oldValue,
              tsgm);
        } else if (key.functionName() == SkyFunctions.DIRECTORY_LISTING_STATE) {
          return checkDirectoryListingStateValue((RootedPath) key.argument(),
              (DirectoryListingStateValue) oldValue);
        } else {
          throw new IllegalStateException("Unexpected key type " + key);
        }
      }
    });
  }

  /**
   * Return a collection of action values which have output files that are not in-sync with
   * the on-disk file value (were modified externally).
   */
  public Collection<SkyKey> getDirtyActionValues(@Nullable final BatchStat batchStatter)
      throws InterruptedException {
    // CPU-bound (usually) stat() calls, plus a fudge factor.
    LOG.info("Accumulating dirty actions");
    final int numOutputJobs = Runtime.getRuntime().availableProcessors() * 4;
    final Set<SkyKey> actionSkyKeys =
        Sets.filter(valuesSupplier.get().keySet(), ACTION_FILTER);
    final Sharder<Pair<SkyKey, ActionExecutionValue>> outputShards =
        new Sharder<>(numOutputJobs, actionSkyKeys.size());

    for (SkyKey key : actionSkyKeys) {
      outputShards.add(Pair.of(key, (ActionExecutionValue) valuesSupplier.get().get(key)));
    }
    LOG.info("Sharded action values for batching");

    ExecutorService executor = Executors.newFixedThreadPool(
        numOutputJobs,
        new ThreadFactoryBuilder().setNameFormat("FileSystem Output File Invalidator %d").build());

    Collection<SkyKey> dirtyKeys = Sets.newConcurrentHashSet();
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("FileSystemValueChecker#getDirtyActionValues");

    modifiedOutputFilesCounter.set(0);
    modifiedOutputFilesIntraBuildCounter.set(0);
    for (List<Pair<SkyKey, ActionExecutionValue>> shard : outputShards) {
      Runnable job = (batchStatter == null)
          ? outputStatJob(dirtyKeys, shard)
          : batchStatJob(dirtyKeys, shard, batchStatter);
      executor.submit(wrapper.wrap(job));
    }

    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    LOG.info("Completed output file stat checks");
    if (interrupted) {
      throw new InterruptedException();
    }
    return dirtyKeys;
  }

  private Runnable batchStatJob(final Collection<SkyKey> dirtyKeys,
                                       final List<Pair<SkyKey, ActionExecutionValue>> shard,
                                       final BatchStat batchStatter) {
    return new Runnable() {
      @Override
      public void run() {
        Map<Artifact, Pair<SkyKey, ActionExecutionValue>> artifactToKeyAndValue = new HashMap<>();
        for (Pair<SkyKey, ActionExecutionValue> keyAndValue : shard) {
          ActionExecutionValue actionValue = keyAndValue.getSecond();
          if (actionValue == null) {
            dirtyKeys.add(keyAndValue.getFirst());
          } else {
            for (Artifact artifact : actionValue.getAllOutputArtifactData().keySet()) {
              artifactToKeyAndValue.put(artifact, keyAndValue);
            }
          }
        }

        List<Artifact> artifacts = ImmutableList.copyOf(artifactToKeyAndValue.keySet());
        List<FileStatusWithDigest> stats;
        try {
          stats = batchStatter.batchStat(/*includeDigest=*/true, /*includeLinks=*/true,
                                         Artifact.asPathFragments(artifacts));
        } catch (IOException e) {
          // Batch stat did not work. Log an exception and fall back on system calls.
          LoggingUtil.logToRemote(Level.WARNING, "Unable to process batch stat", e);
          outputStatJob(dirtyKeys, shard).run();
          return;
        } catch (InterruptedException e) {
          // We handle interrupt in the main thread.
          return;
        }

        Preconditions.checkState(artifacts.size() == stats.size(),
            "artifacts.size() == %s stats.size() == %s", artifacts.size(), stats.size());
        for (int i = 0; i < artifacts.size(); i++) {
          Artifact artifact = artifacts.get(i);
          FileStatusWithDigest stat = stats.get(i);
          Pair<SkyKey, ActionExecutionValue> keyAndValue = artifactToKeyAndValue.get(artifact);
          ActionExecutionValue actionValue = keyAndValue.getSecond();
          SkyKey key = keyAndValue.getFirst();
          FileValue lastKnownData = actionValue.getAllOutputArtifactData().get(artifact);
          try {
            FileValue newData = FileAndMetadataCache.fileValueFromArtifact(artifact, stat, tsgm);
            if (!newData.equals(lastKnownData)) {
              updateIntraBuildModifiedCounter(stat != null ? stat.getLastChangeTime() : -1);
              modifiedOutputFilesCounter.getAndIncrement();
              dirtyKeys.add(key);
            }
          } catch (IOException e) {
            // This is an unexpected failure getting a digest or symlink target.
            modifiedOutputFilesCounter.getAndIncrement();
            dirtyKeys.add(key);
          }
        }
      }
    };
  }

  private void updateIntraBuildModifiedCounter(long time) throws IOException {
    if (lastExecutionTimeRange != null && lastExecutionTimeRange.contains(time)) {
      modifiedOutputFilesIntraBuildCounter.incrementAndGet();
    }
  }

  private Runnable outputStatJob(final Collection<SkyKey> dirtyKeys,
                                 final List<Pair<SkyKey, ActionExecutionValue>> shard) {
    return new Runnable() {
      @Override
      public void run() {
        for (Pair<SkyKey, ActionExecutionValue> keyAndValue : shard) {
          ActionExecutionValue value = keyAndValue.getSecond();
          if (value == null || actionValueIsDirtyWithDirectSystemCalls(value)) {
            dirtyKeys.add(keyAndValue.getFirst());
          }
        }
      }
    };
  }

  /**
   * Returns the number of modified output files inside of dirty actions.
   */
  int getNumberOfModifiedOutputFiles() {
    return modifiedOutputFilesCounter.get();
  }

  /**
   * Returns the number of modified output files that occur during the previous build.
   */
  public int getNumberOfModifiedOutputFilesDuringPreviousBuild() {
    return modifiedOutputFilesIntraBuildCounter.get();
  }

  private boolean actionValueIsDirtyWithDirectSystemCalls(ActionExecutionValue actionValue) {
    boolean isDirty = false;
    for (Map.Entry<Artifact, FileValue> entry :
        actionValue.getAllOutputArtifactData().entrySet()) {
      Artifact artifact = entry.getKey();
      FileValue lastKnownData = entry.getValue();
      try {
        FileValue fileValue = FileAndMetadataCache.fileValueFromArtifact(artifact, null, tsgm);
        if (!fileValue.equals(lastKnownData)) {
          updateIntraBuildModifiedCounter(fileValue.exists()
              ? fileValue.realRootedPath().asPath().getLastModifiedTime()
              : -1);
          modifiedOutputFilesCounter.getAndIncrement();
          isDirty = true;
        }
      } catch (IOException e) {
        // This is an unexpected failure getting a digest or symlink target.
        modifiedOutputFilesCounter.getAndIncrement();
        isDirty = true;
      }
    }
    return isDirty;
  }

  private BatchDirtyResult getDirtyValues(Iterable<SkyKey> values,
                                         Predicate<SkyKey> keyFilter,
                                         final DirtyChecker checker) throws InterruptedException {
    ExecutorService executor = Executors.newFixedThreadPool(DIRTINESS_CHECK_THREADS,
        new ThreadFactoryBuilder().setNameFormat("FileSystem Value Invalidator %d").build());

    final BatchDirtyResult batchResult = new BatchDirtyResult();
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("FilesystemValueChecker#getDirtyValues");
    for (final SkyKey key : values) {
      Preconditions.checkState(keyFilter.apply(key), key);
      final SkyValue value = valuesSupplier.get().get(key);
      executor.execute(wrapper.wrap(new Runnable() {
        @Override
        public void run() {
          if (value == null) {
            // value will be null if the value is in error or part of a cycle.
            // TODO(bazel-team): This is overly conservative.
            batchResult.add(key, /*newValue=*/null);
            return;
          }
          DirtyResult result = checker.check(key, value, tsgm);
          if (result.isDirty()) {
            batchResult.add(key, result.getNewValue());
          }
        }
      }));
    }

    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      throw new InterruptedException();
    }
    return batchResult;
  }

  private static DirtyResult checkFileStateValue(RootedPath rootedPath,
      FileStateValue fileStateValue, TimestampGranularityMonitor tsgm) {
    try {
      FileStateValue newValue = FileStateValue.create(rootedPath, tsgm);
      return newValue.equals(fileStateValue)
          ? DirtyResult.NOT_DIRTY : DirtyResult.dirtyWithNewValue(newValue);
    } catch (InconsistentFilesystemException | IOException e) {
      // TODO(bazel-team): An IOException indicates a failure to get a file digest or a symlink
      // target, not a missing file. Such a failure really shouldn't happen, so failing early
      // may be better here.
      return DirtyResult.DIRTY;
    }
  }

  private static DirtyResult checkDirectoryListingStateValue(RootedPath dirRootedPath,
      DirectoryListingStateValue directoryListingStateValue) {
    try {
      DirectoryListingStateValue newValue = DirectoryListingStateValue.create(dirRootedPath);
      return newValue.equals(directoryListingStateValue)
          ? DirtyResult.NOT_DIRTY : DirtyResult.dirtyWithNewValue(newValue);
    } catch (IOException e) {
      return DirtyResult.DIRTY;
    }
  }

  /**
   * Result of a batch call to {@link DirtyChecker#check}. Partitions the dirty values based on
   * whether we have a new value available for them or not.
   */
  private static class BatchDirtyResult implements Differencer.Diff {

    private final Set<SkyKey> concurrentDirtyKeysWithoutNewValues =
        Collections.newSetFromMap(new ConcurrentHashMap<SkyKey, Boolean>());
    private final ConcurrentHashMap<SkyKey, SkyValue> concurrentDirtyKeysWithNewValues =
        new ConcurrentHashMap<>();

    private void add(SkyKey key, @Nullable SkyValue newValue) {
      if (newValue == null) {
        concurrentDirtyKeysWithoutNewValues.add(key);
      } else {
        concurrentDirtyKeysWithNewValues.put(key, newValue);
      }
    }

    @Override
    public Iterable<SkyKey> changedKeysWithoutNewValues() {
      return concurrentDirtyKeysWithoutNewValues;
    }

    @Override
    public Map<SkyKey, ? extends SkyValue> changedKeysWithNewValues() {
      return concurrentDirtyKeysWithNewValues;
    }
  }

  private static class DirtyResult {

    static final DirtyResult NOT_DIRTY = new DirtyResult(false, null);
    static final DirtyResult DIRTY = new DirtyResult(true, null);

    private final boolean isDirty;
    @Nullable private final SkyValue newValue;

    private DirtyResult(boolean isDirty, @Nullable SkyValue newValue) {
      this.isDirty = isDirty;
      this.newValue = newValue;
    }

    boolean isDirty() {
      return isDirty;
    }

    /**
     * If {@code isDirty()}, then either returns the new value for the value or {@code null} if
     * the new value wasn't computed. In the case where the value is dirty and a new value is
     * available, then the new value can be injected into the skyframe graph. Otherwise, the value
     * should simply be invalidated.
     */
    @Nullable
    SkyValue getNewValue() {
      Preconditions.checkState(isDirty());
      return newValue;
    }

    static DirtyResult dirtyWithNewValue(SkyValue newValue) {
      return new DirtyResult(true, newValue);
    }
  }

  private static interface DirtyChecker {
    DirtyResult check(SkyKey key, SkyValue oldValue, TimestampGranularityMonitor tsgm);
  }
}
