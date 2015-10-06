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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.AutoProfiler.ElapsedTimeReceiver;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker.DirtyResult;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
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
import java.util.concurrent.TimeUnit;
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

  private static final Predicate<SkyKey> ACTION_FILTER =
      SkyFunctionName.functionIs(SkyFunctions.ACTION_EXECUTION);

  private final TimestampGranularityMonitor tsgm;
  @Nullable
  private final Range<Long> lastExecutionTimeRange;
  private final Supplier<Map<SkyKey, SkyValue>> valuesSupplier;
  private AtomicInteger modifiedOutputFilesCounter = new AtomicInteger(0);
  private AtomicInteger modifiedOutputFilesIntraBuildCounter = new AtomicInteger(0);

  FilesystemValueChecker(Supplier<Map<SkyKey, SkyValue>> valuesSupplier,
      TimestampGranularityMonitor tsgm, @Nullable Range<Long> lastExecutionTimeRange) {
    this.valuesSupplier = valuesSupplier;
    this.tsgm = tsgm;
    this.lastExecutionTimeRange = lastExecutionTimeRange;
  }

  FilesystemValueChecker(final MemoizingEvaluator evaluator, TimestampGranularityMonitor tsgm,
      @Nullable Range<Long> lastExecutionTimeRange) {
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

  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys from the backing graph (of the
   * {@link MemoizingEvaluator} given at construction time) that are dirty according to the
   * passed-in {@code dirtinessChecker}.
   */
  Differencer.DiffWithDelta getDirtyKeys(SkyValueDirtinessChecker dirtinessChecker)
      throws InterruptedException {
    return getDirtyValues(valuesSupplier.get().keySet(), dirtinessChecker,
        /*checkMissingValues=*/false);
  }

  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys that are dirty according to the
   * passed-in {@code dirtinessChecker}.
   */
  Differencer.DiffWithDelta getNewAndOldValues(Iterable<SkyKey> keys,
      SkyValueDirtinessChecker dirtinessChecker) throws InterruptedException {
    return getDirtyValues(keys, dirtinessChecker, /*checkMissingValues=*/true);
  }

  /**
   * Return a collection of action values which have output files that are not in-sync with
   * the on-disk file value (were modified externally).
   */
  Collection<SkyKey> getDirtyActionValues(@Nullable final BatchStat batchStatter)
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

    boolean interrupted = ExecutorUtil.interruptibleShutdown(executor);
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
            FileValue newData = ActionMetadataHandler.fileValueFromArtifact(artifact, stat, tsgm);
            if (!newData.equals(lastKnownData)) {
              updateIntraBuildModifiedCounter(stat != null ? stat.getLastChangeTime() : -1,
                  lastKnownData.isSymlink(), newData.isSymlink());
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

  private void updateIntraBuildModifiedCounter(long time, boolean oldWasSymlink,
      boolean newIsSymlink) {
    if (lastExecutionTimeRange != null
        && lastExecutionTimeRange.contains(time)
        && !(oldWasSymlink && newIsSymlink)) {
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

  /** Returns the number of modified output files that occur during the previous build. */
  int getNumberOfModifiedOutputFilesDuringPreviousBuild() {
    return modifiedOutputFilesIntraBuildCounter.get();
  }

  private boolean actionValueIsDirtyWithDirectSystemCalls(ActionExecutionValue actionValue) {
    boolean isDirty = false;
    for (Map.Entry<Artifact, FileValue> entry :
        actionValue.getAllOutputArtifactData().entrySet()) {
      Artifact artifact = entry.getKey();
      FileValue lastKnownData = entry.getValue();
      try {
        FileValue fileValue = ActionMetadataHandler.fileValueFromArtifact(artifact, null, tsgm);
        if (!fileValue.equals(lastKnownData)) {
          updateIntraBuildModifiedCounter(fileValue.exists()
              ? fileValue.realRootedPath().asPath().getLastModifiedTime()
              : -1, lastKnownData.isSymlink(), fileValue.isSymlink());
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

  private BatchDirtyResult getDirtyValues(
      Iterable<SkyKey> values, final SkyValueDirtinessChecker checker,
      final boolean checkMissingValues) throws InterruptedException {
    ExecutorService executor =
        Executors.newFixedThreadPool(
            DIRTINESS_CHECK_THREADS,
            new ThreadFactoryBuilder().setNameFormat("FileSystem Value Invalidator %d").build());

    final BatchDirtyResult batchResult = new BatchDirtyResult();
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("FilesystemValueChecker#getDirtyValues");
    final AtomicInteger numKeysScanned = new AtomicInteger(0);
    final AtomicInteger numKeysChecked = new AtomicInteger(0);
    ElapsedTimeReceiver elapsedTimeReceiver = new ElapsedTimeReceiver() {
        @Override
        public void accept(long elapsedTimeNanos) {
          if (elapsedTimeNanos > 0) {
            LOG.info(String.format("Spent %d ms checking %d filesystem nodes (%d scanned)",
                TimeUnit.MILLISECONDS.convert(elapsedTimeNanos, TimeUnit.NANOSECONDS),
                numKeysChecked.get(),
                numKeysScanned.get()));
          }
        }
    };
    try (AutoProfiler prof = AutoProfiler.create(elapsedTimeReceiver)) {
      for (final SkyKey key : values) {
        final SkyValue value = valuesSupplier.get().get(key);
        executor.execute(
            wrapper.wrap(
                new Runnable() {
                  @Override
                  public void run() {
                    if (value != null || checkMissingValues) {
                      numKeysScanned.incrementAndGet();
                      DirtyResult result = checker.maybeCheck(key, value, tsgm);
                      if (result != null) {
                        numKeysChecked.incrementAndGet();
                        if (result.isDirty()) {
                          batchResult.add(key, value, result.getNewValue());
                        }
                      }
                    }
                  }
                }));
      }

      boolean interrupted = ExecutorUtil.interruptibleShutdown(executor);
      Throwables.propagateIfPossible(wrapper.getFirstThrownError());
      if (interrupted) {
        throw new InterruptedException();
      }
    }
    return batchResult;
  }

  /**
   * Result of a batch call to {@link SkyValueDirtinessChecker#maybeCheck}. Partitions the dirty
   * values based on whether we have a new value available for them or not.
   */
  private static class BatchDirtyResult implements Differencer.DiffWithDelta {

    private final Set<SkyKey> concurrentDirtyKeysWithoutNewValues =
        Collections.newSetFromMap(new ConcurrentHashMap<SkyKey, Boolean>());
    private final ConcurrentHashMap<SkyKey, Delta> concurrentDirtyKeysWithNewAndOldValues =
        new ConcurrentHashMap<>();

    private void add(SkyKey key, @Nullable SkyValue oldValue, @Nullable SkyValue newValue) {
      if (newValue == null) {
        concurrentDirtyKeysWithoutNewValues.add(key);
      } else {
        if (oldValue == null) {
          concurrentDirtyKeysWithNewAndOldValues.put(key, new Delta(newValue));
        } else {
          concurrentDirtyKeysWithNewAndOldValues.put(key, new Delta(oldValue, newValue));
        }
      }
    }

    @Override
    public Collection<SkyKey> changedKeysWithoutNewValues() {
      return concurrentDirtyKeysWithoutNewValues;
    }

    @Override
    public Map<SkyKey, Delta> changedKeysWithNewAndOldValues() {
      return concurrentDirtyKeysWithNewAndOldValues;
    }

    @Override
    public Map<SkyKey, SkyValue> changedKeysWithNewValues() {
      return Delta.newValues(concurrentDirtyKeysWithNewAndOldValues);
    }
  }

}
