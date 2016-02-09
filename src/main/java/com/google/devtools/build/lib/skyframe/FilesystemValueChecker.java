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

import com.google.common.base.Predicate;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFile;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.AutoProfiler.ElapsedTimeReceiver;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker.DirtyResult;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
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
public class FilesystemValueChecker {

  private static final int DIRTINESS_CHECK_THREADS = 200;
  private static final Logger LOG = Logger.getLogger(FilesystemValueChecker.class.getName());

  private static final Predicate<SkyKey> ACTION_FILTER =
      SkyFunctionName.functionIs(SkyFunctions.ACTION_EXECUTION);

  private final TimestampGranularityMonitor tsgm;
  @Nullable
  private final Range<Long> lastExecutionTimeRange;
  private AtomicInteger modifiedOutputFilesCounter = new AtomicInteger(0);
  private AtomicInteger modifiedOutputFilesIntraBuildCounter = new AtomicInteger(0);

  public FilesystemValueChecker(@Nullable TimestampGranularityMonitor tsgm,
      @Nullable Range<Long> lastExecutionTimeRange) {
    this.tsgm = tsgm;
    this.lastExecutionTimeRange = lastExecutionTimeRange;
  }

  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys from the give map that are dirty
   * according to the passed-in {@code dirtinessChecker}.
   */
  // TODO(bazel-team): Refactor these methods so that FilesystemValueChecker only operates on a
  // WalkableGraph.
  Differencer.DiffWithDelta getDirtyKeys(Map<SkyKey, SkyValue> valuesMap,
      SkyValueDirtinessChecker dirtinessChecker) throws InterruptedException {
    return getDirtyValues(new MapBackedValueFetcher(valuesMap), valuesMap.keySet(),
        dirtinessChecker, /*checkMissingValues=*/false);
  }

  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys that are dirty according to the
   * passed-in {@code dirtinessChecker}.
   */
  public Differencer.DiffWithDelta getNewAndOldValues(Map<SkyKey, SkyValue> valuesMap,
      Iterable<SkyKey> keys, SkyValueDirtinessChecker dirtinessChecker)
          throws InterruptedException {
    return getDirtyValues(new MapBackedValueFetcher(valuesMap), keys,
        dirtinessChecker, /*checkMissingValues=*/true);
  }

  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys that are dirty according to the
   * passed-in {@code dirtinessChecker}.
   */
  public Differencer.DiffWithDelta getNewAndOldValues(WalkableGraph walkableGraph,
      Iterable<SkyKey> keys, SkyValueDirtinessChecker dirtinessChecker)
          throws InterruptedException {
    return getDirtyValues(new WalkableGraphBackedValueFetcher(walkableGraph), keys,
        dirtinessChecker, /*checkMissingValues=*/true);
  }

  private static interface ValueFetcher {
    @Nullable
    SkyValue get(SkyKey key);
  }

  private static class WalkableGraphBackedValueFetcher implements ValueFetcher {
    private final WalkableGraph walkableGraph;

    private WalkableGraphBackedValueFetcher(WalkableGraph walkableGraph) {
      this.walkableGraph = walkableGraph;
    }

    @Override
    @Nullable
    public SkyValue get(SkyKey key) {
      return walkableGraph.exists(key) ? walkableGraph.getValue(key) : null;
    }
  }

  private static class MapBackedValueFetcher implements ValueFetcher {
    private final Map<SkyKey, SkyValue> valuesMap;

    private MapBackedValueFetcher(Map<SkyKey, SkyValue> valuesMap) {
      this.valuesMap = valuesMap;
    }

    @Override
    @Nullable
    public SkyValue get(SkyKey key) {
      return valuesMap.get(key);
    }
  }

  /**
   * Return a collection of action values which have output files that are not in-sync with
   * the on-disk file value (were modified externally).
   */
  Collection<SkyKey> getDirtyActionValues(Map<SkyKey, SkyValue> valuesMap,
      @Nullable final BatchStat batchStatter, ModifiedFileSet modifiedOutputFiles)
          throws InterruptedException {
    if (modifiedOutputFiles == ModifiedFileSet.NOTHING_MODIFIED) {
      LOG.info("Not checking for dirty actions since nothing was modified");
      return ImmutableList.of();
    }
    LOG.info("Accumulating dirty actions");
    final int numOutputJobs = Runtime.getRuntime().availableProcessors() * 4;
    final Set<SkyKey> actionSkyKeys = new HashSet<>();
    for (SkyKey key : valuesMap.keySet()) {
      if (ACTION_FILTER.apply(key)) {
        actionSkyKeys.add(key);
      }
    }
    final Sharder<Pair<SkyKey, ActionExecutionValue>> outputShards =
        new Sharder<>(numOutputJobs, actionSkyKeys.size());

    for (SkyKey key : actionSkyKeys) {
      outputShards.add(Pair.of(key, (ActionExecutionValue) valuesMap.get(key)));
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
    ImmutableSet<PathFragment> knownModifiedOutputFiles =
            modifiedOutputFiles == ModifiedFileSet.EVERYTHING_MODIFIED
                    ? null
                    : modifiedOutputFiles.modifiedSourceFiles();
    for (List<Pair<SkyKey, ActionExecutionValue>> shard : outputShards) {
      Runnable job = (batchStatter == null)
          ? outputStatJob(dirtyKeys, shard, knownModifiedOutputFiles)
          : batchStatJob(dirtyKeys, shard, batchStatter, knownModifiedOutputFiles);
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
          final BatchStat batchStatter, final ImmutableSet<PathFragment> knownModifiedOutputFiles) {
    return new Runnable() {
      @Override
      public void run() {
        Map<Artifact, Pair<SkyKey, ActionExecutionValue>> artifactToKeyAndValue = new HashMap<>();
        for (Pair<SkyKey, ActionExecutionValue> keyAndValue : shard) {
          ActionExecutionValue actionValue = keyAndValue.getSecond();
          if (actionValue == null) {
            dirtyKeys.add(keyAndValue.getFirst());
          } else {
            for (ArtifactFile file : actionValue.getAllFileValues().keySet()) {
              // File system value checking for non-Artifacts is not yet implemented.
              if (file instanceof Artifact) {
                Artifact artifact = (Artifact) file;
                if (shouldCheckArtifact(knownModifiedOutputFiles, artifact)) {
                  artifactToKeyAndValue.put(artifact, keyAndValue);
                }
              }
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
          outputStatJob(dirtyKeys, shard, knownModifiedOutputFiles).run();
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
          FileValue lastKnownData = actionValue.getAllFileValues().get(artifact);
          try {
            FileValue newData = ActionMetadataHandler.fileValueFromArtifactFile(artifact, stat,
                tsgm);
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
          final List<Pair<SkyKey, ActionExecutionValue>> shard,
          final ImmutableSet<PathFragment> knownModifiedOutputFiles) {
    return new Runnable() {
      @Override
      public void run() {
        for (Pair<SkyKey, ActionExecutionValue> keyAndValue : shard) {
          ActionExecutionValue value = keyAndValue.getSecond();
          if (value == null
              || actionValueIsDirtyWithDirectSystemCalls(value, knownModifiedOutputFiles)) {
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

  private boolean actionValueIsDirtyWithDirectSystemCalls(ActionExecutionValue actionValue,
      ImmutableSet<PathFragment> knownModifiedOutputFiles) {
    boolean isDirty = false;
    for (Map.Entry<ArtifactFile, FileValue> entry : actionValue.getAllFileValues().entrySet()) {
      ArtifactFile file = entry.getKey();
      FileValue lastKnownData = entry.getValue();
      if (shouldCheckArtifact(knownModifiedOutputFiles, file)) {
        try {
          FileValue fileValue = ActionMetadataHandler.fileValueFromArtifactFile(file, null,
              tsgm);
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
    }
    return isDirty;
  }

  private static boolean shouldCheckArtifact(ImmutableSet<PathFragment> knownModifiedOutputFiles,
      ArtifactFile file) {
    return knownModifiedOutputFiles == null
        || knownModifiedOutputFiles.contains(file.getExecPath());
  }

  private BatchDirtyResult getDirtyValues(ValueFetcher fetcher,
      Iterable<SkyKey> keys, final SkyValueDirtinessChecker checker,
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
      for (final SkyKey key : keys) {
        numKeysScanned.incrementAndGet();
        if (!checker.applies(key)) {
          continue;
        }
        final SkyValue value = fetcher.get(key);
        if (!checkMissingValues && value == null) {
          continue;
        }
        executor.execute(
            wrapper.wrap(
                new Runnable() {
                  @Override
                  public void run() {
                    numKeysChecked.incrementAndGet();
                    DirtyResult result = checker.check(key, value, tsgm);
                    if (result.isDirty()) {
                      batchResult.add(key, value, result.getNewValue());
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
   * Result of a batch call to {@link SkyValueDirtinessChecker#check}. Partitions the dirty
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
