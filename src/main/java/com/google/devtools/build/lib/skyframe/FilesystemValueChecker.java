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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.AutoProfiler.ElapsedTimeReceiver;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker.DirtyResult;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.Differencer;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.FunctionHermeticity;
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
import java.util.NavigableSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * A helper class to find dirty values by accessing the filesystem directly (contrast with
 * {@link DiffAwareness}).
 */
public class FilesystemValueChecker {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Predicate<SkyKey> ACTION_FILTER =
      SkyFunctionName.functionIs(SkyFunctions.ACTION_EXECUTION);

  @Nullable private final TimestampGranularityMonitor tsgm;
  @Nullable private final Range<Long> lastExecutionTimeRange;
  private AtomicInteger modifiedOutputFilesCounter = new AtomicInteger(0);
  private AtomicInteger modifiedOutputFilesIntraBuildCounter = new AtomicInteger(0);
  private final int numThreads;

  public FilesystemValueChecker(
      @Nullable TimestampGranularityMonitor tsgm,
      @Nullable Range<Long> lastExecutionTimeRange,
      int numThreads) {
    this.tsgm = tsgm;
    this.lastExecutionTimeRange = lastExecutionTimeRange;
    this.numThreads = numThreads;
  }
  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys from the give map that are dirty
   * according to the passed-in {@code dirtinessChecker}.
   */
  // TODO(bazel-team): Refactor these methods so that FilesystemValueChecker only operates on a
  // WalkableGraph.
  ImmutableBatchDirtyResult getDirtyKeys(
      Map<SkyKey, SkyValue> valuesMap, SkyValueDirtinessChecker dirtinessChecker)
      throws InterruptedException {
    return getDirtyValues(new MapBackedValueFetcher(valuesMap), valuesMap.keySet(),
        dirtinessChecker, /*checkMissingValues=*/false);
  }

  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys that are dirty according to the
   * passed-in {@code dirtinessChecker}.
   */
  public ImmutableBatchDirtyResult getNewAndOldValues(
      Map<SkyKey, SkyValue> valuesMap,
      Collection<SkyKey> keys,
      SkyValueDirtinessChecker dirtinessChecker)
      throws InterruptedException {
    return getDirtyValues(new MapBackedValueFetcher(valuesMap), keys,
        dirtinessChecker, /*checkMissingValues=*/true);
  }

  /**
   * Returns a {@link Differencer.DiffWithDelta} containing keys that are dirty according to the
   * passed-in {@code dirtinessChecker}.
   */
  public Differencer.DiffWithDelta getNewAndOldValues(
      WalkableGraph walkableGraph,
      Collection<SkyKey> keys,
      SkyValueDirtinessChecker dirtinessChecker)
      throws InterruptedException {
    return getDirtyValues(new WalkableGraphBackedValueFetcher(walkableGraph), keys,
        dirtinessChecker, /*checkMissingValues=*/true);
  }

  private interface ValueFetcher {
    @Nullable
    SkyValue get(SkyKey key) throws InterruptedException;
  }

  private static class WalkableGraphBackedValueFetcher implements ValueFetcher {
    private final WalkableGraph walkableGraph;

    private WalkableGraphBackedValueFetcher(WalkableGraph walkableGraph) {
      this.walkableGraph = walkableGraph;
    }

    @Override
    @Nullable
    public SkyValue get(SkyKey key) throws InterruptedException {
      return walkableGraph.getValue(key);
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
   * Return a collection of action values which have output files that are not in-sync with the
   * on-disk file value (were modified externally).
   */
  Collection<SkyKey> getDirtyActionValues(
      Map<SkyKey, SkyValue> valuesMap,
      @Nullable final BatchStat batchStatter,
      ModifiedFileSet modifiedOutputFiles,
      boolean trustRemoteArtifacts)
      throws InterruptedException {
    if (modifiedOutputFiles == ModifiedFileSet.NOTHING_MODIFIED) {
      logger.atInfo().log("Not checking for dirty actions since nothing was modified");
      return ImmutableList.of();
    }
    logger.atInfo().log("Accumulating dirty actions");
    final int numOutputJobs = Runtime.getRuntime().availableProcessors() * 4;
    final Set<SkyKey> actionSkyKeys = new HashSet<>();
    try (SilentCloseable c = Profiler.instance().profile("getDirtyActionValues.filter_actions")) {
      for (SkyKey key : valuesMap.keySet()) {
        if (ACTION_FILTER.apply(key)) {
          actionSkyKeys.add(key);
        }
      }
    }
    final Sharder<Pair<SkyKey, ActionExecutionValue>> outputShards =
        new Sharder<>(numOutputJobs, actionSkyKeys.size());

    for (SkyKey key : actionSkyKeys) {
      outputShards.add(Pair.of(key, (ActionExecutionValue) valuesMap.get(key)));
    }
    logger.atInfo().log("Sharded action values for batching");

    ExecutorService executor = Executors.newFixedThreadPool(
        numOutputJobs,
        new ThreadFactoryBuilder().setNameFormat("FileSystem Output File Invalidator %d").build());

    Collection<SkyKey> dirtyKeys = Sets.newConcurrentHashSet();
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("FileSystemValueChecker#getDirtyActionValues");

    modifiedOutputFilesCounter.set(0);
    modifiedOutputFilesIntraBuildCounter.set(0);
    final ImmutableSet<PathFragment> knownModifiedOutputFiles =
            modifiedOutputFiles == ModifiedFileSet.EVERYTHING_MODIFIED
                    ? null
                    : modifiedOutputFiles.modifiedSourceFiles();

    // Initialized lazily through a supplier because it is only used to check modified
    // TreeArtifacts, which are not frequently used in builds.
    Supplier<NavigableSet<PathFragment>> sortedKnownModifiedOutputFiles =
      Suppliers.memoize(new Supplier<NavigableSet<PathFragment>>() {
        @Override
        public NavigableSet<PathFragment> get() {
          if (knownModifiedOutputFiles == null) {
            return null;
          } else {
            return ImmutableSortedSet.copyOf(knownModifiedOutputFiles);
          }
        }
      });

    boolean interrupted;
    try (SilentCloseable c = Profiler.instance().profile("getDirtyActionValues.stat_files")) {
      for (List<Pair<SkyKey, ActionExecutionValue>> shard : outputShards) {
        Runnable job =
            (batchStatter == null)
                ? outputStatJob(
                    dirtyKeys,
                    shard,
                    knownModifiedOutputFiles,
                    sortedKnownModifiedOutputFiles,
                    trustRemoteArtifacts)
                : batchStatJob(
                    dirtyKeys,
                    shard,
                    batchStatter,
                    knownModifiedOutputFiles,
                    sortedKnownModifiedOutputFiles,
                    trustRemoteArtifacts);
        Future<?> unused = executor.submit(wrapper.wrap(job));
      }

      interrupted = ExecutorUtil.interruptibleShutdown(executor);
    }
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    logger.atInfo().log("Completed output file stat checks");
    if (interrupted) {
      throw new InterruptedException();
    }
    return dirtyKeys;
  }

  private Runnable batchStatJob(
      final Collection<SkyKey> dirtyKeys,
      final List<Pair<SkyKey, ActionExecutionValue>> shard,
      final BatchStat batchStatter,
      final ImmutableSet<PathFragment> knownModifiedOutputFiles,
      final Supplier<NavigableSet<PathFragment>> sortedKnownModifiedOutputFiles,
      boolean trustRemoteArtifacts) {
    return new Runnable() {
      @Override
      public void run() {
        Map<Artifact, Pair<SkyKey, ActionExecutionValue>> fileToKeyAndValue = new HashMap<>();
        Map<Artifact, Pair<SkyKey, ActionExecutionValue>> treeArtifactsToKeyAndValue =
            new HashMap<>();
        for (Pair<SkyKey, ActionExecutionValue> keyAndValue : shard) {
          ActionExecutionValue actionValue = keyAndValue.getSecond();
          if (actionValue == null) {
            dirtyKeys.add(keyAndValue.getFirst());
          } else {
            for (Artifact artifact : actionValue.getAllFileValues().keySet()) {
              if (!artifact.isMiddlemanArtifact()
                  && shouldCheckFile(knownModifiedOutputFiles, artifact)) {
                fileToKeyAndValue.put(artifact, keyAndValue);
              }
            }

            for (Map.Entry<Artifact, TreeArtifactValue> entry :
                actionValue.getAllTreeArtifactValues().entrySet()) {
              Artifact treeArtifact = entry.getKey();
              TreeArtifactValue tree = entry.getValue();
              for (TreeFileArtifact child : tree.getChildren()) {
                if (shouldCheckFile(knownModifiedOutputFiles, child)) {
                  fileToKeyAndValue.put(child, keyAndValue);
                }
              }
              if (shouldCheckTreeArtifact(sortedKnownModifiedOutputFiles.get(), treeArtifact)) {
                treeArtifactsToKeyAndValue.put(treeArtifact, keyAndValue);
              }
            }
          }
        }

        List<Artifact> artifacts = ImmutableList.copyOf(fileToKeyAndValue.keySet());
        List<FileStatusWithDigest> stats;
        try {
          stats =
              batchStatter.batchStat(
                  /*includeDigest=*/ true,
                  /*includeLinks=*/ true,
                  Artifact.asPathFragments(artifacts));
        } catch (IOException e) {
          // Batch stat did not work. Log an exception and fall back on system calls.
          LoggingUtil.logToRemote(Level.WARNING, "Unable to process batch stat", e);
          logger.atWarning().withCause(e).log("Unable to process batch stat");
          outputStatJob(
                  dirtyKeys,
                  shard,
                  knownModifiedOutputFiles,
                  sortedKnownModifiedOutputFiles,
                  trustRemoteArtifacts)
              .run();
          return;
        } catch (InterruptedException e) {
          // We handle interrupt in the main thread.
          return;
        }

        Preconditions.checkState(
            artifacts.size() == stats.size(),
            "artifacts.size() == %s stats.size() == %s",
            artifacts.size(),
            stats.size());
        for (int i = 0; i < artifacts.size(); i++) {
          Artifact artifact = artifacts.get(i);
          FileStatusWithDigest stat = stats.get(i);
          Pair<SkyKey, ActionExecutionValue> keyAndValue = fileToKeyAndValue.get(artifact);
          ActionExecutionValue actionValue = keyAndValue.getSecond();
          SkyKey key = keyAndValue.getFirst();
          FileArtifactValue lastKnownData = actionValue.getExistingFileArtifactValue(artifact);
          try {
            FileArtifactValue newData =
                ActionMetadataHandler.fileArtifactValueFromArtifact(artifact, stat, tsgm);
            if (newData.couldBeModifiedSince(lastKnownData)) {
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

        // Unfortunately, there exists no facility to batch list directories.
        // We must use direct filesystem calls.
        for (Map.Entry<Artifact, Pair<SkyKey, ActionExecutionValue>> entry :
            treeArtifactsToKeyAndValue.entrySet()) {
          Artifact artifact = entry.getKey();
          if (treeArtifactIsDirty(
              entry.getKey(), entry.getValue().getSecond().getTreeArtifactValue(artifact))) {
            Path path = artifact.getPath();
            // Count the changed directory as one "file".
            // TODO(bazel-team): There are no tests for this codepath.
            try {
              updateIntraBuildModifiedCounter(path.exists() ? path.getLastModifiedTime() : -1);
            } catch (IOException e) {
              // Do nothing here.
            }

            modifiedOutputFilesCounter.getAndIncrement();
            dirtyKeys.add(entry.getValue().getFirst());
          }
        }
      }
    };
  }

  private void updateIntraBuildModifiedCounter(long time) {
    if (lastExecutionTimeRange != null && lastExecutionTimeRange.contains(time)) {
      modifiedOutputFilesIntraBuildCounter.incrementAndGet();
    }
  }

  private Runnable outputStatJob(
      final Collection<SkyKey> dirtyKeys,
      final List<Pair<SkyKey, ActionExecutionValue>> shard,
      final ImmutableSet<PathFragment> knownModifiedOutputFiles,
      final Supplier<NavigableSet<PathFragment>> sortedKnownModifiedOutputFiles,
      boolean trustRemoteArtifacts) {
    return new Runnable() {
      @Override
      public void run() {
        for (Pair<SkyKey, ActionExecutionValue> keyAndValue : shard) {
          ActionExecutionValue value = keyAndValue.getSecond();
          if (value == null
              || actionValueIsDirtyWithDirectSystemCalls(
                  value,
                  knownModifiedOutputFiles,
                  sortedKnownModifiedOutputFiles,
                  trustRemoteArtifacts)) {
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

  private boolean treeArtifactIsDirty(Artifact artifact, TreeArtifactValue value) {
    if (artifact.getPath().isSymbolicLink()) {
      // TreeArtifacts may not be symbolic links.
      return true;
    }

    // There doesn't appear to be any facility to batch list directories... we must
    // do things the 'slow' way.
    try {
      Set<PathFragment> currentDirectoryValue =
          TreeArtifactValue.explodeDirectory(artifact.getPath());
      return !(currentDirectoryValue.isEmpty() && value.isEntirelyRemote())
          && !currentDirectoryValue.equals(value.getChildPaths());
    } catch (IOException e) {
      return true;
    }
  }

  private boolean artifactIsDirtyWithDirectSystemCalls(
      ImmutableSet<PathFragment> knownModifiedOutputFiles,
      boolean trustRemoteArtifacts,
      Map.Entry<? extends Artifact, FileArtifactValue> entry) {
    Artifact file = entry.getKey();
    FileArtifactValue lastKnownData = entry.getValue();
    if (file.isMiddlemanArtifact() || !shouldCheckFile(knownModifiedOutputFiles, file)) {
      return false;
    }
    try {
      FileArtifactValue fileMetadata =
          ActionMetadataHandler.fileArtifactValueFromArtifact(file, null, tsgm);
      boolean trustRemoteValue =
          fileMetadata.getType() == FileStateType.NONEXISTENT
              && lastKnownData.isRemote()
              && trustRemoteArtifacts;
      if (!trustRemoteValue && fileMetadata.couldBeModifiedSince(lastKnownData)) {
        updateIntraBuildModifiedCounter(
            fileMetadata.getType() != FileStateType.NONEXISTENT
                ? file.getPath().getLastModifiedTime(Symlinks.FOLLOW)
                : -1);
        modifiedOutputFilesCounter.getAndIncrement();
        return true;
      }
      return false;
    } catch (IOException e) {
      // This is an unexpected failure getting a digest or symlink target.
      modifiedOutputFilesCounter.getAndIncrement();
      return true;
    }
  }

  private boolean actionValueIsDirtyWithDirectSystemCalls(
      ActionExecutionValue actionValue,
      ImmutableSet<PathFragment> knownModifiedOutputFiles,
      Supplier<NavigableSet<PathFragment>> sortedKnownModifiedOutputFiles,
      boolean trustRemoteArtifacts) {
    boolean isDirty = false;
    for (Map.Entry<Artifact, FileArtifactValue> entry : actionValue.getAllFileValues().entrySet()) {
      if (artifactIsDirtyWithDirectSystemCalls(
          knownModifiedOutputFiles, trustRemoteArtifacts, entry)) {
        isDirty = true;
      }
    }

    for (Map.Entry<Artifact, TreeArtifactValue> entry :
        actionValue.getAllTreeArtifactValues().entrySet()) {
      TreeArtifactValue tree = entry.getValue();

      if (!tree.isEntirelyRemote()) {
        for (Map.Entry<TreeFileArtifact, FileArtifactValue> childEntry :
            tree.getChildValues().entrySet()) {
          if (artifactIsDirtyWithDirectSystemCalls(
              knownModifiedOutputFiles, trustRemoteArtifacts, childEntry)) {
            isDirty = true;
          }
        }
      }

      Artifact treeArtifact = entry.getKey();

      if (shouldCheckTreeArtifact(sortedKnownModifiedOutputFiles.get(), treeArtifact)
          && treeArtifactIsDirty(treeArtifact, entry.getValue())) {
        Path path = treeArtifact.getPath();
        // Count the changed directory as one "file".
        try {
          updateIntraBuildModifiedCounter(path.exists() ? path.getLastModifiedTime() : -1);
        } catch (IOException e) {
          // Do nothing here.
        }

        modifiedOutputFilesCounter.getAndIncrement();
        isDirty = true;
      }
    }

    return isDirty;
  }

  private static boolean shouldCheckFile(ImmutableSet<PathFragment> knownModifiedOutputFiles,
      Artifact artifact) {
    return knownModifiedOutputFiles == null
        || knownModifiedOutputFiles.contains(artifact.getExecPath());
  }

  private static boolean shouldCheckTreeArtifact(
      @Nullable NavigableSet<PathFragment> knownModifiedOutputFiles, Artifact treeArtifact) {
    // If null, everything needs to be checked.
    if (knownModifiedOutputFiles == null) {
      return true;
    }

    // Here we do the following to see whether a TreeArtifact is modified:
    // 1. Sort the set of modified file paths in lexicographical order using TreeSet.
    // 2. Get the first modified output file path that is greater than or equal to the exec path of
    //    the TreeArtifact to check.
    // 3. Check whether the returned file path contains the exec path of the TreeArtifact as a
    //    prefix path.
    PathFragment artifactExecPath = treeArtifact.getExecPath();
    PathFragment headPath = knownModifiedOutputFiles.ceiling(artifactExecPath);

    return headPath != null && headPath.startsWith(artifactExecPath);
  }

  private ImmutableBatchDirtyResult getDirtyValues(
      ValueFetcher fetcher,
      Collection<SkyKey> keys,
      final SkyValueDirtinessChecker checker,
      final boolean checkMissingValues)
      throws InterruptedException {
    ExecutorService executor =
        Executors.newFixedThreadPool(
            numThreads,
            new ThreadFactoryBuilder().setNameFormat("FileSystem Value Invalidator %d").build());

    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("FilesystemValueChecker#getDirtyValues");
    final AtomicInteger numKeysChecked = new AtomicInteger(0);
    MutableBatchDirtyResult batchResult = new MutableBatchDirtyResult(numKeysChecked);
    ElapsedTimeReceiver elapsedTimeReceiver =
        elapsedTimeNanos -> {
          if (elapsedTimeNanos > 0) {
            logger.atInfo().log(
                "Spent %d nanoseconds checking %d filesystem nodes (%d scanned)",
                elapsedTimeNanos, numKeysChecked.get(), keys.size());
          }
        };
    try (AutoProfiler prof = AutoProfiler.create(elapsedTimeReceiver)) {
      for (final SkyKey key : keys) {
        if (!checker.applies(key)) {
          continue;
        }
        Preconditions.checkState(
            key.functionName().getHermeticity() == FunctionHermeticity.NONHERMETIC,
            "Only non-hermetic keys can be dirty roots: %s",
            key);
        executor.execute(
            wrapper.wrap(
                () -> {
                  SkyValue value;
                  try {
                    value = fetcher.get(key);
                  } catch (InterruptedException e) {
                    // Exit fast. Interrupt is handled below on the main thread.
                    return;
                  }
                  if (!checkMissingValues && value == null) {
                    return;
                  }

                  numKeysChecked.incrementAndGet();
                  DirtyResult result = checker.check(key, value, tsgm);
                  if (result.isDirty()) {
                    batchResult.add(key, value, result.getNewValue());
                  }
                }));
      }

      boolean interrupted = ExecutorUtil.interruptibleShutdown(executor);
      Throwables.propagateIfPossible(wrapper.getFirstThrownError());
      if (interrupted) {
        throw new InterruptedException();
      }
    }
    return batchResult.toImmutable();
  }

  static class ImmutableBatchDirtyResult implements Differencer.DiffWithDelta {
    private final Collection<SkyKey> dirtyKeysWithoutNewValues;
    private final Map<SkyKey, Delta> dirtyKeysWithNewAndOldValues;
    private final int numKeysChecked;

    private ImmutableBatchDirtyResult(
        Collection<SkyKey> dirtyKeysWithoutNewValues,
        Map<SkyKey, Delta> dirtyKeysWithNewAndOldValues,
        int numKeysChecked) {
      this.dirtyKeysWithoutNewValues = dirtyKeysWithoutNewValues;
      this.dirtyKeysWithNewAndOldValues = dirtyKeysWithNewAndOldValues;
      this.numKeysChecked = numKeysChecked;
    }

    @Override
    public Collection<SkyKey> changedKeysWithoutNewValues() {
      return dirtyKeysWithoutNewValues;
    }

    @Override
    public Map<SkyKey, Delta> changedKeysWithNewAndOldValues() {
      return dirtyKeysWithNewAndOldValues;
    }

    @Override
    public Map<SkyKey, SkyValue> changedKeysWithNewValues() {
      return Delta.newValues(dirtyKeysWithNewAndOldValues);
    }

    int getNumKeysChecked() {
      return numKeysChecked;
    }
  }

  /**
   * Result of a batch call to {@link SkyValueDirtinessChecker#check}. Partitions the dirty values
   * based on whether we have a new value available for them or not.
   */
  private static class MutableBatchDirtyResult {
    private final Set<SkyKey> concurrentDirtyKeysWithoutNewValues =
        Collections.newSetFromMap(new ConcurrentHashMap<SkyKey, Boolean>());
    private final ConcurrentHashMap<SkyKey, Delta> concurrentDirtyKeysWithNewAndOldValues =
        new ConcurrentHashMap<>();
    private final AtomicInteger numChecked;

    private MutableBatchDirtyResult(AtomicInteger numChecked) {
      this.numChecked = numChecked;
    }

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

    private ImmutableBatchDirtyResult toImmutable() {
      return new ImmutableBatchDirtyResult(
          concurrentDirtyKeysWithoutNewValues,
          concurrentDirtyKeysWithNewAndOldValues,
          numChecked.get());
    }
  }

}
