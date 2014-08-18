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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ExecutorShutdownUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;

import javax.annotation.Nullable;

/**
 * Implements cache for file stat data for all build artifacts.
 * Data is cached for the duration of the build and ensures consistent state
 * of the artifact metadata for the duration of the build.
 *
 * When used by the DatabaseDependencyChecker, persistence of this cache across
 * multiple builds is nonvital. However, this cache is persisted through the use
 * of {@code ArtifactMTimeCache} in order to provide metadata information for
 * incremental build cases.
 */
@ConditionallyThreadSafe // Thread safety is limited by the underlying MetadataCache usage.
public class ArtifactMetadataCache implements MetadataHandler {

  /**
   * Used to store metadata associated with virtual artifact (e.g. middleman).
   */
  private final static class VirtualArtifactFileStatus implements FileStatus {
    final Metadata metadata;
    // Digest should only be null for NO_METADATA_STATUS.
    VirtualArtifactFileStatus (Digest digest) {
      this.metadata = digest == null ? Metadata.NO_METADATA : digest.asMetadata();
    }
    @Override public long getLastChangeTime() { return 0; }
    @Override public long getNodeId() { return 0; }
    @Override public long getLastModifiedTime()  { return 0; }
    @Override public long getSize() { return 0; }
    @Override public boolean isDirectory() { return false; }
    @Override public boolean isFile() { return false; }
    @Override public boolean isSymbolicLink() { return false; }
  }

  /**
   * Used to store exception occurred during stat() syscall.
   */
  protected final static class FileStatusException extends IOException implements FileStatus {
    FileStatusException (IOException e) { super(e); }
    @Override public long getLastChangeTime() throws IOException { throw this; }
    @Override public long getNodeId() throws IOException { throw this; }
    @Override public long getLastModifiedTime() throws IOException { throw this; }
    @Override public long getSize() { return 0; }
    @Override public boolean isDirectory() { return false; }
    @Override public boolean isFile() { return false; }
    @Override public boolean isSymbolicLink() { return false; }
  }

  // Default file status for virtual artifacts.
  private static final FileStatus NO_METADATA_STATUS = new VirtualArtifactFileStatus(null);

  protected final ConcurrentMap<Path, FileStatus> fileStatusMap;
  protected final MetadataCache metadataCache;
  private final TimestampGranularityMonitor granularityMonitor;
  private final BatchStat statter;

  private final PackageUpToDateChecker packageUpToDateChecker;

  /**
   *  Artifacts detected as changed this run. Kept here in case the build is interrupted early, so
   *  that {@link #updateCache} does not miss any artifacts detected as changed in previous builds
   *  (so, which have updated metadata). See javadoc for DependentActionGraph#staleActions.
   */
  private final Set<Artifact> changedArtifacts = Sets.newConcurrentHashSet();

  /**
   * Testing constructor that permits a custom mayHaveChangedChecker.
   */
  public ArtifactMetadataCache(MetadataCache metadataCache,
      TimestampGranularityMonitor timestampGranularityMonitor,
      BatchStat statter, PackageUpToDateChecker upToDateChecker) {
    this.fileStatusMap = new ConcurrentHashMap<Path, FileStatus>(10000);
    this.metadataCache = metadataCache;
    this.granularityMonitor = timestampGranularityMonitor;
    this.statter = statter;
    this.packageUpToDateChecker = Preconditions.checkNotNull(upToDateChecker);
  }

  /**
   * Instantiates ArtifactMetadataCache. Each build should use it exactly once.
   */
  public ArtifactMetadataCache(MetadataCache metadataCache,
      TimestampGranularityMonitor timestampGranularityMonitor,
      PackageUpToDateChecker upToDateChecker) {
    this(metadataCache, timestampGranularityMonitor, null, upToDateChecker);
  }

  /**
   * Hook called at the start of a build.
   */
  @ThreadHostile
  public void beforeBuild() {
  }

  /**
   * Hook called at the end of a build.
   */
  @ThreadHostile
  public void afterBuild() {
  }

  @Override
  public boolean artifactExists(Artifact artifact) {
    return !(getArtifactFileStatus(artifact) instanceof FileStatusException);
  }

  @Override
  public boolean isRegularFile(Artifact artifact) {
    return getArtifactFileStatus(artifact).isFile();
  }

  @Override
  public boolean isInjected(Artifact artifact) throws IOException {
    FileStatus stat = getArtifactFileStatus(artifact);
    if (stat instanceof FileStatusException) {
      throw (FileStatusException) stat;
    }
    return stat instanceof InjectedStat;
  }

  /**
   * Retrieve the artifact's size as a file, returning a cached result when applicable.
   *
   * @return the size of the artifact, in bytes
   * @throws IOException if the file does not exist or there was an error retrieving the size.
   */
  public long getSize(Artifact artifact) throws IOException {
    FileStatus stat = getArtifactFileStatus(artifact);
    if (stat instanceof FileStatusException) {
      throw (FileStatusException) stat;
    }
    return stat.getSize();
  }

  /**
   * Caches all metadata (both FileStatus and Metadata) for given artifats.
   */
  public void cacheMetadata(Iterable<Artifact> artifactList) {
    for (Artifact artifact : artifactList) {
      getMetadataMaybe(artifact);
    }
  }

  /**
   * Called before retrieving a sequence of artifact metadata.
   * This method may or may not be a no-op.
   *
   * @param inputs artifacts for which we'll need metadata.
   */
  public void beforeRetrieval(Iterable<Artifact> inputs) {
  }

  @Override
  public void setDigestForVirtualArtifact(Artifact artifact, Digest digest) {
    Preconditions.checkArgument(isVirtualArtifact(artifact));
    Preconditions.checkNotNull(digest);
    fileStatusMap.put(artifact.getPath(), new VirtualArtifactFileStatus(digest));
  }

  @Override
  public void discardMetadata(Collection<Artifact> artifactList) {
    for (Artifact artifact : artifactList) {
      fileStatusMap.remove(artifact.getPath());
      // Try to delete the file from the metadata cache. We do this unconditionally, even if there
      // was no cached status for it. The metadata cache is more persistent than the
      // ArtifactMetadataCache, and so unconditionally discarding prevents the metadata cache from
      // keeping stale data.
      discardEntryFromMetadataCache(artifact);
    }
  }

  /**
   * Returns metadata for the given artifact or throws an exception if the
   * metadata could not be obtained.
   *
   * @return metadata instance
   * @param forceDigest if true, make sure that the digest is computed accurately.
   *
   * @throws IOException if metadata could not be obtained.
   */
  private Metadata getMetadata(Artifact artifact, boolean forceDigest) throws IOException {
    FileStatus status = getArtifactFileStatus(artifact);
    if (status instanceof FileStatusException) {
      throw (FileStatusException) status;
    } else if (status instanceof VirtualArtifactFileStatus) {
      Preconditions.checkState(!forceDigest, "Don't force digest on virtual artifacts");
      return ((VirtualArtifactFileStatus) status).metadata;
    }
    try {
      return metadataCache.getOrFail(artifact, status, forceDigest);
    } catch (IOException e) {
      // Something went wrong since we were able to access file before. E.g. we failed to
      // calculate digest because file has no read permissions. Or maybe file was deleted during
      // the build. Anyway, update cached status with new exception and proceed accordingly.
      discardEntryFromMetadataCache(artifact);
      fileStatusMap.put(artifact.getPath(), new FileStatusException(e));
      throw e;
    }
  }

  /**
   * Returns digest for the given artifact or null if the digest could not be obtained due to an
   * IOException accessing the file.
   */
  @Nullable
  public byte[] getDigestMaybe(Artifact artifact) {
    try {
      return getMetadata(artifact, /*forceDigest=*/true).digest;
    } catch (IOException e) {
      return null;
    }
  }

  @Override
  public Metadata getMetadata(Artifact artifact) throws IOException {
    return getMetadata(artifact, false);
  }

  @Override
  public Metadata getMetadataMaybe(Artifact artifact) {
    try {
      return getMetadata(artifact, /*forceDigest=*/false);
    } catch (IOException e) {
      return null;
    }
  }

  @Override
  public void injectDigest(ActionInput output, FileStatus stat, byte[] digest) {
    if (!(output instanceof Artifact)) {
      // Storing a non-artifact's data is useless -- no caller will ever see it.
      return;
    }
    Path path = ((Artifact) output).getPath();
    try {
      fileStatusMap.put(path, stat);
      metadataCache.injectDigest(path, stat, digest);
    } catch (IOException e) {
      // Do nothing - we just failed to inject metadata. Real error handling
      // will be done later, when somebody will try to access that file.
    }
  }

  /**
   * Returns artifact file status. File status can be of class FileStatusException
   * in which case it represents an IOException that occurred while obtaining
   * file stat. Once artifact file status is requested, it is permanently cached
   * for the lifetime of the cache instance, unless it is manually discarded.
   */
  private FileStatus getArtifactFileStatus(Artifact artifact) {
    Path path = artifact.getPath();
    FileStatus status = fileStatusMap.get(path);
    if (status == null) {
      if (isVirtualArtifact(artifact)) {
        status = NO_METADATA_STATUS;
      } else {
        try {
          status = path.stat();
        } catch (IOException e) {
          // Overwrite status in cache, storing an exception that occurred during stat() call.
          status = new FileStatusException(e);
          fileStatusMap.put(path, status);
          return status;
        }
      }
      FileStatus oldStatus = fileStatusMap.putIfAbsent(path, status);
      return oldStatus == null ? status : oldStatus;
    } else {
      return status;
    }
  }

  /**
   * Returns the dirty subset of the given artifact collection.
   *
   * <p>These are the files that have changed since the last successful build. The set includes
   * artifacts whose cached file status is obsolete, as well as those that were already dirty in the
   * last build but we failed to build them. See {@link #changedArtifacts}.
   *
   * <p>Prerequisite: this method only returns relevant results if the new build using the data is
   * identical in all but the mtimes of some of the artifacts. It also does not return any useful
   * data if it is run during the build, as its intention is to give a snapshot of which Artifacts
   * have changed since the previous build, and the changing metadata because of an ongoing build
   * can cause inaccurate results.
   *
   * <p>This method uses file system awareness to discover files that haven't changed to reduce the
   * number of files that need to be explicitly checked.
   *
   * <p>Note that this method will not be accurate if the artifact's metadata is updated in some
   * other way before this method is run. Most notably, if there was no metadata entry for a certain
   * artifact, and {@link #getMetadata(Artifact) getMetadata} is called on it before
   * {@link #changedArtifacts}, it will not show up as changed.
   *
   * <p>This method is PERFORMANCE CRITICAL.
   *
   * @param artifacts set of artifacts to be checked for changes.
   * @param modified the known modified source files. if unknown, null.
   * @param artifactsKnownBad set of any artifacts that are known to be changed, and so, to save
   *        time, should not be checked for changes.
   * @return immutable copy of all changed artifacts, including artifactsKnownBad.
   */
  @ThreadCompatible // concurrent runs can create race conditions since this method
                    // modifies its own input data (fileStatusMap).
                    // This method may also spawn threads of its own for
                    // performance reasons.
  public Set<Artifact> updateCache(Collection<Artifact> artifacts,
      @Nullable ModifiedFileSet modified, Set<Artifact> artifactsKnownBad)
      throws InterruptedException {
    ImmutableSet<PathFragment> modifiedFiles = null;
    if (modified != null && !modified.treatEverythingAsModified()) {
      // Say file "s" is a symlink to "f". If we're alerted here that "f" is modified, we still need
      // to trigger a rebuild for Artifacts corresponding to "s".
      modifiedFiles = ImmutableSet.copyOf(modified.modifiedSourceFiles());
    }

    // CPU-bound (usually) stat() calls, plus a fudge factor. Since we are in an "incremental" case,
    // it's likely that the kernel and the filesystem hold in-memory references to the unchanged
    // stat() values.
    final int numOutputJobs = Runtime.getRuntime().availableProcessors() * 2;
    final int numInputJobs = numOutputJobs;
    final AtomicInteger checksum = new AtomicInteger();
    final ExecutorService executor = Executors.newFixedThreadPool(numInputJobs + numOutputJobs,
        new ThreadFactoryBuilder().setNameFormat("Update ArtifactMetadataCache %d").build());

    // From the given set of artifacts, we will construct two list-of-lists
    // which will be used to shard the update job. Size expectations will be
    // overestimates to avoid copying.
    final int numArtifacts = artifacts.size();
    final Sharder<Artifact> inputShards = new Sharder<>(numInputJobs, numArtifacts);
    final Sharder<Artifact> outputShards = new Sharder<>(numOutputJobs, numArtifacts);

    int skippedFileCount = 0;  // Used to help make sure we stat the correct number of files.
    for (Artifact artifact : artifacts) {
      // TODO(bazel-team): There are three basic "artifact buckets" we care about: immutable inputs,
      // mutable inputs, and outputs. The current implementation incorrectly identifies certain
      // immutable inputs as mutable. This weakens the benefits of optimizations like batch
      // statting. Fix these up when possible.

      // Add this artifact to one of the check lists unless it's guaranteed not
      // to have changed due to some pre-checked property.
      if (!artifactMayHaveChanged(artifact, modifiedFiles) || isVirtualArtifact(artifact) ||
          artifactsKnownBad.contains(artifact)) {
        skippedFileCount++;
      } else {
        if (artifact.isSourceArtifact()) {
          inputShards.add(artifact);
        } else {
          outputShards.add(artifact);
        }
      }
    }

    final ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("ArtifactMetadataCache#updateCache");
    // The first set of tasks is the sharded set of output files. When possible, we use batch
    // statting to improve performance.
    if (statter != null) {
      for (List<Artifact> shard : outputShards) {
        executor.submit(wrapper.wrap(batchStatJob(changedArtifacts, shard, checksum)));
      }
    } else {
      for (List<Artifact> shard : outputShards) {
        // No batch stat: Update the same way we did with the input files.
        executor.submit(wrapper.wrap(shardedUpdateJob(changedArtifacts, shard, checksum)));
      }
    }

    // The second set of tasks is the sharded set of input files.
    for (List<Artifact> shard : inputShards) {
      executor.submit(wrapper.wrap(shardedUpdateJob(changedArtifacts, shard, checksum)));
    }
    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      throw new InterruptedException();
    }

    // Important invariant: We updated the right amount of artifacts.
    Preconditions.checkState(checksum.get() == artifacts.size() - skippedFileCount);
    return ImmutableSet.<Artifact>builder().addAll(artifactsKnownBad)
        .addAll(changedArtifacts).build();
  }

  private boolean artifactMayHaveChanged(Artifact artifact,
      @Nullable ImmutableSet<PathFragment> modifiedFiles) {
    if (!artifact.isSourceArtifact()) {
      // We have no way to guarantee that output files haven't changed, because they come from a
      // mutable filesystem.
      return true;
    }

    // Source files can come from smart filesystems, in which case we may be able to guarantee that
    // they haven't changed.
    if (modifiedFiles != null) {
      return modifiedFiles.contains(artifact.getExecPath());
    } else {
      // This check does *not* account for symlinks. If this artifact is a symlink or one of the
      // package-level directories in its path is a symlink, a false return implies only that the
      // link hasn't changed. It *doesn't* imply that the underlying target hasn't changed.
      // Artifacts without owners (those that don't map to to package-contained input files) are
      // always assumed to have potentially changed.
      return artifact.getOwner() == null ||
          packageUpToDateChecker.loadedTargetMayHaveChanged(artifact.getOwner());
    }
  }

  // Note: Please use array lists for shardedOuts, as we will make use of
  // random lookups.
  private Runnable batchStatJob(final Set<Artifact> changedArtifacts,
      final List<Artifact> shardedOuts, final AtomicInteger checksum) {
    return new Runnable() {
      @Override
      public void run() {
        Iterable<PathFragment> outputPaths = Iterables.transform(shardedOuts,
            new Function<Artifact, PathFragment>() {
              @Override
              public PathFragment apply(Artifact artifact) {
                return artifact.getExecPath();
              }
            });
        try {
          List<FileStatusWithDigest> stats =
              statter.batchStat(/*includeDigest=*/false, /*includeLinks=*/false, outputPaths);
          Preconditions.checkState(stats.size() == shardedOuts.size());
          for (int i = 0; i < stats.size(); i++) {
            updateArtifact(changedArtifacts, shardedOuts.get(i), stats.get(i), checksum);
          }
        } catch (IOException e) {
          // Batch stat() did not work. Log an exception and fall back on system calls.
          LoggingUtil.logToRemote(Level.WARNING, "Unable to process batch stat", e);
          shardedUpdateJob(changedArtifacts, shardedOuts, checksum).run();
        } catch (InterruptedException e) {
          // Fall-through: we handle interrupt in the main thread.
        }
      }
    };
  }

  private Runnable shardedUpdateJob(final Set<Artifact> changedArtifacts,
      final List<Artifact> inputArtifacts, final AtomicInteger counter) {
    return new Runnable() {
      @Override
      public void run() {
        for (Artifact inputArtifact : inputArtifacts) {
          try {
            updateArtifact(changedArtifacts, inputArtifact, counter);
          } catch (InterruptedException e) {
            // Fall-through: we handle interrupt in the main thread.
          }
        }
      }
    };
  }

  private void updateArtifact(Set<Artifact> changedArtifacts, Artifact artifact,
                              AtomicInteger counter)
      throws InterruptedException {
    updateArtifact(changedArtifacts, artifact, null, counter);
  }

  private void updateArtifact(Set<Artifact> changedArtifacts, Artifact artifact,
      @Nullable FileStatus newStat, AtomicInteger counter) throws InterruptedException {
    counter.incrementAndGet();

    // Ignore virtual artifacts since they do not have normal mtimes, and will be marked
    // for re-building anyway if any of their dependencies need to be rebuilt.
    if (isVirtualArtifact(artifact)) {
      return;
    }

    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException();
    }

    Path path = artifact.getPath();
    FileStatus oldStatus = fileStatusMap.get(path);
    // Incremental builder now may pass in artifacts which were never processed before, artifacts
    // that were inputs to actions that never ran. Thus, a null oldStatus is acceptable.

    boolean existsInLastBuild = (oldStatus != null);
    long oldMTime = -1;
    if (existsInLastBuild) {
      try {
        oldMTime = oldStatus.getLastModifiedTime();
      } catch (IOException e) {
        // Note that only non-mandatory inputs are allowed to be missing if they were in the
        // last build. This is checked in AbstractBuilder#executeActionIfNeeded().
        existsInLastBuild = false;
      }
    }

    try {
      FileStatus statStatus = newStat != null ? newStat : path.stat();
      long mtime = statStatus.getLastModifiedTime();
      if (artifact.isSourceArtifact()) {
        granularityMonitor.notifyDependenceOnFileTime(mtime);
      }
      if (!existsInLastBuild || oldMTime != mtime) {
        changedArtifacts.add(artifact);
        fileStatusMap.put(path, statStatus);
      }
    } catch (IOException e) {
      if (existsInLastBuild) {
        // For files under the "include" symlink, it is not sound to cache
        // nonexistence throughout the entire build, since they may be "rediscovered" during
        // execution when the symlink is recreated.
        fileStatusMap.remove(path);

        // Add the failed artifact to the changed list anyway. If this was because
        // of a missing file, it may be able to be generated in the build.
        changedArtifacts.add(artifact);
      }
    }
  }

  /**
   * Clear set of changedArtifacts detected this run. Called when state has been updated
   * due to this set, so it is no longer needed to recover state if Blaze is interrupted.
   */
  public void clearChangedArtifacts() {
    changedArtifacts.clear();
  }

  protected void discardEntryFromMetadataCache(Artifact artifact) {
    if (!isVirtualArtifact(artifact)){
      metadataCache.discardCacheEntry(artifact);
    }
  }

  /**
   * Returns true if artifact represents a virtual artifact. Such artifacts
   * do not exist and by default use Metadata.NO_METADATA as their
   * metadata.
   *
   * <p>At this moment the only virtual artifacts in the system are middlemen.
   */
  private static boolean isVirtualArtifact(Artifact artifact) {
    return artifact.isMiddlemanArtifact();
  }

  public MetadataCache getMetadataCache() {
    return metadataCache;
  }
}
