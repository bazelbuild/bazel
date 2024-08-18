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

import static com.google.common.base.MoreObjects.firstNonNull;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.devtools.build.lib.util.LatestObjectMetricExporter;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.Collection;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A basic implementation of {@link SyscallCache} that caches stat and readdir operations, used if
 * no custom cache is set in {@link
 * com.google.devtools.build.lib.runtime.WorkspaceBuilder#setSyscallCache}.
 *
 * <p>Allows non-Skyframe operations (like non-Skyframe globbing) to share a filesystem cache with
 * Skyframe operations, and may be able to answer questions (like the type of a file) based on
 * existing data (like the directory listing of a parent) without filesystem access.
 */
public final class DefaultSyscallCache implements SyscallCache {

  private final Supplier<LoadingCache<Pair<Path, Symlinks>, Object>> statCacheSupplier;
  private final Supplier<LoadingCache<Path, Object>> readdirCacheSupplier;

  @Nullable private final LatestObjectMetricExporter<Cache<?, ?>> statCacheMetricExporter;

  @Nullable private final LatestObjectMetricExporter<Cache<?, ?>> readdirCacheMetricExporter;

  private LoadingCache<Pair<Path, Symlinks>, Object> statCache;

  /* Caches the result of readdir(<path>, Symlinks.NOFOLLOW) calls. */
  private LoadingCache<Path, Object> readdirCache;

  private static final FileStatus NO_STATUS = new FakeFileStatus();

  private DefaultSyscallCache(
      Supplier<LoadingCache<Pair<Path, Symlinks>, Object>> statCacheSupplier,
      Supplier<LoadingCache<Path, Object>> readdirCacheSupplier,
      @Nullable LatestObjectMetricExporter<Cache<?, ?>> statCacheMetricExporter,
      @Nullable LatestObjectMetricExporter<Cache<?, ?>> readdirCacheMetricExporter) {
    this.statCacheSupplier = statCacheSupplier;
    this.readdirCacheSupplier = readdirCacheSupplier;
    this.statCacheMetricExporter = statCacheMetricExporter;
    this.readdirCacheMetricExporter = readdirCacheMetricExporter;
    clear();
  }

  public static Builder newBuilder() {
    return new Builder();
  }

  /** Builder for a per-build filesystem cache. */
  public static final class Builder {
    private static final int UNSET = -1;
    private int maxStats = UNSET;
    private int maxReaddirs = UNSET;
    private int initialCapacity = UNSET;
    private LatestObjectMetricExporter<Cache<?, ?>> statCacheMetricExporter = null;
    private LatestObjectMetricExporter<Cache<?, ?>> readdirCacheMetricExporter = null;

    private Builder() {}

    /** Sets the upper bound of the 'stat' cache. This cache is unbounded by default. */
    @CanIgnoreReturnValue
    public Builder setMaxStats(int maxStats) {
      this.maxStats = maxStats;
      return this;
    }

    /** Sets the upper bound of the 'readdir' cache. This cache is unbounded by default. */
    @CanIgnoreReturnValue
    public Builder setMaxReaddirs(int maxReaddirs) {
      this.maxReaddirs = maxReaddirs;
      return this;
    }

    /** Sets the concurrency level of the caches. */
    @CanIgnoreReturnValue
    public Builder setInitialCapacity(int initialCapacity) {
      this.initialCapacity = initialCapacity;
      return this;
    }

    /**
     * Sets the metric exporter for the 'stat' cache.
     *
     * <p>No metrics are exported by default. If a non-null value is set, the 'stat' cache will
     * record access statistics with some overhead.
     */
    @CanIgnoreReturnValue
    public Builder setStatCacheMetricExporter(
        LatestObjectMetricExporter<Cache<?, ?>> statCacheMetricExporter) {
      this.statCacheMetricExporter = statCacheMetricExporter;
      return this;
    }

    /**
     * Sets the metric exporter for the 'readdir' cache.
     *
     * <p>No metrics are exported by default. If a non-null value is set, the 'readdir' cache will
     * record access statistics with some overhead.
     */
    @CanIgnoreReturnValue
    public Builder setReaddirCacheMetricExporter(
        LatestObjectMetricExporter<Cache<?, ?>> readdirCacheMetricExporter) {
      this.readdirCacheMetricExporter = readdirCacheMetricExporter;
      return this;
    }

    public DefaultSyscallCache build() {
      Caffeine<Object, Object> statCacheBuilder = Caffeine.newBuilder();
      if (maxStats != UNSET) {
        statCacheBuilder.maximumSize(maxStats);
      }
      if (statCacheMetricExporter != null) {
        statCacheBuilder.recordStats();
      }
      Caffeine<Object, Object> readdirCacheBuilder = Caffeine.newBuilder();
      if (maxReaddirs != UNSET) {
        readdirCacheBuilder.maximumSize(maxReaddirs);
      }
      if (readdirCacheMetricExporter != null) {
        readdirCacheBuilder.recordStats();
      }
      if (initialCapacity != UNSET) {
        statCacheBuilder.initialCapacity(initialCapacity);
        readdirCacheBuilder.initialCapacity(initialCapacity);
      }
      return new DefaultSyscallCache(
          () -> statCacheBuilder.build(DefaultSyscallCache::statImpl),
          () -> readdirCacheBuilder.build(DefaultSyscallCache::readdirImpl),
          statCacheMetricExporter,
          readdirCacheMetricExporter);
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public Collection<Dirent> readdir(Path path) throws IOException {
    Object result = readdirCache.get(path);
    if (result instanceof IOException ioException) {
      throw ioException;
    }
    return (Collection<Dirent>) result; // unchecked cast
  }

  @Nullable
  @Override
  public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
    // Try to load a Symlinks.NOFOLLOW result first. Symlinks are rare and this enables sharing the
    // cache for all non-symlink paths.
    Object result = statCache.get(Pair.of(path, Symlinks.NOFOLLOW));
    if (result instanceof IOException ioException) {
      throw ioException;
    }
    FileStatus status = (FileStatus) result;
    if (status != NO_STATUS && symlinks == Symlinks.FOLLOW && status.isSymbolicLink()) {
      result = statCache.get(Pair.of(path, Symlinks.FOLLOW));
      if (result instanceof IOException ioException) {
        throw ioException;
      }
      status = (FileStatus) result;
    }
    return (status == NO_STATUS) ? null : status;
  }

  @Nullable
  @Override
  @SuppressWarnings("unchecked")
  public DirentTypeWithSkip getType(Path path, Symlinks symlinks) throws IOException {
    // Use a cached stat call if we have one. This is done first so that we don't need to iterate
    // over a list of directory entries as we do for cached readdir() entries. We don't ever expect
    // to get a cache hit if symlinks == Symlinks.NOFOLLOW and so we don't bother to check.
    if (symlinks == Symlinks.FOLLOW) {
      Pair<Path, Symlinks> key = Pair.of(path, symlinks);
      Object result = statCache.getIfPresent(key);
      if (result != null && !(result instanceof IOException)) {
        if (result == NO_STATUS) {
          return null;
        }
        return ofStat((FileStatus) result);
      }
    }

    // If this is a root directory, we must stat, there is no parent.
    Path parent = path.getParentDirectory();
    if (parent == null) {
      return ofStat(statIfFound(path, symlinks));
    }

    // Answer based on a cached readdir() call if possible. The cache might already be populated
    // from Skyframe directory lising (DirectoryListingFunction) or by globbing via
    // {@link UnixGlob}. We generally try to avoid following symlinks in readdir() calls as in a
    // directory with many symlinks, these would be resolved basically using a stat anyway and they
    // would be resolved sequentially which can be slow on high-latency file systems. If we request
    // the type of a file with FOLLOW, and find a symlink in the directory, we fall back to doing a
    // stat.
    Object result = readdirCache.getIfPresent(parent);
    if (result != null && !(result instanceof IOException)) {
      String baseName = path.getBaseName();
      for (Dirent dirent : (Collection<Dirent>) result) { // unchecked cast
        // TODO(djasper): Dealing with filesystem case is a bit of a code smell. Figure out a better
        // way to store Dirents, e.g. with names normalized.
        if (path.getFileSystem().isFilePathCaseSensitive() && !dirent.getName().equals(baseName)) {
          continue;
        }
        if (!path.getFileSystem().isFilePathCaseSensitive()
            && !dirent.getName().equalsIgnoreCase(baseName)) {
          continue;
        }
        if (dirent.getType() == Dirent.Type.SYMLINK && symlinks == Symlinks.FOLLOW) {
          // See above: We don't want to follow symlinks with readdir(). Do a stat() instead.
          return ofStat(statIfFound(path, Symlinks.FOLLOW));
        }
        return DirentTypeWithSkip.of(dirent.getType());
      }
      return null;
    }

    return ofStat(statIfFound(path, symlinks));
  }

  @Nullable
  private static DirentTypeWithSkip ofStat(@Nullable FileStatus status) {
    return DirentTypeWithSkip.of(SyscallCache.statusToDirentType(status));
  }

  @Override
  public void clear() {
    // Drop not just the memory of the FileStatus objects but the maps themselves.
    statCache = statCacheSupplier.get();
    readdirCache = readdirCacheSupplier.get();
    if (statCacheMetricExporter != null) {
      statCacheMetricExporter.setLatestInstance(statCache);
    }
    if (readdirCacheMetricExporter != null) {
      readdirCacheMetricExporter.setLatestInstance(readdirCache);
    }
  }

  // This is used because the cache implementations don't allow null.
  private static final class FakeFileStatus implements FileStatus {
    @Override
    public long getLastChangeTime() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getNodeId() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getLastModifiedTime() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isDirectory() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isFile() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isSpecialFile() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isSymbolicLink() {
      throw new UnsupportedOperationException();
    }
  }

  /** Returns {@link FileStatus} or {@link IOException}. */
  private static Object statImpl(Pair<Path, Symlinks> p) {
    try {
      FileStatus stat = p.first.statIfFound(p.second);
      return firstNonNull(stat, NO_STATUS);
    } catch (IOException e) {
      return e;
    }
  }

  /** Returns a collection of {@link Dirent} or {@link IOException}. */
  private static Object readdirImpl(Path p) {
    try {
      // TODO(bazel-team): Consider storing the Collection of Dirent values more compactly by
      // reusing DirectoryEntryListingStateValue#CompactSortedDirents.
      return p.readdir(Symlinks.NOFOLLOW);
    } catch (IOException e) {
      return e;
    }
  }
}
