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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.io.IOException;
import java.util.Collection;

/**
 * A per-build cache of filesystem operations for Skyframe invocations of legacy package loading.
 */
public class PerBuildSyscallCache implements UnixGlob.FilesystemCalls {

  private final LoadingCache<Pair<Path, Symlinks>, Object> statCache;

  /* Caches the result of readdir(<path>, Symlinks.NOFOLLOW) calls. */
  private final LoadingCache<Path, Object> readdirCache;

  private static final FileStatus NO_STATUS = new FakeFileStatus();

  private PerBuildSyscallCache(
      LoadingCache<Pair<Path, Symlinks>, Object> statCache,
      LoadingCache<Path, Object> readdirCache) {
    this.statCache = statCache;
    this.readdirCache = readdirCache;
  }

  public static Builder newBuilder() {
    return new Builder();
  }

  /** Builder for a per-build filesystem cache. */
  public static class Builder {
    private static final int UNSET = -1;
    private int maxStats = UNSET;
    private int maxReaddirs = UNSET;
    private int concurrencyLevel = UNSET;

    private Builder() {
    }

    /** Sets the upper bound of the 'stat' cache. This cache is unbounded by default. */
    public Builder setMaxStats(int maxStats) {
      this.maxStats = maxStats;
      return this;
    }

    /** Sets the upper bound of the 'readdir' cache. This cache is unbounded by default. */
    public Builder setMaxReaddirs(int maxReaddirs) {
      this.maxReaddirs = maxReaddirs;
      return this;
    }

    /** Sets the concurrency level of the caches. */
    public Builder setConcurrencyLevel(int concurrencyLevel) {
      this.concurrencyLevel = concurrencyLevel;
      return this;
    }

    public PerBuildSyscallCache build() {
      CacheBuilder<Object, Object> statCacheBuilder = CacheBuilder.newBuilder();
      if (maxStats != UNSET) {
        statCacheBuilder = statCacheBuilder.maximumSize(maxStats);
      }
      CacheBuilder<Object, Object> readdirCacheBuilder = CacheBuilder.newBuilder();
      if (maxReaddirs != UNSET) {
        readdirCacheBuilder = readdirCacheBuilder.maximumSize(maxReaddirs);
      }
      if (concurrencyLevel != UNSET) {
        statCacheBuilder = statCacheBuilder.concurrencyLevel(concurrencyLevel);
        readdirCacheBuilder = readdirCacheBuilder.concurrencyLevel(concurrencyLevel);
      }
      return new PerBuildSyscallCache(statCacheBuilder.build(newStatLoader()),
          readdirCacheBuilder.build(newReaddirLoader()));
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public Collection<Dirent> readdir(Path path) throws IOException {
    Object result = readdirCache.getUnchecked(path);
    if (result instanceof IOException) {
      throw (IOException) result;
    }
    return (Collection<Dirent>) result; // unchecked cast
  }

  @Override
  public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
    // Try to load a Symlinks.NOFOLLOW result first. Symlinks are rare and this enables sharing the
    // cache for all non-symlink paths.
    Object result = statCache.getUnchecked(Pair.of(path, Symlinks.NOFOLLOW));
    if (result instanceof IOException) {
      throw (IOException) result;
    }
    FileStatus status = (FileStatus) result;
    if (status != NO_STATUS && symlinks == Symlinks.FOLLOW && status.isSymbolicLink()) {
      result = statCache.getUnchecked(Pair.of(path, Symlinks.FOLLOW));
      if (result instanceof IOException) {
        throw (IOException) result;
      }
      status = (FileStatus) result;
    }
    return (status == NO_STATUS) ? null : status;
  }

  @Override
  @SuppressWarnings("unchecked")
  public Dirent.Type getType(Path path, Symlinks symlinks) throws IOException {
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
        return UnixGlob.statusToDirentType((FileStatus) result);
      }
    }

    // If this is a root directory, we must stat, there is no parent.
    Path parent = path.getParentDirectory();
    if (parent == null) {
      return UnixGlob.statusToDirentType(statIfFound(path, symlinks));
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
      for (Dirent dirent : (Collection<Dirent>) result) { // unchecked cast
        // TODO(djasper): Dealing with filesystem case is a bit of a code smell. Figure out a better
        // way to store Dirents, e.g. with names normalized.
        if (path.getFileSystem().isFilePathCaseSensitive()
            && !dirent.getName().equals(path.getBaseName())) {
          continue;
        }
        if (!path.getFileSystem().isFilePathCaseSensitive()
            && !dirent.getName().equalsIgnoreCase(path.getBaseName())) {
          continue;
        }
        if (dirent.getType() == Dirent.Type.SYMLINK && symlinks == Symlinks.FOLLOW) {
          // See above: We don't want to follow symlinks with readdir(). Do a stat() instead.
          return UnixGlob.statusToDirentType(statIfFound(path, Symlinks.FOLLOW));
        }
        return dirent.getType();
      }
      return null;
    }

    return UnixGlob.statusToDirentType(statIfFound(path, symlinks));
  }

  public void clear() {
    statCache.invalidateAll();
    readdirCache.invalidateAll();
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

  /**
   * A {@link CacheLoader} for a cache of stat calls. Input: (path, following_symlinks) Output:
   * FileStatus
   */
  private static CacheLoader<Pair<Path, Symlinks>, Object> newStatLoader() {
    return new CacheLoader<Pair<Path, Symlinks>, Object>() {
      @Override
      public Object load(Pair<Path, Symlinks> p) {
        try {
          FileStatus f = p.first.statIfFound(p.second);
          return (f == null) ? NO_STATUS : f;
        } catch (IOException e) {
          return e;
        }
      }
    };
  }

  /**
   * A {@link CacheLoader} for a cache of readdir calls. Input: path Output: Either Dirents or
   * IOException.
   */
  private static CacheLoader<Path, Object> newReaddirLoader() {
    return new CacheLoader<Path, Object>() {
      @Override
      public Object load(Path p) {
        try {
          // TODO(bazel-team): Consider storing the Collection of Dirent values more compactly
          // by reusing DirectoryEntryListingStateValue#CompactSortedDirents.
          return p.readdir(Symlinks.NOFOLLOW);
        } catch (IOException e) {
          return e;
        }
      }
    };
  }
}
