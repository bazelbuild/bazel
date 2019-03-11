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
  private final LoadingCache<Pair<Path, Symlinks>, Object> readdirCache;

  private static final FileStatus NO_STATUS = new FakeFileStatus();

  private PerBuildSyscallCache(
      LoadingCache<Pair<Path, Symlinks>, Object> statCache,
      LoadingCache<Pair<Path, Symlinks>, Object> readdirCache) {
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
  public Collection<Dirent> readdir(Path path, Symlinks symlinks) throws IOException {
    Object result = readdirCache.getUnchecked(Pair.of(path, symlinks));
    if (result instanceof IOException) {
      throw (IOException) result;
    }
    return (Collection<Dirent>) result;
  }

  @Override
  public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
    Object result = statCache.getUnchecked(Pair.of(path, symlinks));
    if (result instanceof IOException) {
      throw (IOException) result;
    }
    FileStatus status = (FileStatus) result;
    return (status == NO_STATUS) ? null : status;
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
   * A {@link CacheLoader} for a cache of readdir calls. Input: (path, following_symlinks) Output: A
   * union of (Dirents, IOException).
   */
  private static CacheLoader<Pair<Path, Symlinks>, Object> newReaddirLoader() {
    return new CacheLoader<Pair<Path, Symlinks>, Object>() {
      @Override
      public Object load(Pair<Path, Symlinks> p) {
        try {
          // TODO(bazel-team): Consider storing the Collection of Dirent values more compactly
          // by reusing DirectoryEntryListingStateValue#CompactSortedDirents.
          return p.first.readdir(p.second);
        } catch (IOException e) {
          return e;
        }
      }
    };
  }
}
