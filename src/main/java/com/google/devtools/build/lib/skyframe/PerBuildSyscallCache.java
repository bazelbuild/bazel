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
class PerBuildSyscallCache implements UnixGlob.FilesystemCalls {

  private final LoadingCache<Pair<Path, Symlinks>, FileStatus> statCache =
      newStatMap();
  private final LoadingCache<Pair<Path, Symlinks>, Pair<Collection<Dirent>, IOException>>
      readdirCache = newReaddirMap();

  private static final FileStatus NO_STATUS = new FakeFileStatus();

  @Override
  public Collection<Dirent> readdir(Path path, Symlinks symlinks) throws IOException {
    Pair<Collection<Dirent>, IOException> result =
        readdirCache.getUnchecked(Pair.of(path, symlinks));
    Collection<Dirent> entries = result.getFirst();
    if (entries != null) {
      return entries;
    }
    throw result.getSecond();
  }

  @Override
  public FileStatus statNullable(Path path, Symlinks symlinks) {
    FileStatus status = statCache.getUnchecked(Pair.of(path, symlinks));
    return (status == NO_STATUS) ? null : status;
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
    public boolean isSymbolicLink() {
      throw new UnsupportedOperationException();
    }
  }

  /**
   * A cache of stat calls.
   * Input: (path, following_symlinks)
   * Output: FileStatus
   */
  private static LoadingCache<Pair<Path, Symlinks>, FileStatus> newStatMap() {
    return CacheBuilder.newBuilder().build(
        new CacheLoader<Pair<Path, Symlinks>, FileStatus>() {
          @Override
          public FileStatus load(Pair<Path, Symlinks> p) {
            FileStatus f = p.first.statNullable(p.second);
            return (f == null) ? NO_STATUS : f;
          }
        });
  }

  /**
   * A cache of readdir calls.
   * Input: (path, following_symlinks)
   * Output: A union of (Dirents, IOException).
   */
  private static
  LoadingCache<Pair<Path, Symlinks>, Pair<Collection<Dirent>, IOException>> newReaddirMap() {
    return CacheBuilder.newBuilder().build(
        new CacheLoader<Pair<Path, Symlinks>, Pair<Collection<Dirent>, IOException>>() {
          @Override
          public Pair<Collection<Dirent>, IOException> load(Pair<Path, Symlinks> p) {
            try {
              return Pair.of(p.first.readdir(p.second), null);
            } catch (IOException e) {
              return Pair.of(null, e);
            }
          }
        });
  }
}
