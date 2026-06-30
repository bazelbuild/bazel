// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A remote cache for the contents of external repositories. */
public interface RemoteRepoContentsCache {
  /**
   * An entry memoized in a {@link LookupState}. The concrete representation is an implementation
   * detail of the cache, so this is an opaque marker to callers.
   */
  interface OpaqueCacheEntry {}

  /**
   * Per-repository scratch state for {@link #lookupCache} that is meant to be stored in the
   * repository's {@link SkyFunction.Environment.SkyKeyComputeState} so that it survives Skyframe
   * restarts.
   */
  final class LookupState<T extends OpaqueCacheEntry> {
    private final Map<String, T> entries = new ConcurrentHashMap<>();

    /** Returns the entry memoized under the given input hash, or null if there is none. */
    public T get(String inputHash) {
      return entries.get(inputHash);
    }

    /** Memoizes the given entry under the given input hash. */
    public T memoize(String inputHash, T entry) {
      entries.put(inputHash, entry);
      return entry;
    }
  }

  /** Adds a repository that has been fetched locally to the remote cache. */
  void addToCache(
      RepositoryName repoName,
      Path fetchedRepoDir,
      Path fetchedRepoMarkerFile,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws InterruptedException;

  /**
   * Retrieves a repository from the remote cache if possible.
   *
   * <p>Callers have to check {@code env.valuesMissing()} after this method returns.
   *
   * @param lookupState scratch state surviving Skyframe restarts; pass the same instance across
   *     restarts of the repository's evaluation to avoid re-fetching action cache entries
   * @return true if there was a cache hit and the repository has been fetched into the given
   *     directory.
   */
  boolean lookupCache(
      RepositoryName repoName,
      Path repoDir,
      String predeclaredInputHash,
      SkyFunction.Environment env,
      LookupState<?> lookupState)
      throws IOException, InterruptedException;
}
