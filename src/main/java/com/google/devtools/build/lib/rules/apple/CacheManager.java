// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import javax.annotation.Nullable;

/**
 * General cache file manager for mapping one or more keys to host-related path information.
 *
 * <p> Cache management has some notable restrictions:
 * <ul>
 * <li>Each cache entry must have the same number of (string) keys, and one value.</li>
 * <li>An entry, once written to the cache, must be stable between builds. Clearing the cache
 *     requires a full clean of the bazel output directory.</li>
 * </ul>
 *
 * <p> Note that a single cache manager instance is not thread-safe, though multiple threads may
 * hold cache manager instances for the same cache file. As a result, it is possible multiple
 * threads may write the same entry to cache. This is fine, as retrieval from the cache will simply
 * return the first found entry.
 */
class CacheManager {

  private final Path cacheFilePath;
  private boolean cacheFileTouched;

  /**
   * @param outputRoot path to the bazel's output root
   * @param cacheFilename name of the cache file
   */
  CacheManager(Path outputRoot, String cacheFilename) {
    cacheFilePath = outputRoot.getRelative(cacheFilename);
  }
  
  private void touchCacheFile() throws IOException {
    if (!cacheFileTouched) {
      FileSystemUtils.touchFile(cacheFilePath);
      cacheFileTouched = true;
    }
  }
  
  /**
   * Returns the value associated with the given list of string keys from the cache,
   * or null if the entry is not present in the cache. If there is more than one value for the
   * given key, the first value is returned.
   */
  @Nullable
  public String getValue(String... keys) throws IOException {
    Preconditions.checkArgument(keys.length > 0);
    touchCacheFile();

    List<String> keyList = ImmutableList.copyOf(keys);
    Iterable<String> cacheContents =
        FileSystemUtils.readLines(cacheFilePath, StandardCharsets.UTF_8);
    for (String cacheLine : cacheContents) {
      if (cacheLine.isEmpty()) {
        continue;
      }
      List<String> cacheEntry = Splitter.on(':').splitToList(cacheLine);
      List<String> cacheKeys = cacheEntry.subList(0, cacheEntry.size() - 1);
      String cacheValue = cacheEntry.get(cacheEntry.size() - 1);
      if (keyList.size() != cacheKeys.size()) {
        throw new IllegalStateException(
            String.format("cache file %s is malformed. Expected %s keys. line: '%s'",
                cacheFilePath, keyList.size(), cacheLine));
      }
      if (keyList.equals(cacheKeys)) {
        return cacheValue;
      }
    }
    return null;
  }

  /**
   * Write an entry to the cache. An entry consists of one or more keys and a single value.
   * No validation is made regarding whether there are redundant or conflicting entries in the
   * cache; it is thus the responsibility of the caller to ensure that any redundant entries
   * (entries which have the same keys) also have the same value.
   */
  public void writeEntry(List<String> keys, String value) throws IOException {
    Preconditions.checkArgument(!keys.isEmpty());

    touchCacheFile();
    FileSystemUtils.appendLinesAs(cacheFilePath, StandardCharsets.UTF_8,
        Joiner.on(":").join(ImmutableList.builder().addAll(keys).add(value).build()));
  }
}
