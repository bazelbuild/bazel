// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Splitter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Manages the xcrun cache file for mapping sdk identifiers to their root absolute paths. Cache
 * entries are read with {@link #getSdkRoot} and written with {@link #writeSdkRoot}.
 * 
 * <p> Note that multiple threads may hold instances of this class and be interacting with
 * the same cache file. As a result, it is possible multiple threads may write the same entry to
 * cache. This is fine, as {@link #getSdkRoot} will simply return the first found entry. The value
 * for any given key should be stable without a full clean of the bazel output directory (and
 * consequent removal of the cache file).
 */
class XcrunCacheManager {

  private final Path cachePath;
  private final String developerDir;

  /**
   * @param cachePath path to the cache file; this file must exist, but may be empty
   * @param developerDir the developer directory (absolute path) from which to obtain sdk root paths
   */
  XcrunCacheManager(Path cachePath, String developerDir) {
    this.cachePath = cachePath;
    this.developerDir = developerDir;
  }
  
  /**
   * Returns the SDKROOT associated with the given sdk string as its entry exists in the file cache,
   * or null if the entry is not present in the cache. If there is more than one value for the
   * given key, the first value is returned.
   *
   * @param sdkString platform concatenated with version number, as {@code xcrun} accepts
   *     as input, for example "iphoneos9.0"
   */
  @Nullable
  String getSdkRoot(String sdkString) throws IOException {
    Iterable<String> cacheContents = FileSystemUtils.readLines(cachePath, StandardCharsets.UTF_8);
    for (String cacheLine : cacheContents) {
      if (cacheLine.isEmpty()) {
        continue;
      }
      List<String> items = Splitter.on(':').splitToList(cacheLine);
      if (items.size() != 3) {
        throw new IllegalStateException(
            String.format("xcrun cache is malformed, line: '%s'", cacheLine));
      }
      String developerDir = items.get(0);
      String sdkIdentifier = items.get(1);
      String sdkRoot = items.get(2);
      if (sdkIdentifier.equals(sdkString) && developerDir.equals(this.developerDir)) {
        return sdkRoot;
      }
    }
    return null;
  }
  
  /**
   * Write an entry to the cache pairing the given sdk version string with the given SDKROOT.
   * No validation is made regarding whether there are redundant or conflicting entries in the
   * cache; it is thus the responsibility of the caller to ensure that no entry for the given
   * key is present in the cache, prior to calling this. (Entries added to the cache file with
   * the same key as a prior entry will not be used.)
   *
   * @param sdkString platform concatenated with version number, as {@code xcrun} accepts
   *     as input, for example "iphoneos9.0"
   * @param sdkRoot an absolute path to the SDKROOT for the sdk
   */
  void writeSdkRoot(String sdkString, String sdkRoot) throws IOException {
    FileSystemUtils.appendLinesAs(cachePath, StandardCharsets.UTF_8,
        String.format("%s:%s:%s", this.developerDir, sdkString, sdkRoot));
  }
}
