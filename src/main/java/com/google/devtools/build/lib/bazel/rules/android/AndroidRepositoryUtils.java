// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.android;

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.Dirents;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Helper methods for {@code AndroidSdkRepositoryFunction} and {@code AndroidNdkRepositoryFunction}.
 */
final class AndroidRepositoryUtils {
  private static final Pattern PLATFORMS_API_LEVEL_PATTERN = Pattern.compile("android-(\\d+)");

  /** Gets a DirectoryListingValue for {@code dirPath} or returns null. */
  static DirectoryListingValue getDirectoryListing(Path root, PathFragment dirPath, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    try {
      return (DirectoryListingValue)
          env.getValueOrThrow(
              DirectoryListingValue.key(RootedPath.toRootedPath(root, dirPath)),
              InconsistentFilesystemException.class);
    } catch (InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }
  }

  /**
   * Gets the numeric api levels from the contents of the platforms directory in descending order.
   *
   * Note that the directory entries are assumed to match {@code android-[0-9]+}. Any directory
   * entries that are not directories or do not match that pattern are ignored.
   */
  static ImmutableSortedSet<Integer> getApiLevels(Dirents platformsDirectories) {
    ImmutableSortedSet.Builder<Integer> apiLevels = ImmutableSortedSet.reverseOrder();
    for (Dirent platformDirectory : platformsDirectories) {
      if (platformDirectory.getType() != Dirent.Type.DIRECTORY) {
        continue;
      }
      Matcher matcher = PLATFORMS_API_LEVEL_PATTERN.matcher(platformDirectory.getName());
      if (matcher.matches()) {
        apiLevels.add(Integer.parseInt(matcher.group(1)));
      }
    }
    return apiLevels.build();
  }
}
