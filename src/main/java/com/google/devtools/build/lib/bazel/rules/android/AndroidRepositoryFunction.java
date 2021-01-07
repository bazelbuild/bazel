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
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.Dirents;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** This class contains the common logic between Android NDK and SDK repository functions. */
abstract class AndroidRepositoryFunction extends RepositoryFunction {
  private static final Pattern PLATFORMS_API_LEVEL_PATTERN = Pattern.compile("android-(\\d+)");

  /**
   * Android rules depend on the contents in the SDK and NDK existing in the correct locations. This
   * error is thrown when required files or directories are not found or believed to be tampered
   * with.
   */
  abstract void throwInvalidPathException(Path path, Exception e)
      throws RepositoryFunctionException;

  /**
   * Gets a {@link DirectoryListingValue} for {@code dirPath} or returns null.
   *
   * <p>First, we get a {@link FileValue} to check the {@code dirPath} exists and is a directory. If
   * not, we throw an exception.
   */
  final DirectoryListingValue getDirectoryListing(Path root, PathFragment dirPath, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    RootedPath rootedPath = RootedPath.toRootedPath(Root.fromPath(root), dirPath);
    try {
      FileValue dirFileValue =
          (FileValue) env.getValueOrThrow(FileValue.key(rootedPath), IOException.class);
      if (dirFileValue == null) {
        return null;
      }
      if (!dirFileValue.exists() || !dirFileValue.isDirectory()) {
        throwInvalidPathException(
            root,
            new IOException(
                String.format(
                    "Expected directory at %s but it is not a directory or it does not exist.",
                    rootedPath.asPath().getPathString())));
      }
      return (DirectoryListingValue)
          env.getValueOrThrow(
              DirectoryListingValue.key(RootedPath.toRootedPath(Root.fromPath(root), dirPath)),
              InconsistentFilesystemException.class);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }
  }

  /**
   * Gets the numeric api levels from the contents of the platforms directory in descending order.
   *
   * <p>Note that the directory entries are assumed to match {@code android-[0-9]+}. Any directory
   * entries that are not directories or do not match that pattern are ignored.
   */
  static final ImmutableSortedSet<Integer> getApiLevels(Dirents platformsDirectories) {
    ImmutableSortedSet.Builder<Integer> apiLevels = ImmutableSortedSet.reverseOrder();
    for (Dirent platformDirectory : platformsDirectories) {
      if (platformDirectory.getType() == Dirent.Type.DIRECTORY
          || platformDirectory.getType() == Dirent.Type.SYMLINK) {
        Matcher matcher = PLATFORMS_API_LEVEL_PATTERN.matcher(platformDirectory.getName());
        if (matcher.matches()) {
          apiLevels.add(Integer.parseInt(matcher.group(1)));
        }
      }
    }
    return apiLevels.build();
  }
}
