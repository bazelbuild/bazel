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
package com.google.devtools.build.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Android resource and asset directories that can be parsed.
 */
public class UnvalidatedAndroidDirectories {

  private static final Pattern VALID_REGEX = Pattern.compile(".*:.*");

  public static final String EXPECTED_FORMAT = "resources[#resources]:assets[#assets]";

  public static UnvalidatedAndroidDirectories valueOf(String text) {
    return valueOf(text, FileSystems.getDefault());
  }

  @VisibleForTesting
  static UnvalidatedAndroidDirectories valueOf(String text, FileSystem fileSystem) {
    if (!VALID_REGEX.matcher(text).find()) {
      throw new IllegalArgumentException(
          text + " is not in the format '" + EXPECTED_FORMAT + "'");
    }
    String[] parts = text.split(":");
    return new UnvalidatedAndroidDirectories(
        parts.length > 0 ? splitPaths(parts[0], fileSystem) : ImmutableList.<Path>of(),
        parts.length > 1 ? splitPaths(parts[1], fileSystem) : ImmutableList.<Path>of());
  }

  protected static ImmutableList<Path> splitPaths(String pathsString, FileSystem fileSystem) {
    if (pathsString.length() == 0) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Path> paths = new ImmutableList.Builder<>();
    for (String pathString : pathsString.split("#")) {
      paths.add(exists(fileSystem.getPath(pathString)));
    }
    return paths.build();
  }

  protected static Path exists(Path path) {
    if (!Files.exists(path)) {
      throw new IllegalArgumentException(path + " does not exist");
    }
    return path;
  }

  protected final ImmutableList<Path> assetDirs;
  protected final ImmutableList<Path> resourceDirs;

  public UnvalidatedAndroidDirectories(
      ImmutableList<Path> resourceDirs, ImmutableList<Path> assetDirs) {
    this.resourceDirs = resourceDirs;
    this.assetDirs = assetDirs;
  }

  void walk(final AndroidDataPathWalker pathWalker) throws IOException {
    for (Path path : resourceDirs) {
      pathWalker.walkResources(path);
    }
    for (Path path : assetDirs) {
      pathWalker.walkAssets(path);
    }
  }

  @Override
  public String toString() {
    return String.format("UnvalidatedAndroidDirectories(%s, %s)", resourceDirs, assetDirs);
  }

  @Override
  public int hashCode() {
    return Objects.hash(resourceDirs, assetDirs);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof UnvalidatedAndroidDirectories)) {
      return false;
    }
    UnvalidatedAndroidDirectories other = (UnvalidatedAndroidDirectories) obj;
    return Objects.equals(other.resourceDirs, resourceDirs)
        && Objects.equals(other.assetDirs, assetDirs);
  }

  public UnvalidatedAndroidData toData(Path manifest) {
    return new UnvalidatedAndroidData(resourceDirs, assetDirs, manifest);
  }
}
