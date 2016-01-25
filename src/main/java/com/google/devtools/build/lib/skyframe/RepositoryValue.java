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

import com.google.common.base.Objects;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A local view of an external repository.
 */
public class RepositoryValue implements SkyValue {
  private final Path path;
  private final boolean fetchingDelayed;

  private RepositoryValue(Path path, boolean fetchingDelayed) {
    this.path = path;
    this.fetchingDelayed = fetchingDelayed;
  }

  /**
   * Creates an immutable external repository.
   */
  public static RepositoryValue create(Path repositoryDirectory) {
    return new RepositoryValue(repositoryDirectory, false);
  }

  /**
   * Creates a value that represents a repository whose fetching has been delayed by a
   * {@code --nofetch} command line option.
   */
  public static RepositoryValue fetchingDelayed(Path repositoryDirectory) {
    return new RepositoryValue(repositoryDirectory, true);
  }

  /**
   * Returns the path to the directory containing the repository's contents. This directory is
   * guaranteed to exist.  It may contain a full Bazel repository (with a WORKSPACE file,
   * directories, and BUILD files) or simply contain a file (or set of files) for, say, a jar from
   * Maven.
   */
  public Path getPath() {
    return path;
  }

  public boolean isFetchingDelayed() {
    return fetchingDelayed;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }

    if (other instanceof RepositoryValue) {
      RepositoryValue otherValue = (RepositoryValue) other;
      return path.equals(otherValue.path);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(path);
  }

  @Override
  public String toString() {
    return path.getPathString();
  }

  /**
   * Creates a key from the given repository name.
   */
  public static SkyKey key(RepositoryName repository) {
    return new SkyKey(SkyFunctions.REPOSITORY, repository);
  }
}
