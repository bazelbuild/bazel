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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Objects;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A repository's name and directory.
 */
public class RepositoryValue implements SkyValue {
  private final RepositoryName repositoryName;
  private final RepositoryDirectoryValue repositoryDirectory;

  /**
   * Creates a repository with a given name in a certain directory.
   */
  public RepositoryValue(RepositoryName repositoryName, RepositoryDirectoryValue repository) {
    this.repositoryName = repositoryName;
    this.repositoryDirectory = repository;
  }

  /**
   * Returns the path to the repository.
   */
  public Path getPath() {
    return repositoryDirectory.getPath();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (other == null || getClass() != other.getClass()) {
      return false;
    }

    RepositoryValue that = (RepositoryValue) other;
    return Objects.equal(repositoryName, that.repositoryName)
        && Objects.equal(repositoryDirectory, that.repositoryDirectory);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(repositoryName, repositoryDirectory);
  }

  static SkyKey key(RepositoryName repositoryName) {
    return new SkyKey(SkyFunctions.REPOSITORY, repositoryName);
  }
}
