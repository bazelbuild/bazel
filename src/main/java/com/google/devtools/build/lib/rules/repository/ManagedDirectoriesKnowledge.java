// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.repository;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor.WorkspaceFileHeaderListener;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Interface to access the managed directories holder object.
 *
 * <p>Managed directories are user-owned directories, which can be incrementally updated by
 * repository rules, so that the updated files are visible for the actions in the same build.
 *
 * <p>Having managed directories as a separate component (and not SkyValue) allows to skip recording
 * the dependency in Skyframe for each FileStateValue and DirectoryListingStateValue.
 */
public interface ManagedDirectoriesKnowledge extends WorkspaceFileHeaderListener {
  ManagedDirectoriesKnowledge NO_MANAGED_DIRECTORIES =
      new ManagedDirectoriesKnowledge() {
        @Override
        public boolean workspaceHeaderReloaded(
            @Nullable WorkspaceFileValue oldValue, @Nullable WorkspaceFileValue newValue) {
          return false;
        }

        @Nullable
        @Override
        public RepositoryName getOwnerRepository(PathFragment relativePathFragment) {
          return null;
        }

        @Override
        public ImmutableSet<PathFragment> getManagedDirectories(RepositoryName repositoryName) {
          return ImmutableSet.of();
        }
      };

  /**
   * Returns the owning repository for the incrementally updated path, or null.
   *
   * @param relativePathFragment path to check, relative to workspace root
   * @return RepositoryName or null if there is no owning repository
   */
  @Nullable
  RepositoryName getOwnerRepository(PathFragment relativePathFragment);

  /** Returns managed directories for the passed repository. */
  ImmutableSet<PathFragment> getManagedDirectories(RepositoryName repositoryName);
}
