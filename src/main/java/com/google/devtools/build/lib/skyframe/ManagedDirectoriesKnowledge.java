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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Managed directories are user-owned directories, which can be incrementally updated by repository
 * rules, so that the updated files are visible for the actions in the same build.
 *
 * <p>File and directory nodes inside managed directories depend on the relevant {@link
 * com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue} node, so that they are
 * invalidated if the repository definition changes and recomputed based on the contents written by
 * the repository rule.
 *
 * <p>The repository rule will also be re-run if any of the directories it manages is entirely
 * missing. This allows a user to fairly cleanly regenerate managed directories. Whether this is an
 * actually used feature is unknown to the Bazel developers.
 */
public interface ManagedDirectoriesKnowledge {
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
   * Returns true if the multi-mapping of repository -> {managed directory} changes. Such changes
   * trigger a full wipe of the Skyframe graph, similar to a --package_path flag change.
   */
  boolean workspaceHeaderReloaded(
      @Nullable WorkspaceFileValue oldValue, @Nullable WorkspaceFileValue newValue)
      throws AbruptExitException;

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
