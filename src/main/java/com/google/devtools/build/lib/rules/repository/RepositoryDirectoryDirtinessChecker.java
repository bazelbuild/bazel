// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.rules.repository;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A dirtiness checker that:
 *
 * <ul>
 *   <li>Only dirties {@link RepositoryDirectoryValue}s, if they were produced in a {@code
 *       --nofetch} build, so that they are re-created on subsequent {@code --fetch} builds. The
 *       alternative solution would be to reify the value of the flag as a Skyframe value.
 *   <li>Dirties repositories, if their managed directories were changed or do not exist.
 * </ul>
 */
@VisibleForTesting
public class RepositoryDirectoryDirtinessChecker extends SkyValueDirtinessChecker {
  private final Path workspaceRoot;
  private final ManagedDirectoriesKnowledge managedDirectoriesKnowledge;

  public RepositoryDirectoryDirtinessChecker(
      Path workspaceRoot, ManagedDirectoriesKnowledge managedDirectoriesKnowledge) {
    this.workspaceRoot = workspaceRoot;
    this.managedDirectoriesKnowledge = managedDirectoriesKnowledge;
  }

  @Override
  public boolean applies(SkyKey skyKey) {
    return skyKey.functionName().equals(SkyFunctions.REPOSITORY_DIRECTORY);
  }

  @Override
  public SkyValue createNewValue(SkyKey key, @Nullable TimestampGranularityMonitor tsgm) {
    throw new UnsupportedOperationException();
  }

  @Override
  public DirtyResult check(
      SkyKey skyKey, SkyValue skyValue, @Nullable TimestampGranularityMonitor tsgm) {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    RepositoryDirectoryValue repositoryValue = (RepositoryDirectoryValue) skyValue;

    if (!repositoryValue.repositoryExists()) {
      return DirtyResult.notDirty(skyValue);
    }
    if (repositoryValue.isFetchingDelayed()) {
      return DirtyResult.dirty(skyValue);
    }

    if (!Objects.equals(
        managedDirectoriesKnowledge.getManagedDirectories(repositoryName),
        repositoryValue.getManagedDirectories())) {
      return DirtyResult.dirty(skyValue);
    }

    if (!managedDirectoriesExist(workspaceRoot, repositoryValue.getManagedDirectories())) {
      return DirtyResult.dirty(skyValue);
    }
    return DirtyResult.notDirty(skyValue);
  }

  static boolean managedDirectoriesExist(
      Path workspaceRoot, ImmutableSet<PathFragment> managedDirectories) {
    return managedDirectories.stream().allMatch(pf -> workspaceRoot.getRelative(pf).exists());
  }
}
