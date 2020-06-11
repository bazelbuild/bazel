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
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Comparator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/** Managed directories component. {@link ManagedDirectoriesKnowledge} */
public class ManagedDirectoriesKnowledgeImpl implements ManagedDirectoriesKnowledge {
  private final ManagedDirectoriesListener listener;

  private ImmutableSortedMap<PathFragment, RepositoryName> dirToRepoMap = ImmutableSortedMap.of();
  private ImmutableSortedMap<RepositoryName, ImmutableSet<PathFragment>> repoToDirMap =
      ImmutableSortedMap.of();

  public ManagedDirectoriesKnowledgeImpl(ManagedDirectoriesListener listener) {
    this.listener = listener;
  }

  @Override
  @Nullable
  public RepositoryName getOwnerRepository(PathFragment relativePathFragment) {
    Map.Entry<PathFragment, RepositoryName> entry = dirToRepoMap.floorEntry(relativePathFragment);
    if (entry != null && relativePathFragment.startsWith(entry.getKey())) {
      return entry.getValue();
    }
    return null;
  }

  @Override
  public ImmutableSet<PathFragment> getManagedDirectories(RepositoryName repositoryName) {
    ImmutableSet<PathFragment> pathFragments = repoToDirMap.get(repositoryName);
    return pathFragments != null ? pathFragments : ImmutableSet.of();
  }

  @Override
  public boolean workspaceHeaderReloaded(
      @Nullable WorkspaceFileValue oldValue, @Nullable WorkspaceFileValue newValue)
      throws AbruptExitException {
    if (Objects.equals(oldValue, newValue)) {
      listener.onManagedDirectoriesRefreshed(repoToDirMap.keySet());
      return false;
    }
    Map<PathFragment, RepositoryName> oldDirToRepoMap = dirToRepoMap;
    refreshMappings(newValue);
    if (!Objects.equals(oldDirToRepoMap, dirToRepoMap)) {
      listener.onManagedDirectoriesRefreshed(repoToDirMap.keySet());
      return true;
    }
    return false;
  }

  private void refreshMappings(@Nullable WorkspaceFileValue newValue) {
    if (newValue == null) {
      dirToRepoMap = ImmutableSortedMap.of();
      repoToDirMap = ImmutableSortedMap.of();
      return;
    }

    dirToRepoMap = ImmutableSortedMap.copyOf(newValue.getManagedDirectories());

    Map<RepositoryName, Set<PathFragment>> reposMap = Maps.newHashMap();
    for (Map.Entry<PathFragment, RepositoryName> entry : dirToRepoMap.entrySet()) {
      RepositoryName repositoryName = entry.getValue();
      reposMap.computeIfAbsent(repositoryName, name -> Sets.newTreeSet()).add(entry.getKey());
    }

    ImmutableSortedMap.Builder<RepositoryName, ImmutableSet<PathFragment>> reposMapBuilder =
        new ImmutableSortedMap.Builder<>(Comparator.comparing(RepositoryName::getName));
    for (Map.Entry<RepositoryName, Set<PathFragment>> entry : reposMap.entrySet()) {
      reposMapBuilder.put(entry.getKey(), ImmutableSet.copyOf(entry.getValue()));
    }
    repoToDirMap = reposMapBuilder.build();
  }

  /** Interface allows {@link BazelRepositoryModule} to react to managed directories refreshes. */
  public interface ManagedDirectoriesListener {
    void onManagedDirectoriesRefreshed(Set<RepositoryName> repositoryNames)
        throws AbruptExitException;
  }
}
