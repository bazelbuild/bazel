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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.Comparator;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** Managed directories component. {@link ManagedDirectoriesKnowledge} */
public class ManagedDirectoriesKnowledgeImpl implements ManagedDirectoriesKnowledge {
  private final AtomicReference<ImmutableSortedMap<PathFragment, RepositoryName>>
      managedDirectoriesRef = new AtomicReference<>(ImmutableSortedMap.of());
  private final AtomicReference<Map<RepositoryName, ImmutableSet<PathFragment>>> repoToDirMapRef =
      new AtomicReference<>(ImmutableMap.of());

  /**
   * During build commands execution, Skyframe caches the states of files (FileStateValue) and
   * directory listings (DirectoryListingStateValue). In the case when the files/directories are
   * under a managed directory or inside an external repository, evaluation of file/directory
   * listing states requires that the RepositoryDirectoryValue of the owning external repository is
   * evaluated beforehand. (So that the repository rule generates the files.) So there is a
   * dependency on RepositoryDirectoryValue for files under managed directories and external
   * repositories. This dependency is recorded by Skyframe,
   *
   * <p>From the other side, by default Skyframe injects the new values of changed files already at
   * the stage of checking what files have changed. Only the values without any dependencies can be
   * injected into Skyframe. Skyframe can be specifically instructed to not inject new values and
   * only register them as changed.
   *
   * <p>When the values of managed directories change, some files appear to become files under
   * managed directories, or they are no longer files under managed directories. This implies that
   * corresponding file/directory listing states gain the dependency (RepositoryDirectoryValue) or
   * they lose this dependency. In both cases, we should prevent Skyframe from injecting those new
   * values of file/directory listing states at the stage of checking changed files.
   *
   * <p>That is why we need to keep track of the previous value of the managed directories.
   */
  private final AtomicReference<ImmutableSortedMap<PathFragment, RepositoryName>>
      oldManagedDirectoriesRef = new AtomicReference<>(ImmutableSortedMap.of());

  @Override
  @Nullable
  public RepositoryName getOwnerRepository(RootedPath rootedPath, boolean old) {
    PathFragment relativePath = rootedPath.getRootRelativePath();
    NavigableMap<PathFragment, RepositoryName> map =
        old ? oldManagedDirectoriesRef.get() : managedDirectoriesRef.get();
    Map.Entry<PathFragment, RepositoryName> entry = map.floorEntry(relativePath);
    if (entry != null && relativePath.startsWith(entry.getKey())) {
      return entry.getValue();
    }
    return null;
  }

  @Override
  public ImmutableSet<PathFragment> getManagedDirectories(RepositoryName repositoryName) {
    ImmutableSet<PathFragment> pathFragments = repoToDirMapRef.get().get(repositoryName);
    return pathFragments != null ? pathFragments : ImmutableSet.of();
  }

  public void setManagedDirectories(ImmutableMap<PathFragment, RepositoryName> map) {
    oldManagedDirectoriesRef.set(managedDirectoriesRef.get());
    ImmutableSortedMap<PathFragment, RepositoryName> pathsMap = ImmutableSortedMap.copyOf(map);
    managedDirectoriesRef.set(pathsMap);

    Map<RepositoryName, Set<PathFragment>> reposMap = Maps.newHashMap();
    for (Map.Entry<PathFragment, RepositoryName> entry : pathsMap.entrySet()) {
      RepositoryName repositoryName = entry.getValue();
      reposMap.computeIfAbsent(repositoryName, name -> Sets.newTreeSet()).add(entry.getKey());
    }

    ImmutableSortedMap.Builder<RepositoryName, ImmutableSet<PathFragment>> builder =
        new ImmutableSortedMap.Builder<>(Comparator.comparing(RepositoryName::getName));
    for (Map.Entry<RepositoryName, Set<PathFragment>> entry : reposMap.entrySet()) {
      builder.put(entry.getKey(), ImmutableSet.copyOf(entry.getValue()));
    }
    repoToDirMapRef.set(builder.build());
  }
}
