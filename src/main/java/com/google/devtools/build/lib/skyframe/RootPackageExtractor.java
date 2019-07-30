// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.List;
import java.util.function.Consumer;

/** A streaming interface for recursively searching for all packages under a given set of roots. */
public interface RootPackageExtractor {

  /**
   * Recursively search each of the given roots in a repository for packages (while respecting
   * blacklists and exclusions), calling the {@code results} callback as each package is discovered.
   *
   * @param results callback invoked once for each package as it is discovered under a root
   * @param graph skyframe graph used for retrieving the directories under each root
   * @param roots all the filesystem roots to search for packages
   * @param eventHandler receives package-loading errors for any packages loaded by graph queries
   * @param repository the repository under which the roots can be found
   * @param directory starting directory under which to find packages, relative to the roots
   * @param blacklistedSubdirectories directories that will not be searched by policy, relative to
   *     the roots
   * @param excludedSubdirectories directories the user requests not be searched, relative to the
   *     roots
   * @throws InterruptedException if a graph query is interrupted before all roots have been
   *     searched exhaustively
   */
  void streamPackagesFromRoots(
      Consumer<PathFragment> results,
      WalkableGraph graph,
      List<Root> roots,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> blacklistedSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException;
}
