// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;

import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/** Common utilities for dealing with files outside the package roots. */
class ExternalFilesHelper {

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final Set<Path> immutableDirs;

  ExternalFilesHelper(AtomicReference<PathPackageLocator> pkgLocator) {
    this(pkgLocator, ImmutableSet.<Path>of());
  }

  ExternalFilesHelper(AtomicReference<PathPackageLocator> pkgLocator, Set<Path> immutableDirs) {
    this.pkgLocator = pkgLocator;
    this.immutableDirs = immutableDirs;
  }

  private enum FileType {
    // A file inside the package roots.
    INTERNAL_FILE,

    // A file outside the package roots that we may pretend is immutable.
    EXTERNAL_IMMUTABLE_FILE,

    // A file outside the package roots about which we may make no other assumptions.
    EXTERNAL_MUTABLE_FILE,
  }

  private FileType getFileType(RootedPath rootedPath) {
    // TODO(bazel-team): This is inefficient when there are a lot of package roots or there are a
    // lot of immutable directories. Consider either explicitly preventing this case or using a more
    // efficient approach here (e.g. use a trie for determing if a file is under an immutable
    // directory).
    if (!pkgLocator.get().getPathEntries().contains(rootedPath.getRoot())) {
      Path path = rootedPath.asPath();
      for (Path immutableDir : immutableDirs) {
        if (path.startsWith(immutableDir)) {
          return FileType.EXTERNAL_IMMUTABLE_FILE;
        }
      }
      return FileType.EXTERNAL_MUTABLE_FILE;
    }
    return FileType.INTERNAL_FILE;
  }

  public boolean shouldAssumeImmutable(RootedPath rootedPath) {
    return getFileType(rootedPath) == FileType.EXTERNAL_IMMUTABLE_FILE;
  }

  public void maybeAddDepOnBuildId(RootedPath rootedPath, SkyFunction.Environment env) {
   if (getFileType(rootedPath) == FileType.EXTERNAL_MUTABLE_FILE) {
      // For files outside the package roots that are not assumed to be immutable, add a dependency
      // on the build_id. This is sufficient for correctness; all other files will be handled by
      // diff awareness of their respective package path, but these files need to be addressed
      // separately.
      //
      // Using the build_id here seems to introduce a performance concern because the upward
      // transitive closure of these external files will get eagerly invalidated on each
      // incremental build (e.g. if every file had a transitive dependency on the filesystem root,
      // then we'd have a big performance problem). But this a non-issue by design:
      // - We don't add a dependency on the parent directory at the package root boundary, so the
      // only transitive dependencies from files inside the package roots to external files are
      // through symlinks. So the upwards transitive closure of external files is small.
      // - The only way external source files get into the skyframe graph in the first place is
      // through symlinks outside the package roots, which we neither want to encourage nor
      // optimize for since it is not common. So the set of external files is small.
      //
      // The above reasoning doesn't hold for bazel, because external repositories
      // (e.g. new_local_repository) cause lots of external symlinks to be present in the build.
      // So bazel pretends that these external repositories are immutable to avoid the performance
      // penalty described above.
      PrecomputedValue.dependOnBuildId(env);
    }
  }
}
