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

package com.google.devtools.build.lib.buildtool;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.Map;
import java.util.Set;

/**
 * Creates a symlink forest based on a package path map.
 */
class SymlinkForest {
  private final ImmutableMap<PackageIdentifier, Root> packageRoots;
  private final Path execroot;
  private final String prefix;

  SymlinkForest(
      ImmutableMap<PackageIdentifier, Root> packageRoots, Path execroot, String productName) {
    this.packageRoots = packageRoots;
    this.execroot = execroot;
    this.prefix = productName + "-";
  }

  /**
   * Delete all dir trees under a given 'dir' that don't start with a given 'prefix'. Does not
   * follow any symbolic links.
   */
  @VisibleForTesting
  @ThreadSafety.ThreadSafe
  static void deleteTreesBelowNotPrefixed(Path dir, String prefix) throws IOException {
    for (Path p : dir.getDirectoryEntries()) {
      if (!p.getBaseName().startsWith(prefix)) {
        p.deleteTree();
      }
    }
  }

  /**
   * Plant a symlink forest under execution root to ensure sources file are available and up to
   * date. For the main repo: If root package ("//:") is used, link every file and directory under
   * the top-level directory of the main repo. Otherwise, we only link the directories that are used
   * in presented main repo packages. For every external repo: make a such a directory link:
   * <execroot>/<ws_name>/external/<repo_name> --> <output_base>/external/<repo_name>
   */
  void plantSymlinkForest() throws IOException {
    deleteTreesBelowNotPrefixed(execroot, prefix);

    Path mainRepoRoot = null;
    Map<Path, Path> mainRepoLinks = Maps.newHashMap();
    Set<Path> externalRepoLinks = Sets.newHashSet();

    for (Map.Entry<PackageIdentifier, Root> entry : packageRoots.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      if (pkgId.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
        // This isn't a "real" package, don't add it to the symlink tree.
        continue;
      }
      RepositoryName repository = pkgId.getRepository();
      if (repository.isMain() || repository.isDefault()) {
        // If root package of the main repo is required, we record the main repo root so that
        // we can later link everything under main repo's top-level directory. And in this case,
        // we don't need to record other links for directories under the top-level directory any
        // more.
        if (pkgId.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
          mainRepoRoot = entry.getValue().getRelative(pkgId.getSourceRoot());
        }
        if (mainRepoRoot == null) {
          Path execrootLink = execroot.getRelative(pkgId.getPackageFragment().getSegment(0));
          Path sourcePath = entry.getValue().getRelative(pkgId.getSourceRoot().getSegment(0));
          mainRepoLinks.putIfAbsent(execrootLink, sourcePath);
        }
      } else {
        // For other external repositories, generate a symlink to the external repository
        // directory itself.
        // <output_base>/execroot/<main repo name>/external/<external repo name> -->
        // <output_base>/external/<external repo name>
        Path execrootLink = execroot.getRelative(repository.getPathUnderExecRoot());
        Path sourcePath = entry.getValue().getRelative(repository.getSourceRoot());
        if (externalRepoLinks.contains(execrootLink)) {
          continue;
        }
        if (externalRepoLinks.isEmpty()) {
          execroot.getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME).createDirectoryAndParents();
        }
        externalRepoLinks.add(execrootLink);
        execrootLink.createSymbolicLink(sourcePath);
      }
    }
    if (mainRepoRoot != null) {
      // For the main repo top-level directory, generate symlinks to everything in the directory
      // instead of the directory itself.
      for (Path target : mainRepoRoot.getDirectoryEntries()) {
        String baseName = target.getBaseName();
        Path execPath = execroot.getRelative(baseName);
        // Create any links that don't start with bazel-.
        if (!baseName.startsWith(prefix)) {
          execPath.createSymbolicLink(target);
        }
      }
    } else {
      for (Map.Entry<Path, Path> entry : mainRepoLinks.entrySet()) {
        Path link = entry.getKey();
        Path target = entry.getValue();
        link.createSymbolicLink(target);
      }
    }
  }
}
