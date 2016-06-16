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

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Creates a symlink forest based on a package path map.
 */
class SymlinkForest {

  private static final Logger LOG = Logger.getLogger(SymlinkForest.class.getName());
  private static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);

  private final Map<PackageIdentifier, Path> packageRoots;
  private final Path workspace;
  private final String productName;

  SymlinkForest(
      Map<PackageIdentifier, Path> packageRoots, Path workspace, String productName) {
    this.packageRoots = packageRoots;
    this.workspace = workspace;
    this.productName = productName;
  }

  /**
   * Takes a map of directory fragments to root paths, and creates a symlink
   * forest under an existing linkRoot to the corresponding source dirs or
   * files. Symlink are made at the highest dir possible, linking files directly
   * only when needed with nested packages.
   */
  void plantLinkForest() throws IOException {
    deleteExisting();

    // Create a sorted map of all dirs (packages and their ancestors) to sets of their roots.
    // Packages come from exactly one root, but their shared ancestors may come from more.
    // The map is maintained sorted lexicographically, so parents are before their children.
    Map<PackageIdentifier, Set<Path>> dirRootsMap = Maps.newTreeMap();
    for (Map.Entry<PackageIdentifier, Path> entry : packageRoots.entrySet()) {
      PackageIdentifier packageIdentifier = entry.getKey();
      PathFragment pkgDir = packageIdentifier.getPackageFragment();
      Path pkgRoot = entry.getValue();
      for (int i = 0; i <= pkgDir.segmentCount(); i++) {
        PackageIdentifier dir = PackageIdentifier.create(
            packageIdentifier.getRepository(), pkgDir.subFragment(0, i));
        Set<Path> roots = dirRootsMap.get(dir);
        if (roots == null) {
          roots = Sets.newHashSet();
          dirRootsMap.put(dir, roots);
        }
        roots.add(pkgRoot);
      }
    }
    // Now add in roots for all non-pkg dirs that are in between two packages, and missed above.
    for (Map.Entry<PackageIdentifier, Set<Path>> entry : dirRootsMap.entrySet()) {
      PackageIdentifier packageIdentifier = entry.getKey();
      if (!packageRoots.containsKey(packageIdentifier)) {
        PackageIdentifier pkgDir = longestPathPrefix(packageIdentifier, packageRoots.keySet());
        if (pkgDir != null) {
          entry.getValue().add(packageRoots.get(pkgDir));
        }
      }
    }
    // Create output dirs for all dirs that have more than one root and need to be split.
    for (Map.Entry<PackageIdentifier, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey().getPathFragment();
      if (entry.getValue().size() > 1) {
        if (LOG_FINER) {
          LOG.finer("mkdir " + workspace.getRelative(dir));
        }
        FileSystemUtils.createDirectoryAndParents(workspace.getRelative(dir));
      }
    }
    // Make dir links for single rooted dirs.
    for (Map.Entry<PackageIdentifier, Set<Path>> entry : dirRootsMap.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      Path linkRoot = workspace.getRelative(pkgId.getRepository().getPathFragment());
      PathFragment dir = entry.getKey().getPackageFragment();
      Set<Path> roots = entry.getValue();
      // Simple case of one root for this dir.
      if (roots.size() == 1) {
        // Special case: the main repository is not deleted (because it contains symlinks to
        // bazel-out et al) so don't attempt to symlink it in.
        if (pkgId.equals(PackageIdentifier.EMPTY_PACKAGE_IDENTIFIER)) {
          symlinkEmptyPackage(roots.iterator().next());
          continue;
        }
        if (dir.segmentCount() > 0) {
          PackageIdentifier parent = PackageIdentifier.create(
              pkgId.getRepository(), dir.getParentDirectory());
          if (dir.segmentCount() > 0 && dirRootsMap.get(parent).size() == 1) {
            continue;  // skip--an ancestor will link this one in from above
          }
        }

        // This is the top-most dir that can be linked to a single root. Make it so.
        Path root = roots.iterator().next();  // lone root in set
        if (LOG_FINER) {
          LOG.finer("ln -s " + root.getRelative(dir) + " " + linkRoot.getRelative(dir));
        }
        linkRoot.getRelative(dir).createSymbolicLink(root.getRelative(dir));
      }
    }
    // Make links for dirs within packages, skip parent-only dirs.
    for (Map.Entry<PackageIdentifier, Set<Path>> entry : dirRootsMap.entrySet()) {
      Path linkRoot = workspace.getRelative(entry.getKey().getRepository().getPathFragment());
      PathFragment dir = entry.getKey().getPackageFragment();
      if (entry.getValue().size() > 1) {
        // If this dir is at or below a package dir, link in its contents.
        PackageIdentifier pkgDir = longestPathPrefix(entry.getKey(), packageRoots.keySet());
        if (pkgDir != null) {
          Path root = packageRoots.get(pkgDir);
          try {
            Path absdir = root.getRelative(dir);
            if (absdir.isDirectory()) {
              if (LOG_FINER) {
                LOG.finer("ln -s " + absdir + "/* " + linkRoot.getRelative(dir) + "/");
              }
              for (Path target : absdir.getDirectoryEntries()) {
                PackageIdentifier dirent = PackageIdentifier.create(
                    pkgDir.getRepository(), target.relativeTo(root));
                if (!dirRootsMap.containsKey(dirent)) {
                  linkRoot.getRelative(dirent.getPackageFragment()).createSymbolicLink(target);
                }
              }
            } else {
              LOG.fine("Symlink planting skipping dir '" + absdir + "'");
            }
          } catch (IOException e) {
            e.printStackTrace();
          }
          // Otherwise its just an otherwise empty common parent dir.
        }
      }
    }
  }

  private void deleteExisting() throws IOException {
    FileSystemUtils.createDirectoryAndParents(workspace);
    FileSystemUtils.deleteTreesBelowNotPrefixed(workspace,
        new String[] { ".", "_", productName + "-"});
    for (Map.Entry<PackageIdentifier, Path> entry : packageRoots.entrySet()) {
      RepositoryName repo = entry.getKey().getRepository();
      Path repoPath = workspace.getRelative(repo.getPathFragment());
      if (!repo.isMain() && repoPath.exists()) {
        FileSystemUtils.deleteTree(repoPath);
      }
    }
  }

  /**
   * For the top-level directory, generate symlinks to everything in the directory instead of the
   * directory itself.
   */
  private void symlinkEmptyPackage(Path emptyPackagePath) throws IOException {
    for (Path target : emptyPackagePath.getDirectoryEntries()) {
      String baseName = target.getBaseName();
      // Create any links that don't exist yet and don't start with bazel-.
      if (!baseName.startsWith(productName + "-")
          && !workspace.getRelative(baseName).exists()) {
        workspace.getRelative(baseName).createSymbolicLink(target);
      }
    }
  }

  /**
   * Returns the longest prefix from a given set of 'prefixes' that are
   * contained in 'path'. I.e the closest ancestor directory containing path.
   * Returns null if none found.
   */
  static PackageIdentifier longestPathPrefix(
      PackageIdentifier packageIdentifier, Set<PackageIdentifier> prefixes) {
    PathFragment pkg = packageIdentifier.getPackageFragment();
    for (int i = pkg.segmentCount(); i >= 0; i--) {
      PackageIdentifier prefix = PackageIdentifier.create(
          packageIdentifier.getRepository(), pkg.subFragment(0, i));
      if (prefixes.contains(prefix)) {
        return prefix;
      }
    }
    return null;
  }
}
