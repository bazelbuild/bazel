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
import com.google.devtools.build.lib.concurrent.ThreadSafety;
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

  private final ImmutableMap<PathFragment, Path> packageRoots;
  private final Path workspace;
  private final String productName;
  private final String[] prefixes;

  SymlinkForest(
      ImmutableMap<PathFragment, Path> packageRoots, Path workspace, String productName) {
    this.packageRoots = packageRoots;
    this.workspace = workspace;
    this.productName = productName;
    this.prefixes = new String[] { ".", "_", productName + "-"};
  }

  /**
   * Returns the longest prefix from a given set of 'prefixes' that are
   * contained in 'path'. I.e the closest ancestor directory containing path.
   * Returns null if none found.
   */
  @VisibleForTesting
  static PathFragment longestPathPrefix(PathFragment path, Set<PathFragment> prefixes) {
    for (int i = path.segmentCount(); i >= 0; i--) {
      PathFragment prefix = path.subFragment(0, i);
      if (prefixes.contains(prefix)) {
        return prefix;
      }
    }
    return null;
  }

  /**
   * Delete all dir trees under a given 'dir' that don't start with one of a set
   * of given 'prefixes'. Does not follow any symbolic links.
   */
  @VisibleForTesting
  @ThreadSafety.ThreadSafe
  static void deleteTreesBelowNotPrefixed(Path dir, String[] prefixes) throws IOException {
    dirloop:
    for (Path p : dir.getDirectoryEntries()) {
      String name = p.getBaseName();
      for (int i = 0; i < prefixes.length; i++) {
        if (name.startsWith(prefixes[i])) {
          continue dirloop;
        }
      }
      FileSystemUtils.deleteTree(p);
    }
  }

  void plantSymlinkForest() throws IOException {
    deleteTreesBelowNotPrefixed(workspace, prefixes);
    Path emptyPackagePath = null;

    // Create a sorted map of all dirs (packages and their ancestors) to sets of their roots.
    // Packages come from exactly one root, but their shared ancestors may come from more.
    // The map is maintained sorted lexicographically, so parents are before their children.
    Map<PathFragment, Set<Path>> dirRootsMap = Maps.newTreeMap();
    for (Map.Entry<PathFragment, Path> entry : packageRoots.entrySet()) {
      PathFragment pkgDir = entry.getKey();
      Path pkgRoot = entry.getValue();
      if (pkgDir.segmentCount() == 0) {
        emptyPackagePath = entry.getValue();
      }
      for (int i = 1; i <= pkgDir.segmentCount(); i++) {
        PathFragment dir = pkgDir.subFragment(0, i);
        Set<Path> roots = dirRootsMap.get(dir);
        if (roots == null) {
          roots = Sets.newHashSet();
          dirRootsMap.put(dir, roots);
        }
        roots.add(pkgRoot);
      }
    }
    // Now add in roots for all non-pkg dirs that are in between two packages, and missed above.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      if (!packageRoots.containsKey(dir)) {
        PathFragment pkgDir = longestPathPrefix(dir, packageRoots.keySet());
        if (pkgDir != null) {
          entry.getValue().add(packageRoots.get(pkgDir));
        }
      }
    }
    // Create output dirs for all dirs that have more than one root and need to be split.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      if (entry.getValue().size() > 1) {
        if (LOG_FINER) {
          LOG.finer("mkdir " + workspace.getRelative(dir));
        }
        FileSystemUtils.createDirectoryAndParents(workspace.getRelative(dir));
      }
    }
    // Make dir links for single rooted dirs.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      Set<Path> roots = entry.getValue();
      // Simple case of one root for this dir.
      if (roots.size() == 1) {
        if (dir.segmentCount() > 1 && dirRootsMap.get(dir.getParentDirectory()).size() == 1) {
          continue;  // skip--an ancestor will link this one in from above
        }
        // This is the top-most dir that can be linked to a single root. Make it so.
        Path root = roots.iterator().next();  // lone root in set
        if (LOG_FINER) {
          LOG.finer("ln -s " + root.getRelative(dir) + " " + workspace.getRelative(dir));
        }
        workspace.getRelative(dir).createSymbolicLink(root.getRelative(dir));
      }
    }
    // Make links for dirs within packages, skip parent-only dirs.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      if (entry.getValue().size() > 1) {
        // If this dir is at or below a package dir, link in its contents.
        PathFragment pkgDir = longestPathPrefix(dir, packageRoots.keySet());
        if (pkgDir != null) {
          Path root = packageRoots.get(pkgDir);
          try {
            Path absdir = root.getRelative(dir);
            if (absdir.isDirectory()) {
              if (LOG_FINER) {
                LOG.finer("ln -s " + absdir + "/* " + workspace.getRelative(dir) + "/");
              }
              for (Path target : absdir.getDirectoryEntries()) {
                PathFragment p = target.relativeTo(root);
                if (!dirRootsMap.containsKey(p)) {
                  //LOG.finest("ln -s " + target + " " + linkRoot.getRelative(p));
                  workspace.getRelative(p).createSymbolicLink(target);
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

    if (emptyPackagePath != null) {
      // For the top-level directory, generate symlinks to everything in the directory instead of
      // the directory itself.
      for (Path target : emptyPackagePath.getDirectoryEntries()) {
        String baseName = target.getBaseName();
        // Create any links that don't exist yet and don't start with bazel-.
        if (!baseName.startsWith(productName + "-")
            && !workspace.getRelative(baseName).exists()) {
          workspace.getRelative(baseName).createSymbolicLink(target);
        }
      }
    }
  }
}
