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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Creates a symlink forest based on a package path map.
 */
class SymlinkForest {

  private static final Logger logger = Logger.getLogger(SymlinkForest.class.getName());
  private static final boolean LOG_FINER = logger.isLoggable(Level.FINER);

  private final ImmutableMap<PackageIdentifier, Root> packageRoots;
  private final Path execroot;
  private final String workspaceName;
  private final String productName;
  private final String[] prefixes;

  SymlinkForest(
      ImmutableMap<PackageIdentifier, Root> packageRoots,
      Path execroot,
      String productName,
      String workspaceName) {
    this.packageRoots = packageRoots;
    this.execroot = execroot;
    this.workspaceName = workspaceName;
    this.productName = productName;
    this.prefixes = new String[] { ".", "_", productName + "-"};
  }

  /**
   * Returns the longest prefix from a given set of 'prefixes' that are
   * contained in 'path'. I.e the closest ancestor directory containing path.
   * Returns null if none found.
   * @param path
   * @param prefixes
   */
  @VisibleForTesting
  static PackageIdentifier longestPathPrefix(
      PackageIdentifier path, ImmutableSet<PackageIdentifier> prefixes) {
    for (int i = path.getPackageFragment().segmentCount(); i >= 0; i--) {
      PackageIdentifier prefix = createInRepo(path, path.getPackageFragment().subFragment(0, i));
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
      for (String prefix : prefixes) {
        if (name.startsWith(prefix)) {
          continue dirloop;
        }
      }
      p.deleteTree();
    }
  }

  void plantSymlinkForest() throws IOException {
    deleteTreesBelowNotPrefixed(execroot, prefixes);
    // TODO(kchodorow): this can be removed once the execution root is rearranged.
    // Current state: symlink tree was created under execroot/$(basename ws) and then
    // execroot/wsname is symlinked to that. The execution root change creates (and cleans up)
    // subtrees for each repository and has been rolled forward and back several times. Thus, if
    // someone was using a with-execroot-change version of bazel and then switched to this one,
    // their execution root would contain a subtree for execroot/wsname that would never be
    // cleaned up by this version of Bazel.
    Path realWorkspaceDir = execroot.getParentDirectory().getRelative(workspaceName);
    if (!workspaceName.equals(execroot.getBaseName()) && realWorkspaceDir.exists()
        && !realWorkspaceDir.isSymbolicLink()) {
      realWorkspaceDir.deleteTree();
    }

    // Packages come from exactly one root, but their shared ancestors may come from more.
    Map<PackageIdentifier, Set<Root>> dirRootsMap = Maps.newHashMap();
    // Elements in this list are added so that parents come before their children.
    ArrayList<PackageIdentifier> dirsParentsFirst = new ArrayList<>();
    for (Map.Entry<PackageIdentifier, Root> entry : packageRoots.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      if (pkgId.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
        // This isn't a "real" package, don't add it to the symlink tree.
        continue;
      }
      Root pkgRoot = entry.getValue();
      ArrayList<PackageIdentifier> newDirs = new ArrayList<>();
      for (PathFragment fragment = pkgId.getPackageFragment();
          !fragment.isEmpty();
          fragment = fragment.getParentDirectory()) {
        PackageIdentifier dirId = createInRepo(pkgId, fragment);
        Set<Root> roots = dirRootsMap.get(dirId);
        if (roots == null) {
          roots = Sets.newHashSet();
          dirRootsMap.put(dirId, roots);
          newDirs.add(dirId);
        }
        roots.add(pkgRoot);
      }
      Collections.reverse(newDirs);
      dirsParentsFirst.addAll(newDirs);
    }
    // Now add in roots for all non-pkg dirs that are in between two packages, and missed above.
    for (PackageIdentifier dir : dirsParentsFirst) {
      if (!packageRoots.containsKey(dir)) {
        PackageIdentifier pkgId = longestPathPrefix(dir, packageRoots.keySet());
        if (pkgId != null) {
          dirRootsMap.get(dir).add(packageRoots.get(pkgId));
        }
      }
    }
    // Create output dirs for all dirs that have more than one root and need to be split.
    for (PackageIdentifier dir : dirsParentsFirst) {
      if (!dir.getRepository().isMain()) {
        FileSystemUtils.createDirectoryAndParents(
            execroot.getRelative(dir.getRepository().getPathUnderExecRoot()));
      }
      if (dirRootsMap.get(dir).size() > 1) {
        if (LOG_FINER) {
          logger.finer("mkdir " + execroot.getRelative(dir.getPathUnderExecRoot()));
        }
        FileSystemUtils.createDirectoryAndParents(
            execroot.getRelative(dir.getPathUnderExecRoot()));
      }
    }

    // Make dir links for single rooted dirs.
    for (PackageIdentifier dir : dirsParentsFirst) {
      Set<Root> roots = dirRootsMap.get(dir);
      // Simple case of one root for this dir.
      if (roots.size() == 1) {
        PathFragment parent = dir.getPackageFragment().getParentDirectory();
        if (!parent.isEmpty() && dirRootsMap.get(createInRepo(dir, parent)).size() == 1) {
          continue;  // skip--an ancestor will link this one in from above
        }
        // This is the top-most dir that can be linked to a single root. Make it so.
        Root root = roots.iterator().next(); // lone root in set
        if (LOG_FINER) {
          logger.finer(
              "ln -s "
                  + root.getRelative(dir.getSourceRoot())
                  + " "
                  + execroot.getRelative(dir.getPathUnderExecRoot()));
        }
        execroot.getRelative(dir.getPathUnderExecRoot())
            .createSymbolicLink(root.getRelative(dir.getSourceRoot()));
      }
    }
    // Make links for dirs within packages, skip parent-only dirs.
    for (PackageIdentifier dir : dirsParentsFirst) {
      if (dirRootsMap.get(dir).size() > 1) {
        // If this dir is at or below a package dir, link in its contents.
        PackageIdentifier pkgId = longestPathPrefix(dir, packageRoots.keySet());
        if (pkgId != null) {
          Root root = packageRoots.get(pkgId);
          try {
            Path absdir = root.getRelative(dir.getSourceRoot());
            if (absdir.isDirectory()) {
              if (LOG_FINER) {
                logger.finer(
                    "ln -s " + absdir + "/* " + execroot.getRelative(dir.getSourceRoot()) + "/");
              }
              for (Path target : absdir.getDirectoryEntries()) {
                PathFragment p = root.relativize(target);
                if (!dirRootsMap.containsKey(createInRepo(pkgId, p))) {
                  //LOG.finest("ln -s " + target + " " + linkRoot.getRelative(p));
                  execroot.getRelative(p).createSymbolicLink(target);
                }
              }
            } else {
              logger.fine("Symlink planting skipping dir '" + absdir + "'");
            }
          } catch (IOException e) {
            e.printStackTrace();
          }
          // Otherwise its just an otherwise empty common parent dir.
        }
      }
    }

    for (Map.Entry<PackageIdentifier, Root> entry : packageRoots.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      if (!pkgId.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
        continue;
      }
      Path execrootDirectory = execroot.getRelative(pkgId.getPathUnderExecRoot());
      // If there were no subpackages, this directory might not exist yet.
      if (!execrootDirectory.exists()) {
        FileSystemUtils.createDirectoryAndParents(execrootDirectory);
      }
      // For the top-level directory, generate symlinks to everything in the directory instead of
      // the directory itself.
      Path sourceDirectory = entry.getValue().getRelative(pkgId.getSourceRoot());
      for (Path target : sourceDirectory.getDirectoryEntries()) {
        String baseName = target.getBaseName();
        Path execPath = execrootDirectory.getRelative(baseName);
        // Create any links that don't exist yet and don't start with bazel-.
        if (!baseName.startsWith(productName + "-") && !execPath.exists()) {
          execPath.createSymbolicLink(target);
        }
      }
    }

    symlinkCorrectWorkspaceName();
  }

  /**
   * Right now, the execution root is under the basename of the source directory, not the name
   * defined in the WORKSPACE file. Thus, this adds a symlink with the WORKSPACE's workspace name
   * to the old-style execution root.
   * TODO(kchodorow): get rid of this once exec root is always under the WORKSPACE's workspace
   * name.
   * @throws IOException
   */
  private void symlinkCorrectWorkspaceName() throws IOException {
    Path correctDirectory = execroot.getParentDirectory().getRelative(workspaceName);
    if (!correctDirectory.exists()) {
      correctDirectory.createSymbolicLink(execroot);
    }
  }

  private static PackageIdentifier createInRepo(
      PackageIdentifier repo, PathFragment packageFragment) {
    return PackageIdentifier.create(repo.getRepository(), packageFragment);
  }
}
