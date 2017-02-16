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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Creates a symlink forest based on a package path map.
 */
@Immutable
final class SymlinkForest {

  private static final Logger log = Logger.getLogger(SymlinkForest.class.getName());
  private static final boolean LOG_FINER = log.isLoggable(Level.FINER);

  private final ImmutableMap<PackageIdentifier, Path> packageRoots;
  private final Path workspace;
  private final String workspaceName;
  private final String productName;
  private final String[] prefixes;
  private final ImmutableSet<RepositoryName> repositories;
  private final boolean legacyExternalRunfiles;

  private SymlinkForest(
      boolean legacyExternalRunfiles,
      ImmutableMap<PackageIdentifier, Path> packageRoots,
      Path workspace,
      String productName,
      String[] prefixes,
      String workspaceName) {
    this.legacyExternalRunfiles = legacyExternalRunfiles;
    this.packageRoots = packageRoots;
    this.workspace = workspace;
    this.workspaceName = workspaceName;
    this.productName = productName;
    this.prefixes = prefixes;
    ImmutableSet.Builder<RepositoryName> repositoryNameBuilder = ImmutableSet.builder();
    for (PackageIdentifier pkgId : packageRoots.keySet()) {
      repositoryNameBuilder.add(pkgId.getRepository());
    }
    this.repositories = repositoryNameBuilder.build();
  }

  /**
   * Returns the longest prefix from a given set of 'prefixes' that are
   * contained in 'path'. I.e the closest ancestor directory containing path.
   * Returns null if none found.
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
   * Delete all dir trees under each repository root. For the main repository, don't delete trees
   * that start with one of a set of given 'prefixes'. Does not follow any symbolic links.
   */
  @VisibleForTesting
  @ThreadSafety.ThreadSafe
  void deleteTreesBelowNotPrefixed() throws IOException {
    for (RepositoryName repositoryName : Iterables.concat(
        ImmutableList.of(RepositoryName.MAIN), repositories)) {
      Path repositoryExecRoot = workspace.getRelative(repositoryName.getPathUnderExecRoot());
      FileSystemUtils.createDirectoryAndParents(repositoryExecRoot);
      dirloop:
      for (Path p : repositoryExecRoot.getDirectoryEntries()) {
        String name = p.getBaseName();
        for (String prefix : prefixes) {
          if (name.startsWith(prefix)) {
            continue dirloop;
          }
        }
        FileSystemUtils.deleteTree(p);
      }
    }
  }

  private boolean isPackage(PackageIdentifier pkgId) {
    return packageRoots.containsKey(pkgId);
  }

  /**
   * Finds the nearest ancestor package.
   */
  @Nullable
  private PackageIdentifier findParentPackage(PackageIdentifier pkgId) {
    return longestPathPrefix(pkgId, packageRoots.keySet());
  }

  void plantSymlinkForest() throws IOException {
    deleteTreesBelowNotPrefixed();

    // Create a sorted map of all dirs (packages and their ancestors) to sets of their roots.
    // Packages come from exactly one root, but their shared ancestors may come from more.
    // The map is maintained sorted lexicographically, so parents are before their children.
    Map<PackageIdentifier, Set<Path>> dirRootsMap = Maps.newTreeMap();
    for (Map.Entry<PackageIdentifier, Path> entry : packageRoots.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      Path pkgRoot = entry.getValue();
      for (int i = 1; i <= pkgId.getPackageFragment().segmentCount(); i++) {
        if (pkgId.equals(Label.EXTERNAL_PACKAGE_IDENTIFIER)) {
          // This isn't a "real" package, don't add it to the symlink tree.
          continue;
        }
        PackageIdentifier dir = createInRepo(pkgId, pkgId.getPackageFragment().subFragment(0, i));
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
      PackageIdentifier dir = entry.getKey();
      if (!isPackage(dir)) {
        PackageIdentifier parentPackage = findParentPackage(dir);
        if (parentPackage != null) {
          entry.getValue().add(packageRoots.get(parentPackage));
        }
      }
    }
    // Create output dirs for all dirs that have more than one root and need to be split.
    for (Map.Entry<PackageIdentifier, Set<Path>> entry : dirRootsMap.entrySet()) {
      PackageIdentifier dir = entry.getKey();
      // Handle creating top level directories for external repositories here, too.
      if (!dir.getRepository().isMain()) {
        FileSystemUtils.createDirectoryAndParents(
            workspace.getRelative(dir.getRepository().getPathUnderExecRoot()));
      }
      if (entry.getValue().size() > 1) {
        if (LOG_FINER) {
          log.finer("mkdir " + workspace.getRelative(dir.getPathUnderExecRoot()));
        }
        FileSystemUtils.createDirectoryAndParents(
            workspace.getRelative(dir.getPathUnderExecRoot()));
      }
    }

    // Make dir links for single rooted dirs.
    for (Map.Entry<PackageIdentifier, Set<Path>> entry : dirRootsMap.entrySet()) {
      PackageIdentifier dir = entry.getKey();
      Set<Path> roots = entry.getValue();
      // Simple case of one root for this dir.
      if (roots.size() == 1) {
        if (dir.getPackageFragment().segmentCount() > 1
            && dirRootsMap.get(getParent(dir)).size() == 1) {
          continue;  // skip--an ancestor will link this one in from above
        }
        // This is the top-most dir that can be linked to a single root. Make it so.
        Path root = roots.iterator().next();  // lone root in set
        if (LOG_FINER) {
          log.finer("ln -s " + root.getRelative(dir.getPackageFragment()) + " "
              + workspace.getRelative(dir.getPathUnderExecRoot()));
        }
        workspace.getRelative(dir.getPathUnderExecRoot())
            .createSymbolicLink(root.getRelative(dir.getPackageFragment()));
      }
    }
    // Make links for dirs within packages, skip parent-only dirs.
    for (Map.Entry<PackageIdentifier, Set<Path>> entry : dirRootsMap.entrySet()) {
      PackageIdentifier child = entry.getKey();
      if (entry.getValue().size() > 1) {
        // If this dir is at or below a package dir, link in its contents.
        PackageIdentifier parent = longestPathPrefix(child, packageRoots.keySet());
        linkDirectoryEntries(parent, child, dirRootsMap);
      }
    }

    for (Map.Entry<PackageIdentifier, Path> entry : packageRoots.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      if (!pkgId.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
        continue;
      }
      Path execrootDirectory = workspace.getRelative(pkgId.getPathUnderExecRoot());
      // If there were no subpackages, this directory might not exist yet.
      if (!execrootDirectory.exists()) {
        FileSystemUtils.createDirectoryAndParents(execrootDirectory);
      }
      // For the top-level directory, generate symlinks to everything in the directory instead of
      // the directory itself.
      for (Path target : entry.getValue().getDirectoryEntries()) {
        String baseName = target.getBaseName();
        Path execPath = execrootDirectory.getRelative(baseName);
        // Create any links that don't exist yet and don't start with bazel-.
        if (!baseName.startsWith(productName + "-") && !execPath.exists(Symlinks.NOFOLLOW)) {
          execPath.createSymbolicLink(target);
        }
      }
    }

    // Create the external/workspace directory.
    if (legacyExternalRunfiles) {
      workspace.getRelative(Label.EXTERNAL_PACKAGE_NAME).createSymbolicLink(
          workspace.getRelative(Label.EXTERNAL_PATH_PREFIX));
    }
    symlinkCorrectWorkspaceName();
  }

  private void linkDirectoryEntries(
      PackageIdentifier parent, PackageIdentifier child,
      Map<PackageIdentifier, Set<Path>> dirRootsMap) {
    if (parent == null) {
      // No parent package in packageRoots.
      return;
    }
    Path root = packageRoots.get(parent);
    try {
      Path absdir = root.getRelative(child.getPackageFragment());
      if (absdir.isDirectory()) {
        if (LOG_FINER) {
          log.finer("ln -s " + absdir + "/* "
              + workspace.getRelative(child.getPathUnderExecRoot()) + "/");
        }
        for (Path target : absdir.getDirectoryEntries()) {
          PathFragment p = child.getPackageFragment().getRelative(target.getBaseName());
          if (!dirRootsMap.containsKey(createInRepo(parent, p))) {
            PathFragment execFragment = child.getPathUnderExecRoot()
                .getRelative(target.getBaseName());
            workspace.getRelative(execFragment).createSymbolicLink(target);
          }
        }
      } else {
        log.fine("Symlink planting skipping dir '" + absdir + "'");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    // Otherwise its just an otherwise empty common parent dir.
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
    Path correctDirectory = workspace.getParentDirectory().getRelative(workspaceName);
    if (!correctDirectory.exists()) {
      correctDirectory.createSymbolicLink(workspace);
    }
  }

  private static PackageIdentifier getParent(PackageIdentifier packageIdentifier) {
    Preconditions.checkArgument(
        packageIdentifier.getPackageFragment().getParentDirectory() != null);
    return createInRepo(
        packageIdentifier, packageIdentifier.getPackageFragment().getParentDirectory());
  }

  private static PackageIdentifier createInRepo(
      PackageIdentifier repo, PathFragment packageFragment) {
    return PackageIdentifier.create(repo.getRepository(), packageFragment);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private boolean legacyExternalRunfiles = false;
    private ImmutableMap<PackageIdentifier, Path> packageRoots = ImmutableMap.of();
    private Path workspace;
    private String productName;
    private String[] prefixes;
    private String workspaceName;

    Builder setLegacyExternalRunfiles(boolean legacyExternalRunfiles) {
      this.legacyExternalRunfiles = legacyExternalRunfiles;
      return this;
    }

    Builder setPackageRoots(ImmutableMap<PackageIdentifier, Path> packageRoots) {
      this.packageRoots = packageRoots;
      return this;
    }

    Builder setWorkspace(Path workspace) {
      this.workspace = workspace;
      return this;
    }

    Builder setProductName(String productName) {
      this.productName = productName;
      this.prefixes = new String[] { ".", "_", productName + "-"};
      return this;
    }

    Builder setPrefixes(String[] prefixes) {
      this.prefixes = prefixes;
      return this;
    }

    Builder setWorkspaceName(String workspaceName) {
      this.workspaceName = workspaceName;
      return this;
    }

    public SymlinkForest build() {
      Preconditions.checkState(workspace != null);
      Preconditions.checkState(productName != null);
      return new SymlinkForest(
          legacyExternalRunfiles, packageRoots, workspace, productName, prefixes, workspaceName);
    }
  }
}
