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
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
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

/** Creates a symlink forest based on a package path map. */
public class SymlinkForest {

  private static final Logger logger = Logger.getLogger(SymlinkForest.class.getName());
  private static final boolean LOG_FINER = logger.isLoggable(Level.FINER);

  private final ImmutableMap<PackageIdentifier, Root> packageRoots;
  private final Path execroot;
  private final String productName;
  private final String prefix;
  private final ImmutableSortedSet<String> notSymlinkedInExecrootDirectories;
  private final boolean siblingRepositoryLayout;

  /** Constructor for a symlink forest creator without non-symlinked directories parameter. */
  public SymlinkForest(
      ImmutableMap<PackageIdentifier, Root> packageRoots, Path execroot, String productName) {
    this(packageRoots, execroot, productName, ImmutableSortedSet.of(), false);
  }

  /**
   * Constructor for a symlink forest creator; does not perform any i/o.
   *
   * <p>Use {@link #plantSymlinkForest()} to actually create the symlink forest.
   *
   * @param packageRoots source package roots to which to create symlinks
   * @param execroot path where to plant the symlink forest
   * @param productName {@code BlazeRuntime#getProductName()}
   * @param notSymlinkedInExecrootDirectories directories to not symlink in exec root. {@link
   *     com.google.devtools.build.lib.packages.WorkspaceGlobals#dontSymlinkDirectoriesInExecroot(Sequence,
   *     Location, StarlarkThread)}
   */
  public SymlinkForest(
      ImmutableMap<PackageIdentifier, Root> packageRoots,
      Path execroot,
      String productName,
      ImmutableSortedSet<String> notSymlinkedInExecrootDirectories,
      boolean siblingRepositoryLayout) {
    this.packageRoots = packageRoots;
    this.execroot = execroot;
    this.productName = productName;
    this.prefix = productName + "-";
    this.notSymlinkedInExecrootDirectories = notSymlinkedInExecrootDirectories;
    this.siblingRepositoryLayout = siblingRepositoryLayout;
  }

  /**
   * Returns the longest prefix from a given set of 'prefixes' that are contained in 'path'. I.e the
   * closest ancestor directory containing path. Returns null if none found.
   *
   * @param path
   * @param prefixes
   */
  @VisibleForTesting
  static PackageIdentifier longestPathPrefix(
      PackageIdentifier path, Set<PackageIdentifier> prefixes) {
    for (int i = path.getPackageFragment().segmentCount(); i >= 0; i--) {
      PackageIdentifier prefix = createInRepo(path, path.getPackageFragment().subFragment(0, i));
      if (prefixes.contains(prefix)) {
        return prefix;
      }
    }
    return null;
  }

  /**
   * Delete all dir trees under a given 'dir' that don't start with a given 'prefix', and is not
   * special case of not symlinked to exec root directories (those directories are special case of
   * output roots, so they must be kept before commands). Does not follow any symbolic links.
   */
  @VisibleForTesting
  @ThreadSafety.ThreadSafe
  void deleteTreesBelowNotPrefixed(Path dir, String prefix) throws IOException {
    for (Path p : dir.getDirectoryEntries()) {
      if (!p.getBaseName().startsWith(prefix)
          && !notSymlinkedInExecrootDirectories.contains(p.getBaseName())) {
        p.deleteTree();
      }
    }
  }

  private void plantSymlinkForExternalRepo(
      RepositoryName repository, Path source, Set<Path> externalRepoLinks) throws IOException {
    // For external repositories, create one symlink to each external repository
    // directory.
    // From <output_base>/execroot/<main repo name>/external/<external repo name>
    // to   <output_base>/external/<external repo name>
    //
    // However, if --experimental_sibling_repository_layout is true, symlink:
    // From <output_base>/execroot/<external repo name>
    // to   <output_base>/external/<external repo name>
    Path execrootLink = execroot.getRelative(repository.getExecPath(siblingRepositoryLayout));

    if (!siblingRepositoryLayout && externalRepoLinks.isEmpty()) {
      execroot.getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME).createDirectoryAndParents();
    }
    if (!externalRepoLinks.add(execrootLink)) {
      return;
    }
    execrootLink.createSymbolicLink(source);
  }

  private void plantSymlinkForestWithFullMainRepository(Path mainRepoRoot) throws IOException {
    // For the main repo top-level directory, generate symlinks to everything in the directory
    // instead of the directory itself.
    if (siblingRepositoryLayout) {
      execroot.createDirectory();
    }
    for (Path target : mainRepoRoot.getDirectoryEntries()) {
      String baseName = target.getBaseName();
      if (this.notSymlinkedInExecrootDirectories.contains(baseName)) {
        continue;
      }
      Path execPath = execroot.getRelative(baseName);
      // Create any links that don't start with bazel-, and ignore external/ directory if
      // user has it in the source tree because it conflicts with external repository location.
      if (!baseName.startsWith(prefix)
          && (siblingRepositoryLayout
              || !baseName.equals(LabelConstants.EXTERNAL_PATH_PREFIX.getBaseName()))) {
        execPath.createSymbolicLink(target);
        // TODO(jingwen-external): is this creating execroot/io_bazel/external?
      }
    }
  }

  private void plantSymlinkForestWithPartialMainRepository(Map<Path, Path> mainRepoLinks)
      throws IOException, AbruptExitException {
    if (siblingRepositoryLayout) {
      execroot.createDirectory();
    }
    for (Map.Entry<Path, Path> entry : mainRepoLinks.entrySet()) {
      Path link = entry.getKey();
      Path target = entry.getValue();
      if (this.notSymlinkedInExecrootDirectories.contains(target.getBaseName())) {
        throw new AbruptExitException(
            "Directories specified with "
                + "toplevel_output_directories should be ignored and can not be used"
                + " as sources.",
            ExitCode.COMMAND_LINE_ERROR);
      }
      link.createSymbolicLink(target);
    }
  }

  private void plantSymlinkForestMultiPackagePath(
      Map<PackageIdentifier, Root> packageRootsForMainRepo) throws IOException {
    // Packages come from exactly one root, but their shared ancestors may come from more.
    Map<PackageIdentifier, Set<Root>> dirRootsMap = Maps.newHashMap();
    // Elements in this list are added so that parents come before their children.
    ArrayList<PackageIdentifier> dirsParentsFirst = new ArrayList<>();
    for (Map.Entry<PackageIdentifier, Root> entry : packageRootsForMainRepo.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
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
      if (!packageRootsForMainRepo.containsKey(dir)) {
        PackageIdentifier pkgId = longestPathPrefix(dir, packageRootsForMainRepo.keySet());
        if (pkgId != null) {
          dirRootsMap.get(dir).add(packageRootsForMainRepo.get(pkgId));
        }
      }
    }
    // Create output dirs for all dirs that have more than one root and need to be split.
    for (PackageIdentifier dir : dirsParentsFirst) {
      if (!dir.getRepository().isMain()) {
        execroot
            .getRelative(dir.getRepository().getExecPath(siblingRepositoryLayout))
            .createDirectoryAndParents();
      }
      if (dirRootsMap.get(dir).size() > 1) {
        if (LOG_FINER) {
          logger.finer("mkdir " + execroot.getRelative(dir.getExecPath(siblingRepositoryLayout)));
        }
        execroot.getRelative(dir.getExecPath(siblingRepositoryLayout)).createDirectoryAndParents();
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
                  + execroot.getRelative(dir.getExecPath(siblingRepositoryLayout)));
        }
        execroot
            .getRelative(dir.getExecPath(siblingRepositoryLayout))
            .createSymbolicLink(root.getRelative(dir.getSourceRoot()));
      }
    }
    // Make links for dirs within packages, skip parent-only dirs.
    for (PackageIdentifier dir : dirsParentsFirst) {
      if (dirRootsMap.get(dir).size() > 1) {
        // If this dir is at or below a package dir, link in its contents.
        PackageIdentifier pkgId = longestPathPrefix(dir, packageRootsForMainRepo.keySet());
        if (pkgId != null) {
          Root root = packageRootsForMainRepo.get(pkgId);
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

    for (Map.Entry<PackageIdentifier, Root> entry : packageRootsForMainRepo.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      if (!pkgId.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
        continue;
      }
      Path execrootDirectory = execroot.getRelative(pkgId.getExecPath(siblingRepositoryLayout));
      // If there were no subpackages, this directory might not exist yet.
      if (!execrootDirectory.exists()) {
        execrootDirectory.createDirectoryAndParents();
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
  }

  /** Performs the filesystem operations to plant the symlink forest. */
  public void plantSymlinkForest() throws IOException, AbruptExitException {
    deleteTreesBelowNotPrefixed(execroot, prefix);

    if (siblingRepositoryLayout) {
      // Delete execroot/../<symlinks> to directories representing external repositories.
      for (Path p : execroot.getParentDirectory().getDirectoryEntries()) {
        if (p.isSymbolicLink()) {
          p.deleteTree();
        }
      }
    }

    boolean shouldLinkAllTopLevelItems = false;
    Map<Path, Path> mainRepoLinks = Maps.newLinkedHashMap();
    Set<Root> mainRepoRoots = Sets.newLinkedHashSet();
    Set<Path> externalRepoLinks = Sets.newLinkedHashSet();
    Map<PackageIdentifier, Root> packageRootsForMainRepo = Maps.newLinkedHashMap();

    for (Map.Entry<PackageIdentifier, Root> entry : packageRoots.entrySet()) {
      PackageIdentifier pkgId = entry.getKey();
      if (pkgId.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
        // //external is a virtual package regardless , don't add it to the symlink tree.
        // Subpackages of
        // external, like //external/foo, are fine though.
        continue;
      }
      RepositoryName repository = pkgId.getRepository();
      if (repository.isMain() || repository.isDefault()) {
        // Record main repo packages.
        packageRootsForMainRepo.put(entry.getKey(), entry.getValue());

        // Record the root of the packages.
        mainRepoRoots.add(entry.getValue());

        // For single root (single package path) case:
        // If root package of the main repo is required, we record the main repo root so that
        // we can later link everything under the main repo's top-level directory.
        // If root package of the main repo is not required, we only record links for
        // directories under the top-level directory that are used in required packages.
        if (pkgId.getPackageFragment().equals(PathFragment.EMPTY_FRAGMENT)) {
          shouldLinkAllTopLevelItems = true;
        } else {
          String baseName = pkgId.getPackageFragment().getSegment(0);
          if (!siblingRepositoryLayout
              && baseName.equals(LabelConstants.EXTERNAL_PATH_PREFIX.getBaseName())) {
            // ignore external/ directory if user has it in the source tree
            // because it conflicts with external repository location.
            continue;
          }
          Path execrootLink = execroot.getRelative(baseName);
          Path sourcePath = entry.getValue().getRelative(pkgId.getSourceRoot().getSegment(0));
          mainRepoLinks.putIfAbsent(execrootLink, sourcePath);
        }
      } else {
        plantSymlinkForExternalRepo(
            repository,
            entry.getValue().getRelative(repository.getSourceRoot()),
            externalRepoLinks);
      }
    }

    // TODO(bazel-team): Bazel can find packages in multiple paths by specifying --package_paths,
    // we need a more complex algorithm to build execroot in that case. As --package_path will be
    // removed in the future, we should remove the plantSymlinkForestMultiPackagePath
    // implementation when --package_path is gone.
    if (mainRepoRoots.size() > 1) {
      if (!this.notSymlinkedInExecrootDirectories.isEmpty()) {
        throw new AbruptExitException(
            "toplevel_output_directories is "
                + "not supported together with --package_path option.",
            ExitCode.COMMAND_LINE_ERROR);
      }
      plantSymlinkForestMultiPackagePath(packageRootsForMainRepo);
    } else if (shouldLinkAllTopLevelItems) {
      Path mainRepoRoot = Iterables.getOnlyElement(mainRepoRoots).asPath();
      plantSymlinkForestWithFullMainRepository(mainRepoRoot);
    } else {
      plantSymlinkForestWithPartialMainRepository(mainRepoLinks);
    }

    logger.info("Planted symlink forest in " + execroot);
  }

  private static PackageIdentifier createInRepo(
      PackageIdentifier repo, PathFragment packageFragment) {
    return PackageIdentifier.create(repo.getRepository(), packageFragment);
  }
}
