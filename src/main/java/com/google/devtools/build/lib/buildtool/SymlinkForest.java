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
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.SymlinkForest.Code;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

/** Creates a symlink forest based on a package path map. */
public class SymlinkForest {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

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
      ImmutableList.Builder<Path> plantedSymlinks,
      RepositoryName repository,
      Path source,
      Set<Path> externalRepoLinks)
      throws IOException {
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
      execroot.getRelative(LabelConstants.EXTERNAL_PATH_PREFIX).createDirectoryAndParents();
    }
    if (!externalRepoLinks.add(execrootLink)) {
      return;
    }
    execrootLink.createSymbolicLink(source);
    plantedSymlinks.add(execrootLink);
  }

  private void plantSymlinkForestWithFullMainRepository(
      ImmutableList.Builder<Path> plantedSymlinks, Path mainRepoRoot) throws IOException {
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
        plantedSymlinks.add(execPath);
        // TODO(jingwen-external): is this creating execroot/io_bazel/external?
      }
    }
  }

  private void plantSymlinkForestWithPartialMainRepository(
      ImmutableList.Builder<Path> plantedSymlinks, Map<Path, Path> mainRepoLinks)
      throws IOException, AbruptExitException {
    if (siblingRepositoryLayout) {
      execroot.createDirectory();
    }
    for (Map.Entry<Path, Path> entry : mainRepoLinks.entrySet()) {
      Path link = entry.getKey();
      Path target = entry.getValue();
      if (this.notSymlinkedInExecrootDirectories.contains(target.getBaseName())) {
        throw new AbruptExitException(
            detailedSymlinkForestExitCode(
                "Directories specified with toplevel_output_directories should be ignored and can"
                    + " not be used as sources.",
                Code.TOPLEVEL_OUTDIR_USED_AS_SOURCE,
                ExitCode.COMMAND_LINE_ERROR));
      }
      link.createSymbolicLink(target);
      plantedSymlinks.add(link);
    }
  }

  private void plantSymlinkForestMultiPackagePath(
      ImmutableList.Builder<Path> plantedSymlinks,
      Map<PackageIdentifier, Root> packageRootsForMainRepo)
      throws IOException {
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
        logger.atFiner().log(
            "mkdir %s", execroot.getRelative(dir.getExecPath(siblingRepositoryLayout)));
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
          continue; // skip--an ancestor will link this one in from above
        }
        // This is the top-most dir that can be linked to a single root. Make it so.
        Root root = roots.iterator().next(); // lone root in set
        Path link = execroot.getRelative(dir.getExecPath(siblingRepositoryLayout));
        logger.atFiner().log("ln -s %s %s", root.getRelative(dir.getSourceRoot()), link);
        link.createSymbolicLink(root.getRelative(dir.getSourceRoot()));
        plantedSymlinks.add(link);
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
              logger.atFiner().log(
                  "ln -s %s/* %s/", absdir, execroot.getRelative(dir.getSourceRoot()));
              for (Path target : absdir.getDirectoryEntries()) {
                PathFragment p = root.relativize(target);
                if (!dirRootsMap.containsKey(createInRepo(pkgId, p))) {
                  execroot.getRelative(p).createSymbolicLink(target);
                  plantedSymlinks.add(execroot.getRelative(p));
                }
              }
            } else {
              logger.atFine().log("Symlink planting skipping dir '%s'", absdir);
            }
          } catch (IOException e) {
            // TODO(arostovtsev): Why are we swallowing the IOException here instead of letting it
            // be thrown?
            logger.atWarning().withCause(e).log(
                "I/O error while planting symlinks to contents of '%s'",
                root.getRelative(dir.getSourceRoot()));
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
          plantedSymlinks.add(execPath);
        }
      }
    }
  }

  /**
   * Performs the filesystem operations to plant the symlink forest.
   *
   * @return the symlinks that have been planted
   */
  public ImmutableList<Path> plantSymlinkForest() throws IOException, AbruptExitException {
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
    ImmutableList.Builder<Path> plantedSymlinks = ImmutableList.builder();

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
            plantedSymlinks,
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
            detailedSymlinkForestExitCode(
                "toplevel_output_directories is not supported together with --package_path option.",
                Code.TOPLEVEL_OUTDIR_PACKAGE_PATH_CONFLICT,
                ExitCode.COMMAND_LINE_ERROR));
      }
      plantSymlinkForestMultiPackagePath(plantedSymlinks, packageRootsForMainRepo);
    } else if (shouldLinkAllTopLevelItems) {
      Path mainRepoRoot = Iterables.getOnlyElement(mainRepoRoots).asPath();
      plantSymlinkForestWithFullMainRepository(plantedSymlinks, mainRepoRoot);
    } else {
      plantSymlinkForestWithPartialMainRepository(plantedSymlinks, mainRepoLinks);
    }

    logger.atInfo().log("Planted symlink forest in %s", execroot);
    return plantedSymlinks.build();
  }

  private static DetailedExitCode detailedSymlinkForestExitCode(
      String message, Code code, ExitCode exitCode) {
    return DetailedExitCode.of(
        exitCode,
        FailureDetail.newBuilder()
            .setMessage(message)
            .setSymlinkForest(FailureDetails.SymlinkForest.newBuilder().setCode(code))
            .build());
  }

  private static PackageIdentifier createInRepo(
      PackageIdentifier repo, PathFragment packageFragment) {
    return PackageIdentifier.create(repo.getRepository(), packageFragment);
  }
}
