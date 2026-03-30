// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collection;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** Utility methods related to repo fetching. */
public class RepositoryUtils {

  public static final String WORKSPACE_SYMLINK_NAME = "_main";

  private RepositoryUtils() {}

  public static boolean isValidRepoRoot(Path directory) {
    // Keep in sync with //src/main/cpp/workspace_layout.h
    return directory.getRelative(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.WORKSPACE_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME).exists()
        || directory.getRelative(LabelConstants.REPO_FILE_NAME).exists();
  }

  @Nullable
  public static RootedPath getRootedPathFromLabel(Label label, Environment env)
      throws InterruptedException, EvalException {
    SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
    if (pkgLookupValue == null) {
      return null;
    }
    if (!pkgLookupValue.packageExists()) {
      String message = pkgLookupValue.getErrorMsg();
      if (pkgLookupValue == PackageLookupValue.NO_BUILD_FILE_VALUE) {
        message = PackageLookupFunction.explainNoBuildFileValue(label.getPackageIdentifier(), env);
      }
      throw Starlark.errorf("Unable to load package for %s: %s", label, message);
    }

    // And now for the file
    Root packageRoot = pkgLookupValue.getRoot();
    return RootedPath.toRootedPath(packageRoot, label.toPathFragment());
  }

  protected static Path getExternalRepositoryDirectory(BlazeDirectories directories) {
    return directories.getOutputBase().getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
  }

  /**
   * Replants the symlinks under the specified repository directory.
   *
   * <p>Re-writes symlinks that originally point to a path under the workspace or the external root
   * to relative paths. Same-repo symlinks (those pointing back into {@code repoDir}) are replanted
   * to repo-relative paths that work regardless of the repo's physical location. Cross-repo
   * symlinks are replanted to relative paths going through {@code repoDirParentToExternalRoot}.
   *
   * @param repoDir The path to the repository directory.
   * @param workspace The path to the workspace directory.
   * @param externalRepoRoot The path to the root of external repositories.
   * @param repoDirParentToExternalRoot The relative path from the parent directory of {@code
   *     repoDir} to the external root (or a symlink to it), which is used to rewrite cross-repo
   *     symlink targets to relative paths.
   * @return {@code true} if it is safe to reuse the replanted repo from a different physical
   *     location (e.g. a shared repo contents cache). This is the case when there are no cross-repo
   *     symlinks or on Windows with symlinks enabled (where symlinks resolve using logical rather
   *     than physical paths).
   * @throws IOException If an I/O error occurs while replanting the symlinks.
   */
  @CanIgnoreReturnValue
  public static boolean replantSymlinks(
      Path repoDir, Path workspace, Path externalRepoRoot, PathFragment repoDirParentToExternalRoot)
      throws IOException {
    boolean portableSymlinksOnly = true;
    try {
      Collection<Path> symlinks = FileSystemUtils.traverseTree(repoDir, Path::isSymbolicLink);
      Path workspaceSymlinkUnderExternal = externalRepoRoot.getChild(WORKSPACE_SYMLINK_NAME);
      FileSystemUtils.ensureSymbolicLink(workspaceSymlinkUnderExternal, workspace);
      for (Path symlink : symlinks) {
        PathFragment target = symlink.readSymbolicLink();
        PathFragment originalTarget = target;
        // Treat symlinks pointing into the workspace just like cross-repo symlinks by adding a
        // symlink for the main repo to the external root and rewriting symlinks to it.
        if (target.startsWith(workspace.asFragment())) {
          target =
              workspaceSymlinkUnderExternal
                  .asFragment()
                  .getRelative(target.relativeTo(workspace.asFragment()));
        }
        if (!target.startsWith(externalRepoRoot.asFragment())) {
          // This symlink doesn't point into any Bazel repo, including the main repo, and thus its
          // target isn't managed by Bazel. We assume such symlinks are portable across machines
          // on which the repo is relevant (e.g. /lib/ld-linux.so* or /usr/bin/ld).
          continue;
        }
        PathFragment newTarget;
        if (target.startsWith(repoDir.asFragment())) {
          // Same-repo symlink: replant relative within the repo. This is always safe regardless
          // of where the repo is physically located.
          PathFragment targetRelativeToRepo = target.relativeTo(repoDir.asFragment());
          int depth = symlink.relativeTo(repoDir).segmentCount() - 1;
          newTarget = PathFragment.create("../".repeat(depth)).getRelative(targetRelativeToRepo);
        } else {
          // Cross-repo symlink: replant through the parent directory. On Unix, these relative paths
          // only resolve correctly when the repo is at its original location (under the external
          // root), not when moved to a shared cache. This is because symlink resolution is based
          // on physical paths. On Windows, where symlinks are resolved using logical paths, it may
          // be possible to use these symlinks portably, but this would likely require changes to
          // FileFunction to mimic this resolution behavior.
          portableSymlinksOnly = false;
          // Rewrite for consistency even if not portable. A mix of absolute and relative symlinks
          // would result in less predictable behavior and reduced test coverage.
          newTarget =
              PathFragment.create(
                      "../"
                          .repeat(
                              symlink.relativeTo(repoDir.getParentDirectory()).segmentCount() - 1))
                  .getRelative(repoDirParentToExternalRoot)
                  .getRelative(target.relativeTo(externalRepoRoot.asFragment()));
        }
        if (OS.getCurrent() == OS.WINDOWS) {
          // On Windows, FileSystemUtils.ensureSymbolicLink always resolves paths to absolute
          // paths. Use Files.createSymbolicLink here instead to preserve the relative target path.
          var symlinkNioPath = symlink.getFileSystem().getNioPath(symlink.asFragment());
          var newTargetNioPath = symlink.getFileSystem().getNioPath(newTarget);
          if (symlinkNioPath == null || newTargetNioPath == null) {
            portableSymlinksOnly = false;
            continue;
          }
          symlink.delete();
          try {
            Files.createSymbolicLink(symlinkNioPath, newTargetNioPath);
          } catch (IOException e) {
            // Creating a real symlink failed (likely no Developer Mode or symlink privilege).
            // Restore the original symlink/junction and mark as non-portable.
            FileSystemUtils.ensureSymbolicLink(symlink, originalTarget);
            portableSymlinksOnly = false;
          }
        } else {
          FileSystemUtils.ensureSymbolicLink(symlink, newTarget);
        }
      }
    } catch (IOException e) {
      throw new IOException(
          String.format("Failed to rewrite symlinks under %s: %s", repoDir, e.getMessage()), e);
    }
    return portableSymlinksOnly;
  }
}
