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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Static utilities for managing output directory symlinks.
 */
public class OutputDirectoryLinksUtils {
  // Used in getPrettyPath() method below.
  private static final String[] LINKS = { "bin", "genfiles", "includes" };

  private static final String NO_CREATE_SYMLINKS_PREFIX = "/";

  public static ImmutableList<String> getOutputSymlinkNames(String productName,
      String symlinkPrefix) {
    ImmutableList.Builder<String> builder = ImmutableList.<String>builder();
    // TODO(b/35234395): This symlink is created for backwards compatiblity, remove it once
    // we're sure it won't cause any other issues.
    builder.add(productName + "-out");
    if (!productName.equals(symlinkPrefix)) {
      builder.add(symlinkPrefix + "out");
    }
    return builder.build();
  }

  private static String execRootSymlink(String symlinkPrefix, String workspaceName) {
    return symlinkPrefix + workspaceName;
  }
  /**
   * Attempts to create convenience symlinks in the workspaceDirectory and in
   * execRoot to the output area and to the configuration-specific output
   * directories. Issues a warning if it fails, e.g. because workspaceDirectory
   * is readonly.
   */
  static void createOutputDirectoryLinks(String workspaceName,
      Path workspace, Path execRoot, Path outputPath,
      EventHandler eventHandler, @Nullable BuildConfiguration targetConfig,
      String symlinkPrefix, String productName) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }
    List<String> failures = new ArrayList<>();

    // Make the two non-specific links from the workspace to the output area,
    // and the configuration-specific links in both the workspace and the execution root dirs.
    // IMPORTANT: Keep in sync with removeOutputDirectoryLinks below.
    for (String outputSymlinkName : getOutputSymlinkNames(productName, symlinkPrefix)) {
      createLink(workspace, outputSymlinkName, outputPath, failures);
    }

    // Points to execroot
    createLink(workspace, execRootSymlink(
        symlinkPrefix, workspace.getBaseName()), execRoot, failures);
    RepositoryName repositoryName = RepositoryName.createFromValidStrippedName(workspaceName);
    if (targetConfig != null) {
      createLink(workspace, symlinkPrefix + "bin",
          targetConfig.getBinDirectory(repositoryName).getPath(), failures);
      createLink(workspace, symlinkPrefix + "testlogs",
          targetConfig.getTestLogsDirectory(repositoryName).getPath(), failures);
      createLink(workspace, symlinkPrefix + "genfiles",
          targetConfig.getGenfilesDirectory(repositoryName).getPath(), failures);
    }

    if (!failures.isEmpty()) {
      eventHandler.handle(Event.warn(String.format(
          "failed to create one or more convenience symlinks for prefix '%s':\n  %s",
          symlinkPrefix, Joiner.on("\n  ").join(failures))));
    }
  }

  /**
   * Returns a convenient path to the specified file, relativizing it and using output-dir symlinks
   * if possible.  Otherwise, return a path relative to the workspace directory if possible.
   * Otherwise, return the absolute path.
   *
   * <p>This method must be called after the symlinks are created at the end of a build. If called
   * before, the pretty path may be incorrect if the symlinks end up pointing somewhere new.
   */
  public static PathFragment getPrettyPath(Path file, String workspaceName,
      Path workspaceDirectory, String symlinkPrefix, String productName) {
    for (String link : LINKS) {
      PathFragment result = relativize(file, workspaceDirectory, symlinkPrefix + link);
      if (result != null) {
        return result;
      }
    }

    PathFragment result = relativize(file, workspaceDirectory,
        execRootSymlink(symlinkPrefix, workspaceName));
    if (result != null) {
      return result;
    }

    ImmutableList<String> outputSymlinkNames = getOutputSymlinkNames(productName, symlinkPrefix);
    checkArgument(!outputSymlinkNames.isEmpty());
    result = relativize(file, workspaceDirectory, outputSymlinkNames.get(0));
    if (result != null) {
      return result;
    }

    return file.asFragment();
  }

  // Helper to getPrettyPath.  Returns file, relativized w.r.t. the referent of
  // "linkname", or null if it was a not a child.
  private static PathFragment relativize(Path file, Path workspaceDirectory, String linkname) {
    PathFragment link = PathFragment.create(linkname);
    try {
      Path dir = workspaceDirectory.getRelative(link);
      PathFragment levelOneLinkTarget = dir.readSymbolicLink();
      if (levelOneLinkTarget.isAbsolute() &&
          file.startsWith(dir = file.getRelative(levelOneLinkTarget))) {
        return link.getRelative(file.relativeTo(dir));
      }
    } catch (IOException e) {
      /* ignore */
    }
    return null;
  }

  /**
   * Attempts to remove the convenience symlinks in the workspace directory.
   *
   * <p>Issues a warning if it fails, e.g. because workspaceDirectory is readonly.
   * Also cleans up any child directories created by a custom prefix.
   *
   * @param workspace the runtime's workspace
   * @param eventHandler the error eventHandler
   * @param symlinkPrefix the symlink prefix which should be removed
   * @param productName the product name
   */
  public static void removeOutputDirectoryLinks(String workspaceName, Path workspace,
      EventHandler eventHandler, String symlinkPrefix, String productName) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return;
    }
    List<String> failures = new ArrayList<>();

    for (String outputSymlinkName : getOutputSymlinkNames(productName, symlinkPrefix)) {
      removeLink(workspace, outputSymlinkName, failures);
    }
    removeLink(workspace, execRootSymlink(symlinkPrefix, workspaceName), failures);
    removeLink(workspace, execRootSymlink(symlinkPrefix, workspace.getBaseName()), failures);
    removeLink(workspace, symlinkPrefix + "bin", failures);
    removeLink(workspace, symlinkPrefix + "testlogs", failures);
    removeLink(workspace, symlinkPrefix + "genfiles", failures);
    FileSystemUtils.removeDirectoryAndParents(workspace, PathFragment.create(symlinkPrefix));
    if (!failures.isEmpty()) {
      eventHandler.handle(Event.warn(String.format(
          "failed to remove one or more convenience symlinks for prefix '%s':\n  %s", symlinkPrefix,
          Joiner.on("\n  ").join(failures))));
    }
  }

  /**
   * Helper to createOutputDirectoryLinks that creates a symlink from base + name to target.
   */
  private static boolean createLink(Path base, String name, Path target, List<String> failures) {
    try {
      FileSystemUtils.createDirectoryAndParents(target);
    } catch (IOException e) {
      failures.add(String.format("cannot create directory %s: %s",
          target.getPathString(), e.getMessage()));
      return false;
    }
    try {
      FileSystemUtils.ensureSymbolicLink(base.getRelative(name), target);
    } catch (IOException e) {
      failures.add(String.format("cannot create symbolic link %s -> %s:  %s",
          name, target.getPathString(), e.getMessage()));
      return false;
    }

    return true;
  }

  /**
   * Helper to removeOutputDirectoryLinks that removes one of the Blaze convenience symbolic links.
   */
  private static boolean removeLink(Path base, String name, List<String> failures) {
    Path link = base.getRelative(name);
    try {
      if (link.exists(Symlinks.NOFOLLOW)) {
        ExecutionTool.logger.finest("Removing " + link);
        link.delete();
      }
      return true;
    } catch (IOException e) {
      failures.add(String.format("%s: %s", name, e.getMessage()));
      return false;
    }
  }
}
