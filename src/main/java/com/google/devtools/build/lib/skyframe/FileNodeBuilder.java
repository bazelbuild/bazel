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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * A {@link NodeBuilder} for {@link FileNode}s.
 *
 * <p>Most of the complexity in the implementation is associated to handling symlinks. Namely,
 * this class makes sure that {@code FileNode}s corresponding to symlinks are correctly invalidated
 * if the destination of the symlink is invalidated. Directory symlinks are also covered.
 */
public class FileNodeBuilder implements NodeBuilder {

  private final TimestampGranularityMonitor tsgm;
  private final AtomicReference<PathPackageLocator> pkgLocator;

  /**
   * Construct FileNodeBuilder.
   *
   * @param tsgm used to notify the system about timestamp granularity dependencies.
   * @param pkgLocator the package locator
   */
  public FileNodeBuilder(TimestampGranularityMonitor tsgm,
      AtomicReference<PathPackageLocator> pkgLocator) {
    this.tsgm = tsgm;
    this.pkgLocator = pkgLocator;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws FileNodeBuilderException {
    RootedPath rootedPath = (RootedPath) nodeKey.getNodeName();
    Path path = rootedPath.asPath();
    Path root = rootedPath.getRoot();
    PathFragment relativePath = rootedPath.getRelativePath();
    RootedPath realRootedPath = rootedPath;
    FileStatus stat = null;
    PathFragment symlinkTarget = null;
    FileNode parentNode = null;

    // Declare dependencies on the parent directory, but only if the current file is not the
    // filesystem root or a package path root. The former has no parent and the latter is treated
    // opaquely and handled by skyframe's DiffAwareness interface.
    if (!relativePath.equals(PathFragment.EMPTY_FRAGMENT)) {
      RootedPath parentRootedPath = RootedPath.toRootedPath(root,
          relativePath.getParentDirectory());
      // Unconditional dependency on the parent directory file.
      parentNode = (FileNode) env.getDep(FileNode.key(parentRootedPath));
      if (parentNode == null) {
        return null;
      }

      // If any ancestor is a symlink, mark dependency on the corresponding real file.
      if (!parentNode.realRootedPath().equals(parentNode.rootedPath())) {
        // Note that we don't take the real path of the file itself, but rather join the real path
        // of the parent directory and the filename. This is to ensure that if the file is a
        // symlink, we depend on the symlink itself rather than its target (in case the symlink
        // changes). For example, say "a/" is a directory, "a/f" is a file, "a/s" is a symlink
        // to "f" (i.e., "a/f"), and "b" is a symlink to "a". Then FileNode("b/s") needs to
        // depend on FileNode("a/s") and not just FileNode("a/f"), because if "a/s" changes,
        // "b/s" needs to be invalidated.
        PathFragment baseName = new PathFragment(relativePath.getBaseName());
        RootedPath parentRealRootedPath = parentNode.realRootedPath();
        RootedPath rootedPathToLookup = RootedPath.toRootedPath(parentRealRootedPath.getRoot(),
            parentRealRootedPath.getRelativePath().getRelative(baseName));
        FileNode realFileNode = (FileNode) env.getDep(FileNode.key(rootedPathToLookup));
        if (realFileNode == null) {
          return null;
        }
        realRootedPath = realFileNode.realRootedPath();
      }
    }

    // If the file is a symlink, mark dependency on the target of the symlink.
    try {
      stat = path.statIfFound(Symlinks.NOFOLLOW);
    } catch (IOException e) {
      throw new FileNodeBuilderException(nodeKey, e);
    }
    if (stat != null && stat.isSymbolicLink()) {
      // TODO(bazel-team): Disallow symlinks here that escape the containing package root or point
      // to an absolute path [skyframe-loading]
      try {
        symlinkTarget = path.readSymbolicLink();
      } catch (IOException exception) {
        throw new FileNodeBuilderException(nodeKey, exception);
      }
      Path symlinkPath;
      if (!relativePath.equals(PathFragment.EMPTY_FRAGMENT)) {
        Preconditions.checkNotNull(parentNode);
        symlinkPath = parentNode.realRootedPath().asPath().getRelative(symlinkTarget);
      } else {
        try {
          symlinkPath = path.resolveSymbolicLinks();
        } catch (IOException exception) {
          throw new FileNodeBuilderException(nodeKey, exception);
        }
      }

      // We need a dependency on the symlink target, which may be under a different package root.
      RootedPath symlinkRootedPath = toRootedPathMaybeUnderRoot(symlinkPath, pkgLocator.get());
      NodeKey symlinkNodeKey = FileNode.key(symlinkRootedPath);
      FileNode symlinkNode = (FileNode) env.getDep(symlinkNodeKey);
      if (symlinkNode == null) {
        return null;
      }
      realRootedPath = symlinkNode.realRootedPath();
    }

    FileNode fileNode;
    try {
      fileNode = FileNode.nodeForRootedPath(rootedPath, realRootedPath, tsgm,
          FileStatusWithDigestAdapter.adapt(stat), symlinkTarget);
    } catch (IOException e) {
      throw new FileNodeBuilderException(nodeKey, e);
    }

    if ((fileNode.isFile() || fileNode.isDirectory()) && parentNode != null &&
        !parentNode.isDirectory()) {
      String message = (fileNode.isFile() ? "file " : "directory ") + fileNode.rootedPath().asPath()
          + " exists but its parent directory " + parentNode.rootedPath().asPath()
          + " doesn't exist.";
      throw new FileNodeBuilderException(nodeKey, new InconsistentFilesystemException(message));
    }

    // For files outside the package roots, add a dependency on the build_id. This is sufficient
    // for correctness; all other files will be handled by diff awareness of their respective
    // package path, but these files need to be addressed separately.
    //
    // Using the build_id here seems to introduce a performance concern because the upward
    // transitive closure of these external files will get eagerly invalidated on each incremental
    // build (e.g. if every file had a transitive dependency on the filesystem root, then we'd have
    // a big performance problem). But this a non-issue by design:
    // - We don't add a dependency on the parent directory at the package root boundary, so the
    // only transitive dependencies from files inside the package roots to external files are
    // through symlinks. So the upwards transitive closure of external files is small.
    // - The only way external source files get into the skyframe graph in the first place is
    // through symlinks outside the package roots, which we neither want to encourage nor optimize
    // for since it is not common. So the set of external files is small.
    if (!pkgLocator.get().getPathEntries().contains(rootedPath.getRoot())) {
      UUID buildId = BuildVariableNode.BUILD_ID.get(env);
      if (buildId == null) {
        return null;
      }
    }

    return fileNode;
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Returns a rooted path representing {@code path} under one of the package roots, or under the
   * filesystem root if it's not under any package root.
   */
  private static RootedPath toRootedPathMaybeUnderRoot(Path path, PathPackageLocator pkgLocator) {
    for (Path root : pkgLocator.getPathEntries()) {
      if (path.startsWith(root)) {
        return RootedPath.toRootedPath(root, path);
      }
    }
    return RootedPath.toRootedPath(path.getFileSystem().getRootDirectory(), path);
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link FileNodeBuilder#build}.
   */
  private static final class FileNodeBuilderException extends NodeBuilderException {
    public FileNodeBuilderException(NodeKey key, IOException e) {
      super(key, e);
    }

    public FileNodeBuilderException(NodeKey key, InconsistentFilesystemException e) {
      super(key, e, /*isTransient=*/true);
    }
  }
}
