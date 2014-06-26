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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.util.LinkedHashSet;
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

  private final AtomicReference<PathPackageLocator> pkgLocator;

  public FileNodeBuilder(AtomicReference<PathPackageLocator> pkgLocator) {
    this.pkgLocator = pkgLocator;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws FileNodeBuilderException {
    RootedPath rootedPath = (RootedPath) nodeKey.getNodeName();
    RootedPath realRootedPath = rootedPath;
    FileStateNode realFileStateNode = null;
    PathFragment relativePath = rootedPath.getRelativePath();

    // Resolve ancestor symlinks, but only if the current file is not the filesystem root or a
    // package path root. The former has no parent and the latter is treated opaquely and handled
    // by skyframe's DiffAwareness interface. Note that this is the first thing we do - if an
    // ancestor is part of a symlink cycle, we want to detect that quickly as it gives a more
    // informative error message than we'd get doing bogus filesystem operations.
    if (!relativePath.equals(PathFragment.EMPTY_FRAGMENT)) {
      Pair<RootedPath, FileStateNode> resolvedState =
          resolveFromAncestors(rootedPath, env, nodeKey);
      if (resolvedState == null) {
        return null;
      }
      realRootedPath = resolvedState.getFirst();
      realFileStateNode = resolvedState.getSecond();
    }

    FileStateNode fileStateNode = (FileStateNode) env.getDep(FileStateNode.key(rootedPath));
    if (fileStateNode == null) {
      return null;
    }
    if (realFileStateNode == null) {
      realFileStateNode = fileStateNode;
    }

    LinkedHashSet<RootedPath> seenPaths = Sets.newLinkedHashSet();
    while (realFileStateNode.getType().equals(FileStateNode.Type.SYMLINK)) {
      if (!seenPaths.add(realRootedPath)) {
        FileSymlinkCycleException fileSymlinkCycleException =
            makeFileSymlinkCycleException(realRootedPath, seenPaths);
        if (env.getDep(FileSymlinkCycleUniquenessNode.key(fileSymlinkCycleException.getCycle()))
            == null) {
          // Note that this dependency is merely to ensure that each unique cycle gets reported
          // exactly once.
          return null;
        }
        throw new FileNodeBuilderException(nodeKey, fileSymlinkCycleException);
      }
      Pair<RootedPath, FileStateNode> resolvedState = getSymlinkTargetRootedPath(realRootedPath,
          realFileStateNode.getSymlinkTarget(), env, nodeKey);
      if (resolvedState == null) {
        return null;
      }
      realRootedPath = resolvedState.getFirst();
      realFileStateNode = resolvedState.getSecond();
    }
    return FileNode.node(rootedPath, fileStateNode, realRootedPath, realFileStateNode);
  }

  /**
   * Returns the path and file state of {@code rootedPath}, accounting for ancestor symlinks, or
   * {@code null} if there was a missing dep.
   */
  @Nullable
  private Pair<RootedPath, FileStateNode> resolveFromAncestors(RootedPath rootedPath,
      Environment env, NodeKey key) throws FileNodeBuilderException {
    PathFragment relativePath = rootedPath.getRelativePath();
    RootedPath parentRootedPath = RootedPath.toRootedPath(rootedPath.getRoot(),
        relativePath.getParentDirectory());
    FileNode parentFileNode = (FileNode) env.getDep(FileNode.key(parentRootedPath));
    if (parentFileNode == null) {
      return null;
    }
    PathFragment baseName = new PathFragment(relativePath.getBaseName());
    RootedPath parentRealRootedPath = parentFileNode.realRootedPath();
    RootedPath realRootedPath = RootedPath.toRootedPath(parentRealRootedPath.getRoot(),
        parentRealRootedPath.getRelativePath().getRelative(baseName));
    FileStateNode realFileStateNode =
        (FileStateNode) env.getDep(FileStateNode.key(realRootedPath));
    if (realFileStateNode == null) {
      return null;
    }
    if (realFileStateNode.getType() != FileStateNode.Type.NONEXISTENT
        && !parentFileNode.isDirectory()) {
      String type = realFileStateNode.getType().toString().toLowerCase();
      String message = type + " " + rootedPath.asPath() + " exists but its parent "
          + "directory " + parentFileNode.realRootedPath().asPath() + " doesn't exist.";
      throw new FileNodeBuilderException(key, new InconsistentFilesystemException(message));
    }
    return Pair.of(realRootedPath, realFileStateNode);
  }

  /**
   * Returns the symlink target and file state of {@code rootedPath}'s symlink to
   * {@code symlinkTarget}, accounting for ancestor symlinks, or {@code null} if there was a
   * missing dep.
   */
  @Nullable
  private Pair<RootedPath, FileStateNode> getSymlinkTargetRootedPath(RootedPath rootedPath,
      PathFragment symlinkTarget, Environment env, NodeKey key) throws FileNodeBuilderException {
    if (symlinkTarget.isAbsolute()) {
      Path path = rootedPath.asPath().getFileSystem().getRootDirectory().getRelative(
          symlinkTarget);
      return resolveFromAncestors(
          RootedPath.toRootedPathMaybeUnderRoot(path, pkgLocator.get().getPathEntries()), env,
          key);
    }
    Path path = rootedPath.asPath();
    Path symlinkTargetPath;
    if (path.getParentDirectory() != null) {
      RootedPath parentRootedPath = RootedPath.toRootedPathMaybeUnderRoot(
          path.getParentDirectory(), pkgLocator.get().getPathEntries());
      FileNode parentFileNode = (FileNode) env.getDep(FileNode.key(parentRootedPath));
      if (parentFileNode == null) {
        return null;
      }
      symlinkTargetPath = parentFileNode.realRootedPath().asPath().getRelative(symlinkTarget);
    } else {
      // This means '/' is a symlink to 'symlinkTarget'.
      symlinkTargetPath = path.getRelative(symlinkTarget);
    }
    RootedPath symlinkTargetRootedPath = RootedPath.toRootedPathMaybeUnderRoot(symlinkTargetPath,
        pkgLocator.get().getPathEntries());
    return resolveFromAncestors(symlinkTargetRootedPath, env, key);
  }

  private FileSymlinkCycleException makeFileSymlinkCycleException(RootedPath startOfCycle,
      Iterable<RootedPath> symlinkPaths) {
    boolean inPathToCycle = true;
    ImmutableList.Builder<RootedPath> pathToCycleBuilder = ImmutableList.builder();
    ImmutableList.Builder<RootedPath> cycleBuilder = ImmutableList.builder();
    for (RootedPath path : symlinkPaths) {
      if (path.equals(startOfCycle)) {
        inPathToCycle = false;
      }
      if (inPathToCycle) {
        pathToCycleBuilder.add(path);
      } else {
        cycleBuilder.add(path);
      }
    }
    return new FileSymlinkCycleException(pathToCycleBuilder.build(), cycleBuilder.build());
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link FileNodeBuilder#build}.
   */
  private static final class FileNodeBuilderException extends NodeBuilderException {

    public FileNodeBuilderException(NodeKey key, InconsistentFilesystemException e) {
      super(key, e, /*isTransient=*/true);
    }

    public FileNodeBuilderException(NodeKey key, FileSymlinkCycleException e) {
      super(key, e);
    }
  }
}
