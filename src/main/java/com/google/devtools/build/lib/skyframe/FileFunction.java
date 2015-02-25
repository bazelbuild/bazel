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
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.LinkedHashSet;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} for {@link FileValue}s.
 *
 * <p>Most of the complexity in the implementation is associated to handling symlinks. Namely,
 * this class makes sure that {@code FileValue}s corresponding to symlinks are correctly invalidated
 * if the destination of the symlink is invalidated. Directory symlinks are also covered.
 */
public class FileFunction implements SkyFunction {

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final ExternalFilesHelper externalFilesHelper;

  public FileFunction(AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFilesHelper externalFilesHelper) {
    this.pkgLocator = pkgLocator;
    this.externalFilesHelper = externalFilesHelper;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws FileFunctionException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();
    RootedPath realRootedPath = rootedPath;
    FileStateValue realFileStateValue = null;
    PathFragment relativePath = rootedPath.getRelativePath();

    // Resolve ancestor symlinks, but only if the current file is not the filesystem root (has no
    // parent) or a package path root (treated opaquely and handled by skyframe's DiffAwareness
    // interface) or otherwise assumed to be immutable (handling ancestors would add dependencies
    // too aggressively). Note that this is the first thing we do - if an ancestor is part of a
    // symlink cycle, we want to detect that quickly as it gives a more informative error message
    // than we'd get doing bogus filesystem operations.
    if (!relativePath.equals(PathFragment.EMPTY_FRAGMENT)
        && !externalFilesHelper.shouldAssumeImmutable(rootedPath)) {
      Pair<RootedPath, FileStateValue> resolvedState =
          resolveFromAncestors(rootedPath, env);
      if (resolvedState == null) {
        return null;
      }
      realRootedPath = resolvedState.getFirst();
      realFileStateValue = resolvedState.getSecond();
    }

    FileStateValue fileStateValue = (FileStateValue) env.getValue(FileStateValue.key(rootedPath));
    if (fileStateValue == null) {
      return null;
    }
    if (realFileStateValue == null) {
      realFileStateValue = fileStateValue;
    }

    LinkedHashSet<RootedPath> seenPaths = Sets.newLinkedHashSet();
    while (realFileStateValue.getType().equals(FileStateValue.Type.SYMLINK)) {
      if (!seenPaths.add(realRootedPath)) {
        FileSymlinkCycleException fileSymlinkCycleException =
            makeFileSymlinkCycleException(realRootedPath, seenPaths);
        if (env.getValue(FileSymlinkCycleUniquenessValue.key(fileSymlinkCycleException.getCycle()))
            == null) {
          // Note that this dependency is merely to ensure that each unique cycle gets reported
          // exactly once.
          return null;
        }
        throw new FileFunctionException(fileSymlinkCycleException);
      }
      Pair<RootedPath, FileStateValue> resolvedState = getSymlinkTargetRootedPath(realRootedPath,
          realFileStateValue.getSymlinkTarget(), env);
      if (resolvedState == null) {
        return null;
      }
      realRootedPath = resolvedState.getFirst();
      realFileStateValue = resolvedState.getSecond();
    }
    return FileValue.value(rootedPath, fileStateValue, realRootedPath, realFileStateValue);
  }

  /**
   * Returns the path and file state of {@code rootedPath}, accounting for ancestor symlinks, or
   * {@code null} if there was a missing dep.
   */
  @Nullable
  private Pair<RootedPath, FileStateValue> resolveFromAncestors(RootedPath rootedPath,
      Environment env) throws FileFunctionException {
    PathFragment relativePath = rootedPath.getRelativePath();
    RootedPath realRootedPath = rootedPath;
    FileValue parentFileValue = null;
    if (!relativePath.equals(PathFragment.EMPTY_FRAGMENT)) {
      RootedPath parentRootedPath = RootedPath.toRootedPath(rootedPath.getRoot(),
          relativePath.getParentDirectory());
      parentFileValue = (FileValue) env.getValue(FileValue.key(parentRootedPath));
      if (parentFileValue == null) {
        return null;
      }
      PathFragment baseName = new PathFragment(relativePath.getBaseName());
      RootedPath parentRealRootedPath = parentFileValue.realRootedPath();
      realRootedPath = RootedPath.toRootedPath(parentRealRootedPath.getRoot(),
          parentRealRootedPath.getRelativePath().getRelative(baseName));
    }
    FileStateValue realFileStateValue =
        (FileStateValue) env.getValue(FileStateValue.key(realRootedPath));
    if (realFileStateValue == null) {
      return null;
    }
    if (realFileStateValue.getType() != FileStateValue.Type.NONEXISTENT
        && parentFileValue != null && !parentFileValue.isDirectory()) {
      String type = realFileStateValue.getType().toString().toLowerCase();
      String message = type + " " + rootedPath.asPath() + " exists but its parent "
          + "directory " + parentFileValue.realRootedPath().asPath() + " doesn't exist.";
      throw new FileFunctionException(new InconsistentFilesystemException(message),
          Transience.TRANSIENT);
    }
    return Pair.of(realRootedPath, realFileStateValue);
  }

  /**
   * Returns the symlink target and file state of {@code rootedPath}'s symlink to
   * {@code symlinkTarget}, accounting for ancestor symlinks, or {@code null} if there was a
   * missing dep.
   */
  @Nullable
  private Pair<RootedPath, FileStateValue> getSymlinkTargetRootedPath(RootedPath rootedPath,
      PathFragment symlinkTarget, Environment env) throws FileFunctionException {
    if (symlinkTarget.isAbsolute()) {
      Path path = rootedPath.asPath().getFileSystem().getRootDirectory().getRelative(
          symlinkTarget);
      return resolveFromAncestors(
          RootedPath.toRootedPathMaybeUnderRoot(path, pkgLocator.get().getPathEntries()), env);
    }
    Path path = rootedPath.asPath();
    Path symlinkTargetPath;
    if (path.getParentDirectory() != null) {
      RootedPath parentRootedPath = RootedPath.toRootedPathMaybeUnderRoot(
          path.getParentDirectory(), pkgLocator.get().getPathEntries());
      FileValue parentFileValue = (FileValue) env.getValue(FileValue.key(parentRootedPath));
      if (parentFileValue == null) {
        return null;
      }
      symlinkTargetPath = parentFileValue.realRootedPath().asPath().getRelative(symlinkTarget);
    } else {
      // This means '/' is a symlink to 'symlinkTarget'.
      symlinkTargetPath = path.getRelative(symlinkTarget);
    }
    RootedPath symlinkTargetRootedPath = RootedPath.toRootedPathMaybeUnderRoot(symlinkTargetPath,
        pkgLocator.get().getPathEntries());
    return resolveFromAncestors(symlinkTargetRootedPath, env);
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
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link FileFunction#compute}.
   */
  private static final class FileFunctionException extends SkyFunctionException {

    public FileFunctionException(InconsistentFilesystemException e, Transience transience) {
      super(e, transience);
    }

    public FileFunctionException(FileSymlinkCycleException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
