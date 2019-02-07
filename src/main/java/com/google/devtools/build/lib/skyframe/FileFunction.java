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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeSet;
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
  @Nullable private final NonexistentFileReceiver nonexistentFileReceiver;

  /** Temporary interface to help track down why files are missing in some cases. */
  public interface NonexistentFileReceiver {
    void accept(
        RootedPath rootedPath,
        RootedPath realRootedPath,
        RootedPath parentRootedPath,
        FileValue parentFileValue);
  }

  public FileFunction(AtomicReference<PathPackageLocator> pkgLocator) {
    this(pkgLocator, null);
  }

  FileFunction(
      AtomicReference<PathPackageLocator> pkgLocator,
      @Nullable NonexistentFileReceiver nonexistentFileReceiver) {
    this.pkgLocator = pkgLocator;
    this.nonexistentFileReceiver = nonexistentFileReceiver;
  }

  @Override
  public FileValue compute(SkyKey skyKey, Environment env)
      throws FileFunctionException, InterruptedException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();
    RootedPath realRootedPath = null;
    FileStateValue realFileStateValue = null;
    PathFragment relativePath = rootedPath.getRootRelativePath();

    // Resolve ancestor symlinks, but only if the current file is not the filesystem root (has no
    // parent) or a package path root (treated opaquely and handled by skyframe's DiffAwareness
    // interface). Note that this is the first thing we do - if an ancestor is part of a
    // symlink cycle, we want to detect that quickly as it gives a more informative error message
    // than we'd get doing bogus filesystem operations.
    if (!relativePath.equals(PathFragment.EMPTY_FRAGMENT)) {
      Pair<RootedPath, FileStateValue> resolvedState = resolveFromAncestors(rootedPath, env);
      if (resolvedState == null) {
        return null;
      }
      realRootedPath = resolvedState.getFirst();
      realFileStateValue = resolvedState.getSecond();
      if (realFileStateValue.getType() == FileStateType.NONEXISTENT) {
        return FileValue.value(
            rootedPath,
            FileStateValue.NONEXISTENT_FILE_STATE_NODE,
            realRootedPath,
            realFileStateValue);
      }
    }

    FileStateValue fileStateValue;
    if (rootedPath.equals(realRootedPath)) {
      fileStateValue = Preconditions.checkNotNull(realFileStateValue, rootedPath);
    } else {
      fileStateValue = (FileStateValue) env.getValue(FileStateValue.key(rootedPath));
      if (fileStateValue == null) {
        return null;
      }
    }

    if (realFileStateValue == null) {
      realRootedPath = rootedPath;
      realFileStateValue = fileStateValue;
    }

    ArrayList<RootedPath> symlinkChain = new ArrayList<>();
    TreeSet<Path> orderedSeenPaths = Sets.newTreeSet();
    while (realFileStateValue.getType().isSymlink()) {
      symlinkChain.add(realRootedPath);
      orderedSeenPaths.add(realRootedPath.asPath());
      Pair<RootedPath, FileStateValue> resolvedState = getSymlinkTargetRootedPath(realRootedPath,
          realFileStateValue.getSymlinkTarget(), orderedSeenPaths, symlinkChain, env);
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
  private Pair<RootedPath, FileStateValue> resolveFromAncestors(
      RootedPath rootedPath, Environment env) throws InterruptedException {
    PathFragment relativePath = rootedPath.getRootRelativePath();
    RootedPath realRootedPath = rootedPath;
    FileValue parentFileValue = null;
    PathFragment parentDirectory = relativePath.getParentDirectory();
    if (parentDirectory != null) {
      RootedPath parentRootedPath = RootedPath.toRootedPath(rootedPath.getRoot(), parentDirectory);

      parentFileValue = (FileValue) env.getValue(FileValue.key(parentRootedPath));
      if (parentFileValue == null) {
        return null;
      }
      PathFragment baseName = PathFragment.create(relativePath.getBaseName());
      RootedPath parentRealRootedPath = parentFileValue.realRootedPath();
      realRootedPath =
          RootedPath.toRootedPath(
              parentRealRootedPath.getRoot(),
              parentRealRootedPath.getRootRelativePath().getRelative(baseName));

      if (!parentFileValue.exists() || !parentFileValue.isDirectory()) {
        if (nonexistentFileReceiver != null) {
          nonexistentFileReceiver.accept(
              rootedPath, realRootedPath, parentRootedPath, parentFileValue);
        }
        return Pair.of(realRootedPath, FileStateValue.NONEXISTENT_FILE_STATE_NODE);
      }
    }
    FileStateValue realFileStateValue =
        (FileStateValue)
            env.getValue(FileStateValue.key(realRootedPath));

    if (realFileStateValue == null) {
      return null;
    }
    return Pair.of(realRootedPath, realFileStateValue);
  }

  /**
   * Returns the symlink target and file state of {@code rootedPath}'s symlink to {@code
   * symlinkTarget}, accounting for ancestor symlinks, or {@code null} if there was a missing dep.
   */
  @Nullable
  private Pair<RootedPath, FileStateValue> getSymlinkTargetRootedPath(
      RootedPath rootedPath,
      PathFragment symlinkTarget,
      TreeSet<Path> orderedSeenPaths,
      Iterable<RootedPath> symlinkChain,
      Environment env)
      throws FileFunctionException, InterruptedException {
    RootedPath symlinkTargetRootedPath;
    if (symlinkTarget.isAbsolute()) {
      Path path = rootedPath.asPath().getFileSystem().getPath(symlinkTarget);
      symlinkTargetRootedPath =
          RootedPath.toRootedPathMaybeUnderRoot(path, pkgLocator.get().getPathEntries());
    } else {
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
      symlinkTargetRootedPath = RootedPath.toRootedPathMaybeUnderRoot(symlinkTargetPath,
          pkgLocator.get().getPathEntries());
    }
    // Suppose we have a symlink chain p -> p1 -> p2 -> ... pK. We want to determine the fully
    // resolved path, if any, of p. This entails following the chain and noticing if there's a
    // symlink issue. There three sorts of issues:
    // (i) Symlink cycle:
    //   p -> p1 -> p2 -> p1
    // (ii) Unbounded expansion caused by a symlink to a descendant of a member of the chain:
    //   p -> a/b -> c/d -> a/b/e
    // (iii) Unbounded expansion caused by a symlink to an ancestor of a member of the chain:
    //   p -> a/b -> c/d -> a
    //
    // We can detect all three of these symlink issues by following the chain and deciding if each
    // new element is problematic. Here is our incremental algorithm:
    //
    // Suppose we encounter the symlink target p and we have already encountered all the paths in P:
    //   If p is in P then we have a found a cycle (i).
    //   If p is a descendant of any path p' in P then we have unbounded expansion (ii).
    //   If p is an ancestor of any path p' in P then we have unbounded expansion (iii).
    // We can check for these cases efficiently (read: sublinear time) by finding the extremal
    // candidate p' for (ii) and (iii).
    Path symlinkTargetPath = symlinkTargetRootedPath.asPath();
    SkyKey uniquenessKey = null;
    FileSymlinkException fse = null;
    Path seenFloorPath = orderedSeenPaths.floor(symlinkTargetPath);
    Path seenCeilingPath = orderedSeenPaths.ceiling(symlinkTargetPath);
    if (orderedSeenPaths.contains(symlinkTargetPath)) {
      // 'rootedPath' is a symlink to a previous element in the symlink chain (i).
      Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
          CycleUtils.splitIntoPathAndChain(
              isPathPredicate(symlinkTargetRootedPath.asPath()), symlinkChain);
      FileSymlinkCycleException fsce =
          new FileSymlinkCycleException(pathAndChain.getFirst(), pathAndChain.getSecond());
      uniquenessKey = FileSymlinkCycleUniquenessFunction.key(fsce.getCycle());
      fse = fsce;
    } else if (seenFloorPath != null && symlinkTargetPath.startsWith(seenFloorPath)) {
      // 'rootedPath' is a symlink to a descendant of a previous element in the symlink chain (ii).
      Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
          CycleUtils.splitIntoPathAndChain(
              isPathPredicate(seenFloorPath),
              ImmutableList.copyOf(
                  Iterables.concat(symlinkChain, ImmutableList.of(symlinkTargetRootedPath))));
      uniquenessKey = FileSymlinkInfiniteExpansionUniquenessFunction.key(pathAndChain.getSecond());
      fse = new FileSymlinkInfiniteExpansionException(
          pathAndChain.getFirst(), pathAndChain.getSecond());
    } else if (seenCeilingPath != null && seenCeilingPath.startsWith(symlinkTargetPath)) {
      // 'rootedPath' is a symlink to an ancestor of a previous element in the symlink chain (iii).
      Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
          CycleUtils.splitIntoPathAndChain(
              isPathPredicate(seenCeilingPath),
              ImmutableList.copyOf(
                  Iterables.concat(symlinkChain, ImmutableList.of(symlinkTargetRootedPath))));
      uniquenessKey = FileSymlinkInfiniteExpansionUniquenessFunction.key(pathAndChain.getSecond());
      fse =
          new FileSymlinkInfiniteExpansionException(
              pathAndChain.getFirst(), pathAndChain.getSecond());
    }
    if (uniquenessKey != null) {
      if (env.getValue(uniquenessKey) == null) {
        // Note that this dependency is merely to ensure that each unique symlink error gets
        // reported exactly once.
        return null;
      }
      throw new FileFunctionException(
          Preconditions.checkNotNull(fse, rootedPath), Transience.PERSISTENT);
    }

    return resolveFromAncestors(symlinkTargetRootedPath, env);
  }

  private static final Predicate<RootedPath> isPathPredicate(final Path path) {
    return new Predicate<RootedPath>() {
      @Override
      public boolean apply(RootedPath rootedPath) {
        return rootedPath.asPath().equals(path);
      }
    };
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
    public FileFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }
  }
}
