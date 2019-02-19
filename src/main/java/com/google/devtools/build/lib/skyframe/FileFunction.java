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
 * <p>Most of the complexity in the implementation results from wanting incremental correctness in
 * the presence of symlinks, esp. ancestor directory symlinks.
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

    // Suppose we have a path p. One of the goals of FileFunction is to resolve the "real path", if
    // any, of p. The basic algorithm is to use the fully resolved path of p's parent directory to
    // determine the fully resolved path of p. This is complicated when symlinks are involved, and
    // is especially complicated when ancestor directory symlinks are involved.
    //
    // Since FileStateValues are the roots of invalidation, care has to be taken to ensuring we
    // declare the proper FileStateValue deps. As a concrete example, let p = a/b and imagine (i) a
    // is a direct symlink to c and also (ii) c/b is an existing file. Among other direct deps, we
    // want to have a direct dep on FileStateValue(c/b), since that's the node that will be changed
    // if the actual contents of a/b (aka c/b) changes. To rephrase: a dep on FileStateValue(a/b)
    // won't do anything productive since that path will never be in the Skyframe diff.
    //
    // In the course of resolving the real path of p, there will be a logical chain of paths we
    // consider. Going with the example from above, the full chain of paths we consider is
    // [a/b, c/b].
    ArrayList<RootedPath> logicalChain = new ArrayList<>();
    // Same contents as 'logicalChain', except stored as an sorted TreeSet for efficiency reasons.
    // See the usage in checkPathSeenDuringPartialResolutionInternal.
    TreeSet<Path> sortedLogicalChain = Sets.newTreeSet();

    // Fully resolve the path of the parent directory, but only if the current file is not the
    // filesystem root (has no parent) or a package path root (treated opaquely and handled by
    // skyframe's DiffAwareness interface).
    //
    // This entails resolving ancestor symlinks fully. Note that this is the first thing we do - if
    // an ancestor is part of a symlink cycle, we want to detect that quickly as it gives a more
    // informative error message than we'd get doing bogus filesystem operations.
    PartialResolutionResult resolveFromAncestorsResult =
        resolveFromAncestors(rootedPath, sortedLogicalChain, logicalChain, env);
    if (resolveFromAncestorsResult == null) {
      return null;
    }
    RootedPath rootedPathFromAncestors = resolveFromAncestorsResult.rootedPath;
    FileStateValue fileStateValueFromAncestors = resolveFromAncestorsResult.fileStateValue;
    if (fileStateValueFromAncestors.getType() == FileStateType.NONEXISTENT) {
      return FileValue.value(
          ImmutableList.copyOf(logicalChain),
          rootedPath,
          FileStateValue.NONEXISTENT_FILE_STATE_NODE,
          rootedPathFromAncestors,
          fileStateValueFromAncestors);
    }

    RootedPath realRootedPath = rootedPathFromAncestors;
    FileStateValue realFileStateValue = fileStateValueFromAncestors;

    while (realFileStateValue.getType().isSymlink()) {
      PartialResolutionResult getSymlinkTargetRootedPathResult =
          getSymlinkTargetRootedPath(
              realRootedPath,
              realFileStateValue.getSymlinkTarget(),
              sortedLogicalChain,
              logicalChain,
              env);
      if (getSymlinkTargetRootedPathResult == null) {
        return null;
      }
      realRootedPath = getSymlinkTargetRootedPathResult.rootedPath;
      realFileStateValue = getSymlinkTargetRootedPathResult.fileStateValue;
    }

    return FileValue.value(
        ImmutableList.copyOf(logicalChain),
        rootedPath,
        // TODO(b/123922036): This is a bug. Should be 'fileStateValueFromAncestors'.
        fileStateValueFromAncestors,
        realRootedPath,
        realFileStateValue);
  }

  private static RootedPath getParent(RootedPath childRootedPath) {
    return RootedPath.toRootedPath(
        childRootedPath.getRoot(), childRootedPath.getRootRelativePath().getParentDirectory());
  }

  private static RootedPath getChild(RootedPath parentRootedPath, String baseName) {
    return RootedPath.toRootedPath(
        parentRootedPath.getRoot(), parentRootedPath.getRootRelativePath().getChild(baseName));
  }

  private RootedPath toRootedPath(Path path) {
    return RootedPath.toRootedPathMaybeUnderRoot(path, pkgLocator.get().getPathEntries());
  }

  /**
   * Returns the path and file state of {@code rootedPath}, accounting for ancestor symlinks, or
   * {@code null} if there was a missing dep.
   */
  @Nullable
  private PartialResolutionResult resolveFromAncestors(
      RootedPath rootedPath,
      TreeSet<Path> sortedLogicalChain,
      ArrayList<RootedPath> logicalChain,
      Environment env)
      throws InterruptedException, FileFunctionException {
    PathFragment parentDirectory = rootedPath.getRootRelativePath().getParentDirectory();
    return parentDirectory != null
        ? resolveFromAncestorsWithParent(
            rootedPath, parentDirectory, sortedLogicalChain, logicalChain, env)
        : resolveFromAncestorsNoParent(rootedPath, sortedLogicalChain, logicalChain, env);
  }

  @Nullable
  private PartialResolutionResult resolveFromAncestorsWithParent(
      RootedPath rootedPath,
      PathFragment parentDirectory,
      TreeSet<Path> sortedLogicalChain,
      ArrayList<RootedPath> logicalChain,
      Environment env)
      throws InterruptedException, FileFunctionException {
    PathFragment relativePath = rootedPath.getRootRelativePath();
    RootedPath rootedPathFromAncestors;
    String baseName = relativePath.getBaseName();
    RootedPath parentRootedPath = RootedPath.toRootedPath(rootedPath.getRoot(), parentDirectory);

    FileValue parentFileValue = (FileValue) env.getValue(FileValue.key(parentRootedPath));
    if (parentFileValue == null) {
      return null;
    }
    rootedPathFromAncestors = getChild(parentFileValue.realRootedPath(), baseName);

    if (!parentFileValue.exists() || !parentFileValue.isDirectory()) {
      if (nonexistentFileReceiver != null) {
        nonexistentFileReceiver.accept(
            rootedPath, rootedPathFromAncestors, parentRootedPath, parentFileValue);
      }
      return new PartialResolutionResult(
          rootedPathFromAncestors, FileStateValue.NONEXISTENT_FILE_STATE_NODE);
    }

    for (RootedPath parentPartialRootedPath : parentFileValue.logicalChainDuringResolution()) {
      checkAndNotePathSeenDuringPartialResolution(
          getChild(parentPartialRootedPath, baseName), sortedLogicalChain, logicalChain, env);
      if (env.valuesMissing()) {
        return null;
      }
    }

    FileStateValue fileStateValueFromAncestors =
        (FileStateValue) env.getValue(FileStateValue.key(rootedPathFromAncestors));
    if (fileStateValueFromAncestors == null) {
      return null;
    }

    return new PartialResolutionResult(rootedPathFromAncestors, fileStateValueFromAncestors);
  }

  @Nullable
  private PartialResolutionResult resolveFromAncestorsNoParent(
      RootedPath rootedPath,
      TreeSet<Path> sortedLogicalChain,
      ArrayList<RootedPath> logicalChain,
      Environment env)
      throws InterruptedException, FileFunctionException {
    checkAndNotePathSeenDuringPartialResolution(rootedPath, sortedLogicalChain, logicalChain, env);
    if (env.valuesMissing()) {
      return null;
    }
    FileStateValue realFileStateValue =
        (FileStateValue) env.getValue(FileStateValue.key(rootedPath));
    if (realFileStateValue == null) {
      return null;
    }
    return new PartialResolutionResult(rootedPath, realFileStateValue);
  }

  private static final class PartialResolutionResult {
    private final RootedPath rootedPath;
    private final FileStateValue fileStateValue;

    private PartialResolutionResult(RootedPath rootedPath, FileStateValue fileStateValue) {
      this.rootedPath = rootedPath;
      this.fileStateValue = fileStateValue;
    }
  }

  /**
   * Returns the symlink target and file state of {@code rootedPath}'s symlink to {@code
   * symlinkTarget}, accounting for ancestor symlinks, or {@code null} if there was a missing dep.
   */
  @Nullable
  private PartialResolutionResult getSymlinkTargetRootedPath(
      RootedPath rootedPath,
      PathFragment symlinkTarget,
      TreeSet<Path> sortedLogicalChain,
      ArrayList<RootedPath> logicalChain,
      Environment env)
      throws FileFunctionException, InterruptedException {
    Path path = rootedPath.asPath();
    Path symlinkTargetPath;
    if (symlinkTarget.isAbsolute()) {
      symlinkTargetPath = path.getRelative(symlinkTarget);
    } else {
      Path parentPath = path.getParentDirectory();
      symlinkTargetPath =
          parentPath != null
              ? parentPath.getRelative(symlinkTarget)
              : path.getRelative(symlinkTarget);
    }
    RootedPath symlinkTargetRootedPath = toRootedPath(symlinkTargetPath);
    checkPathSeenDuringPartialResolution(
        symlinkTargetRootedPath, sortedLogicalChain, logicalChain, env);
    if (env.valuesMissing()) {
      return null;
    }
    // The symlink target could have a different parent directory, which itself could be a directory
    // symlink (or have an ancestor directory symlink)!
    return resolveFromAncestors(symlinkTargetRootedPath, sortedLogicalChain, logicalChain, env);
  }

  private void checkAndNotePathSeenDuringPartialResolution(
      RootedPath rootedPath,
      TreeSet<Path> sortedLogicalChain,
      ArrayList<RootedPath> logicalChain,
      Environment env)
      throws FileFunctionException, InterruptedException {
    Path path = rootedPath.asPath();
    checkPathSeenDuringPartialResolutionInternal(
        rootedPath, path, sortedLogicalChain, logicalChain, env);
    sortedLogicalChain.add(path);
    logicalChain.add(rootedPath);
  }

  private void checkPathSeenDuringPartialResolution(
      RootedPath rootedPath,
      TreeSet<Path> sortedLogicalChain,
      ArrayList<RootedPath> logicalChain,
      Environment env)
      throws FileFunctionException, InterruptedException {
    checkPathSeenDuringPartialResolutionInternal(
        rootedPath, rootedPath.asPath(), sortedLogicalChain, logicalChain, env);
  }

  private void checkPathSeenDuringPartialResolutionInternal(
      RootedPath rootedPath,
      Path path,
      TreeSet<Path> sortedLogicalChain,
      ArrayList<RootedPath> logicalChain,
      Environment env)
      throws FileFunctionException, InterruptedException {
    // We are about to perform another step of partial real path resolution. 'logicalChain' is the
    // chain of paths we've considered so far, and 'rootedPath' / 'path' is the proposed next path
    // we consider.
    //
    // Before we proceed with 'rootedPath', we need to ensure there won't be a problem. There are
    // three sorts of issues, all stemming from symlinks:
    //   (i) Symlink cycle:
    //     p -> p1 -> p2 -> p1
    //   (ii) Unbounded expansion caused by a symlink to a descendant of a member of the chain:
    //     p -> a/b -> c/d -> a/b/e
    //   (iii) Unbounded expansion caused by a symlink to an ancestor of a member of the chain:
    //     p -> a/b -> c/d -> a
    //
    // We can detect all three of these symlink issues inspection of the proposed new element. Here
    // is our incremental algorithm:
    //
    //   If 'path' is in 'sortedLogicalChain' then we have a found a cycle (i).
    //   If 'path' is a descendant of any path p in 'sortedLogicalChain' then we have unbounded
    //   expansion (ii).
    //   If 'path' is an ancestor of any path p in 'sortedLogicalChain' then we have unbounded
    //   expansion (iii).
    // We can check for these cases efficiently (read: sublinear time) by finding the extremal
    // candidate p for (ii) and (iii).
    SkyKey uniquenessKey = null;
    FileSymlinkException fse = null;
    Path seenFloorPath = sortedLogicalChain.floor(path);
    Path seenCeilingPath = sortedLogicalChain.ceiling(path);
    if (sortedLogicalChain.contains(path)) {
      // 'rootedPath' is [transitively] a symlink to a previous element in the symlink chain (i).
      Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
          CycleUtils.splitIntoPathAndChain(isPathPredicate(path), logicalChain);
      FileSymlinkCycleException fsce =
          new FileSymlinkCycleException(pathAndChain.getFirst(), pathAndChain.getSecond());
      uniquenessKey = FileSymlinkCycleUniquenessFunction.key(fsce.getCycle());
      fse = fsce;
    } else if (seenFloorPath != null && path.startsWith(seenFloorPath)) {
      // 'rootedPath' is [transitively] a symlink to a descendant of a previous element in the
      // symlink chain (ii).
      Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
          CycleUtils.splitIntoPathAndChain(
              isPathPredicate(seenFloorPath),
              ImmutableList.copyOf(Iterables.concat(logicalChain, ImmutableList.of(rootedPath))));
      uniquenessKey = FileSymlinkInfiniteExpansionUniquenessFunction.key(pathAndChain.getSecond());
      fse = new FileSymlinkInfiniteExpansionException(
          pathAndChain.getFirst(), pathAndChain.getSecond());
    } else if (seenCeilingPath != null && seenCeilingPath.startsWith(path)) {
      // 'rootedPath' is [transitively] a symlink to an ancestor of a previous element in the
      // symlink chain (iii).
      Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
          CycleUtils.splitIntoPathAndChain(
              isPathPredicate(seenCeilingPath),
              ImmutableList.copyOf(Iterables.concat(logicalChain, ImmutableList.of(rootedPath))));
      uniquenessKey = FileSymlinkInfiniteExpansionUniquenessFunction.key(pathAndChain.getSecond());
      fse =
          new FileSymlinkInfiniteExpansionException(
              pathAndChain.getFirst(), pathAndChain.getSecond());
    }
    if (uniquenessKey != null) {
      // Note that this dependency is merely to ensure that each unique symlink error gets
      // reported exactly once.
      env.getValue(uniquenessKey);
      if (env.valuesMissing()) {
        return;
      }
      throw new FileFunctionException(
          Preconditions.checkNotNull(fse, rootedPath), Transience.PERSISTENT);
    }
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
