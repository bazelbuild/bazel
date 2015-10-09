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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

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
  private final TimestampGranularityMonitor tsgm;
  private final ExternalFilesHelper externalFilesHelper;

  public FileFunction(AtomicReference<PathPackageLocator> pkgLocator,
      TimestampGranularityMonitor tsgm,
      ExternalFilesHelper externalFilesHelper) {
    this.pkgLocator = pkgLocator;
    this.tsgm = tsgm;
    this.externalFilesHelper = externalFilesHelper;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws FileFunctionException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();
    RootedPath realRootedPath = null;
    FileStateValue realFileStateValue = null;
    PathFragment relativePath = rootedPath.getRelativePath();

    // Resolve ancestor symlinks, but only if the current file is not the filesystem root (has no
    // parent) or a package path root (treated opaquely and handled by skyframe's DiffAwareness
    // interface). Note that this is the first thing we do - if an ancestor is part of a
    // symlink cycle, we want to detect that quickly as it gives a more informative error message
    // than we'd get doing bogus filesystem operations.
    if (!relativePath.equals(PathFragment.EMPTY_FRAGMENT)) {
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
      realRootedPath = rootedPath;
      realFileStateValue = fileStateValue;
    } else if (rootedPath.equals(realRootedPath) && !fileStateValue.equals(realFileStateValue)) {
      String message = String.format(
          "Some filesystem operations implied %s was a %s but others made us think it was a %s",
          rootedPath.asPath().getPathString(),
          fileStateValue.prettyPrint(),
          realFileStateValue.prettyPrint());
      throw new FileFunctionException(new InconsistentFilesystemException(message),
          Transience.TRANSIENT);
    }

    ArrayList<RootedPath> symlinkChain = new ArrayList<>();
    TreeSet<Path> orderedSeenPaths = Sets.newTreeSet();
    while (realFileStateValue.getType().equals(FileStateValue.Type.SYMLINK)) {
      symlinkChain.add(realRootedPath);
      orderedSeenPaths.add(realRootedPath.asPath());
      if (externalFilesHelper.shouldAssumeImmutable(realRootedPath)) {
        // If the file is assumed to be immutable, we want to resolve the symlink chain without
        // adding dependencies since we don't care about incremental correctness.
        try {
          Path realPath = rootedPath.asPath().resolveSymbolicLinks();
          realRootedPath = RootedPath.toRootedPathMaybeUnderRoot(realPath,
              pkgLocator.get().getPathEntries());
          realFileStateValue = FileStateValue.create(realRootedPath, tsgm);
        } catch (IOException e) {
          RootedPath root = RootedPath.toRootedPath(
              rootedPath.asPath().getFileSystem().getRootDirectory(),
              rootedPath.asPath().getFileSystem().getRootDirectory());
          return FileValue.value(
              rootedPath, fileStateValue,
              root, FileStateValue.NONEXISTENT_FILE_STATE_NODE);
        } catch (InconsistentFilesystemException e) {
          throw new FileFunctionException(e, Transience.TRANSIENT);
        }
      } else {
        Pair<RootedPath, FileStateValue> resolvedState = getSymlinkTargetRootedPath(realRootedPath,
            realFileStateValue.getSymlinkTarget(), orderedSeenPaths, symlinkChain, env);
        if (resolvedState == null) {
          return null;
        }
        realRootedPath = resolvedState.getFirst();
        realFileStateValue = resolvedState.getSecond();
      }
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
    // We only resolve ancestors if the file is not assumed to be immutable (handling ancestors
    // would be too aggressive).
    if (!externalFilesHelper.shouldAssumeImmutable(rootedPath)
        && !relativePath.equals(PathFragment.EMPTY_FRAGMENT)) {
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
      if (!parentFileValue.exists()) {
        return Pair.of(realRootedPath, FileStateValue.NONEXISTENT_FILE_STATE_NODE);
      }
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
          + "path " + parentFileValue.realRootedPath().asPath() + " isn't an existing directory.";
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
      PathFragment symlinkTarget, TreeSet<Path> orderedSeenPaths,
      Iterable<RootedPath> symlinkChain, Environment env) throws FileFunctionException {
    RootedPath symlinkTargetRootedPath;
    if (symlinkTarget.isAbsolute()) {
      Path path = rootedPath.asPath().getFileSystem().getRootDirectory().getRelative(
          symlinkTarget);
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
    Path symlinkTargetPath = symlinkTargetRootedPath.asPath();
    Path existingFloorPath = orderedSeenPaths.floor(symlinkTargetPath);
    // Here is a brief argument that the following logic is correct.
    //
    // Any path 'p' in the symlink chain that is no larger than 'symlinkTargetPath' is one of:
    //   (i)   'symlinkTargetPath'
    //   (ii)   a smaller sibling 's' of 'symlinkTargetPath' or a sibling of an ancestor of
    //         'symlinkTargetPath'
    //   (iii)  an ancestor 'a' of 'symlinkTargetPath'
    //   (iv) something else (e.g. a smaller sibling of an ancestor of 'symlinkTargetPath')
    // If the largest 'p' is 'symlinkTarget' itself then 'existingFloorPath' will be that and we
    // have found cycle. Otherwise, if there is such a 's' then 'existingFloorPath' will be the
    // largest one. But the presence of any such 's' in the symlink chain implies an infinite
    // expansion, which we would have already noticed. On the other hand, if there is such an 'a'
    // then 'existingFloorPath' will be the largest one that and we definitely have found an
    // infinite symlink expansion. Otherwise, if there is no such 'a', then the presence of
    // 'symlinkTargetPath' doesn't create an infinite symlink expansion.
    if (existingFloorPath != null && symlinkTargetPath.startsWith(existingFloorPath)) {
      SkyKey uniquenessKey;
      FileSymlinkException fse;
      if (symlinkTargetPath.equals(existingFloorPath)) {
        Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
            CycleUtils.splitIntoPathAndChain(
                isPathPredicate(symlinkTargetRootedPath.asPath()), symlinkChain);
        FileSymlinkCycleException fsce =
            new FileSymlinkCycleException(pathAndChain.getFirst(), pathAndChain.getSecond());
        uniquenessKey = FileSymlinkCycleUniquenessValue.key(fsce.getCycle());
        fse = fsce;
      } else {
        Pair<ImmutableList<RootedPath>, ImmutableList<RootedPath>> pathAndChain =
            CycleUtils.splitIntoPathAndChain(
                isPathPredicate(existingFloorPath),
                ImmutableList.copyOf(
                    Iterables.concat(symlinkChain, ImmutableList.of(symlinkTargetRootedPath))));
        uniquenessKey = FileSymlinkInfiniteExpansionUniquenessValue.key(pathAndChain.getSecond());
        fse = new FileSymlinkInfiniteExpansionException(
            pathAndChain.getFirst(), pathAndChain.getSecond());
      }
      if (env.getValue(uniquenessKey) == null) {
        // Note that this dependency is merely to ensure that each unique symlink error gets
        // reported exactly once.
        return null;
      }
      throw new FileFunctionException(fse);
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

    public FileFunctionException(InconsistentFilesystemException e, Transience transience) {
      super(e, transience);
    }

    public FileFunctionException(FileSymlinkException e) {
      super(e, Transience.PERSISTENT);
    }

    public FileFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }
  }
}
