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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Dirent.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException4;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * RecursiveDirectoryTraversalFunction traverses the subdirectories of a directory, looking for
 * and loading packages, and builds up a value from these packages in a manner customized by
 * classes that derive from it.
 */
abstract class RecursiveDirectoryTraversalFunction
    <TVisitor extends RecursiveDirectoryTraversalFunction.Visitor, TReturn> {
  private static final String SENTINEL_FILE_NAME_FOR_NOT_TRAVERSING_SYMLINKS =
      "DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN";

  private final BlazeDirectories directories;

  protected RecursiveDirectoryTraversalFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  /**
   * Returned from {@link #visitDirectory} if its {@code recursivePkgKey} is a symlink or not a
   * directory, or if a dependency value lookup returns an error.
   */
  protected abstract TReturn getEmptyReturn();

  /**
   * Called by {@link #visitDirectory}, which will next call {@link Visitor#visitPackageValue} if
   * the {@code recursivePkgKey} specifies a directory with a package, and which will lastly be
   * provided to {@link #aggregateWithSubdirectorySkyValues} to compute the {@code TReturn} value
   * returned by {@link #visitDirectory}.
   */
  protected abstract TVisitor getInitialVisitor();

  /**
   * Called by {@link #visitDirectory} to get the {@link SkyKey}s associated with recursive
   * computation in subdirectories of {@code subdirectory}, excluding directories in
   * {@code excludedSubdirectoriesBeneathSubdirectory}, all of which must be proper subdirectories
   * of {@code subdirectory}.
   */
  protected abstract SkyKey getSkyKeyForSubdirectory(
      RepositoryName repository, RootedPath subdirectory,
      ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory);

  /**
   * Called by {@link #visitDirectory} to compute the {@code TReturn} value it returns, as a
   * function of {@code visitor} and the {@link SkyValue}s computed for subdirectories
   * of the directory specified by {@code recursivePkgKey}, contained in
   * {@code subdirectorySkyValues}.
   */
  protected abstract TReturn aggregateWithSubdirectorySkyValues(
      TVisitor visitor, Map<SkyKey, SkyValue> subdirectorySkyValues);

  /**
   * A type of value used by {@link #visitDirectory} as it checks for a package in the directory
   * specified by {@code recursivePkgKey}; if such a package exists, {@link #visitPackageValue}
   * is called.
   *
   * <p>The value is then provided to {@link #aggregateWithSubdirectorySkyValues} to compute the
   * value returned by {@link #visitDirectory}.
   */
  interface Visitor {

    /**
     * Called iff the directory contains a package. Provides an {@link Environment} {@code env}
     * so that the visitor may do additional lookups. {@link Environment#valuesMissing} will be
     * checked afterwards.
     */
    void visitPackageValue(Package pkg, Environment env);
  }

  /**
   * Looks in the directory specified by {@code recursivePkgKey} for a package, does some work
   * as specified by {@link Visitor} if such a package exists, then recursively does work in each
   * non-excluded subdirectory as specified by {@link #getSkyKeyForSubdirectory}, and finally
   * aggregates the {@link Visitor} value along with values from each subdirectory as specified
   * by {@link #aggregateWithSubdirectorySkyValues}, and returns that aggregation.
   *
   * <p>Returns null if {@code env.valuesMissing()} is true, checked after each call to one of
   * {@link RecursiveDirectoryTraversalFunction}'s abstract methods that were given {@code env}.
   * (And after each of {@code visitDirectory}'s own uses of {@code env}, of course.)
   */
  TReturn visitDirectory(RecursivePkgKey recursivePkgKey, Environment env) {
    RootedPath rootedPath = recursivePkgKey.getRootedPath();
    BlacklistedPackagePrefixesValue blacklist =
        (BlacklistedPackagePrefixesValue) env.getValue(BlacklistedPackagePrefixesValue.key());
    if (blacklist == null) {
      return null;
    }
    Set<PathFragment> excludedPaths =
        Sets.union(recursivePkgKey.getExcludedPaths(), blacklist.getPatterns());
    Path root = rootedPath.getRoot();
    PathFragment rootRelativePath = rootedPath.getRelativePath();

    SkyKey fileKey = FileValue.key(rootedPath);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileKey, InconsistentFilesystemException.class,
          FileSymlinkException.class, IOException.class);
    } catch (InconsistentFilesystemException | FileSymlinkException | IOException e) {
      return reportErrorAndReturn("Failed to get information about path", e, rootRelativePath,
          env.getListener());
    }
    if (fileValue == null) {
      return null;
    }

    if (!fileValue.isDirectory()) {
      return getEmptyReturn();
    }

    PackageIdentifier packageId = PackageIdentifier.create(
        recursivePkgKey.getRepository(), rootRelativePath);

    if (packageId.getRepository().isDefault()
      && fileValue.isSymlink()
      && fileValue.getUnresolvedLinkTarget().startsWith(directories.getOutputBase().asFragment())) {
      // Symlinks back to the output base are not traversed so that we avoid convenience symlinks.
      // Note that it's not enough to just check for the convenience symlinks themselves, because
      // if the value of --symlink_prefix changes, the old symlinks are left in place. This
      // algorithm also covers more creative use cases where people create convenience symlinks
      // somewhere in the directory tree manually.
      return getEmptyReturn();
    }

    SkyKey pkgLookupKey = PackageLookupValue.key(packageId);
    SkyKey dirListingKey = DirectoryListingValue.key(rootedPath);
    Map<SkyKey,
        ValueOrException4<
            NoSuchPackageException,
            InconsistentFilesystemException,
            FileSymlinkException,
            IOException>> pkgLookupAndDirectoryListingDeps = env.getValuesOrThrow(
                ImmutableList.of(pkgLookupKey, dirListingKey),
                NoSuchPackageException.class,
                InconsistentFilesystemException.class,
                FileSymlinkException.class,
                IOException.class);
    if (env.valuesMissing()) {
      return null;
    }
    PackageLookupValue pkgLookupValue;
    try {
      pkgLookupValue = (PackageLookupValue) Preconditions.checkNotNull(
          pkgLookupAndDirectoryListingDeps.get(pkgLookupKey).get(), "%s %s", recursivePkgKey,
          pkgLookupKey);
    } catch (NoSuchPackageException | InconsistentFilesystemException e) {
      return reportErrorAndReturn("Failed to load package", e, rootRelativePath,
          env.getListener());
    } catch (IOException | FileSymlinkException e) {
      throw new IllegalStateException(e);
    }

    TVisitor visitor = getInitialVisitor();
    if (pkgLookupValue.packageExists()) {
      if (pkgLookupValue.getRoot().equals(root)) {
        Package pkg = null;
        try {
          PackageValue pkgValue = (PackageValue)
              env.getValueOrThrow(PackageValue.key(packageId), NoSuchPackageException.class);
          if (pkgValue == null) {
            return null;
          }
          pkg = pkgValue.getPackage();
          if (pkg.containsErrors()) {
            env
                .getListener()
                .handle(
                    Event.error("package contains errors: " + rootRelativePath.getPathString()));
          }
        } catch (NoSuchPackageException e) {
          // The package had errors, but don't fail-fast as there might be subpackages below the
          // current directory.
          env
              .getListener()
              .handle(Event.error("package contains errors: " + rootRelativePath.getPathString()));
        }
        if (pkg != null) {
          visitor.visitPackageValue(pkg, env);
          if (env.valuesMissing()) {
            return null;
          }
        }
      }
      // The package lookup succeeded, but was under a different root. We still, however, need to
      // recursively consider subdirectories. For example:
      //
      //  Pretend --package_path=rootA/workspace:rootB/workspace and these are the only files:
      //    rootA/workspace/foo/
      //    rootA/workspace/foo/bar/BUILD
      //    rootB/workspace/foo/BUILD
      //  If we're doing a recursive package lookup under 'rootA/workspace' starting at 'foo', note
      //  that even though the package 'foo' is under 'rootB/workspace', there is still a package
      //  'foo/bar' under 'rootA/workspace'.
    }

    DirectoryListingValue dirListingValue;
    try {
      dirListingValue = (DirectoryListingValue) Preconditions.checkNotNull(
          pkgLookupAndDirectoryListingDeps.get(dirListingKey).get(), "%s %s", recursivePkgKey,
          dirListingKey);
    } catch (InconsistentFilesystemException | IOException e) {
      return reportErrorAndReturn("Failed to list directory contents", e, rootRelativePath,
          env.getListener());
    } catch (FileSymlinkException e) {
      // DirectoryListingFunction only throws FileSymlinkCycleException when FileFunction throws it,
      // but FileFunction was evaluated for rootedPath above, and didn't throw there. It shouldn't
      // be able to avoid throwing there but throw here.
      throw new IllegalStateException("Symlink cycle found after not being found for \""
          + rootedPath + "\"");
    } catch (NoSuchPackageException e) {
      throw new IllegalStateException(e);
    }

    boolean followSymlinks = shouldFollowSymlinksWhenTraversing(dirListingValue.getDirents());
    List<SkyKey> childDeps = new ArrayList<>();
    for (Dirent dirent : dirListingValue.getDirents()) {
      Type type = dirent.getType();
      if (type != Type.DIRECTORY
          && (type != Type.SYMLINK || (type == Type.SYMLINK && !followSymlinks))) {
        // Non-directories can never host packages. Symlinks to non-directories are weeded out at
        // the next level of recursion when we check if its FileValue is a directory. This is slower
        // if there are a lot of symlinks in the tree, but faster if there are only a few, which is
        // the case most of the time.
        //
        // We are not afraid of weird symlink structure here: both cyclical ones and ones that give
        // rise to infinite directory trees are diagnosed by FileValue.
        continue;
      }
      String basename = dirent.getName();
      if (rootRelativePath.equals(PathFragment.EMPTY_FRAGMENT)
          && PathPackageLocator.DEFAULT_TOP_LEVEL_EXCLUDES.contains(basename)) {
        continue;
      }
      PathFragment subdirectory = rootRelativePath.getRelative(basename);

      // If this subdirectory is one of the excluded paths, don't recurse into it.
      if (excludedPaths.contains(subdirectory)) {
        continue;
      }

      // If we have an excluded path that isn't below this subdirectory, we shouldn't pass that
      // excluded path to our evaluation of the subdirectory, because the exclusion can't
      // possibly match anything beneath the subdirectory.
      //
      // For example, if we're currently evaluating directory "a", are looking at its subdirectory
      // "a/b", and we have an excluded path "a/c/d", there's no need to pass the excluded path
      // "a/c/d" to our evaluation of "a/b".
      //
      // This strategy should help to get more skyframe sharing. Consider the example above. A
      // subsequent request of "a/b/...", without any excluded paths, will be a cache hit.
      //
      // TODO(bazel-team): Replace the excludedPaths set with a trie or a SortedSet for better
      // efficiency.
      ImmutableSet<PathFragment> excludedSubdirectoriesBeneathThisSubdirectory =
          PathFragment.filterPathsStartingWith(excludedPaths, subdirectory);
      RootedPath subdirectoryRootedPath = RootedPath.toRootedPath(root, subdirectory);
      childDeps.add(getSkyKeyForSubdirectory(recursivePkgKey.getRepository(),
          subdirectoryRootedPath, excludedSubdirectoriesBeneathThisSubdirectory));
    }
    Map<SkyKey, SkyValue> subdirectorySkyValues = env.getValues(childDeps);
    if (env.valuesMissing()) {
      return null;
    }
    return aggregateWithSubdirectorySkyValues(visitor, subdirectorySkyValues);
  }

  private static boolean shouldFollowSymlinksWhenTraversing(Dirents dirents) {
    for (Dirent dirent : dirents) {
      // This is a specical sentinel file whose existence tells Blaze not to follow symlinks when
      // recursively traversing through this directory.
      //
      // This admittedly ugly feature is used to support workspaces with directories with weird
      // symlink structures that aren't intended to be consumed by Blaze.
      if (dirent.getName().equals(SENTINEL_FILE_NAME_FOR_NOT_TRAVERSING_SYMLINKS)) {
        return false;
      }
    }
    return true;
  }

  // Ignore all errors in traversal and return an empty value.
  private TReturn reportErrorAndReturn(String errorPrefix, Exception e,
      PathFragment rootRelativePath, EventHandler handler) {
    handler.handle(Event.warn(errorPrefix + ", for " + rootRelativePath
        + ", skipping: " + e.getMessage()));
    return getEmptyReturn();
  }
}
