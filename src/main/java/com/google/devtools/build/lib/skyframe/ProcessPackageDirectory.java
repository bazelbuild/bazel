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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Processes a directory that may contain a package and subdirectories for the benefit of processes
 * that traverse directories recursively, looking for packages.
 */
public class ProcessPackageDirectory {
  private static final String SENTINEL_FILE_NAME_FOR_NOT_TRAVERSING_SYMLINKS =
      "DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN";

  interface SkyKeyTransformer {
    SkyKey makeSkyKey(
        RepositoryName repository,
        RootedPath subdirectory,
        ImmutableSet<PathFragment> excludedSubdirectoriesBeneathSubdirectory);
  }

  private final BlazeDirectories directories;
  private final SkyKeyTransformer skyKeyTransformer;

  ProcessPackageDirectory(BlazeDirectories directories, SkyKeyTransformer skyKeyTransformer) {
    this.directories = directories;
    this.skyKeyTransformer = skyKeyTransformer;
  }

  /**
   * Examines {@code rootedPath} to see if it is the location of a package, and to see if it has any
   * subdirectory children that should also be examined. Returns a {@link
   * ProcessPackageDirectoryResult}, or {@code null} if required dependencies were missing.
   */
  @Nullable
  ProcessPackageDirectoryResult getPackageExistenceAndSubdirDeps(
      RootedPath rootedPath,
      RepositoryName repositoryName,
      SkyFunction.Environment env,
      Set<PathFragment> excludedPaths)
      throws InterruptedException {
    PathFragment rootRelativePath = rootedPath.getRootRelativePath();

    SkyKey fileKey = FileValue.key(rootedPath);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileKey, IOException.class);
    } catch (IOException e) {
      return reportErrorAndReturn(
          "Failed to get information about path", e, rootRelativePath, env.getListener());
    }
    if (env.valuesMissing()) {
      return null;
    }

    if (!fileValue.isDirectory()) {
      return ProcessPackageDirectoryResult.EMPTY_RESULT;
    }

    PackageIdentifier packageId = PackageIdentifier.create(repositoryName, rootRelativePath);

    if ((packageId.getRepository().isDefault() || packageId.getRepository().isMain())
        && fileValue.isSymlink()
        && fileValue
            .getUnresolvedLinkTarget()
            .startsWith(directories.getExecRootBase().asFragment())) {
      // Symlinks back to the execroot are not traversed so that we avoid convenience symlinks.
      // Note that it's not enough to just check for the convenience symlinks themselves,
      // because if the value of --symlink_prefix changes, the old symlinks are left in place. This
      // algorithm also covers more creative use cases where people create convenience symlinks
      // somewhere in the directory tree manually.
      return ProcessPackageDirectoryResult.EMPTY_RESULT;
    }

    SkyKey pkgLookupKey = PackageLookupValue.key(packageId);
    SkyKey dirListingKey = DirectoryListingValue.key(rootedPath);
    Map<
            SkyKey,
            ValueOrException2<
                NoSuchPackageException, IOException>>
        pkgLookupAndDirectoryListingDeps =
            env.getValuesOrThrow(
                ImmutableList.of(pkgLookupKey, dirListingKey),
                NoSuchPackageException.class,
                IOException.class);
    if (env.valuesMissing()) {
      return null;
    }
    PackageLookupValue pkgLookupValue;
    try {
      pkgLookupValue =
          (PackageLookupValue)
              Preconditions.checkNotNull(
                  pkgLookupAndDirectoryListingDeps.get(pkgLookupKey).get(),
                  "%s %s %s",
                  rootedPath,
                  repositoryName,
                  pkgLookupKey);
    } catch (NoSuchPackageException | InconsistentFilesystemException e) {
      return reportErrorAndReturn("Failed to load package", e, rootRelativePath, env.getListener());
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    DirectoryListingValue dirListingValue;
    try {
      dirListingValue =
          (DirectoryListingValue)
              Preconditions.checkNotNull(
                  pkgLookupAndDirectoryListingDeps.get(dirListingKey).get(),
                  "%s %s %s",
                  rootedPath,
                  repositoryName,
                  dirListingKey);
    } catch (FileSymlinkException e) {
      // DirectoryListingFunction only throws FileSymlinkCycleException when FileFunction throws it,
      // but FileFunction was evaluated for rootedPath above, and didn't throw there. It shouldn't
      // be able to avoid throwing there but throw here.
      throw new IllegalStateException(
          "Symlink cycle found after not being found for \"" + rootedPath + "\"");
    } catch (IOException e) {
      return reportErrorAndReturn(
          "Failed to list directory contents", e, rootRelativePath, env.getListener());
    } catch (NoSuchPackageException e) {
      throw new IllegalStateException(e);
    }
    return new ProcessPackageDirectoryResult(
        pkgLookupValue.packageExists() && pkgLookupValue.getRoot().equals(rootedPath.getRoot()),
        getSubdirDeps(dirListingValue, rootedPath, repositoryName, excludedPaths));
  }

  private Iterable<SkyKey> getSubdirDeps(
      DirectoryListingValue dirListingValue,
      RootedPath rootedPath,
      RepositoryName repositoryName,
      Set<PathFragment> excludedPaths) {
    Root root = rootedPath.getRoot();
    PathFragment rootRelativePath = rootedPath.getRootRelativePath();
    boolean followSymlinks = shouldFollowSymlinksWhenTraversing(dirListingValue.getDirents());
    List<SkyKey> childDeps = new ArrayList<>();
    for (Dirent dirent : dirListingValue.getDirents()) {
      Dirent.Type type = dirent.getType();
      if (type != Dirent.Type.DIRECTORY && (type != Dirent.Type.SYMLINK || !followSymlinks)) {
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
      PathFragment subdirectory = rootRelativePath.getRelative(basename);
      if (subdirectory.equals(Label.EXTERNAL_PACKAGE_NAME)) {
        // Not a real package.
        continue;
      }

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
          excludedPaths
              .stream()
              .filter(pathFragment -> pathFragment.startsWith(subdirectory))
              .collect(toImmutableSet());
      RootedPath subdirectoryRootedPath = RootedPath.toRootedPath(root, subdirectory);
      childDeps.add(
          skyKeyTransformer.makeSkyKey(
              repositoryName,
              subdirectoryRootedPath,
              excludedSubdirectoriesBeneathThisSubdirectory));
    }
    return childDeps;
  }

  private static ProcessPackageDirectoryResult reportErrorAndReturn(
      String errorPrefix, Exception e, PathFragment rootRelativePath, EventHandler handler) {
    handler.handle(
        Event.error(errorPrefix + ", for " + rootRelativePath + ", skipping: " + e.getMessage()));
    return ProcessPackageDirectoryResult.EMPTY_RESULT;
  }

  private static boolean shouldFollowSymlinksWhenTraversing(Dirents dirents) {
    for (Dirent dirent : dirents) {
      // This is a special sentinel file whose existence tells Blaze not to follow symlinks when
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
}
