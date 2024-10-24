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


import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.io.FileSymlinkException;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionException;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionUniquenessFunction;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.io.ProcessPackageDirectoryException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Processes a directory that may contain a package and subdirectories for the benefit of processes
 * that traverse directories recursively, looking for packages.
 */
public final class ProcessPackageDirectory {
  private static final String SENTINEL_FILE_NAME_FOR_NOT_TRAVERSING_SYMLINKS =
      "DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN";

  /** Produces a {@link SkyKey} for the recursive traversal into the specified subdirectory. */
  public interface SkyKeyTransformer {
    SkyKey makeSkyKey(
        RepositoryName repository,
        RootedPath subdirectory,
        IgnoredSubdirectories excludedSubdirectoriesBeneathSubdirectory);
  }

  private final BlazeDirectories directories;
  private final SkyKeyTransformer skyKeyTransformer;

  public ProcessPackageDirectory(
      BlazeDirectories directories, SkyKeyTransformer skyKeyTransformer) {
    this.directories = directories;
    this.skyKeyTransformer = skyKeyTransformer;
  }

  /**
   * Examines {@code rootedPath} to see if it is the location of a package, and to see if it has any
   * subdirectory children that should also be examined. Returns a {@link
   * ProcessPackageDirectoryResult}, or {@code null} if required dependencies were missing.
   */
  @Nullable
  public ProcessPackageDirectoryResult getPackageExistenceAndSubdirDeps(
      RootedPath rootedPath,
      RepositoryName repositoryName,
      IgnoredSubdirectories excludedPaths,
      SkyFunction.Environment env)
      throws InterruptedException, ProcessPackageDirectorySkyFunctionException {
    PathFragment rootRelativePath = rootedPath.getRootRelativePath();

    SkyKey fileKey = FileValue.key(rootedPath);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileKey, IOException.class);
    } catch (InconsistentFilesystemException e) {
      throw new ProcessPackageDirectorySkyFunctionException(rootedPath, e);
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

    if (fileValue.unboundedAncestorSymlinkExpansionChain() != null) {
      SkyKey uniquenessKey =
          FileSymlinkInfiniteExpansionUniquenessFunction.key(
              fileValue.unboundedAncestorSymlinkExpansionChain());
      env.getValue(uniquenessKey);
      if (env.valuesMissing()) {
        return null;
      }

      FileSymlinkInfiniteExpansionException symlinkException =
          new FileSymlinkInfiniteExpansionException(
              fileValue.pathToUnboundedAncestorSymlinkExpansionChain(),
              fileValue.unboundedAncestorSymlinkExpansionChain());
      return reportErrorAndReturn(
          symlinkException.getMessage(), symlinkException, rootRelativePath, env.getListener());
    }

    PackageIdentifier packageId = PackageIdentifier.create(repositoryName, rootRelativePath);

    if (packageId.getRepository().isMain() && isConvenienceSymlink(fileValue, rootedPath, env)) {
      return ProcessPackageDirectoryResult.EMPTY_RESULT;
    }

    if (env.valuesMissing()) {
      return null;
    }

    SkyKey pkgLookupKey = PackageLookupValue.key(packageId);
    SkyKey dirListingKey = DirectoryListingValue.key(rootedPath);
    SkyframeLookupResult pkgLookupAndDirectoryListingDeps =
        env.getValuesAndExceptions(ImmutableList.of(pkgLookupKey, dirListingKey));
    PackageLookupValue pkgLookupValue;
    try {
      pkgLookupValue =
          (PackageLookupValue)
              pkgLookupAndDirectoryListingDeps.getOrThrow(
                  pkgLookupKey,
                  NoSuchPackageException.class,
                  InconsistentFilesystemException.class);
    } catch (NoSuchPackageException e) {
      return reportErrorAndReturn("Failed to load package", e, rootRelativePath, env.getListener());
    } catch (InconsistentFilesystemException e) {
      throw new ProcessPackageDirectorySkyFunctionException(rootedPath, e);
    }
    DirectoryListingValue dirListingValue;
    try {
      dirListingValue =
          (DirectoryListingValue)
              pkgLookupAndDirectoryListingDeps.getOrThrow(dirListingKey, IOException.class);
    } catch (FileSymlinkException e) {
      // DirectoryListingFunction only throws FileSymlinkCycleException when FileFunction throws it,
      // but FileFunction was evaluated for rootedPath above, and didn't throw there. It shouldn't
      // be able to avoid throwing there but throw here.
      throw new IllegalStateException(
          "Symlink cycle found after not being found for \"" + rootedPath + "\"", e);
    } catch (IOException e) {
      return reportErrorAndReturn(
          "Failed to list directory contents", e, rootRelativePath, env.getListener());
    }
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (env.valuesMissing()) {
      return null;
    }
    Preconditions.checkNotNull(
        pkgLookupValue, "%s %s %s", rootedPath, repositoryName, pkgLookupKey);
    Preconditions.checkNotNull(
        dirListingValue, "%s %s %s", rootedPath, repositoryName, dirListingKey);
    return new ProcessPackageDirectoryResult(
        pkgLookupValue.packageExists() && pkgLookupValue.getRoot().equals(rootedPath.getRoot()),
        getSubdirDeps(
            dirListingValue,
            rootedPath,
            repositoryName,
            excludedPaths,
            starlarkSemantics.getBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT)),
        /*additionalValuesToAggregate=*/ ImmutableMap.of());
  }

  // Note that it's not enough to just check for the convenience symlinks themselves,
  // because if the value of --symlink_prefix changes, the old symlinks are left in place. It
  // is also not sufficient to check whether the symlink points to a directory in the current
  // exec root, since this can change between bazel invocations. Therefore we check if the
  // suffix of the symlink source suggests it is a convenience symlink, then see if the symlink
  // target is in a directory that looks like an execroot. This algorithm also covers more
  // creative use cases where people create convenience symlinks somewhere in the directory
  // tree manually.
  private boolean isConvenienceSymlink(
      FileValue fileValue, RootedPath rootedPath, SkyFunction.Environment env)
      throws InterruptedException {
    if (!fileValue.isSymlink()) {
      return false;
    }

    PathFragment linkTarget = fileValue.getUnresolvedLinkTarget();

    if (linkTarget.startsWith(directories.getExecRootBase().asFragment())) {
      return true;
    }

    PathFragment rootRelativePath = rootedPath.getRootRelativePath();
    Root root = rootedPath.getRoot();

    if (rootRelativePath.getBaseName().endsWith("-bin") && isInExecRoot(linkTarget, root, 4, env)) {
      return true;
    }

    if (rootRelativePath.getBaseName().endsWith("-genfiles")
        && isInExecRoot(linkTarget, root, 4, env)) {
      return true;
    }

    if (rootRelativePath.getBaseName().endsWith("-out") && isInExecRoot(linkTarget, root, 2, env)) {
      return true;
    }

    if (rootRelativePath.getBaseName().endsWith("-testlogs")
        && isInExecRoot(linkTarget, root, 4, env)) {
      return true;
    }

    if (rootRelativePath
            .getBaseName()
            .endsWith("-" + directories.getWorkingDirectory().getBaseName())
        && isInExecRoot(linkTarget, root, 1, env)) {
      return true;
    }

    return false;
  }

  private boolean isInExecRoot(PathFragment path, Root root, int depth, SkyFunction.Environment env)
      throws InterruptedException {
    int segmentCount = path.segmentCount();

    if (segmentCount <= depth) {
      return false;
    }

    PathFragment candidateExecRoot = path.subFragment(0, segmentCount - depth);

    if (!candidateExecRoot.getBaseName().equals("execroot")) {
      return false;
    }

    Root absoluteRoot = Root.absoluteRoot(root.getFileSystem());
    RootedPath doNotBuildPath =
        RootedPath.toRootedPath(absoluteRoot, candidateExecRoot.getChild("DO_NOT_BUILD_HERE"));
    FileValue doNotBuildValue = (FileValue) env.getValue(FileValue.key(doNotBuildPath));
    if (doNotBuildValue == null) {
      return false;
    }

    return doNotBuildValue.exists();
  }

  private Iterable<SkyKey> getSubdirDeps(
      DirectoryListingValue dirListingValue,
      RootedPath rootedPath,
      RepositoryName repositoryName,
      IgnoredSubdirectories excludedPaths,
      boolean siblingRepositoryLayout) {
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
      if (!siblingRepositoryLayout && subdirectory.equals(LabelConstants.EXTERNAL_PACKAGE_NAME)) {
        // Subpackages under //external can be processed only when
        // --experimental_sibling_repository_layout is set.
        continue;
      }

      // If this subdirectory is one of the excluded paths, don't recurse into it.
      if (excludedPaths.matchingEntry(subdirectory) != null) {
        continue;
      }

      childDeps.add(
          skyKeyTransformer.makeSkyKey(
              repositoryName,
              RootedPath.toRootedPath(root, subdirectory),
              excludedPaths.filterForDirectory(subdirectory)));
    }
    return childDeps;
  }

  /**
   * Returns the 'excludedPaths' set to use when recursing below this subdirectory. If we have an
   * excluded path that isn't below this subdirectory, we shouldn't pass that excluded path to our
   * evaluation of the subdirectory, because the exclusion can't possibly match anything beneath the
   * subdirectory.
   *
   * <p>For example, if we're currently evaluating directory "a", are looking at its subdirectory
   * "a/b", and we have an excluded path "a/c/d", there's no need to pass the excluded path "a/c/d"
   * to our evaluation of "a/b". This strategy should help to get more skyframe sharing. In our
   * example, a subsequent request of "a/b/...", without any excluded paths, will be a cache hit.
   *
   * <p>TODO(bazel-team): Replace the excludedPaths set with a trie or a SortedSet for better
   * efficiency.
   */
  public static IgnoredSubdirectories getExcludedSubdirectoriesBeneathSubdirectory(
      PathFragment subdirectory, IgnoredSubdirectories excludedPaths) {
    return excludedPaths.filterForDirectory(subdirectory);
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

  /** Wraps {@link InconsistentFilesystemException} in {@link ProcessPackageDirectoryException}. */
  public static final class ProcessPackageDirectorySkyFunctionException
      extends SkyFunctionException {
    public ProcessPackageDirectorySkyFunctionException(
        RootedPath directory, InconsistentFilesystemException e) {
      super(new ProcessPackageDirectoryException(directory, e), Transience.PERSISTENT);
    }

    @Override
    public boolean isCatastrophic() {
      return true;
    }
  }
}
