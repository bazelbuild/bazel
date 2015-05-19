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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.RecursivePkgValue.RecursivePkgKey;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Dirent.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * RecursivePkgFunction builds up the set of packages underneath a given directory
 * transitively.
 *
 * <p>Example: foo/BUILD, foo/sub/x, foo/subpkg/BUILD would yield transitive packages "foo" and
 * "foo/subpkg".
 */
public class RecursivePkgFunction implements SkyFunction {

  private static final Order ORDER = Order.STABLE_ORDER;

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    RecursivePkgKey recursivePkgKey = (RecursivePkgKey) skyKey.argument();
    RootedPath rootedPath = recursivePkgKey.getRootedPath();
    Path root = rootedPath.getRoot();
    PathFragment rootRelativePath = rootedPath.getRelativePath();
    Set<PathFragment> excludedPaths = recursivePkgKey.getExcludedPaths();

    SkyKey fileKey = FileValue.key(rootedPath);
    FileValue fileValue = (FileValue) env.getValue(fileKey);
    if (fileValue == null) {
      return null;
    }

    if (!fileValue.isDirectory()) {
      return new RecursivePkgValue(NestedSetBuilder.<String>emptySet(ORDER));
    }

    if (fileValue.isSymlink()) {
      // We do not follow directory symlinks when we look recursively for packages. It also
      // prevents symlink loops.
      return new RecursivePkgValue(NestedSetBuilder.<String>emptySet(ORDER));
    }

    PackageIdentifier packageId = PackageIdentifier.createInDefaultRepo(
        rootRelativePath.getPathString());
    PackageLookupValue pkgLookupValue =
        (PackageLookupValue) env.getValue(PackageLookupValue.key(packageId));
    if (pkgLookupValue == null) {
      return null;
    }

    NestedSetBuilder<String> packages = new NestedSetBuilder<>(ORDER);

    if (pkgLookupValue.packageExists()) {
      if (pkgLookupValue.getRoot().equals(root)) {
        try {
          PackageValue pkgValue = (PackageValue)
              env.getValueOrThrow(PackageValue.key(packageId), NoSuchPackageException.class);
          if (pkgValue == null) {
            return null;
          }
          packages.add(pkgValue.getPackage().getName());
        } catch (NoSuchPackageException e) {
          // The package had errors, but don't fail-fast as there might be subpackages below the
          // current directory.
          env.getListener().handle(Event.error(
              "package contains errors: " + rootRelativePath.getPathString()));
          if (e.getPackage() != null) {
            packages.add(e.getPackage().getName());
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

    DirectoryListingValue dirValue = (DirectoryListingValue)
        env.getValue(DirectoryListingValue.key(rootedPath));
    if (dirValue == null) {
      return null;
    }

    List<SkyKey> childDeps = Lists.newArrayList();
    for (Dirent dirent : dirValue.getDirents()) {
      if (dirent.getType() != Type.DIRECTORY) {
        // Non-directories can never host packages, and we do not follow symlinks (see above).
        continue;
      }
      String basename = dirent.getName();
      if (rootRelativePath.equals(PathFragment.EMPTY_FRAGMENT)
          && PathPackageLocator.DEFAULT_TOP_LEVEL_EXCLUDES.contains(basename)) {
        continue;
      }
      PathFragment subdirectory = rootRelativePath.getRelative(basename);

      // If this subdirectory is one of the excluded paths, don't do package lookups in it.
      if (excludedPaths.contains(subdirectory)) {
        continue;
      }

      // If we have an excluded path that isn't below this subdirectory, we shouldn't pass that
      // excluded path to our recursive package lookup of the subdirectory, because the exclusion
      // can't possibly match anything beneath the subdirectory.
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
      ImmutableSet.Builder<PathFragment> excludedSubdirectoriesBeneathThisSubdirectory =
          ImmutableSet.builder();
      for (PathFragment excludedPath : excludedPaths) {
        if (excludedPath.startsWith(subdirectory)) {
          excludedSubdirectoriesBeneathThisSubdirectory.add(excludedPath);
        }
      }
      SkyKey req = RecursivePkgValue.key(RootedPath.toRootedPath(root,
          subdirectory), excludedSubdirectoriesBeneathThisSubdirectory.build());
      childDeps.add(req);
    }
    Map<SkyKey, SkyValue> childValueMap = env.getValues(childDeps);
    if (env.valuesMissing()) {
      return null;
    }
    // Aggregate the transitive subpackages.
    for (SkyValue childValue : childValueMap.values()) {
      if (childValue != null) {
        packages.addTransitive(((RecursivePkgValue) childValue).getPackages());
      }
    }
    return new RecursivePkgValue(packages.build());
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
