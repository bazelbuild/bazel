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

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Dirent.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.regex.Pattern;

import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} for {@link GlobValue}s.
 *
 * <p>This code drives the glob matching process.
 */
public final class GlobFunction implements SkyFunction {

  private final Cache<String, Pattern> regexPatternCache =
      CacheBuilder.newBuilder().maximumSize(10000).concurrencyLevel(4).build();

  private final boolean alwaysUseDirListing;

  public GlobFunction(boolean alwaysUseDirListing) {
    this.alwaysUseDirListing = alwaysUseDirListing;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws GlobFunctionException {
    GlobDescriptor glob = (GlobDescriptor) skyKey.argument();

    // Note that the glob's package is assumed to exist which implies that the package's BUILD file
    // exists which implies that the package's directory exists.
    PathFragment globSubdir = glob.getSubdir();
    if (!globSubdir.equals(PathFragment.EMPTY_FRAGMENT)) {
      PackageLookupValue globSubdirPkgLookupValue = (PackageLookupValue) env.getValue(
          PackageLookupValue.key(PackageIdentifier.create(
              glob.getPackageId().getRepository(),
              glob.getPackageId().getPackageFragment().getRelative(globSubdir))));
      if (globSubdirPkgLookupValue == null) {
        return null;
      }
      if (globSubdirPkgLookupValue.packageExists()) {
        // We crossed the package boundary, that is, pkg/subdir contains a BUILD file and thus
        // defines another package, so glob expansion should not descend into that subdir.
        return GlobValue.EMPTY;
      }
    }

    String pattern = glob.getPattern();
    // Split off the first path component of the pattern.
    int slashPos = pattern.indexOf('/');
    String patternHead;
    String patternTail;
    if (slashPos == -1) {
      patternHead = pattern;
      patternTail = null;
    } else {
      // Substrings will share the backing array of the original glob string. That should be fine.
      patternHead = pattern.substring(0, slashPos);
      patternTail = pattern.substring(slashPos + 1);
    }

    NestedSetBuilder<PathFragment> matches = NestedSetBuilder.stableOrder();

    // "**" also matches an empty segment, so try the case where it is not present.
    if ("**".equals(patternHead)) {
      if (patternTail == null) {
        // Recursive globs aren't supposed to match the package's directory.
        if (!glob.excludeDirs() && !globSubdir.equals(PathFragment.EMPTY_FRAGMENT)) {
          matches.add(globSubdir);
        }
      } else {
        SkyKey globKey = GlobValue.internalKey(glob.getPackageId(), glob.getPackageRoot(),
            globSubdir, patternTail, glob.excludeDirs());
        GlobValue globValue = (GlobValue) env.getValue(globKey);
        if (globValue == null) {
          return null;
        }
        matches.addTransitive(globValue.getMatches());
      }
    }

    PathFragment dirPathFragment = glob.getPackageId().getPackageFragment().getRelative(globSubdir);
    RootedPath dirRootedPath = RootedPath.toRootedPath(glob.getPackageRoot(), dirPathFragment);
    if (alwaysUseDirListing || containsGlobs(patternHead)) {
      // Pattern contains globs, so a directory listing is required.
      //
      // Note that we have good reason to believe the directory exists: if this is the
      // top-level directory of the package, the package's existence implies the directory's
      // existence; if this is a lower-level directory in the package, then we got here from
      // previous directory listings. Filesystem operations concurrent with build could mean the
      // directory no longer exists, but DirectoryListingFunction handles that gracefully.
      DirectoryListingValue listingValue = (DirectoryListingValue)
          env.getValue(DirectoryListingValue.key(dirRootedPath));
      if (listingValue == null) {
        return null;
      }

      for (Dirent dirent : listingValue.getDirents()) {
        Type direntType = dirent.getType();
        String fileName = dirent.getName();

        boolean isDirectory = (direntType == Dirent.Type.DIRECTORY);

        if (!UnixGlob.matches(patternHead, fileName, regexPatternCache)) {
          continue;
        }

        if (direntType == Dirent.Type.SYMLINK) {
          // TODO(bazel-team): Consider extracting the symlink resolution logic.
          // For symlinks, look up the corresponding FileValue. This ensures that if the symlink
          // changes and "switches types" (say, from a file to a directory), this value will be
          // invalidated.
          RootedPath symlinkRootedPath = RootedPath.toRootedPath(glob.getPackageRoot(),
              dirPathFragment.getRelative(fileName));
          FileValue symlinkFileValue = (FileValue) env.getValue(FileValue.key(symlinkRootedPath));
          if (symlinkFileValue == null) {
            continue;
          }
          if (!symlinkFileValue.isSymlink()) {
            throw new GlobFunctionException(new InconsistentFilesystemException(
                "readdir and stat disagree about whether " + symlinkRootedPath.asPath()
                    + " is a symlink."), Transience.TRANSIENT);
          }
          if (!symlinkFileValue.exists()) {
            continue;
          }
          isDirectory = symlinkFileValue.isDirectory();
        }

        String subdirPattern = "**".equals(patternHead) ? glob.getPattern() : patternTail;
        addFile(fileName, glob, subdirPattern, patternTail == null, isDirectory,
            matches, env);
      }
    } else {
      // Pattern does not contain globs, so a direct stat is enough.
      String fileName = patternHead;
      RootedPath fileRootedPath = RootedPath.toRootedPath(glob.getPackageRoot(),
          dirPathFragment.getRelative(fileName));
      FileValue fileValue = (FileValue) env.getValue(FileValue.key(fileRootedPath));
      if (fileValue == null) {
        return null;
      }
      if (fileValue.exists()) {
        addFile(fileName, glob, patternTail, patternTail == null,
            fileValue.isDirectory(), matches, env);
      }
    }

    if (env.valuesMissing()) {
      return null;
    }

    NestedSet<PathFragment> matchesBuilt = matches.build();
    // Use the same value to represent that we did not match anything.
    if (matchesBuilt.isEmpty()) {
      return GlobValue.EMPTY;
    }
    return new GlobValue(matchesBuilt);
  }

  /**
   * Returns true if the given pattern contains globs.
   */
  private boolean containsGlobs(String pattern) {
    return pattern.contains("*") || pattern.contains("?");
  }

  /**
   * Includes the given file/directory in the glob.
   *
   * <p>{@code fileName} must exist.
   *
   * <p>{@code isDirectory} must be true iff the file is a directory.
   *
   * <p>{@code directResult} must be set if the file should be included in the result set
   * directly rather than recursed into if it is a directory.
   */
  private void addFile(String fileName, GlobDescriptor glob, String subdirPattern,
      boolean directResult, boolean isDirectory, NestedSetBuilder<PathFragment> matches,
      Environment env) {
    if (isDirectory && subdirPattern != null) {
      // This is a directory, and the pattern covers files under that directory.
      SkyKey subdirGlobKey = GlobValue.internalKey(glob.getPackageId(), glob.getPackageRoot(),
          glob.getSubdir().getRelative(fileName), subdirPattern, glob.excludeDirs());
      GlobValue subdirGlob = (GlobValue) env.getValue(subdirGlobKey);
      if (subdirGlob == null) {
        return;
      }
      matches.addTransitive(subdirGlob.getMatches());
    }

    if (directResult && !(isDirectory && glob.excludeDirs())) {
      if (isDirectory) {
        // TODO(bazel-team): Refactor. This is basically inlined code from the next recursion level.
        // Ensure that subdirectories that contain other packages are not picked up.
        PathFragment directory = glob.getPackageId().getPackageFragment()
            .getRelative(glob.getSubdir()).getRelative(fileName);
        PackageLookupValue pkgLookupValue = (PackageLookupValue)
            env.getValue(PackageLookupValue.key(PackageIdentifier.create(
                glob.getPackageId().getRepository(), directory)));
        if (pkgLookupValue == null) {
          return;
        }
        if (pkgLookupValue.packageExists()) {
          // The file is a directory and contains another package.
          return;
        }
      }
      matches.add(glob.getSubdir().getRelative(fileName));
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link GlobFunction#compute}.
   */
  private static final class GlobFunctionException extends SkyFunctionException {
    public GlobFunctionException(InconsistentFilesystemException e, Transience transience) {
      super(e, transience);
    }
  }
}
