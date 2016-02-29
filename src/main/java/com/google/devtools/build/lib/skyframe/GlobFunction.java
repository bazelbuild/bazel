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
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.Preconditions;
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

import java.util.Map;
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
      PackageLookupValue globSubdirPkgLookupValue =
          (PackageLookupValue)
              env.getValue(
                  PackageLookupValue.key(
                      PackageIdentifier.create(
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

    boolean globMatchesBareFile = patternTail == null;

    // "**" also matches an empty segment, so try the case where it is not present.
    if ("**".equals(patternHead)) {
      if (globMatchesBareFile) {
        // Recursive globs aren't supposed to match the package's directory.
        if (!glob.excludeDirs() && !globSubdir.equals(PathFragment.EMPTY_FRAGMENT)) {
          matches.add(globSubdir);
        }
      } else {
        SkyKey globKey =
            GlobValue.internalKey(
                glob.getPackageId(),
                glob.getPackageRoot(),
                globSubdir,
                patternTail,
                glob.excludeDirs());
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
      String subdirPattern = "**".equals(patternHead) ? glob.getPattern() : patternTail;
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

      // In order to batch Skyframe requests, we do three passes over the directory:
      // (1) Process every dirent, keeping track of values we need to request if the dirent cannot
      //     be processed with current information (symlink targets and subdirectory globs/package
      //     lookups for some subdirectories).
      // (2) Get those values and process the symlinks, keeping track of subdirectory globs/package
      //     lookups we may need to request in case the symlink's target is a directory.
      // (3) Process the necessary subdirectories.
      int direntsSize = listingValue.getDirents().size();
      Map<Dirent, SkyKey> symlinkFileMap = Maps.newHashMapWithExpectedSize(direntsSize);
      Map<Dirent, SkyKey> firstPassSubdirMap = Maps.newHashMapWithExpectedSize(direntsSize);
      // First pass: do normal files and collect SkyKeys to request.
      for (Dirent dirent : listingValue.getDirents()) {
        Type direntType = dirent.getType();
        String fileName = dirent.getName();
        if (!UnixGlob.matches(patternHead, fileName, regexPatternCache)) {
          continue;
        }

        if (direntType == Dirent.Type.SYMLINK) {
          // TODO(bazel-team): Consider extracting the symlink resolution logic.
          // For symlinks, look up the corresponding FileValue. This ensures that if the symlink
          // changes and "switches types" (say, from a file to a directory), this value will be
          // invalidated. We also need the target's type to properly process the symlink.
          symlinkFileMap.put(
              dirent,
              FileValue.key(
                  RootedPath.toRootedPath(
                      glob.getPackageRoot(), dirPathFragment.getRelative(fileName))));
          continue;
        }

        if (direntType == Dirent.Type.DIRECTORY) {
          SkyKey keyToRequest = getSkyKeyForSubdir(fileName, glob, subdirPattern);
          if (keyToRequest != null) {
            firstPassSubdirMap.put(dirent, keyToRequest);
          }
        } else if (globMatchesBareFile) {
          matches.add(glob.getSubdir().getRelative(fileName));
        }
      }

      Map<SkyKey, SkyValue> firstPassAndSymlinksResult =
          env.getValues(Iterables.concat(firstPassSubdirMap.values(), symlinkFileMap.values()));
      if (env.valuesMissing()) {
        return null;
      }
      // Second pass: do symlinks, and maybe collect further SkyKeys if targets are directories.
      Map<Dirent, SkyKey> symlinkSubdirMap = Maps.newHashMapWithExpectedSize(symlinkFileMap.size());
      for (Map.Entry<Dirent, SkyKey> direntAndKey : symlinkFileMap.entrySet()) {
        Dirent dirent = direntAndKey.getKey();
        String fileName = dirent.getName();
        Preconditions.checkState(dirent.getType() == Dirent.Type.SYMLINK, direntAndKey);
        FileValue symlinkFileValue =
            Preconditions.checkNotNull(
                (FileValue) firstPassAndSymlinksResult.get(direntAndKey.getValue()), direntAndKey);
        if (!symlinkFileValue.isSymlink()) {
          throw new GlobFunctionException(
              new InconsistentFilesystemException(
                  "readdir and stat disagree about whether "
                      + ((RootedPath) direntAndKey.getValue().argument()).asPath()
                      + " is a symlink."),
              Transience.TRANSIENT);
        }
        if (!symlinkFileValue.exists()) {
          continue;
        }
        if (symlinkFileValue.isDirectory()) {
          SkyKey keyToRequest = getSkyKeyForSubdir(fileName, glob, subdirPattern);
          if (keyToRequest != null) {
            symlinkSubdirMap.put(dirent, keyToRequest);
          }
        } else if (globMatchesBareFile) {
          matches.add(glob.getSubdir().getRelative(fileName));
        }
      }

      Map<SkyKey, SkyValue> secondResult = env.getValues(symlinkSubdirMap.values());
      if (env.valuesMissing()) {
        return null;
      }
      // Third pass: do needed subdirectories.
      for (Map.Entry<Dirent, SkyKey> direntAndKey :
          Iterables.concat(firstPassSubdirMap.entrySet(), symlinkSubdirMap.entrySet())) {
        Dirent dirent = direntAndKey.getKey();
        String fileName = dirent.getName();
        SkyKey key = direntAndKey.getValue();
        SkyValue valueRequested =
            symlinkSubdirMap.containsKey(dirent)
                ? secondResult.get(key)
                : firstPassAndSymlinksResult.get(key);
        Preconditions.checkNotNull(valueRequested, direntAndKey);
        addSubdirMatchesFromSkyValue(fileName, glob, matches, valueRequested);
      }
    } else {
      // Pattern does not contain globs, so a direct stat is enough.
      String fileName = patternHead;
      RootedPath fileRootedPath =
          RootedPath.toRootedPath(glob.getPackageRoot(), dirPathFragment.getRelative(fileName));
      FileValue fileValue = (FileValue) env.getValue(FileValue.key(fileRootedPath));
      if (fileValue == null) {
        return null;
      }
      if (fileValue.exists()) {
        if (fileValue.isDirectory()) {
          SkyKey keyToRequest = getSkyKeyForSubdir(fileName, glob, patternTail);
          if (keyToRequest != null) {
            SkyValue valueRequested = env.getValue(keyToRequest);
            if (env.valuesMissing()) {
              return null;
            }
            addSubdirMatchesFromSkyValue(fileName, glob, matches, valueRequested);
          }
        } else if (globMatchesBareFile) {
          matches.add(glob.getSubdir().getRelative(fileName));
        }
      }
    }

    Preconditions.checkState(!env.valuesMissing(), skyKey);

    NestedSet<PathFragment> matchesBuilt = matches.build();
    // Use the same value to represent that we did not match anything.
    if (matchesBuilt.isEmpty()) {
      return GlobValue.EMPTY;
    }
    return new GlobValue(matchesBuilt);
  }

  /** Returns true if the given pattern contains globs. */
  private static boolean containsGlobs(String pattern) {
    return pattern.contains("*") || pattern.contains("?");
  }

  /**
   * Includes the given file/directory in the glob.
   *
   * <p>{@code fileName} must exist.
   *
   * <p>{@code isDirectory} must be true iff the file is a directory.
   *
   * <p>Returns a {@link SkyKey} for a value that is needed to compute the files that will be added
   * to {@code matches}, or {@code null} if no additional value is needed. The returned value should
   * be opaquely passed to {@link #addSubdirMatchesFromSkyValue}.
   */
  private static SkyKey getSkyKeyForSubdir(
      String fileName, GlobDescriptor glob, String subdirPattern) {
    if (subdirPattern == null) {
      if (glob.excludeDirs()) {
        return null;
      } else {
        return PackageLookupValue.key(
            PackageIdentifier.create(
                glob.getPackageId().getRepository(),
                glob.getPackageId()
                    .getPackageFragment()
                    .getRelative(glob.getSubdir())
                    .getRelative(fileName)));
      }
    } else {
      // There is some more pattern to match. Get the glob for the subdirectory. Note that this
      // directory may also match directly in the case of a pattern that starts with "**", but that
      // match will be found in the subdirectory glob.
      return GlobValue.internalKey(
          glob.getPackageId(),
          glob.getPackageRoot(),
          glob.getSubdir().getRelative(fileName),
          subdirPattern,
          glob.excludeDirs());
    }
  }

  /**
   * Add matches to {@code matches} coming from the directory {@code fileName} if appropriate.
   *
   * <p>{@code valueRequested} must be the SkyValue whose key was returned by
   * {@link #getSkyKeyForSubdir} for these parameters.
   */
  private static void addSubdirMatchesFromSkyValue(
      String fileName,
      GlobDescriptor glob,
      NestedSetBuilder<PathFragment> matches,
      SkyValue valueRequested) {
    if (valueRequested instanceof GlobValue) {
      matches.addTransitive(((GlobValue) valueRequested).getMatches());
    } else {
      Preconditions.checkState(
          valueRequested instanceof PackageLookupValue,
          "%s is not a GlobValue or PackageLookupValue (%s %s)",
          valueRequested,
          fileName,
          glob);
      if (!((PackageLookupValue) valueRequested).packageExists()) {
        matches.add(glob.getSubdir().getRelative(fileName));
      }
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
