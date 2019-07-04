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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} for {@link GlobValue}s.
 *
 * <p>This code drives the glob matching process.
 */
public final class GlobFunction implements SkyFunction {

  private final ConcurrentHashMap<String, Pattern> regexPatternCache = new ConcurrentHashMap<>();

  private final boolean alwaysUseDirListing;

  public GlobFunction(boolean alwaysUseDirListing) {
    this.alwaysUseDirListing = alwaysUseDirListing;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws GlobFunctionException, InterruptedException {
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
      } else if (globSubdirPkgLookupValue
          instanceof PackageLookupValue.IncorrectRepositoryReferencePackageLookupValue) {
        // We crossed a repository boundary, so glob expansion should not descend into that subdir.
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
      SkyKey directoryListingKey = DirectoryListingValue.key(dirRootedPath);
      DirectoryListingValue listingValue = null;

      boolean patternHeadIsStarStar = "**".equals(patternHead);
      if (patternHeadIsStarStar) {
        // "**" also matches an empty segment, so try the case where it is not present.
        if (globMatchesBareFile) {
          // Recursive globs aren't supposed to match the package's directory.
          if (!glob.excludeDirs() && !globSubdir.equals(PathFragment.EMPTY_FRAGMENT)) {
            matches.add(globSubdir);
          }
        } else {
          // Optimize away a Skyframe restart by requesting the DirectoryListingValue dep and
          // recursive GlobValue dep in a single batch.

          SkyKey keyForRecursiveGlobInCurrentDirectory =
              GlobValue.internalKey(
                  glob.getPackageId(),
                  glob.getPackageRoot(),
                  globSubdir,
                  patternTail,
                  glob.excludeDirs());
          Map<SkyKey, SkyValue> listingAndRecursiveGlobMap =
              env.getValues(
                  ImmutableList.of(keyForRecursiveGlobInCurrentDirectory, directoryListingKey));
          if (env.valuesMissing()) {
            return null;
          }
          GlobValue globValue =
              (GlobValue) listingAndRecursiveGlobMap.get(keyForRecursiveGlobInCurrentDirectory);
          matches.addTransitive(globValue.getMatches());
          listingValue =
              (DirectoryListingValue) listingAndRecursiveGlobMap.get(directoryListingKey);
        }
      }

      if (listingValue == null) {
        listingValue = (DirectoryListingValue) env.getValue(directoryListingKey);
        if (listingValue == null) {
          return null;
        }
      }

      // Now that we have the directory listing, we do three passes over it so as to maximize
      // skyframe batching:
      // (1) Process every dirent, keeping track of values we need to request if the dirent cannot
      //     be processed with current information (symlink targets and subdirectory globs/package
      //     lookups for some subdirectories).
      // (2) Get those values and process the symlinks, keeping track of subdirectory globs/package
      //     lookups we may need to request in case the symlink's target is a directory.
      // (3) Process the necessary subdirectories.
      int direntsSize = listingValue.getDirents().size();
      Map<SkyKey, Dirent> symlinkFileMap = Maps.newHashMapWithExpectedSize(direntsSize);
      Map<SkyKey, Dirent> subdirMap = Maps.newHashMapWithExpectedSize(direntsSize);
      Map<Dirent, Object> sortedResultMap = Maps.newTreeMap();
      String subdirPattern = patternHeadIsStarStar ? glob.getPattern() : patternTail;
      // First pass: do normal files and collect SkyKeys to request for subdirectories and symlinks.
      for (Dirent dirent : listingValue.getDirents()) {
        Dirent.Type direntType = dirent.getType();
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
              FileValue.key(
                  RootedPath.toRootedPath(
                      glob.getPackageRoot(), dirPathFragment.getRelative(fileName))),
              dirent);
          continue;
        }

        if (direntType == Dirent.Type.DIRECTORY) {
          SkyKey keyToRequest = getSkyKeyForSubdir(fileName, glob, subdirPattern);
          if (keyToRequest != null) {
            subdirMap.put(keyToRequest, dirent);
          }
        } else if (globMatchesBareFile) {
          sortedResultMap.put(dirent, glob.getSubdir().getRelative(fileName));
        }
      }

      Map<SkyKey, SkyValue> subdirAndSymlinksResult =
          env.getValues(Sets.union(subdirMap.keySet(), symlinkFileMap.keySet()));
      if (env.valuesMissing()) {
        return null;
      }
      Map<SkyKey, Dirent> symlinkSubdirMap = Maps.newHashMapWithExpectedSize(symlinkFileMap.size());
      // Second pass: process the symlinks and subdirectories from the first pass, and maybe
      // collect further SkyKeys if fully resolved symlink targets are themselves directories.
      // Also process any known directories.
      for (Map.Entry<SkyKey, SkyValue> lookedUpKeyAndValue : subdirAndSymlinksResult.entrySet()) {
        if (symlinkFileMap.containsKey(lookedUpKeyAndValue.getKey())) {
          FileValue symlinkFileValue = (FileValue) lookedUpKeyAndValue.getValue();
          if (!symlinkFileValue.isSymlink()) {
            throw new GlobFunctionException(
                new InconsistentFilesystemException(
                    "readdir and stat disagree about whether "
                        + ((RootedPath) lookedUpKeyAndValue.getKey().argument()).asPath()
                        + " is a symlink."),
                Transience.TRANSIENT);
          }
          if (!symlinkFileValue.exists()) {
            continue;
          }
          Dirent dirent = symlinkFileMap.get(lookedUpKeyAndValue.getKey());
          String fileName = dirent.getName();
          if (symlinkFileValue.isDirectory()) {
            SkyKey keyToRequest = getSkyKeyForSubdir(fileName, glob, subdirPattern);
            if (keyToRequest != null) {
              symlinkSubdirMap.put(keyToRequest, dirent);
            }
          } else if (globMatchesBareFile) {
            sortedResultMap.put(dirent, glob.getSubdir().getRelative(fileName));
          }
        } else {
          processSubdir(lookedUpKeyAndValue, subdirMap, glob, sortedResultMap);
        }
      }

      Map<SkyKey, SkyValue> symlinkSubdirResult = env.getValues(symlinkSubdirMap.keySet());
      if (env.valuesMissing()) {
        return null;
      }
      // Third pass: do needed subdirectories of symlinked directories discovered during the second
      // pass.
      for (Map.Entry<SkyKey, SkyValue> lookedUpKeyAndValue : symlinkSubdirResult.entrySet()) {
        processSubdir(lookedUpKeyAndValue, symlinkSubdirMap, glob, sortedResultMap);
      }
      for (Map.Entry<Dirent, Object> fileMatches : sortedResultMap.entrySet()) {
        addToMatches(fileMatches.getValue(), matches);
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
            Object fileMatches = getSubdirMatchesFromSkyValue(fileName, glob, valueRequested);
            if (fileMatches != null) {
              addToMatches(fileMatches, matches);
            }
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

  private static void processSubdir(
      Map.Entry<SkyKey, SkyValue> keyAndValue,
      Map<SkyKey, Dirent> subdirMap,
      GlobDescriptor glob,
      Map<Dirent, Object> sortedResultMap) {
    Dirent dirent = Preconditions.checkNotNull(subdirMap.get(keyAndValue.getKey()), keyAndValue);
    String fileName = dirent.getName();
    Object dirMatches = getSubdirMatchesFromSkyValue(fileName, glob, keyAndValue.getValue());
    if (dirMatches != null) {
      sortedResultMap.put(dirent, dirMatches);
    }
  }

  /** Returns true if the given pattern contains globs. */
  private static boolean containsGlobs(String pattern) {
    return pattern.contains("*") || pattern.contains("?");
  }

  @SuppressWarnings("unchecked") // cast to NestedSet<PathFragment>
  private static void addToMatches(Object toAdd, NestedSetBuilder<PathFragment> matches) {
    if (toAdd instanceof PathFragment) {
      matches.add((PathFragment) toAdd);
    } else {
      matches.addTransitive((NestedSet<PathFragment>) toAdd);
    }
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
   * be opaquely passed to {@link #getSubdirMatchesFromSkyValue}.
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
   * Returns matches coming from the directory {@code fileName} if appropriate, either an individual
   * file or a nested set of files.
   *
   * <p>{@code valueRequested} must be the SkyValue whose key was returned by
   * {@link #getSkyKeyForSubdir} for these parameters.
   */
  @Nullable
  private static Object getSubdirMatchesFromSkyValue(
      String fileName,
      GlobDescriptor glob,
      SkyValue valueRequested) {
    if (valueRequested instanceof GlobValue) {
      return ((GlobValue) valueRequested).getMatches();
    } else {
      Preconditions.checkState(
          valueRequested instanceof PackageLookupValue,
          "%s is not a GlobValue or PackageLookupValue (%s %s)",
          valueRequested,
          fileName,
          glob);
      PackageLookupValue packageLookupValue = (PackageLookupValue) valueRequested;
      if (packageLookupValue.packageExists()) {
        // This is a separate package, so ignore it.
        return null;
      } else if (packageLookupValue
          instanceof PackageLookupValue.IncorrectRepositoryReferencePackageLookupValue) {
        // This is a separate repository, so ignore it.
        return null;
      } else {
        return glob.getSubdir().getRelative(fileName);
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
