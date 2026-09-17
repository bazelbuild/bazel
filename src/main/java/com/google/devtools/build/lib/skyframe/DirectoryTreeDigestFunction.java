// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;

/** A {@link SkyFunction} for {@link DirectoryTreeDigestValue}s. */
public final class DirectoryTreeDigestFunction implements SkyFunction {
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, DirectoryTreeDigestFunctionException {
    Map<String, Pattern> patternCache = new HashMap<>();
    DirectoryTreeDigestValue.Key key = (DirectoryTreeDigestValue.Key) skyKey;
    RootedPath rootedPath = key.rootedPath();
    if (excludes(rootedPath, key.globBase(), key.excludes(), patternCache)) {
      // The path we are trying to compute a digest for is excluded.
      // This should only happen at the very beginning/root of a tree digest as the subsequent
      // computation of digests for child nodes should be excluded before they are asked to be
      // computed. Eg. user asks to watch /some/path and excludes everything under it ('**').  This
      // would be a nonsensical action, so throw an error.
      throw new DirectoryTreeDigestFunctionException(
          new FileNotFoundException(
              String.format(
                  "Tried to compute the digest of path '%s' but this path was filtered out by glob"
                      + " exclude base '%s' and excludes: %s",
                  rootedPath, key.globBase(), key.excludes())));
    }
    DirectoryListingValue dirListingValue =
        (DirectoryListingValue) env.getValue(DirectoryListingValue.key(rootedPath));
    if (dirListingValue == null) {
      return null;
    }

    // Get the names of entries directly in this directory, and sort them. This sets the basis for
    // subsequent digests.
    ImmutableSet<String> sortedDirents =
        StreamSupport.stream(dirListingValue.getDirents().spliterator(), /* parallel= */ false)
            .map(Dirent::getName)
            .filter(
                entry -> {
                  String path = rootedPath.getRootRelativePath().getRelative(entry).toString();
                  return !excludes(path, key.globBase(), key.excludes(), patternCache);
                })
            .sorted()
            .collect(toImmutableSet());

    // Turn each entry into a FileValue.
    ImmutableList<Pair<RootedPath, FileValue>> fileValues =
        getFileValues(env, sortedDirents, rootedPath);
    if (fileValues == null) {
      return null;
    }

    // For each entry that is a directory (or a symlink to a directory), find its own
    // DirectoryTreeDigestValue.
    ImmutableList<String> subDirTreeDigests = getSubDirTreeDigests(env, fileValues, key);
    if (subDirTreeDigests == null) {
      return null;
    }

    // Finally, we're ready to digest everything together!
    Fingerprint fp = new Fingerprint();
    fp.addStrings(sortedDirents);
    fp.addStrings(subDirTreeDigests);
    try {
      for (Pair<RootedPath, FileValue> rootedPathAndFileValue : fileValues) {
        RootedPath direntRootedPath = rootedPathAndFileValue.getFirst();
        FileValue fileValue = rootedPathAndFileValue.getSecond();
        fp.addInt(fileValue.realFileStateValue().getType().ordinal());
        if (fileValue.isFile()) {
          byte[] digest = fileValue.realFileStateValue().getDigest();
          if (digest == null) {
            // Fast digest not available, or it would have been in the FileValue.
            digest = fileValue.realRootedPath(direntRootedPath).asPath().getDigest();
          }
          fp.addBytes(digest);
        }
      }
    } catch (IOException e) {
      throw new DirectoryTreeDigestFunctionException(e);
    }

    return DirectoryTreeDigestValue.of(fp.hexDigestAndReset());
  }

  @Nullable
  private static ImmutableList<Pair<RootedPath, FileValue>> getFileValues(
      Environment env, ImmutableSet<String> sortedDirents, RootedPath rootedPath)
      throws InterruptedException {
    ImmutableSet<FileKey> fileValueKeys =
        sortedDirents.stream()
            .map(
                dirent ->
                    FileValue.key(
                        RootedPath.toRootedPath(
                            rootedPath.getRoot(),
                            rootedPath.getRootRelativePath().getRelative(dirent))))
            .collect(toImmutableSet());
    SkyframeLookupResult result = env.getValuesAndExceptions(fileValueKeys);
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableList<Pair<RootedPath, FileValue>> fileValues =
        fileValueKeys.stream()
            .map(k -> Pair.of((RootedPath) k.argument(), (FileValue) result.get(k)))
            .collect(toImmutableList());
    if (env.valuesMissing()) {
      return null;
    }
    return fileValues;
  }

  @Nullable
  private static ImmutableList<String> getSubDirTreeDigests(
      Environment env,
      ImmutableList<Pair<RootedPath, FileValue>> fileValues,
      DirectoryTreeDigestValue.Key key)
      throws InterruptedException {
    ImmutableSet<SkyKey> dirTreeDigestValueKeys =
        fileValues.stream()
            .filter(p -> p.getSecond().isDirectory())
            .map(
                p ->
                    DirectoryTreeDigestValue.key(
                        /* rootedPath= */ p.getSecond().realRootedPath(p.getFirst()),
                        /* globBase= */ key.globBase(),
                        /* excludes= */ key.excludes()))
            .collect(toImmutableSet());
    SkyframeLookupResult result = env.getValuesAndExceptions(dirTreeDigestValueKeys);
    if (env.valuesMissing()
        || dirTreeDigestValueKeys.stream().map(result::get).anyMatch(Objects::isNull)) {
      return null;
    }
    return dirTreeDigestValueKeys.stream()
        .map(result::get)
        .map(DirectoryTreeDigestValue.class::cast)
        .map(DirectoryTreeDigestValue::hexDigest)
        .collect(toImmutableList());
  }

  /** Returns if the given {@code rootedPath} would be filtered/excluded out. */
  public static boolean excludes(
      RootedPath rootedPath,
      RootedPath globBase,
      ImmutableList<String> excludes,
      Map<String, Pattern> patternCache) {
    // Are we comparing the same roots?
    if (!rootedPath.getRoot().equals(globBase.getRoot())) {
      return false;
    }
    String path = rootedPath.getRootRelativePath().toString();
    return excludes(path, globBase, excludes, patternCache);
  }

  /** Returns if the given {@code path} would be filtered/excluded out. */
  public static boolean excludes(
      String path,
      RootedPath globBase,
      ImmutableList<String> excludes,
      Map<String, Pattern> patternCache) {
    PathFragment baseExclude = globBase.getRootRelativePath();
    for (String exclude : excludes) {
      String excludePattern = baseExclude.getRelative(exclude).toString();
      if (UnixGlob.matches(excludePattern, path, patternCache)) {
        return true;
      }
    }
    return false;
  }

  private static final class DirectoryTreeDigestFunctionException extends SkyFunctionException {
    public DirectoryTreeDigestFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
