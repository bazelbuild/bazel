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
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;

/** A {@link SkyFunction} for {@link DirectoryTreeDigestValue}s. */
public final class DirectoryTreeDigestFunction implements SkyFunction {
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, DirectoryTreeDigestFunctionException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();
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
    ImmutableList<String> subDirTreeDigests = getSubDirTreeDigests(env, fileValues);
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
    ImmutableSet<FileValue.Key> fileValueKeys =
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
      Environment env, ImmutableList<Pair<RootedPath, FileValue>> fileValues)
      throws InterruptedException {
    ImmutableSet<SkyKey> dirTreeDigestValueKeys =
        fileValues.stream()
            .filter(p -> p.getSecond().isDirectory())
            .map(p -> DirectoryTreeDigestValue.key(p.getSecond().realRootedPath(p.getFirst())))
            .collect(toImmutableSet());
    SkyframeLookupResult result = env.getValuesAndExceptions(dirTreeDigestValueKeys);
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableList<String> dirTreeDigests =
        dirTreeDigestValueKeys.stream()
            .map(result::get)
            .map(DirectoryTreeDigestValue.class::cast)
            .map(DirectoryTreeDigestValue::hexDigest)
            .collect(toImmutableList());
    if (env.valuesMissing()) {
      return null;
    }
    return dirTreeDigests;
  }

  private static final class DirectoryTreeDigestFunctionException extends SkyFunctionException {
    public DirectoryTreeDigestFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
