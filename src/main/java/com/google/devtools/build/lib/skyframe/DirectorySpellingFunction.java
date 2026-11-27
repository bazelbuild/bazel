// Copyright 2026 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.DirectorySpellingValue.NonCanonical;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.text.Normalizer;
import javax.annotation.Nullable;

/** A {@link SkyFunction} for {@link DirectorySpellingValue}s. */
public final class DirectorySpellingFunction implements SkyFunction {

  @Nullable
  @Override
  public DirectorySpellingValue compute(SkyKey skyKey, Environment env)
      throws DirectorySpellingFunctionException, InterruptedException {
    RootedPath directory = (RootedPath) skyKey.argument();
    PathFragment relativePath = directory.getRootRelativePath();
    if (relativePath.isEmpty()) {
      return DirectorySpellingValue.CANONICAL;
    }

    RootedPath parent = directory.getParentDirectory();
    SkyKey parentSpellingKey = DirectorySpellingValue.key(parent);
    SkyKey parentListingKey = DirectoryListingValue.key(parent);
    SkyframeLookupResult deps =
        env.getValuesAndExceptions(ImmutableList.of(parentSpellingKey, parentListingKey));
    DirectorySpellingValue parentSpelling;
    DirectoryListingValue parentListing;
    try {
      parentSpelling =
          (DirectorySpellingValue) deps.getOrThrow(parentSpellingKey, IOException.class);
      parentListing = (DirectoryListingValue) deps.getOrThrow(parentListingKey, IOException.class);
    } catch (IOException e) {
      throw new DirectorySpellingFunctionException(e);
    }
    if (parentSpelling == null || parentListing == null) {
      return null;
    }

    String baseName = relativePath.getBaseName();
    if (parentSpelling == DirectorySpellingValue.CANONICAL
        && parentListing.getDirents().maybeGetDirent(baseName) != null) {
      return DirectorySpellingValue.CANONICAL;
    }

    PathFragment canonicalParent =
        parentSpelling instanceof NonCanonical(PathFragment canonicalPath)
            ? canonicalPath
            : parent.getRootRelativePath();
    if (canonicalParent == null) {
      return new NonCanonical(null);
    }
    String canonicalBaseName = findUniqueLooseMatch(parentListing.getDirents(), baseName);
    return new NonCanonical(
        canonicalBaseName != null ? canonicalParent.getChild(canonicalBaseName) : null);
  }

  /**
   * Returns the name of the only entry that matches {@code baseName} up to case and Unicode
   * normalization, or null if there is no such entry or more than one.
   */
  @Nullable
  static String findUniqueLooseMatch(Iterable<Dirent> dirents, String baseName) {
    String looseBaseName = loosen(baseName);
    String match = null;
    for (Dirent dirent : dirents) {
      if (loosen(dirent.getName()).equalsIgnoreCase(looseBaseName)) {
        if (match != null) {
          return null;
        }
        match = dirent.getName();
      }
    }
    return match;
  }

  private static String loosen(String internalString) {
    return Normalizer.normalize(
        StringEncoding.internalToUnicode(internalString), Normalizer.Form.NFC);
  }

  private static final class DirectorySpellingFunctionException extends SkyFunctionException {
    DirectorySpellingFunctionException(IOException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
