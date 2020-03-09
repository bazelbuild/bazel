// Copyright 2019 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.RootedPathAndCasing;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/** SkyFunction for {@link PathCasingLookupValue}s. */
public final class PathCasingLookupFunction implements SkyFunction {

  @Override
  public PathCasingLookupValue compute(SkyKey skyKey, Environment env)
      throws PathComponentIsNotDirectoryException, InterruptedException {
    RootedPathAndCasing arg = (RootedPathAndCasing) skyKey.argument();
    if (arg.getPath().getRootRelativePath().isEmpty()) {
      // This is a Root, e.g. "[/foo/bar]/[]".
      // As of 2019-12-04, PathCasingLookupValue is not used anywhere. But it's planned to be used
      // in PackageLookupFunction to validate the package part's casing, so the RootedPath's Root's
      // casing doesn't even matter, so if the relative part is empty then for our use case this is
      // a correctly cased RootedPath.
      return PathCasingLookupValue.GOOD;
    }

    RootedPath parent = arg.getPath().getParentDirectory();
    Preconditions.checkNotNull(parent, arg.getPath());
    Preconditions.checkNotNull(parent.getRootRelativePath(), arg.getPath());

    SkyKey parentCasingKey = PathCasingLookupValue.key(parent);
    SkyKey parentFileKey = FileValue.key(parent);
    SkyKey childFileKey = FileValue.key(arg.getPath());
    Map<SkyKey, SkyValue> values =
        env.getValues(ImmutableList.of(parentCasingKey, parentFileKey, childFileKey));
    if (env.valuesMissing()) {
      return null;
    }
    if (!((PathCasingLookupValue) values.get(parentCasingKey)).isCorrect()) {
      // Parent's casing is bad, so this path's casing is also bad.
      return PathCasingLookupValue.BAD;
    }

    FileValue parentFile = (FileValue) values.get(parentFileKey);
    if (!parentFile.exists()) {
      // Parent's casing is good, because it's missing.
      // That means this path is also missing, so by definition its casing is good.
      return PathCasingLookupValue.GOOD;
    }
    if (!parentFile.isDirectory()) {
      // Parent's casing is good, but it's not a directory.
      throw new PathComponentIsNotDirectoryException(
          new IOException(
              "Cannot check path casing of "
                  + arg.getPath()
                  + ": its parent exists but is not a directory"));
    }

    FileValue childFile = (FileValue) values.get(childFileKey);
    if (!childFile.exists()) {
      // Parent's casing is good, but this file is missing.
      // That means this path is missing, so by definition its casing is good.
      return PathCasingLookupValue.GOOD;
    }

    // The parent file must exist, otherwise the DirectoryListingFunction will throw.
    SkyKey parentListingKey = DirectoryListingValue.key(parent);
    DirectoryListingValue parentList = (DirectoryListingValue) env.getValue(parentListingKey);
    if (parentList == null) {
      return null;
    }

    String expected = arg.getPath().getRootRelativePath().getBaseName();
    // We already handled RootedPaths with empty relative part.
    Preconditions.checkState(!Strings.isNullOrEmpty(expected), arg.getPath());

    Dirent child = parentList.getDirents().maybeGetDirent(expected);
    return (child != null && expected.equals(child.getName()))
        ? PathCasingLookupValue.GOOD
        : PathCasingLookupValue.BAD;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** Thrown if a non-terminal path component exists, but it's not a directory. */
  public static final class PathComponentIsNotDirectoryException extends SkyFunctionException {
    public PathComponentIsNotDirectoryException(IOException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
