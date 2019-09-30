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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.RootedPathAndCasing;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import java.util.Map;

/** SkyFunction for {@link PathCasingLookupValue}s. */
public final class PathCasingLookupFunction implements SkyFunction {

  @Override
  public PathCasingLookupValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    RootedPathAndCasing arg = (RootedPathAndCasing) skyKey.argument();
    RootedPath parent = arg.getPath().getParentDirectory();

    if (parent == null) {
      Path argPath = arg.getPath().asPath();
      Path parentPath = argPath.getParentDirectory();
      if (parentPath == null) {
        String driveStr = argPath.getDriveStr();
        return driveStr.length() == 1 || Character.isUpperCase(driveStr.charAt(0))
            ? PathCasingLookupValue.GOOD
            : PathCasingLookupValue.BAD;
      }
      parent = RootedPath.toRootedPath(Root.absoluteRoot(argPath.getFileSystem()), parentPath);
    }

    SkyKey pathCasingKey = PathCasingLookupValue.key(parent);
    SkyKey dirListKey = DirectoryListingValue.key(parent);
    Map<SkyKey, SkyValue> values = env.getValues(ImmutableList.of(pathCasingKey, dirListKey));
    if (env.valuesMissing()) {
      return null;
    }

    PathCasingLookupValue parentCasing = (PathCasingLookupValue) values.get(pathCasingKey);
    if (!parentCasing.isCorrect()) {
      return PathCasingLookupValue.BAD;
    }

    DirectoryListingValue parentList = (DirectoryListingValue) values.get(dirListKey);
    String expected = arg.getPath().getRootRelativePath().getBaseName();
    if (Strings.isNullOrEmpty(expected)) {
      expected = arg.getPath().asPath().getBaseName();
    }
    Dirent child = parentList.getDirents().maybeGetDirent(expected);

    return child != null && expected.equals(child.getName())
        ? PathCasingLookupValue.GOOD
        : PathCasingLookupValue.BAD;  // or missing
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
