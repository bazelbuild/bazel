// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;

/** SkyFunction for {@link RootedPathCasingValue}s. */
public final class RootedPathCasingFunction implements SkyFunction {

  @Override
  public RootedPathCasingValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    var path = ((RootedPathCasingValue.Key) skyKey).argument();
    if (path.getRootRelativePath().isEmpty()) {
      return RootedPathCasingValue.MATCH;
    }

    // Detect the actual file system case of the path's basename by listing the entries of its
    // parent directory.
    var parent = path.getParentDirectory();
    var childFileKey = FileValue.key(path);
    var parentCasingKey = RootedPathCasingValue.key(parent);
    var parentListingKey = DirectoryListingValue.key(parent);
    var result =
        env.getValuesAndExceptions(
            ImmutableList.of(childFileKey, parentCasingKey, parentListingKey));
    var childFileValue = (FileValue) env.getValue(childFileKey);
    var parentCasingValue = (RootedPathCasingValue) result.get(parentCasingKey);
    var parentListingValue = (DirectoryListingValue) result.get(parentListingKey);
    if (childFileValue == null || parentCasingValue == null || parentListingValue == null) {
      return null;
    }

    if (!childFileValue.exists()) {
      return RootedPathCasingValue.MATCH;
    }
    var expectedBasename = path.getRootRelativePath().getBaseName();
    if (parentListingValue.getDirents().maybeGetDirent(expectedBasename) != null
        && parentCasingValue instanceof RootedPathCasingValue.Match) {
      return RootedPathCasingValue.MATCH;
    }

    String expectedBasenameUnicode = internalToUnicode(expectedBasename);
    String actualBasename = null;
    for (Dirent dirent : parentListingValue.getDirents()) {
      if (internalToUnicode(dirent.getName()).equalsIgnoreCase(expectedBasenameUnicode)) {
        actualBasename = dirent.getName();
        break;
      }
    }
    return new RootedPathCasingValue.NoMatch(
        parentCasingValue,
        actualBasename != null
            ? actualBasename
            : "<unknown differently cased version of '%s'>".formatted(expectedBasename));
  }
}
