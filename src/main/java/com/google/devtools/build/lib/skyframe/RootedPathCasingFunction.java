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

import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;

/** SkyFunction for {@link RootedPathCasingValue}s. */
public final class RootedPathCasingFunction implements SkyFunction {

  @Override
  public RootedPathCasingValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, SkyFunctionException {
    var path = ((RootedPathCasingValue.Key) skyKey).argument();
    // This function is currently only used to detect non-canonical casing of labels, so we treat
    // the root prefixes as canonical.
    if (path.getRootRelativePath().isEmpty()) {
      return RootedPathCasingValue.CANONICAL;
    }

    var fileStateValue = (FileStateValue) env.getValue(FileStateValue.key(path));
    if (fileStateValue == null) {
      return null;
    }
    if (!fileStateValue.exists()) {
      // If the path doesn't exist, we can't tell if it's canonical or not. But it also doesn't
      // matter since we don't derive any other paths from a non-existent path.
      return RootedPathCasingValue.CANONICAL;
    }

    var parentCasingValue =
        (RootedPathCasingValue) env.getValue(RootedPathCasingValue.key(path.getParentDirectory()));
    if (parentCasingValue == null) {
      return null;
    }
    // File system access is guarded by the FileStateValue dependency above.
    String canonicalBasename;
    try {
      canonicalBasename = path.asPath().canonicalizeCase().getBaseName();
    } catch (IOException e) {
      throw new RootedPathCasingFunctionException(e);
    }
    if (parentCasingValue instanceof RootedPathCasingValue.Canonical
        && canonicalBasename.equals(path.asPath().getBaseName())) {
      return RootedPathCasingValue.CANONICAL;
    }
    return new RootedPathCasingValue.NonCanonical(parentCasingValue, canonicalBasename);
  }

  private static class RootedPathCasingFunctionException extends SkyFunctionException {
    RootedPathCasingFunctionException(Exception cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
