// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;

/** Common parent class of SkyKeys that wrap a {@link RecursivePkgKey}. */
public abstract class RecursivePkgSkyKey extends RecursivePkgKey implements SkyKey {
  public RecursivePkgSkyKey(
      RepositoryName repositoryName, RootedPath rootedPath, IgnoredSubdirectories excludedPaths) {
    super(repositoryName, rootedPath, excludedPaths);
  }

  @Override
  public String toString() {
    return functionName() + " " + super.toString();
  }

  @Override
  public boolean equals(Object o) {
    return super.equals(o)
        && o instanceof RecursivePkgSkyKey recursivePkgSkyKey
        && recursivePkgSkyKey.functionName().equals(functionName());
  }

  /** Don't bother to memoize hashCode because {@link RecursivePkgKey#hashCode} is cheap enough. */
  @Override
  public int hashCode() {
    return 37 * super.hashCode() + functionName().hashCode();
  }
}
