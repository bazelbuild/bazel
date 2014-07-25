// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * This value represents the result of looking up all the packages under a given package path root,
 * starting at a given directory.
 */
@Immutable
@ThreadSafe
public class RecursivePkgValue implements SkyValue {

  private final NestedSet<String> packages;

  public RecursivePkgValue(NestedSet<String> packages) {
    this.packages = packages;
  }

  /**
   * Create a transitive package lookup request.
   */
  @ThreadSafe
  public static SkyKey key(RootedPath rootedPath) {
    return new SkyKey(SkyFunctions.RECURSIVE_PKG, rootedPath);
  }

  public NestedSet<String> getPackages() {
    return packages;
  }
}
