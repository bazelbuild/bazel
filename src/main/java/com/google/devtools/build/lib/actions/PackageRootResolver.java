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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * Represents logic that evaluates the root of the package containing path.
 */
public interface PackageRootResolver {

  /**
   * Returns mapping from execPath to Root. Some roots can equal null if the corresponding
   * package can't be found. Returns null if for some reason we can't evaluate it.
   */
  @Nullable
  Map<PathFragment, Root> findPackageRoots(Iterable<PathFragment> execPaths);
}
