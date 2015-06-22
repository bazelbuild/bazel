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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;

/**
 * Support for resolving {@code package/...} target patterns.
 */
public interface RecursivePackageProvider extends PackageProvider {

  /**
   * Returns the names of all the packages under a given directory.
   * @param directory a {@link RootedPath} specifying the directory to search
   * @param excludedSubdirectories a set of {@link PathFragment}s, all of which are beneath
   *     {@code directory}, specifying transitive subdirectories to exclude
   */
  Iterable<PathFragment> getPackagesUnderDirectory(RootedPath directory,
      ImmutableSet<PathFragment> excludedSubdirectories);
}
