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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Map;
import java.util.Set;

/**
 * An interface for getting multiple existing packages for the purpose of finding labels
 * that cross subpackage boundaries.
 *
 * This is a separate interface so that Skyframe handle the implicit dependencies on these
 * packages.
 */
public interface BulkPackageLocatorForCrossingSubpackageBoundaries {
  /**
   * Given a set of {@link PathFragment}s of potential packages corresponding to label names with
   * slashes in them, returns a map from {@link PathFragment} to {@link Path} for those that are
   * existing packages. If any these label names do indeed conflict with an existing package,
   * then the entire enclosing package is in error because it has a label that crosses a
   * subpackage boundary.
   */
  Map<PathFragment, Path> getExistingPackages(
      Set<PathFragment> labelsMaybeCrossingSubpackageBoundaries)
          throws InterruptedException;
}
