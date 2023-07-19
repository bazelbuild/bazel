// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Optional;

/**
 * {@link PackageRoots} with a single source root that does not want symlink forest creation, and
 * whose {@link PackageRootLookup} returns the unique source root for any given package identifier.
 */
public class PackageRootsNoSymlinkCreation implements PackageRoots {
  private final Root sourceRoot;

  public PackageRootsNoSymlinkCreation(Root sourcePath) {
    this.sourceRoot = sourcePath;
  }

  @Override
  public Optional<ImmutableMap<PackageIdentifier, Root>> getPackageRootsMap() {
    return Optional.empty();
  }

  @Override
  public PackageRootLookup getPackageRootLookup() {
    return packageIdentifier -> sourceRoot;
  }
}
