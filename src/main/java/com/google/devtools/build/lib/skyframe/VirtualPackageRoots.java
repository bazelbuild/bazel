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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Root;

/** {@link PackageRoots} with a single, virtual source root. */
final class VirtualPackageRoots implements PackageRoots {
  private final Root virtualSourceRoot;

  VirtualPackageRoots(Root virtualSourceRoot) {
    this.virtualSourceRoot = checkNotNull(virtualSourceRoot);
  }

  @Override
  public ImmutableMap<PackageIdentifier, Root> getPackageRootsMap() {
    // Tells SymlinkForest to link all top-level path entries under the virtual source root.
    return ImmutableMap.of(PackageIdentifier.EMPTY_PACKAGE_ID, virtualSourceRoot);
  }

  @Override
  public PackageRootLookup getPackageRootLookup() {
    return packageIdentifier -> virtualSourceRoot;
  }
}
