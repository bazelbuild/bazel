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
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * A {@link PackageRoots} backed by a map of package identifiers to paths. A symlink forest must be
 * planted for execution.
 */
public class MapAsPackageRoots implements PackageRoots {
  private final ImmutableMap<PackageIdentifier, Root> packageRootsMap;

  MapAsPackageRoots(ImmutableMap<PackageIdentifier, Root> packageRootsMap) {
    this.packageRootsMap = packageRootsMap;
  }

  @Override
  public Optional<ImmutableMap<PackageIdentifier, Root>> getPackageRootsMap() {
    return Optional.of(packageRootsMap);
  }

  @Override
  public PackageRootLookup getPackageRootLookup() {
    Map<Root, Root> rootMap = new HashMap<>();
    Map<PackageIdentifier, Root> realPackageRoots = new HashMap<>();
    for (Map.Entry<PackageIdentifier, Root> entry : packageRootsMap.entrySet()) {
      Root root = rootMap.get(entry.getValue());
      if (root == null) {
        root = entry.getValue();
        rootMap.put(entry.getValue(), root);
      }
      realPackageRoots.put(entry.getKey(), root);
    }
    return realPackageRoots::get;
  }
}
