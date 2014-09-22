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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Utility code for asset catalogs.
 */
class AssetCatalogs {
  private AssetCatalogs() {}

  static final FileType CONTAINER_TYPE = FileType.of(".xcassets");

  /**
   * Returns all {@code *.xcassets} directories that contain the items from {@code artifacts}.
   */
  static Iterable<PathFragment> xcassetsDirs(Iterable<Artifact> artifacts) {
    ImmutableSet.Builder<PathFragment> xcassetsDirs = new ImmutableSet.Builder<>();
    for (Artifact assetCatalogFile : artifacts) {
      xcassetsDirs.addAll(
          ObjcCommon.nearestContainerMatching(CONTAINER_TYPE, assetCatalogFile).asSet());
    }
    return xcassetsDirs.build();
  }
}
