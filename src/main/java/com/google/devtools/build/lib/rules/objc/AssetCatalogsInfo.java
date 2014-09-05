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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;

/**
 * Contains information specific to asset catalogs for a single rule.
 */
class AssetCatalogsInfo {
  private final Iterable<PathFragment> xcassetsDirs;
  private final Iterable<Artifact> allXcassets;
  private final Iterable<Artifact> notInXcassetsDir;

  private AssetCatalogsInfo(Iterable<PathFragment> xcassetsDirs,
      Iterable<Artifact> allXcassets, Iterable<Artifact> notInXcassetsDir) {
    this.xcassetsDirs = Preconditions.checkNotNull(xcassetsDirs);
    this.allXcassets = Preconditions.checkNotNull(allXcassets);
    this.notInXcassetsDir = Preconditions.checkNotNull(notInXcassetsDir);
  }

  /**
   * Returns all the artifacts specified by the {@code asset_catalogs} attribute.
   */
  public Iterable<Artifact> getAllXcassets() {
    return allXcassets;
  }

  /**
   * Returns all {@code *.xcassets} directories that contain the items returned by
   * {@link #getAllXcassets()}.
   */
  public Iterable<PathFragment> getXcassetsDirs() {
    return xcassetsDirs;
  }

  /**
   * Returns all Artifacts specified in the xcassets attribute that are not contained in a
   * *.xcassets directory. It is considered a rule error for any Artifact to fall in this category.
   */
  public Iterable<Artifact> getNotInXcassetsDir() {
    return notInXcassetsDir;
  }

  /**
   * Creates an instance populated with the asset catalog information of the rule corresponding to
   * some {@code RuleContext}.
   */
  public static AssetCatalogsInfo fromRule(RuleContext context) {
    ImmutableSet.Builder<PathFragment> xcassetsDirs = new ImmutableSet.Builder<>();
    ImmutableList.Builder<Artifact> notInXcassetsDir = new ImmutableList.Builder<>();
    Iterable<Artifact> allXcassets =
        context.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET);
    for (Artifact assetCatalogFile : allXcassets) {
      PathFragment path = assetCatalogFile.getExecPath();
      do {
        path = path.getParentDirectory();
      } while (path != null && !path.getBaseName().endsWith(".xcassets"));
      if (path == null) {
        notInXcassetsDir.add(assetCatalogFile);
      } else {
        xcassetsDirs.add(path);
      }
    }
    return new AssetCatalogsInfo(xcassetsDirs.build(), allXcassets, notInXcassetsDir.build());
  }
}
