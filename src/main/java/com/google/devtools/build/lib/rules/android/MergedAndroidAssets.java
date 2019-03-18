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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import java.util.Objects;

/** Android assets that have been merged together with their dependencies. */
public class MergedAndroidAssets extends ParsedAndroidAssets {
  private final Artifact mergedAssets;
  private final AssetDependencies assetDependencies;

  static MergedAndroidAssets mergeFrom(
      AndroidDataContext dataContext, ParsedAndroidAssets parsed, AssetDependencies deps)
      throws InterruptedException {

    Artifact mergedAssets =
        dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_ASSETS_ZIP);

    BusyBoxActionBuilder builder = BusyBoxActionBuilder.create(dataContext, "MERGE_ASSETS");
    if (dataContext.throwOnResourceConflict()) {
      builder.addFlag("--throwOnAssetConflict");
    }

    builder
        .addOutput("--assetsOutput", mergedAssets)
        .addInput(
            "--primaryData",
            AndroidDataConverter.PARSED_ASSET_CONVERTER.map(parsed),
            Iterables.concat(parsed.getAssets(), ImmutableList.of(parsed.getSymbols())))
        .addTransitiveFlag(
            "--directData",
            deps.getDirectParsedAssets(),
            AndroidDataConverter.PARSED_ASSET_CONVERTER)
        .addTransitiveFlag(
            "--data", deps.getTransitiveParsedAssets(), AndroidDataConverter.PARSED_ASSET_CONVERTER)
        .addTransitiveInputValues(deps.getTransitiveAssets())
        .addTransitiveInputValues(deps.getTransitiveSymbols())
        .buildAndRegister("Merging Android assets", "AndroidAssetMerger");

    return of(parsed, mergedAssets, deps);
  }

  static MergedAndroidAssets of(
      ParsedAndroidAssets parsed, Artifact mergedAssets, AssetDependencies assetDependencies) {
    return new MergedAndroidAssets(parsed, mergedAssets, assetDependencies);
  }

  private MergedAndroidAssets(
      ParsedAndroidAssets parsed, Artifact mergedAssets, AssetDependencies assetDependencies) {
    super(parsed);
    this.mergedAssets = mergedAssets;
    this.assetDependencies = assetDependencies;
  }

  AndroidAssetsInfo toProvider() {
    return assetDependencies.toInfo(this);
  }

  public Artifact getMergedAssets() {
    return mergedAssets;
  }

  public AssetDependencies getAssetDependencies() {
    return assetDependencies;
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object)) {
      return false;
    }

    MergedAndroidAssets other = (MergedAndroidAssets) object;

    return mergedAssets.equals(other.mergedAssets)
        && assetDependencies.equals(other.assetDependencies);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), mergedAssets, assetDependencies);
  }
}
