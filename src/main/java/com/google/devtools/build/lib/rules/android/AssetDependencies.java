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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

/**
 * Contains transitive asset dependencies for a target.
 *
 * <p>In addition to NestedSets of transitive artifacts, we also keep transitive NestedSets of
 * direct and transitive parsed assets. These NestedSets contain the same information (and it may
 * appear in both the direct and transitive parsed asset NestedSets); we need to keep them both
 * separately to record relationships between assets, asset directories, and symbol artifacts and to
 * distinguish between direct and transitive assets.
 */
public class AssetDependencies {
  private final boolean neverlink;
  private final NestedSet<ParsedAndroidAssets> directParsedAssets;
  private final NestedSet<ParsedAndroidAssets> transitiveParsedAssets;
  private final NestedSet<Artifact> transitiveAssets;
  private final NestedSet<Artifact> transitiveSymbols;

  public static AssetDependencies fromRuleDeps(RuleContext ruleContext, boolean neverlink) {
    return fromProviders(
        AndroidCommon.getTransitivePrerequisites(
            ruleContext, Mode.TARGET, AndroidAssetsInfo.PROVIDER),
        neverlink);
  }

  public static AssetDependencies fromProviders(
      Iterable<AndroidAssetsInfo> providers, boolean neverlink) {
    NestedSetBuilder<ParsedAndroidAssets> direct = NestedSetBuilder.naiveLinkOrder();
    NestedSetBuilder<ParsedAndroidAssets> transitive = NestedSetBuilder.naiveLinkOrder();
    NestedSetBuilder<Artifact> assets = NestedSetBuilder.naiveLinkOrder();
    NestedSetBuilder<Artifact> symbols = NestedSetBuilder.naiveLinkOrder();

    for (AndroidAssetsInfo info : providers) {
      direct.addTransitive(info.getDirectParsedAssets());
      transitive.addTransitive(info.getTransitiveParsedAssets());
      assets.addTransitive(info.getAssets());
      symbols.addTransitive(info.getSymbols());
    }

    return of(neverlink, direct.build(), transitive.build(), assets.build(), symbols.build());
  }

  public static AssetDependencies empty() {
    return of(
        false,
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER));
  }

  @VisibleForTesting
  static AssetDependencies of(
      boolean neverlink,
      NestedSet<ParsedAndroidAssets> directParsedAssets,
      NestedSet<ParsedAndroidAssets> transitiveParsedAssets,
      NestedSet<Artifact> transitiveAssets,
      NestedSet<Artifact> transitiveSymbols) {
    return new AssetDependencies(
        neverlink, directParsedAssets, transitiveParsedAssets, transitiveAssets, transitiveSymbols);
  }

  private AssetDependencies(
      boolean neverlink,
      NestedSet<ParsedAndroidAssets> directParsedAssets,
      NestedSet<ParsedAndroidAssets> transitiveParsedAssets,
      NestedSet<Artifact> transitiveAssets,
      NestedSet<Artifact> transitiveSymbols) {
    this.neverlink = neverlink;
    this.directParsedAssets = directParsedAssets;
    this.transitiveParsedAssets = transitiveParsedAssets;
    this.transitiveAssets = transitiveAssets;
    this.transitiveSymbols = transitiveSymbols;
  }

  /** Creates a new AndroidAssetInfo using the passed assets as the direct dependency. */
  public AndroidAssetsInfo toInfo(MergedAndroidAssets assets) {
    if (neverlink) {
      return AndroidAssetsInfo.empty(assets.getLabel());
    }

    // Create a new object to avoid passing around unwanted merge information to the provider
    ParsedAndroidAssets parsedAssets = new ParsedAndroidAssets(assets);

    return AndroidAssetsInfo.of(
        assets.getLabel(),
        assets.getMergedAssets(),
        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, parsedAssets),
        NestedSetBuilder.<ParsedAndroidAssets>naiveLinkOrder()
            .addTransitive(transitiveParsedAssets)
            .addTransitive(directParsedAssets)
            .build(),
        NestedSetBuilder.<Artifact>naiveLinkOrder()
            .addTransitive(transitiveAssets)
            .addAll(assets.getAssets())
            .build(),
        NestedSetBuilder.<Artifact>naiveLinkOrder()
            .addTransitive(transitiveSymbols)
            .add(assets.getSymbols())
            .build());
  }

  /** Creates a new AndroidAssetsInfo from this target's dependencies, without any local assets. */
  public AndroidAssetsInfo toInfo(Label label) {
    if (neverlink) {
      return AndroidAssetsInfo.empty(label);
    }

    return AndroidAssetsInfo.of(
        label,
        null,
        directParsedAssets,
        transitiveParsedAssets,
        transitiveAssets,
        transitiveSymbols);
  }

  public NestedSet<ParsedAndroidAssets> getDirectParsedAssets() {
    return directParsedAssets;
  }

  public NestedSet<ParsedAndroidAssets> getTransitiveParsedAssets() {
    return transitiveParsedAssets;
  }

  public NestedSet<Artifact> getTransitiveAssets() {
    return transitiveAssets;
  }

  public NestedSet<Artifact> getTransitiveSymbols() {
    return transitiveSymbols;
  }
}
