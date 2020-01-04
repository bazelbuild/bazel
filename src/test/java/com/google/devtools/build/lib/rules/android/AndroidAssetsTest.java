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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link AndroidAssets} */
@RunWith(JUnit4.class)
public class AndroidAssetsTest extends ResourceTestBase {
  @Test
  public void testParseAapt2() throws Exception {
    RuleContext ruleContext = getRuleContext();
    AndroidAssets assets = getLocalAssets();

    ParsedAndroidAssets parsed = assets.parse(AndroidDataContext.forNative(ruleContext));

    // Assets should be unchanged
    assertThat(parsed.getAssets()).isEqualTo(assets.getAssets());
    assertThat(parsed.getAssetRoots()).isEqualTo(assets.getAssetRoots());

    // Label should be correct
    assertThat(parsed.getLabel()).isEqualTo(ruleContext.getLabel());

    // There should be compiled symbols
    assertThat(parsed.getCompiledSymbols()).isNotNull();

    // Symbols and compiled symbols files should be created from raw assets
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ assets.getAssets(),
        /* outputs = */ ImmutableList.of(parsed.getSymbols()));
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ assets.getAssets(),
        /* outputs = */ ImmutableList.of(parsed.getCompiledSymbols()));
  }

  @Test
  public void testMergeNoDeps() throws Exception {
    RuleContext ruleContext = getRuleContext();
    ParsedAndroidAssets parsed = getLocalAssets().parse(AndroidDataContext.forNative(ruleContext));
    MergedAndroidAssets merged = assertMerge(ruleContext, parsed, AssetDependencies.empty());

    // The assets can be correctly built into a provider
    AndroidAssetsInfo info = merged.toProvider();
    assertThat(info.getLabel()).isEqualTo(merged.getLabel());

    // The provider just has the local values
    assertThat(info.getAssets()).containsExactlyElementsIn(merged.getAssets()).inOrder();
    assertThat(info.getSymbols()).containsExactly(merged.getSymbols());
    assertThat(info.getDirectParsedAssets()).containsExactly(parsed);
    assertThat(info.getTransitiveParsedAssets()).isEmpty();
  }

  @Test
  public void testMergeNeverlink() throws Exception {
    RuleContext ruleContext = getRuleContext();
    ParsedAndroidAssets parsed = getLocalAssets().parse(AndroidDataContext.forNative(ruleContext));
    AssetDependencies deps = makeDeps(ruleContext, /* neverlink = */ true);

    MergedAndroidAssets merged = assertMerge(ruleContext, parsed, deps);

    AndroidAssetsInfo info = merged.toProvider();
    assertThat(info.getLabel()).isEqualTo(merged.getLabel());

    // The provider should be empty because of neverlinking
    assertThat(info.getAssets()).isEmpty();
    assertThat(info.getSymbols()).isEmpty();
    assertThat(info.getDirectParsedAssets()).isEmpty();
    assertThat(info.getTransitiveParsedAssets()).isEmpty();
    assertThat(info.getCompiledSymbols()).isEmpty();
  }

  @Test
  public void testMergeAapt2() throws Exception {
    RuleContext ruleContext = getRuleContext();
    ParsedAndroidAssets parsed = getLocalAssets().parse(AndroidDataContext.forNative(ruleContext));
    AssetDependencies deps = makeDeps(ruleContext, /* neverlink = */ false);

    MergedAndroidAssets merged = assertMerge(ruleContext, parsed, deps);

    AndroidAssetsInfo info = merged.toProvider();
    assertThat(info.getLabel()).isEqualTo(merged.getLabel());

    // The provider should have transitive and direct deps
    assertThat(info.getAssets())
        .containsExactlyElementsIn(Iterables.concat(parsed.getAssets(), deps.getTransitiveAssets()))
        .inOrder();
    assertThat(info.getSymbols())
        .containsExactlyElementsIn(
            Iterables.concat(ImmutableList.of(parsed.getSymbols()), deps.getTransitiveSymbols()))
        .inOrder();
    assertThat(info.getCompiledSymbols())
        .containsExactlyElementsIn(
            Iterables.concat(
                ImmutableList.of(parsed.getCompiledSymbols()),
                deps.getTransitiveCompiledSymbols()));
    assertThat(info.getDirectParsedAssets()).containsExactly(parsed).inOrder();
    assertThat(info.getTransitiveParsedAssets())
        .containsExactlyElementsIn(
            Iterables.concat(deps.getTransitiveParsedAssets(), deps.getDirectParsedAssets()))
        .inOrder();
  }

  private AssetDependencies makeDeps(RuleContext ruleContext, boolean neverlink) {
    ParsedAndroidAssets firstDirect = getDependencyAssets(ruleContext, "first_direct");
    ParsedAndroidAssets secondDirect = getDependencyAssets(ruleContext, "second_direct");
    ParsedAndroidAssets firstTransitive = getDependencyAssets(ruleContext, "first_transitive");
    ParsedAndroidAssets secondTransitive = getDependencyAssets(ruleContext, "second_transitive");

    return AssetDependencies.of(
        neverlink,
        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, firstDirect, secondDirect),
        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, firstTransitive, secondTransitive),
        NestedSetBuilder.wrap(
            Order.NAIVE_LINK_ORDER,
            Iterables.concat(
                firstDirect.getAssets(),
                secondDirect.getAssets(),
                firstTransitive.getAssets(),
                secondTransitive.getAssets())),
        NestedSetBuilder.wrap(
            Order.NAIVE_LINK_ORDER,
            ImmutableList.of(
                firstDirect.getSymbols(),
                secondDirect.getSymbols(),
                firstTransitive.getSymbols(),
                secondTransitive.getSymbols())),
        NestedSetBuilder.wrap(
            Order.NAIVE_LINK_ORDER,
            ImmutableList.of(
                firstDirect.getCompiledSymbols(),
                secondDirect.getCompiledSymbols(),
                firstTransitive.getCompiledSymbols(),
                secondTransitive.getCompiledSymbols())));
  }

  private MergedAndroidAssets assertMerge(
      RuleContext ruleContext, ParsedAndroidAssets parsed, AssetDependencies deps)
      throws InterruptedException {
    MergedAndroidAssets merged =
        MergedAndroidAssets.mergeFrom(AndroidDataContext.forNative(ruleContext), parsed, deps);

    // Inherited values should be unchanged
    assertThat(new ParsedAndroidAssets(merged)).isEqualTo(parsed);

    // The raw assets should be used to merge
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ ImmutableList.<Artifact>builder()
            .addAll(merged.getAssets())
            .add(merged.getSymbols())
            .addAll(deps.getTransitiveAssets())
            .addAll(deps.getTransitiveSymbols())
            .build(),
        /* outputs = */ ImmutableList.of(merged.getMergedAssets()));

    return merged;
  }

  private AndroidAssets getLocalAssets() {
    return new AndroidAssets(
        ImmutableList.of(getResource("asset_1"), getResource("asset_2")),
        ImmutableList.of(PathFragment.create("asset_dir")),
        "asset_dir");
  }

  private ParsedAndroidAssets getDependencyAssets(RuleContext ruleContext, String depName) {
    return ParsedAndroidAssets.of(
        new AndroidAssets(
            ImmutableList.of(getResource(depName + "_asset_1"), getResource(depName + "_asset_2")),
            ImmutableList.of(PathFragment.create(depName)),
            depName),
        getResource("symbols_for_" + depName),
        getResource("compiled_symbols_for_" + depName),
        ruleContext.getLabel());
  }

  private RuleContext getRuleContext() throws Exception {
    return getRuleContextForActionTesting(
        scratchConfiguredTarget("pkg", "r", "android_library(name='r')"));
  }
}
