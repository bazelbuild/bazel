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
import com.google.devtools.build.lib.rules.android.AndroidAssetsTest.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidAssetsTest.WithoutPlatforms;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Tests {@link AndroidAssets} */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidAssetsTest extends ResourceTestBase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidAssetsTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidAssetsTest {
    @Override
    protected boolean platformBasedToolchains() {
      return true;
    }
  }

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
    assertThat(info.getAssets().toList()).containsExactlyElementsIn(merged.getAssets()).inOrder();
    assertThat(info.getSymbols().toList()).containsExactly(merged.getSymbols());
    assertThat(info.getDirectParsedAssets().toList()).containsExactly(parsed);
    assertThat(info.getTransitiveParsedAssets().toList()).isEmpty();
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
    assertThat(info.getAssets().toList()).isEmpty();
    assertThat(info.getSymbols().toList()).isEmpty();
    assertThat(info.getDirectParsedAssets().toList()).isEmpty();
    assertThat(info.getTransitiveParsedAssets().toList()).isEmpty();
    assertThat(info.getCompiledSymbols().toList()).isEmpty();
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
    assertThat(info.getAssets().toList())
        .containsExactlyElementsIn(
            Iterables.concat(parsed.getAssets(), deps.getTransitiveAssets().toList()))
        .inOrder();
    assertThat(info.getSymbols().toList())
        .containsExactlyElementsIn(
            Iterables.concat(
                ImmutableList.of(parsed.getSymbols()), deps.getTransitiveSymbols().toList()))
        .inOrder();
    assertThat(info.getCompiledSymbols().toList())
        .containsExactlyElementsIn(
            Iterables.concat(
                ImmutableList.of(parsed.getCompiledSymbols()),
                deps.getTransitiveCompiledSymbols().toList()));
    assertThat(info.getDirectParsedAssets().toList()).containsExactly(parsed).inOrder();
    assertThat(info.getTransitiveParsedAssets().toList())
        .containsExactlyElementsIn(
            Iterables.concat(
                deps.getTransitiveParsedAssets().toList(), deps.getDirectParsedAssets().toList()))
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
            .addAll(deps.getTransitiveAssets().toList())
            .addAll(deps.getTransitiveSymbols().toList())
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
