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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link AndroidAssets} */
@RunWith(JUnit4.class)
public class AndroidAssetsTest extends ResourceTestBase {
  @Test
  public void testParse() throws Exception {
    RuleContext ruleContext = getRuleContext();
    AndroidAssets assets =
        new AndroidAssets(
            ImmutableList.of(getResource("asset_1"), getResource("asset_2")),
            ImmutableList.of(PathFragment.create("asset_dir")));
    ParsedAndroidAssets parsed = assets.parse(ruleContext);

    // Assets should be unchanged
    assertThat(parsed.getAssets()).isEqualTo(assets.getAssets());
    assertThat(parsed.getAssetRoots()).isEqualTo(assets.getAssetRoots());

    // Label should be correct
    assertThat(parsed.getLabel()).isEqualTo(ruleContext.getLabel());

    // Symbols file should be created from raw assets
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ assets.getAssets(),
        /* outputs = */ ImmutableList.of(parsed.getSymbols()));
  }

  private RuleContext getRuleContext() throws Exception {
    return getRuleContextForActionTesting(
        scratchConfiguredTarget("pkg", "r", "android_library(name='r')"));
  }
}
