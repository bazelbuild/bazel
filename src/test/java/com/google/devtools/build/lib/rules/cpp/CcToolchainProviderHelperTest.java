// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@code CcToolchainProviderHelper} */
@RunWith(JUnit4.class)
public class CcToolchainProviderHelperTest extends BuildViewTestCase {

  @Before
  public void createFooFooCcLibraryForRuleContext() throws IOException {
    scratch.file("foo/BUILD", "cc_library(name = 'foo')");
  }

  private RuleContext getRuleContext() throws Exception {
    return getRuleContext(getConfiguredTarget("//foo:foo"));
  }

  @Test
  public void getDynamicRuntimeLinkMiddleman_disableMiddlemanArtifacts() throws Exception {
    useConfiguration("--noexperimental_enable_aggregating_middleman");
    RuleContext ruleContext = getRuleContext();

    NestedSetBuilder<Artifact> nonEmptyBuilder = NestedSetBuilder.stableOrder();
    nonEmptyBuilder.add(getSourceArtifact("foo.h"));
    Artifact middleman =
        CcToolchainProviderHelper.getDynamicRuntimeLinkMiddleman(
            ruleContext, "purposePrefix", "runtimeSolibDirBase", "solibDirectory", nonEmptyBuilder);
    assertThat(middleman).isNull();
  }

  @Test
  public void getStaticRuntimeLinkMiddleman_disableMiddlemanArtifacts() throws Exception {
    useConfiguration("--noexperimental_enable_aggregating_middleman");
    RuleContext ruleContext = getRuleContext();

    NestedSet<Artifact> nonEmptySet =
        NestedSetBuilder.<Artifact>stableOrder().add(getSourceArtifact("foo.h")).build();
    Artifact middleman =
        CcToolchainProviderHelper.getStaticRuntimeLinkMiddleman(
            ruleContext, "purposePrefix", /* staticRuntimeLib= */ null, nonEmptySet);
    assertThat(middleman).isNull();
  }
}
