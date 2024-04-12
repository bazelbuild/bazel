// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationResult;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FlagSetsFunctionTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.build();
  }

  @Test
  public void flagSetsFunction_returns_modified_buildOptions() throws Exception {
    rewriteWorkspace("workspace(name = 'my_workspace')");
    scratch.file(
        "test/PROJECT.scl", "test_config = ['--platforms=//buildenv/platforms/android:x86']");
    scratch.file("test/BUILD");
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    // given original BuildOptions and a valid key
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    PathFragment projectFile = PathFragment.createAlreadyNormalized("//test/PROJECT.scl");
    FlagSetValue.Key key = FlagSetValue.Key.create(projectFile, "test_config", buildOptions);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the modified BuildOptions
    assertThat(flagSetsValue.getTopLevelBuildOptions().get(PlatformOptions.class).platforms)
        .containsExactly(Label.parseCanonical("//buildenv/platforms/android:x86"));
  }

  @Test
  public void given_unknown_sclConfig_flagSetsFunction_returns_original_buildOptions()
      throws Exception {
    rewriteWorkspace("workspace(name = 'my_workspace')");
    scratch.file(
        "test/PROJECT.scl", "test_config = ['--platforms=//buildenv/platforms/android:x86']");
    scratch.file("test/BUILD");
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    // given valid project file but a nonexistent scl config
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    PathFragment projectFile = PathFragment.createAlreadyNormalized("//test/PROJECT.scl");
    FlagSetValue.Key key = FlagSetValue.Key.create(projectFile, "unknown_config", buildOptions);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the original BuildOptions
    assertThat(flagSetsValue.getTopLevelBuildOptions()).isEqualTo(buildOptions);
  }

  @Test
  public void flagSetsFunction_returns_origional_buildOptions() throws Exception {
    // given original BuildOptions and an empty scl config name
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    PathFragment projectFile = PathFragment.create("test/PROJECT.scl");
    FlagSetValue.Key key = FlagSetValue.Key.create(projectFile, "", buildOptions);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the original BuildOptions
    assertThat(flagSetsValue.getTopLevelBuildOptions()).isEqualTo(buildOptions);
  }

  private FlagSetValue executeFunction(FlagSetValue.Key key) throws Exception {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    EvaluationResult<FlagSetValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /* keepGoing= */ false, reporter);
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty())));
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key);
  }
}
