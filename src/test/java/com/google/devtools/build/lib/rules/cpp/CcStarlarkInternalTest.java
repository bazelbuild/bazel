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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the {@code cc_internal} Starlark module. */
@RunWith(JUnit4.class)
public final class CcStarlarkInternalTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addStarlarkAccessibleTopLevels(CcStarlarkInternal.NAME, new CcStarlarkInternal());
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void testBuildInfoArtifacts() throws Exception {
    scratch.file(
        "bazel_internal/test_rules/cc/rule.bzl",
        "def _impl(ctx):",
        "  artifacts = cc_internal.get_build_info(ctx)",
        "  return [DefaultInfo(files = depset(artifacts))]",
        "build_info_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {'stamp': attr.int()},",
        ")");
    scratch.file(
        "bazel_internal/test_rules/cc/BUILD",
        "load(':rule.bzl', 'build_info_rule')",
        "build_info_rule(name = 'stamped', stamp = 1,)",
        "build_info_rule(name = 'unstamped', stamp = 0,)");
    assertThat(
            prettyArtifactNames(
                getConfiguredTarget("//bazel_internal/test_rules/cc:stamped")
                    .getProvider(FileProvider.class)
                    .getFilesToBuild()))
        .containsExactly("build-info-nonvolatile.h", "build-info-volatile.h");
    assertThat(
            prettyArtifactNames(
                getConfiguredTarget("//bazel_internal/test_rules/cc:unstamped")
                    .getProvider(FileProvider.class)
                    .getFilesToBuild()))
        .containsExactly("build-info-redacted.h");
  }
}
