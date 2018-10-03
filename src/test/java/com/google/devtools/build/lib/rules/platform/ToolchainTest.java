// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link Toolchain}. */
@RunWith(JUnit4.class)
public class ToolchainTest extends BuildViewTestCase {

  @Before
  public void createConstraints() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "constraint_value(name = 'bar',",
        "    constraint_setting = ':basic',",
        "    )");
  }

  @Test
  public void testToolchain() throws Exception {
    scratch.file(
        "toolchain/toolchain_def.bzl",
        "def _impl(ctx):",
        "  toolchain = platform_common.ToolchainInfo(",
        "      data = ctx.attr.data)",
        "  return [toolchain]",
        "toolchain_def = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'data': attr.string()})");
    scratch.file(
        "toolchain/BUILD",
        "load(':toolchain_def.bzl', 'toolchain_def')",
        "toolchain_type(name = 'demo_toolchain')",
        "toolchain(",
        "  name = 'toolchain1',",
        "  toolchain_type = ':demo_toolchain',",
        "  exec_compatible_with = ['//constraint:foo'],",
        "  target_compatible_with = ['//constraint:bar'],",
        "  toolchain = ':toolchain_def1')",
        "toolchain_def(",
        "  name = 'toolchain_def1',",
        "  data = 'foo')");

    ConfiguredTarget target = getConfiguredTarget("//toolchain:toolchain1");
    assertThat(target).isNotNull();

    DeclaredToolchainInfo provider = target.getProvider(DeclaredToolchainInfo.class);
    assertThat(provider).isNotNull();
    assertThat(provider.toolchainType())
        .isEqualTo(ToolchainTypeInfo.create(makeLabel("//toolchain:demo_toolchain")));

    ConstraintSettingInfo basicConstraintSetting =
        ConstraintSettingInfo.create(makeLabel("//constraint:basic"));
    assertThat(provider.execConstraints().get(basicConstraintSetting))
        .isEqualTo(
            ConstraintValueInfo.create(basicConstraintSetting, makeLabel("//constraint:foo")));
    assertThat(provider.targetConstraints().get(basicConstraintSetting))
        .isEqualTo(
            ConstraintValueInfo.create(basicConstraintSetting, makeLabel("//constraint:bar")));

    assertThat(provider.toolchainLabel()).isEqualTo(makeLabel("//toolchain:toolchain_def1"));
  }
}
