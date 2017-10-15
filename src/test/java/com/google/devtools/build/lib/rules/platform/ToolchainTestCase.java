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

import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import org.junit.Before;

/** Utility methods for setting up platform and toolchain related tests. */
public abstract class ToolchainTestCase extends SkylarkTestCase {

  public PlatformInfo linuxPlatform;
  public PlatformInfo macPlatform;

  public ConstraintSettingInfo setting;
  public ConstraintValueInfo linuxConstraint;
  public ConstraintValueInfo macConstraint;

  public Label testToolchainType;

  @Before
  public void createConstraints() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'os')",
        "constraint_value(name = 'linux',",
        "    constraint_setting = ':os')",
        "constraint_value(name = 'mac',",
        "    constraint_setting = ':os')",
        "platform(name = 'linux_plat',",
        "    constraint_values = [':linux'])",
        "platform(name = 'mac_plat',",
        "    constraint_values = [':mac'])");

    setting = ConstraintSettingInfo.create(makeLabel("//constraint:os"));
    linuxConstraint = ConstraintValueInfo.create(setting, makeLabel("//constraint:linux"));
    macConstraint = ConstraintValueInfo.create(setting, makeLabel("//constraint:mac"));

    linuxPlatform =
        PlatformInfo.builder()
            .setLabel(makeLabel("//platforms:target_platform"))
            .addConstraint(linuxConstraint)
            .build();
    macPlatform =
        PlatformInfo.builder()
            .setLabel(makeLabel("//platforms:host_platform"))
            .addConstraint(macConstraint)
            .build();
  }

  @Before
  public void createToolchains() throws Exception {
    rewriteWorkspace("register_toolchains('//toolchain:toolchain_1', '//toolchain:toolchain_2')");

    scratch.file(
        "toolchain/BUILD",
        "load(':toolchain_def.bzl', 'test_toolchain')",
        "toolchain_type(name = 'test_toolchain')",
        "toolchain(",
        "    name = 'toolchain_1',",
        "    toolchain_type = ':test_toolchain',",
        "    exec_compatible_with = ['//constraint:linux'],",
        "    target_compatible_with = ['//constraint:mac'],",
        "    toolchain = ':test_toolchain_1')",
        "toolchain(",
        "    name = 'toolchain_2',",
        "    toolchain_type = ':test_toolchain',",
        "    exec_compatible_with = ['//constraint:mac'],",
        "    target_compatible_with = ['//constraint:linux'],",
        "    toolchain = ':test_toolchain_2')",
        "test_toolchain(",
        "  name='test_toolchain_1',",
        "  data = 'foo')",
        "test_toolchain(",
        "  name='test_toolchain_2',",
        "  data = 'bar')");
    scratch.file(
        "toolchain/toolchain_def.bzl",
        "def _impl(ctx):",
        "  toolchain = platform_common.ToolchainInfo(",
        "      data = ctx.attr.data)",
        "  return [toolchain]",
        "test_toolchain = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'data': attr.string()})");

    testToolchainType = makeLabel("//toolchain:test_toolchain");
  }
}
