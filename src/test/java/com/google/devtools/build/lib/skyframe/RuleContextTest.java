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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.analysis.testing.ToolchainContextSubject.assertThat;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for toolchains computed in BuildViewTestCase. */
@RunWith(JUnit4.class)
public class RuleContextTest extends ToolchainTestCase {

  @Test
  public void testToolchains() throws Exception {
    mockToolsConfig.create("x/BUILD", "mock_toolchain_rule(name='x')");
    useConfiguration("--host_platform=//platforms:linux", "--platforms=//platforms:mac");
    RuleContext ruleContext = getRuleContext(getConfiguredTarget("//x"));
    assertThat(ruleContext.getToolchainContext().resolvedToolchainLabels())
        .contains(Label.parseCanonical("//toolchain:toolchain_1_impl"));

    assertThat(ruleContext.getToolchainContext()).hasToolchainType("//toolchain:test_toolchain");
    ToolchainInfo toolchain =
        ruleContext.getToolchainInfo(Label.parseCanonical("//toolchain:test_toolchain"));
    assertThat(toolchain.getValue("data")).isEqualTo("foo");
  }

  @Test
  public void testTargetPlatformHasConstraint_mac() throws Exception {
    scratch.file("a/BUILD", "filegroup(name = 'a')");
    useConfiguration("--platforms=//platforms:mac");
    RuleContext ruleContext = getRuleContext(getConfiguredTarget("//a"));
    assertThat(ruleContext.targetPlatformHasConstraint(macConstraint)).isTrue();
    assertThat(ruleContext.targetPlatformHasConstraint(linuxConstraint)).isFalse();
  }

  @Test
  public void testTargetPlatformHasConstraint_linux() throws Exception {
    scratch.file("a/BUILD", "filegroup(name = 'a')");
    useConfiguration("--platforms=//platforms:linux");
    RuleContext ruleContext = getRuleContext(getConfiguredTarget("//a"));
    assertThat(ruleContext.targetPlatformHasConstraint(macConstraint)).isFalse();
    assertThat(ruleContext.targetPlatformHasConstraint(linuxConstraint)).isTrue();
  }

  @Test
  public void testTestonlyToolchain_allowed() throws Exception {
    createTestonlyToolchain();

    scratch.file(
        "p0/BUILD",
        "load('//foo:rule_def.bzl', 'foo_rule')",
        "foo_rule(",
        "    name = 'p0',",
        "    testonly = True,",
        ")");
    // This should succeed.
    getConfiguredTarget("//p0:p0");
  }

  @Test
  public void testTestonlyToolchain_invalid() throws Exception {
    createTestonlyToolchain();

    checkError(
        "p0",
        "p0",
        // error:
        "non-test target '//p0:p0' depends on testonly target",
        // build file:
        "load('//foo:rule_def.bzl', 'foo_rule')",
        "foo_rule(",
        "    name = 'p0',",
        "    testonly = False,", // False is the default, we set it here for clarity.
        ")");
  }

  private void createTestonlyToolchain() throws Exception {
    // Define a custom rule with a testonly toolchain.
    scratch.file(
        "foo/toolchain_def.bzl",
        "def _impl(ctx):",
        "  return [platform_common.ToolchainInfo()]",
        "foo_toolchain = rule(",
        "    implementation = _impl,",
        "    attrs = {})");
    scratch.file(
        "foo/rule_def.bzl",
        "def _impl(ctx):",
        "    pass",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    toolchains = ['//foo:toolchain_type'])");
    scratch.file("foo/BUILD", "toolchain_type(name = 'toolchain_type')");
    // Create an instance of the toolchain.
    scratch.file(
        "bar/BUILD",
        "load('//foo:toolchain_def.bzl', 'foo_toolchain')",
        "toolchain(",
        "  name = 'foo_toolchain_impl',",
        "  toolchain_type = '//foo:toolchain_type',",
        "  toolchain = ':foo_toolchain_def')",
        "foo_toolchain(",
        "    name = 'foo_toolchain_def',",
        "    testonly = True,",
        ")");

    useConfiguration("--extra_toolchains=//bar:all");
  }
}
