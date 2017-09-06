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

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.MockPlatformSupport;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for toolchains computed in BuildViewTestCase. */
@RunWith(JUnit4.class)
public class RuleContextTest extends ToolchainTestCase {

  @Test
  public void testMockRuleContextHasToolchains() throws Exception {
    MockPlatformSupport.setup(mockToolsConfig);
    mockToolsConfig.create("x/BUILD", "mock_toolchain_rule(name='x')");
    useConfiguration(
        "--experimental_host_platform=//constraint:linux_plat",
        "--experimental_platforms=//constraint:mac_plat");
    RuleContext ruleContext = getRuleContext(getConfiguredTarget("//x"));
    assertThat(ruleContext.getToolchainContext().getResolvedToolchainLabels())
        .contains(Label.parseAbsolute("//toolchain:test_toolchain_1"));
  }
}
