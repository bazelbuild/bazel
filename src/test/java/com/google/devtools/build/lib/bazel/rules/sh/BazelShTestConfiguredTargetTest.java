// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.sh;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for sh_test configured target. */
@RunWith(JUnit4.class)
public class BazelShTestConfiguredTargetTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    scratch.file("BUILD", "sh_test(name = 'test', srcs = ['test.sh'])");
  }

  @Test
  public void testCoverageOutputGenerator() throws Exception {
    reporter.removeHandler(failFastHandler);
    ConfiguredTarget ct = getConfiguredTarget("//:test");
    assertThat(getRuleContext(ct).getPrerequisite(":lcov_merger")).isNull();
  }

  @Test
  public void testCoverageOutputGeneratorCoverageMode() throws Exception {
    useConfiguration("--collect_code_coverage");
    reporter.removeHandler(failFastHandler);
    ConfiguredTarget ct = getConfiguredTarget("//:test");
    assertThat(getRuleContext(ct).getPrerequisite(":lcov_merger").getLabel().toString())
        .isEqualTo("@bazel_tools//tools/test:lcov_merger");
  }

  @Test
  public void testNonWindowsWrapper() throws Exception {
    assertThat(getTestRunnerAction("//:test").getArguments().get(0)).endsWith("test-setup.sh");
  }

  @Test
  public void testWindowsWrapper() throws Exception {
    scratch.file(
        "platforms/BUILD",
        "platform(name = 'windows', constraint_values = ['@platforms//os:windows'])");
    useConfiguration("--host_platform=//platforms:windows");

    assertThat(getTestRunnerAction("//:test").getArguments().get(0)).endsWith("test_wrapper_bin");
  }

  private TestRunnerAction getTestRunnerAction(String label) throws Exception {
    return (TestRunnerAction) Iterables.getOnlyElement(getActions(label, TestRunnerAction.class));
  }
}
