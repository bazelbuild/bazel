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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for sh_test configured target. */
@RunWith(JUnit4.class)
public class BazelShTestConfiguredTargetTest extends BuildViewTestCase {
  @Test
  public void testCoverageOutputGenerator() throws Exception {
    scratch.file("sh/test/BUILD", "sh_test(name = 'foo_test', srcs = ['foo_test.sh'])");
    reporter.removeHandler(failFastHandler);
    ConfiguredTarget ct = getConfiguredTarget("//sh/test:foo_test");
    assertThat(getRuleContext(ct).getPrerequisite(":lcov_merger", Mode.HOST)).isNull();
  }

  @Test
  public void testCoverageOutputGeneratorCoverageMode() throws Exception {
    useConfiguration("--collect_code_coverage");
    scratch.file("sh/test/BUILD", "sh_test(name = 'foo_test', srcs = ['foo_test.sh'])");
    reporter.removeHandler(failFastHandler);
    ConfiguredTarget ct = getConfiguredTarget("//sh/test:foo_test");
    assertThat(getRuleContext(ct).getPrerequisite(":lcov_merger", Mode.HOST).getLabel().toString())
        .isEqualTo(
            "@bazel_tools//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Main");
  }
}
