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
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ToolchainResolutionValue} and {@link ToolchainResolutionFunction}. */
@RunWith(JUnit4.class)
public class ToolchainResolutionFunctionTest extends ToolchainTestCase {

  private EvaluationResult<ToolchainResolutionValue> invokeToolchainResolution(SkyKey key)
      throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }

  // TODO(katre): Current toolchain resolution does not actually check the constraints, it just
  // returns the first toolchain available.
  @Test
  public void testResolution() throws Exception {
    SkyKey key =
        ToolchainResolutionValue.key(targetConfig, testToolchainType, targetPlatform, hostPlatform);
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result).hasNoError();

    ToolchainResolutionValue toolchainResolutionValue = result.get(key);
    assertThat(toolchainResolutionValue.toolchainLabel())
        .isEqualTo(makeLabel("//toolchain:test_toolchain_1"));
  }

  @Test
  public void testResolution_noneFound() throws Exception {
    // Clear the toolchains.
    rewriteWorkspace();

    SkyKey key =
        ToolchainResolutionValue.key(targetConfig, testToolchainType, targetPlatform, hostPlatform);
    EvaluationResult<ToolchainResolutionValue> result = invokeToolchainResolution(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no matching toolchain found for //toolchain:test_toolchain");
  }

  @Test
  public void testToolchainResolutionValue_equalsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_1")),
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_1")))
        .addEqualityGroup(
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_2")),
            ToolchainResolutionValue.create(makeLabel("//test:toolchain_impl_2")));
  }
}
