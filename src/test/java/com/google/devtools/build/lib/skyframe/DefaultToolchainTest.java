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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for default toolchains. */
@RunWith(JUnit4.class)
public class DefaultToolchainTest extends ToolchainTestCase {
  @Test
  public void testDefaultCcToolchainIsPresent() throws Exception {
    SkyKey toolchainKey = RegisteredToolchainsValue.key(targetConfig);
    EvaluationResult<RegisteredToolchainsValue> result =
        requestToolchainsFromSkyframe(toolchainKey);
    ImmutableList<DeclaredToolchainInfo> declaredToolchains =
        result.get(toolchainKey).registeredToolchains();
    List<Label> labels = collectToolchainLabels(declaredToolchains);
    assertThat(
            labels
                .stream()
                .anyMatch(
                    toolchainLabel ->
                        toolchainLabel.toString().contains("//tools/cpp:dummy_cc_toolchain_impl")))
        .isTrue();
  }
}
