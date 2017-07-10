// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildConfiguration}'s integration with Skylark. */
@RunWith(JUnit4.class)
public final class BuildConfigurationSkylarkTest extends BuildViewTestCase {

  @Test
  public void testSkylarkWithTestEnvOptions() throws Exception {
    useConfiguration("--test_env=TEST_ENV_VAR=my_value");
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/config_test.bzl",
        "def _test_rule_impl(ctx):",
        "   return struct(test_env = ctx.configuration.test_env)",
        "test_rule = rule(implementation = _test_rule_impl,",
        "   attrs = {},",
        ")");

    scratch.file(
        "examples/config_skylark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('/examples/rule/config_test', 'test_rule')",
        "test_rule(",
        "    name = 'my_target',",
        ")");

    ConfiguredTarget skylarkTarget = getConfiguredTarget("//examples/config_skylark:my_target");
    assertThat(((SkylarkDict) skylarkTarget.get("test_env")).get("TEST_ENV_VAR"))
        .isEqualTo("my_value");
  }
}
