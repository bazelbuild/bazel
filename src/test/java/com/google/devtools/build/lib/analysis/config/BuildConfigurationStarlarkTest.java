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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.syntax.Dict;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildConfiguration}'s integration with Starlark. */
@RunWith(JUnit4.class)
public final class BuildConfigurationStarlarkTest extends BuildViewTestCase {

  @Test
  public void testStarlarkWithTestEnvOptions() throws Exception {
    useConfiguration("--test_env=TEST_ENV_VAR=my_value");
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/config_test.bzl",
        "MyInfo = provider()",
        "def _test_rule_impl(ctx):",
        "   return MyInfo(test_env = ctx.configuration.test_env)",
        "test_rule = rule(implementation = _test_rule_impl,",
        "   attrs = {},",
        ")");

    scratch.file(
        "examples/config_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:config_test.bzl', 'test_rule')",
        "test_rule(",
        "    name = 'my_target',",
        ")");

    ConfiguredTarget starlarkTarget = getConfiguredTarget("//examples/config_starlark:my_target");
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//examples/rule:config_test.bzl", ImmutableMap.of()), "MyInfo");
    StructImpl myInfo = (StructImpl) starlarkTarget.get(key);
    assertThat(((Dict) myInfo.getValue("test_env")).get("TEST_ENV_VAR")).isEqualTo("my_value");
  }
}
