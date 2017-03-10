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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link Platform}. */
@RunWith(JUnit4.class)
public class PlatformTest extends BuildViewTestCase {

  @Test
  public void testPlatform() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "platform(name = 'plat1',",
        "    constraint_values = [",
        "       ':foo',",
        "    ])");

    ConfiguredTarget platform = getConfiguredTarget("//constraint:plat1");
    assertThat(platform).isNotNull();

    PlatformProvider provider = platform.getProvider(PlatformProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.constraints()).hasSize(1);
    ConstraintSettingProvider constraintSettingProvider =
        ConstraintSettingProvider.create(makeLabel("//constraint:basic"));
    ConstraintValueProvider constraintValueProvider =
        ConstraintValueProvider.create(constraintSettingProvider, makeLabel("//constraint:foo"));
    assertThat(provider.constraints())
        .containsExactlyEntriesIn(
            ImmutableMap.of(constraintSettingProvider, constraintValueProvider));
    assertThat(provider.remoteExecutionProperties()).isEmpty();
  }

  @Test
  public void testPlatform_overlappingConstraintValueError() throws Exception {
    checkError(
        "constraint",
        "plat1",
        "Duplicate constraint_values for constraint_setting //constraint:basic: "
            + "//constraint:foo, //constraint:bar",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "constraint_value(name = 'bar',",
        "    constraint_setting = ':basic',",
        "    )",
        "platform(name = 'plat1',",
        "    constraint_values = [",
        "       ':foo',",
        "       ':bar',",
        "    ])");
  }

  @Test
  public void testPlatform_remoteExecution() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )",
        "platform(name = 'plat1',",
        "    constraint_values = [",
        "       ':foo',",
        "    ],",
        "    remote_execution_properties = {",
        "        'foo': 'val1',",
        "        'bar': 'val2',",
        "    },",
        ")");

    ConfiguredTarget platform = getConfiguredTarget("//constraint:plat1");
    assertThat(platform).isNotNull();

    PlatformProvider provider = platform.getProvider(PlatformProvider.class);
    assertThat(provider).isNotNull();
    assertThat(provider.remoteExecutionProperties())
        .containsExactlyEntriesIn(ImmutableMap.of("foo", "val1", "bar", "val2"));
  }
}
