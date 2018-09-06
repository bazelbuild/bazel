// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Strings;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformInfoApi;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Skylark API for {@link PlatformInfoApi} providers. */
@RunWith(JUnit4.class)
public class PlatformInfoApiTest extends BuildViewTestCase {

  @Test
  public void testPlatform() throws Exception {
    platformBuilder().addConstraint("basic", "value1").build();
    assertNoEvents();

    PlatformInfoApi platformInfo = fetchPlatformInfo();
    assertThat(platformInfo).isNotNull();
    ConstraintSettingInfo constraintSetting =
        ConstraintSettingInfo.create(makeLabel("//foo:basic"));
    ConstraintValueInfo constraintValue =
        ConstraintValueInfo.create(constraintSetting, makeLabel("//foo:value1"));
    assertThat(platformInfo.constraints().get(constraintSetting)).isEqualTo(constraintValue);
    assertThat(platformInfo.remoteExecutionProperties()).isEmpty();
  }

  @Test
  public void testPlatform_overlappingConstraintValueError() throws Exception {
    checkError(
        "foo",
        "my_platform",
        "Duplicate constraint_values detected: "
            + "constraint_setting //foo:basic has "
            + "[//foo:value1, //foo:value2]",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'value1',",
        "    constraint_setting = ':basic',",
        "    )",
        "constraint_value(name = 'value2',",
        "    constraint_setting = ':basic',",
        "    )",
        "platform(name = 'my_platform',",
        "    constraint_values = [",
        "       ':value1',",
        "       ':value2',",
        "    ])");
  }

  @Test
  public void testPlatform_remoteExecution() throws Exception {
    platformBuilder().setRemoteExecutionProperties("foo: val1").build();
    assertNoEvents();

    PlatformInfoApi platformInfo = fetchPlatformInfo();
    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("foo: val1");
  }

  PlatformBuilder platformBuilder() {
    return new PlatformBuilder();
  }

  final class PlatformBuilder {
    private final Multimap<String, String> constraints = HashMultimap.create();
    private String remoteExecutionProperties = "";

    public PlatformBuilder addConstraint(String setting, String value) {
      this.constraints.put(setting, value);
      return this;
    }

    public PlatformBuilder setRemoteExecutionProperties(String value) {
      this.remoteExecutionProperties = value;
      return this;
    }

    public void build() throws Exception {
      ImmutableList.Builder<String> lines = ImmutableList.builder();

      // Add the constraint settings.
      for (String name : constraints.keySet()) {
        lines.add("constraint_setting(name = '" + name + "')");
      }

      // Add the constraint values.
      for (Map.Entry<String, String> entry : constraints.entries()) {
        lines.add(
            "constraint_value(",
            "  name = '" + entry.getValue() + "',",
            "  constraint_setting = ':" + entry.getKey() + "',",
            ")");
      }

      // Add the platform.
      lines.add("platform(", "  name = 'my_platform',");
      if (!constraints.isEmpty()) {
        lines.add("  constraint_values = [");
        for (String name : constraints.values()) {
          lines.add("    ':" + name + "',");
        }
        lines.add("  ],");
      }
      if (!Strings.isNullOrEmpty(remoteExecutionProperties)) {
        lines.add("  remote_execution_properties = '" + remoteExecutionProperties + "',");
      }
      lines.add(")");

      scratch.file("foo/BUILD", lines.build().toArray(new String[] {}));
    }
  }

  @Nullable
  PlatformInfoApi fetchPlatformInfo() throws Exception {
    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:my_platform");
    return PlatformProviderUtils.platform(myRuleTarget);
  }
}
