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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Skylark API for {@link PlatformInfo} providers. */
@RunWith(JUnit4.class)
public class PlatformInfoApiTest extends BuildViewTestCase {

  @Test
  public void testPlatform() throws Exception {
    constraintBuilder("//foo:basic").addConstraintValue("value1").write();
    platformBuilder("//foo:my_platform").addConstraint("value1").write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
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
    List<String> lines =
        new ImmutableList.Builder<String>()
            .addAll(
                constraintBuilder("//foo:basic")
                    .addConstraintValue("value1")
                    .addConstraintValue("value2")
                    .lines())
            .addAll(
                platformBuilder("//foo:my_platform")
                    .addConstraint("value1")
                    .addConstraint("value2")
                    .lines())
            .build();

    checkError(
        "foo",
        "my_platform",
        "Duplicate constraint_values detected: "
            + "constraint_setting //foo:basic has [//foo:value1, //foo:value2]",
        lines.toArray(new String[] {}));
  }

  @Test
  public void testPlatform_remoteExecution() throws Exception {
    platformBuilder("//foo:my_platform").setRemoteExecutionProperties("foo: val1").write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("foo: val1");
  }

  ConstraintBuilder constraintBuilder(String name) {
    return new ConstraintBuilder(name);
  }

  final class ConstraintBuilder {
    private final Label label;
    private final List<String> constraintValues = new ArrayList<>();
    private String defaultConstraintValue = null;

    public ConstraintBuilder(String name) {
      this.label = Label.parseAbsoluteUnchecked(name);
    }

    public ConstraintBuilder defaultConstraintValue(String defaultConstraintValue) {
      this.defaultConstraintValue = defaultConstraintValue;
      this.constraintValues.add(defaultConstraintValue);
      return this;
    }

    public ConstraintBuilder addConstraintValue(String constraintValue) {
      this.constraintValues.add(constraintValue);
      return this;
    }

    public List<String> lines() {
      ImmutableList.Builder<String> lines = ImmutableList.builder();

      // Add the constraint setting.
      lines.add("constraint_setting(name = '" + label.getName() + "',");
      if (!Strings.isNullOrEmpty(defaultConstraintValue)) {
        lines.add("  default_constraint_value = ':" + defaultConstraintValue + "',");
      }
      lines.add(")");

      // Add the constraint values.
      for (String constraintValue : constraintValues) {
        lines.add(
            "constraint_value(",
            "  name = '" + constraintValue + "',",
            "  constraint_setting = ':" + label.getName() + "',",
            ")");
      }

      return lines.build();
    }

    public void write() throws Exception {
      List<String> lines = lines();
      String filename = label.getPackageFragment().getRelative("BUILD").getPathString();
      scratch.appendFile(filename, lines.toArray(new String[] {}));
    }
  }

  PlatformBuilder platformBuilder(String name) {
    return new PlatformBuilder(name);
  }

  final class PlatformBuilder {
    private final Label label;
    private final List<String> constraintValues = new ArrayList<>();
    private String remoteExecutionProperties = "";

    public PlatformBuilder(String name) {
      this.label = Label.parseAbsoluteUnchecked(name);
    }

    public PlatformBuilder addConstraint(String value) {
      this.constraintValues.add(value);
      return this;
    }

    public PlatformBuilder setRemoteExecutionProperties(String value) {
      this.remoteExecutionProperties = value;
      return this;
    }

    public List<String> lines() {
      ImmutableList.Builder<String> lines = ImmutableList.builder();

      lines.add("platform(", "  name = '" + label.getName() + "',");
      lines.add("  constraint_values = [");
      for (String name : constraintValues) {
        lines.add("    ':" + name + "',");
      }
      lines.add("  ],");
      if (!Strings.isNullOrEmpty(remoteExecutionProperties)) {
        lines.add("  remote_execution_properties = '" + remoteExecutionProperties + "',");
      }
      lines.add(")");

      return lines.build();
    }

    public void write() throws Exception {
      List<String> lines = lines();
      String filename = label.getPackageFragment().getRelative("BUILD").getPathString();
      scratch.appendFile(filename, lines.toArray(new String[] {}));
    }
  }

  @Nullable
  ConstraintSettingInfo fetchConstraintSettingInfo(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    return PlatformProviderUtils.constraintSetting(target);
  }

  @Nullable
  PlatformInfo fetchPlatformInfo(String platformLabel) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(platformLabel);
    return PlatformProviderUtils.platform(target);
  }
}
