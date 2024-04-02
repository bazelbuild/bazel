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
package com.google.devtools.build.lib.analysis.platform;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo.ExecPropertiesException;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link PlatformInfo}. */
@RunWith(JUnit4.class)
public class PlatformInfoTest extends BuildViewTestCase {

  @Test
  public void builder() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s1"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s2"));

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.addConstraint(
        ConstraintValueInfo.create(setting1, Label.parseCanonicalUnchecked("//constraint:v1")));
    builder.addConstraint(
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:v2")));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.constraints().has(setting1)).isTrue();
    assertThat(platformInfo.constraints().get(setting1).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v1"));
    assertThat(platformInfo.constraints().has(setting2)).isTrue();
    assertThat(platformInfo.constraints().get(setting2).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v2"));
  }

  @Test
  public void constraints_parentPlatform_noOverlaps() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s1"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s2"));
    ConstraintSettingInfo setting3 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s3"));

    PlatformInfo parent =
        PlatformInfo.builder()
            .addConstraint(
                ConstraintValueInfo.create(
                    setting1, Label.parseCanonicalUnchecked("//constraint:v1")))
            .build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.addConstraint(
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:v2")));
    builder.addConstraint(
        ConstraintValueInfo.create(setting3, Label.parseCanonicalUnchecked("//constraint:v3")));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.constraints().has(setting1)).isTrue();
    assertThat(platformInfo.constraints().get(setting1).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v1"));
    assertThat(platformInfo.constraints().has(setting2)).isTrue();
    assertThat(platformInfo.constraints().get(setting2).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v2"));
    assertThat(platformInfo.constraints().has(setting3)).isTrue();
    assertThat(platformInfo.constraints().get(setting3).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v3"));
  }

  @Test
  public void constraints_parentPlatform_overlaps() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s1"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s2"));
    ConstraintSettingInfo setting3 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:s3"));

    PlatformInfo parent =
        PlatformInfo.builder()
            .addConstraint(
                ConstraintValueInfo.create(
                    setting1, Label.parseCanonicalUnchecked("//constraint:v1")))
            .build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.addConstraint(
        ConstraintValueInfo.create(setting1, Label.parseCanonicalUnchecked("//constraint:v1a")));
    builder.addConstraint(
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:v2")));
    builder.addConstraint(
        ConstraintValueInfo.create(setting3, Label.parseCanonicalUnchecked("//constraint:v3")));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.constraints().get(setting1).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v1a"));
    assertThat(platformInfo.constraints().get(setting2).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v2"));
    assertThat(platformInfo.constraints().get(setting3).label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint:v3"));
  }

  @Test
  public void constraints_overlappingError() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:basic"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:complex"));
    ConstraintSettingInfo setting3 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:single"));

    PlatformInfo.Builder builder = PlatformInfo.builder();

    builder.addConstraint(
        ConstraintValueInfo.create(setting1, Label.parseCanonicalUnchecked("//constraint:value1")));
    builder.addConstraint(
        ConstraintValueInfo.create(setting1, Label.parseCanonicalUnchecked("//constraint:value2")));

    builder.addConstraint(
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:value3")));
    builder.addConstraint(
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:value4")));
    builder.addConstraint(
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:value5")));

    builder.addConstraint(
        ConstraintValueInfo.create(setting3, Label.parseCanonicalUnchecked("//constraint:value6")));

    ConstraintCollection.DuplicateConstraintException exception =
        assertThrows(
            ConstraintCollection.DuplicateConstraintException.class, () -> builder.build());
    assertThat(exception)
        .hasMessageThat()
        .contains(
            "Duplicate constraint values detected: "
                + "constraint_setting //constraint:basic has "
                + "[//constraint:value1, //constraint:value2], "
                + "constraint_setting //constraint:complex has "
                + "[//constraint:value3, //constraint:value4, //constraint:value5]");
  }

  @Test
  public void remoteExecutionProperties() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setRemoteExecutionProperties("properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("properties");
  }

  @Test
  public void remoteExecutionProperties_parentPlatform_keep() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder().setRemoteExecutionProperties("parent properties").build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("parent properties");
  }

  @Test
  public void remoteExecutionProperties_parentPlatform_override() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder().setRemoteExecutionProperties("parent properties").build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.setRemoteExecutionProperties("child properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child properties");
    assertThat(platformInfo.execProperties()).isEmpty();
  }

  @Test
  public void remoteExecutionProperties_parentPlatform_merge() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder().setRemoteExecutionProperties("parent properties").build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.setRemoteExecutionProperties("child {PARENT_REMOTE_EXECUTION_PROPERTIES} properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties())
        .isEqualTo("child parent properties properties");
  }

  @Test
  public void remoteExecutionProperties_parentPlatform_merge_noParent() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setRemoteExecutionProperties("child {PARENT_REMOTE_EXECUTION_PROPERTIES} properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child  properties");
  }

  @Test
  public void remoteExecutionProperties_parentPlatform_merge_parentNotSet() throws Exception {
    PlatformInfo parent = PlatformInfo.builder().build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.setRemoteExecutionProperties("child {PARENT_REMOTE_EXECUTION_PROPERTIES} properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child  properties");
  }

  @Test
  public void remoteExecutionProperties_parentSpecifiesExecProperties_error() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder()
            .setLabel(Label.parseCanonicalUnchecked("//foo:parent_platform"))
            .setExecProperties(ImmutableMap.of("elem1", "value1"))
            .build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.setRemoteExecutionProperties("props");

    ExecPropertiesException exception = assertThrows(ExecPropertiesException.class, builder::build);
    assertThat(exception)
        .hasMessageThat()
        .contains(
            "Platform specifies remote_execution_properties but its parent specifies"
                + " exec_properties. Prefer exec_properties over the deprecated"
                + " remote_execution_properties.");
  }

  @Test
  public void execProperties_empty() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setExecProperties(ImmutableMap.of());
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.execProperties()).isNotNull();
    assertThat(platformInfo.execProperties()).isEmpty();
  }

  @Test
  public void execProperties_one() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setExecProperties(ImmutableMap.of("elem1", "value1"));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.execProperties()).isNotNull();
    assertThat(platformInfo.execProperties()).containsExactly("elem1", "value1");
  }

  @Test
  public void execProperties_parentPlatform_keep() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder().setExecProperties(ImmutableMap.of("parent", "properties")).build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.execProperties()).containsExactly("parent", "properties");
  }

  @Test
  public void execProperties_parentPlatform_inheritance() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder()
            .setExecProperties(
                ImmutableMap.of("p1", "keep", "p2", "delete", "p3", "parent", "p4", "del2"))
            .build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    PlatformInfo platformInfo =
        builder.setExecProperties(ImmutableMap.of("p2", "", "p3", "child", "p4", "")).build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.execProperties()).containsExactly("p1", "keep", "p3", "child");
  }

  @Test
  public void execProperties_remoteExecProperties_error() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setExecProperties(ImmutableMap.of("elem1", "value1"));
    builder.setRemoteExecutionProperties("props");

    ExecPropertiesException exception = assertThrows(ExecPropertiesException.class, builder::build);
    assertThat(exception)
        .hasMessageThat()
        .contains(
            "Platform contains both remote_execution_properties and exec_properties. Prefer"
                + " exec_properties over the deprecated remote_execution_properties.");
  }

  @Test
  public void execProperties_parentSpecifiesRemoteExecutionProperties_error() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder()
            .setLabel(Label.parseCanonicalUnchecked("//foo:parent_platform"))
            .setRemoteExecutionProperties("props")
            .build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.setExecProperties(ImmutableMap.of("elem1", "value1"));

    ExecPropertiesException exception = assertThrows(ExecPropertiesException.class, builder::build);
    assertThat(exception)
        .hasMessageThat()
        .contains(
            "Platform specifies exec_properties but its parent //foo:parent_platform specifies"
                + " remote_execution_properties. Prefer exec_properties over the deprecated"
                + " remote_execution_properties.");
  }

  @Test
  public void flags_empty() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    // Don't add any flags
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.flags()).isEmpty();
  }

  @Test
  public void flags() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.addFlags(ImmutableList.of("--cpu=k8", "--//starlark:flag=other"));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.flags()).containsExactly("--cpu=k8", "--//starlark:flag=other");
  }

  @Test
  public void flags_parentPlatform_keep() throws Exception {
    PlatformInfo parent = PlatformInfo.builder().addFlags(ImmutableList.of("--cpu=k8")).build();
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.addFlags(ImmutableList.of("--//starlark:flag=other"));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.flags())
        .containsExactly("--cpu=k8", "--//starlark:flag=other")
        .inOrder();
  }

  @Test
  public void flags_parentPlatform_inheritance() throws Exception {
    PlatformInfo parent = PlatformInfo.builder().addFlags(ImmutableList.of("--cpu=arm")).build();
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.addFlags(ImmutableList.of("--cpu=k8"));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.flags()).containsExactly("--cpu=arm", "--cpu=k8").inOrder();
  }

  @Test
  public void equalsTester() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:basic"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseCanonicalUnchecked("//constraint:other"));

    ConstraintValueInfo value1 =
        ConstraintValueInfo.create(setting1, Label.parseCanonicalUnchecked("//constraint:value1"));
    ConstraintValueInfo value2 =
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:value2"));
    ConstraintValueInfo value3 =
        ConstraintValueInfo.create(setting2, Label.parseCanonicalUnchecked("//constraint:value3"));

    new EqualsTester()
        .addEqualityGroup(
            // Base case.
            PlatformInfo.builder()
                .setLabel(Label.parseCanonicalUnchecked("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value2)
                .build(),
            PlatformInfo.builder()
                .setLabel(Label.parseCanonicalUnchecked("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value2)
                .build())
        .addEqualityGroup(
            // Different label.
            PlatformInfo.builder()
                .setLabel(Label.parseCanonicalUnchecked("//platform/plat2"))
                .addConstraint(value1)
                .addConstraint(value2)
                .build())
        .addEqualityGroup(
            // Extra constraint.
            PlatformInfo.builder()
                .setLabel(Label.parseCanonicalUnchecked("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value3)
                .build())
        .addEqualityGroup(
            // Missing constraint.
            PlatformInfo.builder()
                .setLabel(Label.parseCanonicalUnchecked("//platform/plat1"))
                .addConstraint(value1)
                .build())
        .addEqualityGroup(
            // Different remote exec properties.
            PlatformInfo.builder()
                .setLabel(Label.parseCanonicalUnchecked("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value2)
                .setRemoteExecutionProperties("foo")
                .build())
        .testEquals();
  }
}
