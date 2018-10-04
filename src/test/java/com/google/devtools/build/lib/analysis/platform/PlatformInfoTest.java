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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link PlatformInfo}. */
@RunWith(JUnit4.class)
public class PlatformInfoTest extends BuildViewTestCase {

  @Test
  public void platformInfo() throws Exception {
    ConstraintSettingInfo setting1 = ConstraintSettingInfo.create(makeLabel("//constraint:s1"));
    ConstraintSettingInfo setting2 = ConstraintSettingInfo.create(makeLabel("//constraint:s2"));

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.addConstraint(ConstraintValueInfo.create(setting1, makeLabel("//constraint:v1")));
    builder.addConstraint(ConstraintValueInfo.create(setting2, makeLabel("//constraint:v2")));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.constraints().has(setting1)).isTrue();
    assertThat(platformInfo.constraints().get(setting1).label())
        .isEqualTo(makeLabel("//constraint:v1"));
    assertThat(platformInfo.constraints().has(setting2)).isTrue();
    assertThat(platformInfo.constraints().get(setting2).label())
        .isEqualTo(makeLabel("//constraint:v2"));
  }

  @Test
  public void platformInfo_remoteExecutionProperties() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setRemoteExecutionProperties("properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("properties");
  }

  @Test
  public void platformInfo_parentPlatform_noOverlaps() throws Exception {
    ConstraintSettingInfo setting1 = ConstraintSettingInfo.create(makeLabel("//constraint:s1"));
    ConstraintSettingInfo setting2 = ConstraintSettingInfo.create(makeLabel("//constraint:s2"));
    ConstraintSettingInfo setting3 = ConstraintSettingInfo.create(makeLabel("//constraint:s3"));

    PlatformInfo parent =
        PlatformInfo.builder()
            .addConstraint(ConstraintValueInfo.create(setting1, makeLabel("//constraint:v1")))
            .build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.addConstraint(ConstraintValueInfo.create(setting2, makeLabel("//constraint:v2")));
    builder.addConstraint(ConstraintValueInfo.create(setting3, makeLabel("//constraint:v3")));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.constraints().has(setting1)).isTrue();
    assertThat(platformInfo.constraints().get(setting1).label())
        .isEqualTo(makeLabel("//constraint:v1"));
    assertThat(platformInfo.constraints().has(setting2)).isTrue();
    assertThat(platformInfo.constraints().get(setting2).label())
        .isEqualTo(makeLabel("//constraint:v2"));
    assertThat(platformInfo.constraints().has(setting3)).isTrue();
    assertThat(platformInfo.constraints().get(setting3).label())
        .isEqualTo(makeLabel("//constraint:v3"));
  }

  @Test
  public void platformInfo_parentPlatform_overlaps() throws Exception {
    ConstraintSettingInfo setting1 = ConstraintSettingInfo.create(makeLabel("//constraint:s1"));
    ConstraintSettingInfo setting2 = ConstraintSettingInfo.create(makeLabel("//constraint:s2"));
    ConstraintSettingInfo setting3 = ConstraintSettingInfo.create(makeLabel("//constraint:s3"));

    PlatformInfo parent =
        PlatformInfo.builder()
            .addConstraint(ConstraintValueInfo.create(setting1, makeLabel("//constraint:v1")))
            .build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.addConstraint(ConstraintValueInfo.create(setting1, makeLabel("//constraint:v1a")));
    builder.addConstraint(ConstraintValueInfo.create(setting2, makeLabel("//constraint:v2")));
    builder.addConstraint(ConstraintValueInfo.create(setting3, makeLabel("//constraint:v3")));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.constraints().get(setting1).label())
        .isEqualTo(makeLabel("//constraint:v1a"));
    assertThat(platformInfo.constraints().get(setting2).label())
        .isEqualTo(makeLabel("//constraint:v2"));
    assertThat(platformInfo.constraints().get(setting3).label())
        .isEqualTo(makeLabel("//constraint:v3"));
  }

  @Test
  public void platformInfo_parentPlatform_keepRemoteExecutionProperties() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder().setRemoteExecutionProperties("parent properties").build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("parent properties");
  }

  @Test
  public void platformInfo_parentPlatform_overrideRemoteExecutionProperties() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder().setRemoteExecutionProperties("parent properties").build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.setRemoteExecutionProperties("child properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child properties");
  }

  @Test
  public void platformInfo_parentPlatform_mergeRemoteExecutionProperties() throws Exception {
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
  public void platformInfo_parentPlatform_mergeRemoteExecutionProperties_noParent()
      throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setRemoteExecutionProperties("child {PARENT_REMOTE_EXECUTION_PROPERTIES} properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child  properties");
  }

  @Test
  public void platformInfo_parentPlatform_mergeRemoteExecutionProperties_parentNotSet()
      throws Exception {
    PlatformInfo parent = PlatformInfo.builder().build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    builder.setRemoteExecutionProperties("child {PARENT_REMOTE_EXECUTION_PROPERTIES} properties");
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child  properties");
  }

  @Test
  public void platformInfo_overlappingConstraintsError() throws Exception {
    ConstraintSettingInfo setting1 = ConstraintSettingInfo.create(makeLabel("//constraint:basic"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(makeLabel("//constraint:complex"));
    ConstraintSettingInfo setting3 = ConstraintSettingInfo.create(makeLabel("//constraint:single"));

    PlatformInfo.Builder builder = PlatformInfo.builder();

    builder.addConstraint(ConstraintValueInfo.create(setting1, makeLabel("//constraint:value1")));
    builder.addConstraint(ConstraintValueInfo.create(setting1, makeLabel("//constraint:value2")));

    builder.addConstraint(ConstraintValueInfo.create(setting2, makeLabel("//constraint:value3")));
    builder.addConstraint(ConstraintValueInfo.create(setting2, makeLabel("//constraint:value4")));
    builder.addConstraint(ConstraintValueInfo.create(setting2, makeLabel("//constraint:value5")));

    builder.addConstraint(ConstraintValueInfo.create(setting3, makeLabel("//constraint:value6")));

    PlatformInfo.DuplicateConstraintException exception =
        assertThrows(PlatformInfo.DuplicateConstraintException.class, () -> builder.build());
    assertThat(exception)
        .hasMessageThat()
        .contains(
            "Duplicate constraint_values detected: "
                + "constraint_setting //constraint:basic has "
                + "[//constraint:value1, //constraint:value2], "
                + "constraint_setting //constraint:complex has "
                + "[//constraint:value3, //constraint:value4, //constraint:value5]");
  }

  @Test
  public void platformInfo_equalsTester() throws Exception {
    ConstraintSettingInfo setting1 = ConstraintSettingInfo.create(makeLabel("//constraint:basic"));
    ConstraintSettingInfo setting2 = ConstraintSettingInfo.create(makeLabel("//constraint:other"));

    ConstraintValueInfo value1 =
        ConstraintValueInfo.create(setting1, makeLabel("//constraint:value1"));
    ConstraintValueInfo value2 =
        ConstraintValueInfo.create(setting2, makeLabel("//constraint:value2"));
    ConstraintValueInfo value3 =
        ConstraintValueInfo.create(setting2, makeLabel("//constraint:value3"));

    new EqualsTester()
        .addEqualityGroup(
            // Base case.
            PlatformInfo.builder()
                .setLabel(makeLabel("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value2)
                .build(),
            PlatformInfo.builder()
                .setLabel(makeLabel("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value2)
                .build())
        .addEqualityGroup(
            // Different label.
            PlatformInfo.builder()
                .setLabel(makeLabel("//platform/plat2"))
                .addConstraint(value1)
                .addConstraint(value2)
                .build())
        .addEqualityGroup(
            // Extra constraint.
            PlatformInfo.builder()
                .setLabel(makeLabel("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value3)
                .build())
        .addEqualityGroup(
            // Missing constraint.
            PlatformInfo.builder()
                .setLabel(makeLabel("//platform/plat1"))
                .addConstraint(value1)
                .build())
        .addEqualityGroup(
            // Different remote exec properties.
            PlatformInfo.builder()
                .setLabel(makeLabel("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value2)
                .setRemoteExecutionProperties("foo")
                .build())
        .testEquals();
  }
}
