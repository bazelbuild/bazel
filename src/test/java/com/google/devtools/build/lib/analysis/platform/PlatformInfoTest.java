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

import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
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
    assertThat(platformInfo.execProperties()).isEmpty();
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

  @Test
  public void platformInfo_execProperties_empty() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setExecProperties(ImmutableMap.of());
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.execProperties()).isNotNull();
    assertThat(platformInfo.execProperties()).isEmpty();
  }

  @Test
  public void platformInfo_execProperties_one() throws Exception {
    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setExecProperties(ImmutableMap.of("elem1", "value1"));
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.execProperties()).isNotNull();
    assertThat(platformInfo.execProperties()).containsExactly("elem1", "value1");
  }

  @Test
  public void platformInfo_parentPlatform_keepExecProperties() throws Exception {
    PlatformInfo parent =
        PlatformInfo.builder().setExecProperties(ImmutableMap.of("parent", "properties")).build();

    PlatformInfo.Builder builder = PlatformInfo.builder();
    builder.setParent(parent);
    PlatformInfo platformInfo = builder.build();

    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.execProperties()).containsExactly("parent", "properties");
  }

  @Test
  public void platformInfo_parentPlatform_inheritanceRules() throws Exception {
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
  public void platformInfo_constructor() throws Exception {
    scratch.file(
        "test/platform/my_platform.bzl",
        "def _impl(ctx):",
        "  constraints = [val[platform_common.ConstraintValueInfo] "
            + "for val in ctx.attr.constraints]",
        "  platform = platform_common.PlatformInfo(",
        "      label = ctx.label, constraint_values = constraints)",
        "  return [platform]",
        "my_platform = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'constraints': attr.label_list(providers = [platform_common.ConstraintValueInfo])",
        "  }",
        ")");
    scratch.file(
        "test/constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        ")");
    scratch.file(
        "test/platform/BUILD",
        "load('//test/platform:my_platform.bzl', 'my_platform')",
        "my_platform(name = 'custom',",
        "    constraints = [",
        "       '//test/constraint:foo',",
        "    ],",
        ")");

    setSkylarkSemanticsOptions("--experimental_platforms_api");
    ConfiguredTarget platform = getConfiguredTarget("//test/platform:custom");
    assertThat(platform).isNotNull();

    PlatformInfo provider = PlatformProviderUtils.platform(platform);
    assertThat(provider).isNotNull();
    assertThat(provider.label()).isEqualTo(makeLabel("//test/platform:custom"));
    ConstraintSettingInfo constraintSetting =
        ConstraintSettingInfo.create(makeLabel("//test/constraint:basic"));
    ConstraintValueInfo constraintValue =
        ConstraintValueInfo.create(constraintSetting, makeLabel("//test/constraint:foo"));
    assertThat(provider.constraints().get(constraintSetting)).isEqualTo(constraintValue);
  }

  @Test
  public void platformInfo_constructor_parent() throws Exception {
    scratch.file(
        "test/platform/my_platform.bzl",
        "def _impl(ctx):",
        "  constraints = [val[platform_common.ConstraintValueInfo] "
            + "for val in ctx.attr.constraints]",
        "  platform = platform_common.PlatformInfo(",
        "      label = ctx.label, constraint_values = constraints)",
        "  return [platform]",
        "my_platform = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'constraints': attr.label_list(providers = [platform_common.ConstraintValueInfo])",
        "  }",
        ")");
    scratch.file(
        "test/constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_setting(name = 'complex')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        ")",
        "constraint_value(name = 'bar',",
        "    constraint_setting = ':basic',",
        ")",
        "constraint_value(name = 'baz',",
        "    constraint_setting = ':complex',",
        ")");
    scratch.file(
        "test/platform/BUILD",
        "load('//test/platform:my_platform.bzl', 'my_platform')",
        "platform(",
        "    name='parent',",
        "    constraint_values = [",
        "       '//test/constraint:foo',",
        "    ],",
        ")",
        "my_platform(name = 'custom',",
        "    constraints = [",
        "       '//test/constraint:bar',",
        "       '//test/constraint:baz',",
        "    ],",
        ")");

    setSkylarkSemanticsOptions("--experimental_platforms_api");
    ConfiguredTarget platform = getConfiguredTarget("//test/platform:custom");
    assertThat(platform).isNotNull();

    PlatformInfo provider = PlatformProviderUtils.platform(platform);
    assertThat(provider).isNotNull();
    assertThat(provider.label()).isEqualTo(makeLabel("//test/platform:custom"));

    // Check that overrides work.
    ConstraintSettingInfo constraintSetting =
        ConstraintSettingInfo.create(makeLabel("//test/constraint:basic"));
    ConstraintValueInfo constraintValue =
        ConstraintValueInfo.create(constraintSetting, makeLabel("//test/constraint:bar"));
    assertThat(provider.constraints().get(constraintSetting)).isEqualTo(constraintValue);

    // Check that inheritance works.
    ConstraintSettingInfo constraintSetting2 =
        ConstraintSettingInfo.create(makeLabel("//test/constraint:complex"));
    ConstraintValueInfo constraintValue2 =
        ConstraintValueInfo.create(constraintSetting2, makeLabel("//test/constraint:baz"));
    assertThat(provider.constraints().get(constraintSetting2)).isEqualTo(constraintValue2);
  }

  @Test
  public void platformInfo_constructor_error_duplicateConstraints() throws Exception {
    scratch.file(
        "test/platform/my_platform.bzl",
        "def _impl(ctx):",
        "  platform = platform_common.PlatformInfo()",
        "  return [platform]",
        "my_platform = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'constraints': attr.label_list(providers = [platform_common.ConstraintValueInfo])",
        "  }",
        ")");
    scratch.file(
        "test/constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        ")");
    setSkylarkSemanticsOptions("--experimental_platforms_api");
    checkError(
        "test/platform",
        "custom",
        "Label '//test/constraint:foo' is duplicated in the 'constraints' attribute of rule"
            + " 'custom'",
        "load('//test/platform:my_platform.bzl', 'my_platform')",
        "my_platform(name = 'custom',",
        "    constraints = [",
        "       '//test/constraint:foo',",
        "       '//test/constraint:foo',",
        "    ],",
        ")");
  }
}
