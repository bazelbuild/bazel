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
import static com.google.devtools.build.lib.testutil.MoreAsserts.expectThrows;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link PlatformInfo}. */
@RunWith(JUnit4.class)
public class PlatformInfoTest extends BuildViewTestCase {

  @Before
  public void createPlatform() throws Exception {
    scratch.file(
        "constraint/BUILD",
        "constraint_setting(name = 'basic')",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        "    )");
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
        expectThrows(PlatformInfo.DuplicateConstraintException.class, () -> builder.build());
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
                .build(),
            PlatformInfo.builder()
                .setLabel(makeLabel("//platform/plat1"))
                .addConstraint(value1)
                .addConstraint(value2)
                .setRemoteExecutionProperties("key=val") // execution properties are ignored.
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
        .testEquals();
  }

  @Test
  public void platformInfoConstructor() throws Exception {
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
        "test/platform/BUILD",
        "load('//test/platform:my_platform.bzl', 'my_platform')",
        "my_platform(name = 'custom',",
        "    constraints = [",
        "       '//constraint:foo',",
        "    ])");

    ConfiguredTarget platform = getConfiguredTarget("//test/platform:custom");
    assertThat(platform).isNotNull();

    PlatformInfo provider = PlatformProviderUtils.platform(platform);
    assertThat(provider).isNotNull();
    assertThat(provider.label()).isEqualTo(makeLabel("//test/platform:custom"));
    assertThat(provider.constraints()).hasSize(1);
    ConstraintSettingInfo constraintSetting =
        ConstraintSettingInfo.create(makeLabel("//constraint:basic"));
    ConstraintValueInfo constraintValue =
        ConstraintValueInfo.create(constraintSetting, makeLabel("//constraint:foo"));
    assertThat(provider.constraints()).containsExactly(constraintValue);
    assertThat(provider.remoteExecutionProperties()).isNull();
  }

  @Test
  public void platformInfoConstructor_error_duplicateConstraints() throws Exception {
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
    checkError(
        "test/platform",
        "custom",
        "Label '//constraint:foo' is duplicated in the 'constraints' attribute of rule 'custom'",
        "load('//test/platform:my_platform.bzl', 'my_platform')",
        "my_platform(name = 'custom',",
        "    constraints = [",
        "       '//constraint:foo',",
        "       '//constraint:foo',",
        "    ])");
  }

  @Test
  public void proxyTemplateVariableInfo() throws Exception {
    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  return struct(",
        "      providers = [ctx.attr._cc_toolchain[platform_common.TemplateVariableInfo]])",
        "crule = rule(_impl, attrs = { '_cc_toolchain': attr.label(default=Label('//a:a')) })");

    scratch.file("a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='a')",
        "crule(name='r')",
        "genrule(name='g', srcs=[], outs=['go'], toolchains=[':r'], cmd='VAR $(CC)')");

    SpawnAction action = (SpawnAction) getGeneratingAction(getConfiguredTarget("//a:g"), "a/go");
    assertThat(action.getArguments().get(2)).containsMatch("VAR .*gcc");
  }

  @Test
  public void makeVariableInfo() throws Exception {
    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  return struct(",
        "      variables = ctx.attr._cc_toolchain[platform_common.TemplateVariableInfo].variables)",
        "crule = rule(_impl, attrs = { '_cc_toolchain': attr.label(default=Label('//a:a')) })");

    scratch.file("a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='a')",
        "crule(name='r')");
    ConfiguredTarget ct = getConfiguredTarget("//a:r");

    @SuppressWarnings("unchecked")
    Map<String, String> makeVariables = (Map<String, String>) ct.get("variables");
    assertThat(makeVariables).containsKey("CC_FLAGS");
  }
}
