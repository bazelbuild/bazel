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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
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
  public void templateVariableInfo() throws Exception {
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

  @Test
  public void templateVariableInfoConstructor() throws Exception {
    scratch.file(
        "a/rule.bzl",
        "def _consumer_impl(ctx):",
        "  return struct(",
        "      var = ctx.attr.supplier[platform_common.TemplateVariableInfo]",
        "          .variables[ctx.attr.var])",
        "def _supplier_impl(ctx):",
        "  return [platform_common.TemplateVariableInfo({ctx.attr.var: ctx.attr.value})]",
        "consumer = rule(_consumer_impl,",
        "    attrs = { 'var': attr.string(), 'supplier': attr.label() })",
        "supplier = rule(_supplier_impl,",
        "    attrs = { 'var': attr.string(), 'value': attr.string() })");

    scratch.file("a/BUILD",
        "load(':rule.bzl', 'consumer', 'supplier')",
        "consumer(name='consumer', supplier=':supplier', var='cherry')",
        "supplier(name='supplier', var='cherry', value='ontop')");

    ConfiguredTarget consumer = getConfiguredTarget("//a:consumer");
    @SuppressWarnings("unchecked") String value = (String) consumer.get("var");
    assertThat(value).isEqualTo("ontop");

    ConfiguredTarget supplier = getConfiguredTarget("//a:supplier");
    assertThat(supplier.get(TemplateVariableInfo.PROVIDER).getVariables())
        .containsExactly("cherry", "ontop");
  }
}
