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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import net.starlark.java.eval.Starlark;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ConstraintSetting} and {@link ConstraintValue}. */
@RunWith(JUnit4.class)
public class ConstraintTest extends BuildViewTestCase {

  @Before
  public void createConstraints() throws Exception {
    scratch.file(
        "constraint/BUILD",
        """
        constraint_setting(name = "basic")

        constraint_value(
            name = "foo",
            constraint_setting = ":basic",
        )

        constraint_value(
            name = "bar",
            constraint_setting = ":basic",
        )
        """);
  }

  @Test
  public void testConstraint() throws Exception {
    ConfiguredTarget setting = getConfiguredTarget("//constraint:basic");
    assertThat(setting).isNotNull();

    ConstraintSettingInfo constraintSettingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(constraintSettingInfo).isNotNull();
    assertThat(constraintSettingInfo).isNotNull();
    assertThat(constraintSettingInfo.label()).isEqualTo(Label.parseCanonical("//constraint:basic"));
    assertThat(constraintSettingInfo.hasDefaultConstraintValue()).isFalse();
    assertThat(constraintSettingInfo.defaultConstraintValue()).isNull();

    ConfiguredTarget fooValue = getConfiguredTarget("//constraint:foo");
    assertThat(fooValue).isNotNull();

    ConstraintValueInfo fooConstraintValueInfo = PlatformProviderUtils.constraintValue(fooValue);
    assertThat(fooConstraintValueInfo).isNotNull();
    assertThat(fooConstraintValueInfo.constraint().label())
        .isEqualTo(Label.parseCanonical("//constraint:basic"));
    assertThat(fooConstraintValueInfo.label()).isEqualTo(Label.parseCanonical("//constraint:foo"));

    ConfiguredTarget barValue = getConfiguredTarget("//constraint:bar");
    assertThat(barValue).isNotNull();

    ConstraintValueInfo barConstraintValueInfo = PlatformProviderUtils.constraintValue(barValue);
    assertThat(barConstraintValueInfo.constraint().label())
        .isEqualTo(Label.parseCanonical("//constraint:basic"));
    assertThat(barConstraintValueInfo.label()).isEqualTo(Label.parseCanonical("//constraint:bar"));
  }

  @Test
  public void testConstraint_defaultValue() throws Exception {
    scratch.file(
        "constraint_default/BUILD",
        """
        constraint_setting(
            name = "basic",
            default_constraint_value = ":foo",
        )

        constraint_value(
            name = "foo",
            constraint_setting = ":basic",
        )

        constraint_value(
            name = "bar",
            constraint_setting = ":basic",
        )
        """);

    ConfiguredTarget setting = getConfiguredTarget("//constraint_default:basic");
    assertThat(setting).isNotNull();
    ConstraintSettingInfo constraintSettingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(constraintSettingInfo).isNotNull();
    assertThat(constraintSettingInfo.hasDefaultConstraintValue()).isTrue();

    ConfiguredTarget fooValue = getConfiguredTarget("//constraint_default:foo");
    assertThat(fooValue).isNotNull();
    ConstraintValueInfo fooConstraintValueInfo = PlatformProviderUtils.constraintValue(fooValue);
    assertThat(fooConstraintValueInfo).isNotNull();

    assertThat(constraintSettingInfo.defaultConstraintValue()).isEqualTo(fooConstraintValueInfo);
  }

  @Test
  public void testConstraint_defaultValue_differentPackageFails() throws Exception {
    scratch.file(
        "other/BUILD",
        """
        constraint_value(
            name = "other",
            constraint_setting = "//constraint_default:basic",
        )
        """);
    checkError(
        "constraint_default",
        "basic",
        "same package",
        "constraint_setting(name = 'basic',",
        "    default_constraint_value = '//other:other',",
        ")");
  }

  @Test
  public void testConstraint_defaultValue_nonExistentTargetFails() throws Exception {
    checkError(
        "constraint_default",
        "basic",
        "default constraint value '//constraint_default:food' does not exist",
        "constraint_setting(name = 'basic',",
        "    default_constraint_value = ':food',",
        "    )",
        "constraint_value(name = 'foo',",
        "    constraint_setting = ':basic',",
        ")");
  }

  @Test
  public void testConstraint_defaultValue_starlark() throws Exception {
    setBuildLanguageOptions("--experimental_platforms_api=true");
    scratch.file(
        "constraint_default/BUILD",
        """
        constraint_setting(
            name = "basic",
            default_constraint_value = ":foo",
        )

        constraint_value(
            name = "foo",
            constraint_setting = ":basic",
        )
        """);

    scratch.file(
        "verify/verify.bzl",
        """
        result = provider()

        def _impl(ctx):
            constraint_setting = ctx.attr.constraint_setting[platform_common.ConstraintSettingInfo]
            default_value = constraint_setting.default_constraint_value
            has_default_value = constraint_setting.has_default_constraint_value
            return [result(
                default_value = default_value,
                has_default_value = has_default_value,
            )]

        verify = rule(
            implementation = _impl,
            attrs = {
                "constraint_setting": attr.label(
                    providers = [platform_common.ConstraintSettingInfo],
                ),
            },
        )
        """);
    scratch.file(
        "verify/BUILD",
        """
        load(":verify.bzl", "verify")

        verify(
            name = "verify",
            constraint_setting = "//constraint_default:basic",
        )
        """);

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//verify:verify");
    StructImpl info =
        (StructImpl)
            myRuleTarget.get(
                new StarlarkProvider.Key(Label.parseCanonical("//verify:verify.bzl"), "result"));

    @SuppressWarnings("unchecked")
    ConstraintValueInfo defaultConstraintValue =
        (ConstraintValueInfo) info.getValue("default_value");
    assertThat(defaultConstraintValue).isNotNull();
    assertThat(defaultConstraintValue.label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint_default:foo"));
    assertThat(defaultConstraintValue.constraint().label())
        .isEqualTo(Label.parseCanonicalUnchecked("//constraint_default:basic"));

    boolean hasConstraintValue = (boolean) info.getValue("has_default_value");
    assertThat(hasConstraintValue).isTrue();
  }

  @Test
  public void testConstraint_defaultValue_notSet_starlark() throws Exception {
    setBuildLanguageOptions("--experimental_platforms_api=true");
    scratch.file("constraint_default/BUILD", "constraint_setting(name = 'basic')");

    scratch.file(
        "verify/verify.bzl",
        """
        result = provider()

        def _impl(ctx):
            constraint_setting = ctx.attr.constraint_setting[platform_common.ConstraintSettingInfo]
            default_value = constraint_setting.default_constraint_value
            has_default_value = constraint_setting.has_default_constraint_value
            return [result(
                default_value = default_value,
                has_default_value = has_default_value,
            )]

        verify = rule(
            implementation = _impl,
            attrs = {
                "constraint_setting": attr.label(
                    providers = [platform_common.ConstraintSettingInfo],
                ),
            },
        )
        """);
    scratch.file(
        "verify/BUILD",
        """
        load(":verify.bzl", "verify")

        verify(
            name = "verify",
            constraint_setting = "//constraint_default:basic",
        )
        """);

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//verify:verify");
    StructImpl info =
        (StructImpl)
            myRuleTarget.get(
                new StarlarkProvider.Key(Label.parseCanonical("//verify:verify.bzl"), "result"));

    assertThat(info.getValue("default_value")).isEqualTo(Starlark.NONE);

    boolean hasConstraintValue = (boolean) info.getValue("has_default_value");
    assertThat(hasConstraintValue).isFalse();
  }
}
