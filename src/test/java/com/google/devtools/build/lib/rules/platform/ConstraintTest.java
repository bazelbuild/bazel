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
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

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
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//verify:verify.bzl")), "result"));

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
  public void testConstraint_refinesConstraintValue() throws Exception {
    scratch.file(
        "libc/BUILD",
        """
        constraint_setting(name = "libc")

        constraint_value(
            name = "glibc",
            constraint_setting = ":libc",
        )
        """);
    scratch.file(
        "libc/glibc/BUILD",
        """
        constraint_setting(
            name = "version",
            refines_constraint_value = "//libc:glibc",
        )

        constraint_value(
            name = "2.42",
            constraint_setting = ":version",
        )
        """);

    ConfiguredTarget setting = getConfiguredTarget("//libc/glibc:version");
    assertThat(setting).isNotNull();
    ConstraintSettingInfo settingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(settingInfo).isNotNull();
    assertThat(settingInfo.refinedConstraintValue().label())
        .isEqualTo(Label.parseCanonical("//libc:glibc"));
    assertThat(settingInfo.refinementChain()).containsExactly(Label.parseCanonical("//libc:glibc"));
  }

  @Test
  public void testConstraint_refinesConstraintValue_alias() throws Exception {
    scratch.file(
        "libc/BUILD",
        """
        constraint_setting(name = "libc")

        constraint_value(
            name = "glibc",
            constraint_setting = ":libc",
        )

        alias(
            name = "glibc_alias",
            actual = ":glibc",
        )
        """);
    scratch.file(
        "libc/glibc/BUILD",
        """
        constraint_setting(
            name = "version",
            refines_constraint_value = "//libc:glibc_alias",
        )

        constraint_value(
            name = "2.42",
            constraint_setting = ":version",
        )
        """);

    ConfiguredTarget setting = getConfiguredTarget("//libc/glibc:version");
    ConstraintSettingInfo settingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(settingInfo).isNotNull();
    // The alias is resolved to the underlying constraint_value.
    assertThat(settingInfo.refinedConstraintValue().label())
        .isEqualTo(Label.parseCanonical("//libc:glibc"));
    assertThat(settingInfo.refinementChain()).containsExactly(Label.parseCanonical("//libc:glibc"));
  }

  @Test
  public void testConstraint_refinesConstraintValue_chain() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        constraint_setting(name = "a")

        constraint_value(
            name = "a1",
            constraint_setting = ":a",
        )
        """);
    scratch.file(
        "b/BUILD",
        """
        constraint_setting(
            name = "b",
            refines_constraint_value = "//a:a1",
        )

        constraint_value(
            name = "b1",
            constraint_setting = ":b",
        )
        """);
    scratch.file(
        "c/BUILD",
        """
        constraint_setting(
            name = "c",
            refines_constraint_value = "//b:b1",
        )

        constraint_value(
            name = "c1",
            constraint_setting = ":c",
        )
        """);

    ConfiguredTarget setting = getConfiguredTarget("//c:c");
    ConstraintSettingInfo settingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(settingInfo).isNotNull();
    assertThat(settingInfo.refinementChain())
        .containsExactly(Label.parseCanonical("//b:b1"), Label.parseCanonical("//a:a1"))
        .inOrder();
  }

  @Test
  public void testConstraint_refinesConstraintValue_withDefault() throws Exception {
    scratch.file(
        "libc/BUILD",
        """
        constraint_setting(name = "libc")

        constraint_value(
            name = "glibc",
            constraint_setting = ":libc",
        )
        """);
    scratch.file(
        "libc/glibc/BUILD",
        """
        constraint_setting(
            name = "version",
            default_constraint_value = ":unknown",
            refines_constraint_value = "//libc:glibc",
        )

        constraint_value(
            name = "unknown",
            constraint_setting = ":version",
        )

        constraint_value(
            name = "2.42",
            constraint_setting = ":version",
        )
        """);

    ConfiguredTarget setting = getConfiguredTarget("//libc/glibc:version");
    ConstraintSettingInfo settingInfo = PlatformProviderUtils.constraintSetting(setting);
    assertThat(settingInfo).isNotNull();
    assertThat(settingInfo.hasDefaultConstraintValue()).isTrue();
    assertThat(settingInfo.refinedConstraintValue().label())
        .isEqualTo(Label.parseCanonical("//libc:glibc"));
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
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//verify:verify.bzl")), "result"));

    assertThat(info.getValue("default_value")).isEqualTo(Starlark.NONE);

    boolean hasConstraintValue = (boolean) info.getValue("has_default_value");
    assertThat(hasConstraintValue).isFalse();
  }
}
