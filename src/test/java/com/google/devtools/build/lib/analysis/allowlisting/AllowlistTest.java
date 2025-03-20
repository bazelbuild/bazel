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

package com.google.devtools.build.lib.analysis.allowlisting;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the Allowlist methods. */
@RunWith(JUnit4.class)
public final class AllowlistTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    return builder.addRuleDefinition(AllowlistDummyRule.DEFINITION).build();
  }

  @Test
  public void testDirectPackage() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//direct",
            ],
        )
        """);
    scratch.file("direct/BUILD", "rule_with_allowlist(name='x')");
    getConfiguredTarget("//direct:x");
    assertNoEvents();
  }

  @Test
  public void testRecursivePackage() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//recursive/...",
            ],
        )
        """);
    scratch.file("recursive/x/BUILD", "rule_with_allowlist(name='y')");
    getConfiguredTarget("//recursive/x:y");
    assertNoEvents();
  }

  @Test
  public void testAbsentPackage() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//somethingelse/...",
            ],
        )
        """);
    checkError("absent", "x", "Dummy is not available.", "rule_with_allowlist(name='x')");
  }

  @Test
  public void testCatchAll() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//...",
            ],
        )
        """);
    scratch.file("notingroup/BUILD", "rule_with_allowlist(name='x')");
    getConfiguredTarget("//notingroup:x");
    assertNoEvents();
  }

  @Test
  public void testEmptyPackageGroup() throws Exception {
    scratch.file("allowlist/BUILD", "package_group(name='allowlist', packages=[])");
    checkError("x", "x", "Dummy is not available.", "rule_with_allowlist(name='x')");
  }

  @Test
  public void testNonExistentPackageGroup() throws Exception {
    checkError(
        "x",
        "x",
        "every rule of type rule_with_allowlist implicitly depends upon the target"
            + " '//allowlist:allowlist', but this target could not be found because of: no such"
            + " package 'allowlist': BUILD file not found",
        "rule_with_allowlist(name='x')");
  }

  @Test
  public void testIncludes() throws Exception {
    scratch.file(
        "suballowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//x",
            ],
        )
        """);
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            includes = [
                "//suballowlist:allowlist",
            ],
            packages = [
            ],
        )
        """);
    scratch.file(
        "x/BUILD",
        """
        rule_with_allowlist(
            name = "x",
        )
        """);
    getConfiguredTarget("//x:x");
    assertNoEvents();
  }

  @Test
  public void targetInAllowlist_targetAsStringParameter() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//direct",
            ],
        )
        """);
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  target = '//direct:rule_from_allowlist'",
        "  target_in_allowlist ="
            + " ctx.attr._allowlist_test[PackageSpecificationInfo].contains(target)",
        "  if not target_in_allowlist:",
        "    fail('Target should be in the allowlist')",
        "  return []",
        "custom_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    '_allowlist_test': attr.label(",
        "      default = '//allowlist:allowlist',",
        "      cfg = 'exec',",
        "      providers = ['PackageSpecificationInfo']",
        "    ),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        """
        load("//test:rule.bzl", "custom_rule")

        custom_rule(name = "allowlist_rule")
        """);

    getConfiguredTarget("//test:allowlist_rule");

    assertNoEvents();
  }

  @Test
  public void targetInAllowlist_targetAsLabelParameter() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//test",
            ],
        )
        """);
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  target = ctx.label",
        "  target_in_allowlist ="
            + " ctx.attr._allowlist_test[PackageSpecificationInfo].contains(target)",
        "  if not target_in_allowlist:",
        "    fail('Target should be in the allowlist')",
        "  return []",
        "custom_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    '_allowlist_test': attr.label(",
        "      default = '//allowlist:allowlist',",
        "      cfg = 'exec',",
        "      providers = [PackageSpecificationInfo]",
        "    ),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        """
        load("//test:rule.bzl", "custom_rule")

        custom_rule(name = "allowlist_rule")
        """);

    getConfiguredTarget("//test:allowlist_rule");

    assertNoEvents();
  }

  @Test
  public void targetNotInAllowlist() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//direct",
            ],
        )
        """);
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  target = '//non_direct:rule_not_from_allowlist'",
        "  target_in_allowlist ="
            + " ctx.attr._allowlist_test[PackageSpecificationInfo].contains(target)",
        "  if target_in_allowlist:",
        "    fail('Target should not be in the allowlist')",
        "  return []",
        "custom_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    '_allowlist_test': attr.label(",
        "      default = '//allowlist:allowlist',",
        "      cfg = 'exec',",
        "      providers = [PackageSpecificationInfo]",
        "    ),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        """
        load("//test:rule.bzl", "custom_rule")

        custom_rule(name = "allowlist_rule")
        """);

    getConfiguredTarget("//test:allowlist_rule");

    assertNoEvents();
  }

  @Test
  public void targetNotInAllowlist_negativePath() throws Exception {
    scratch.file(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "-//direct",
            ],
        )
        """);
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  target = '//direct:rule_from_allowlist'",
        "  target_in_allowlist ="
            + " ctx.attr._allowlist_test[PackageSpecificationInfo].contains(target)",
        "  if target_in_allowlist:",
        "    fail('Target should not be in the allowlist (negative path)')",
        "  return []",
        "custom_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    '_allowlist_test': attr.label(",
        "      default = '//allowlist:allowlist',",
        "      cfg = 'exec',",
        "      providers = [PackageSpecificationInfo]",
        "    ),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        """
        load("//test:rule.bzl", "custom_rule")

        custom_rule(name = "allowlist_rule")
        """);

    getConfiguredTarget("//test:allowlist_rule");

    assertNoEvents();
  }
}
