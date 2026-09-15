// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.stream;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import net.starlark.java.annot.StarlarkMethod;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests Starlark API for Java rules. */
@RunWith(TestParameterInjector.class)
public class JavaStarlarkApiTest extends BuildViewTestCase {

  @Test // not to be Starlarkified: tests native functionality
  @TestParameters({
    "{module: java_config, api: use_ijars}",
    "{module: java_config, api: disallow_java_import_exports}",
    "{module: java_config, api: enforce_explicit_java_test_deps}",
    "{module: java_config, api: use_header_compilation}",
    "{module: java_config, api: generate_java_deps}",
    "{module: java_config, api: reduce_java_classpath}",
  })
  public void testNoArgsPrivateAPIsAreIndeedPrivate(String module, String api) throws Exception {
    scratch.file(
        "foo/custom_rule.bzl",
        "def _impl(ctx):",
        "  java_config = ctx.fragments.java",
        "  " + module + "." + api + "()",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java']",
        ")");
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(name = "custom")
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:custom");

    assertContainsEvent(
        "Error in " + api + ": file '//foo:custom_rule.bzl' cannot use private API");
  }

  @Test // not to be Starlarkified: tests native functionality
  @TestParameters({
    "{api: create_header_compilation_action}",
    "{api: create_compilation_action}",
    "{api: target_kind}",
    "{api: collect_native_deps_dirs}",
    "{api: get_runtime_classpath_for_archive}",
    "{api: check_provider_instances}",
    "{api: _google_legacy_api_enabled}",
    "{api: _check_java_toolchain_is_declared_on_rule}",
    "{api: tokenize_javacopts}",
  })
  public void testJavaCommonPrivateApis_areNotVisibleToPublicStarlark(String api) throws Exception {
    // validate that this api is present on the module, so this test fails when the API is deleted
    var unused =
        stream(JavaCommonApi.class.getDeclaredMethods())
            .filter(method -> method.isAnnotationPresent(StarlarkMethod.class))
            .filter(method -> method.getAnnotation(StarlarkMethod.class).name().equals(api))
            .findAny()
            .orElseThrow(
                () -> new IllegalArgumentException("API not declared on java_common: " + api));
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_common." + api + "()",
        "  return []",
        "custom_rule = rule(implementation = _impl)");
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "custom_rule")

        custom_rule(name = "custom")
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:custom");

    assertContainsEvent("no field or method '" + api + "'");
  }

  @Test // not to be Starlarkified: tests native functionality
  public void testProviderValidationPrintsProviderName() throws Exception {
    scratch.file(
        "foo/rule.bzl",
        """
        load("@rules_java//java/common:java_info.bzl", "JavaInfo")
        load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

        def _impl(ctx):
            cc_info = ctx.attr.dep[CcInfo]
            JavaInfo(output_jar = None, compile_jar = None, deps = [cc_info])
            return []

        myrule = rule(
            implementation = _impl,
            attrs = {"dep": attr.label()},
            fragments = [],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load(":rule.bzl", "myrule")

        cc_library(name = "cc_lib")

        myrule(
            name = "myrule",
            dep = ":cc_lib",
        )
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:myrule");

    assertContainsEvent("got element of type CcInfo, want JavaInfo");
  }

  @Test // not to be Starlarkified: tests native functionality
  public void testNativeJavaInfoPrintableType_isJavaInfo() {
    String type = JavaStarlarkCommon.printableType(JavaInfo.EMPTY_JAVA_INFO_FOR_TESTING);

    assertThat(type).isEqualTo("JavaInfo");
  }
}
