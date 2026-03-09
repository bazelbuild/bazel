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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import com.google.testing.junit.testparameterinjector.TestParameters.TestParametersValues;
import com.google.testing.junit.testparameterinjector.TestParametersValuesProvider;
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests Starlark API for Java rules. */
@RunWith(TestParameterInjector.class)
public class JavaStarlarkApiTest extends BuildViewTestCase {

  private static final String PLATFORMS_PACKAGE_PATH = "my/java/platforms";

  private final String targetPlatform;
  private final String targetOs;
  private final String targetCpu;

  @TestParameters(valuesProvider = PlatformsParametersProvider.class)
  public JavaStarlarkApiTest(String platform, String os, String cpu) {
    this.targetPlatform = platform;
    this.targetOs = os;
    this.targetCpu = cpu;
  }

  public void setupTargetPlatform() throws Exception {
    JavaTestUtil.setupPlatform(
        getAnalysisMock(),
        mockToolsConfig,
        scratch,
        PLATFORMS_PACKAGE_PATH,
        targetPlatform,
        targetOs,
        targetCpu);
  }

  @Before
  public void setupMyInfo() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");
  }

  @Override
  protected void useConfiguration(String... args) throws Exception {
    // Must actually define the platform before using it in a flag.
    setupTargetPlatform();
    super.useConfiguration(
        ObjectArrays.concat(
            args,
            new String[] {
              "--platforms=//" + PLATFORMS_PACKAGE_PATH + ":" + targetPlatform,
              "--extra_execution_platforms=//" + PLATFORMS_PACKAGE_PATH + ":" + targetPlatform
            },
            String.class));
  }

  @Test
  public void testPackSourcesWithExternalResourceArtifact() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  out = ctx.actions.declare_file('output.jar')",
        "  java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo]",
        "  java_common.pack_sources(",
        "    ctx.actions,",
        "    java_toolchain = java_toolchain,",
        "    output_source_jar = out,",
        "    sources = ctx.files.srcs,",
        "  )",
        "  return [DefaultInfo(files = depset([out]))]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files = True),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java']",
        ")");
    scratch.file("my_other_repo/MODULE.bazel", "module(name='other_repo')");
    scratch.file("my_other_repo/external-file.txt");
    scratch.file("my_other_repo/BUILD", "exports_files(['external-file.txt'])");
    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'other_repo')",
        "local_path_override(module_name = 'other_repo', path = 'my_other_repo')");
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(
            name = "custom",
            srcs = [
                "internal-file.txt",
                "@other_repo//:external-file.txt",
            ],
        )
        """);
    invalidatePackages();

    List<String> arguments =
        ((SpawnAction) getGeneratingAction(getConfiguredTarget("//foo:custom"), "foo/output.jar"))
            .getArguments();

    assertThat(arguments)
        .containsAtLeast(
            "--resources",
            "foo/internal-file.txt:foo/internal-file.txt",
            "external/other_repo+/external-file.txt:external-file.txt")
        .inOrder();
  }









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
    setBuildLanguageOptions("--experimental_builtins_injection_override=+java_import");
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "def _impl(ctx):",
        "  java_config = ctx.fragments.java",
        "  java_toolchain = ctx.toolchains['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'].java",
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

  private static class PlatformsParametersProvider extends TestParametersValuesProvider {

    @Override
    public List<TestParametersValues> provideValues(Context context) {
      ImmutableList.Builder<TestParametersValues> parameters = ImmutableList.builder();
      parameters
          .add(
              TestParametersValues.builder()
                  .name("linux")
                  .addParameter("platform", "linux-x86_64")
                  .addParameter("os", "linux")
                  .addParameter("cpu", "x86_64")
                  .build())
          .add(
              TestParametersValues.builder()
                  .name("darwin")
                  .addParameter("platform", "darwin-x86_64")
                  .addParameter("os", "macos")
                  .addParameter("cpu", "x86_64")
                  .build());
      // building for windows is only supported in Bazel
      if (AnalysisMock.get().isThisBazel()) {
        parameters.add(
            TestParametersValues.builder()
                .name("windows")
                .addParameter("platform", "windows-x86_64")
                .addParameter("os", "windows")
                .addParameter("cpu", "x86_64")
                .build());
      }
      return parameters.build();
    }
  }
}
