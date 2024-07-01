// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.TestConstants.TOOLS_REPOSITORY;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.java.JavaTestUtil;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/** Tests of bazel java rules. */
@RunWith(Parameterized.class)
public final class JavaConfiguredTargetsTest extends BuildViewTestCase {

  // we use a custom package to avoid overwriting the default
  private static final String PLATFORMS_PACKAGE_PATH = "my/java/platforms";

  @Parameter(0)
  public String targetPlatform;

  @Parameter(1)
  public String targetOs;

  @Parameter(2)
  public String targetCpu;

  @Parameters
  public static List<Object[]> platformParameters() {
    return Arrays.asList(
        new Object[][] {
          {"linux-default", "linux", "x86_64"},
          {"windows-default", "windows", "x86_64"},
          {"darwin-x86", "macos", "x86_64"},
        });
  }

  @Override
  protected void useConfiguration(String... args) throws Exception {
    JavaTestUtil.setupPlatform(
        getAnalysisMock(),
        mockToolsConfig,
        scratch,
        PLATFORMS_PACKAGE_PATH,
        targetPlatform,
        targetOs,
        targetCpu);
    super.useConfiguration(
        ObjectArrays.concat(
            args, "--platforms=//" + PLATFORMS_PACKAGE_PATH + ":" + targetPlatform));
  }

  @Test
  public void testResourceStripPrefix() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        java_binary(
            name = "bin",
            srcs = ["Foo.java"],
            main_class = "Foo",
            resource_strip_prefix = "a/path/to/strip",
            resources = ["path/to/strip/bar.props"],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//a:bin");

    assertThat(target).isNotNull();
    String resourceJarArgs =
        Joiner.on(" ").join(getGeneratingSpawnActionArgs(getBinArtifact("bin.jar", target)));
    assertThat(resourceJarArgs).contains("--resources a/path/to/strip/bar.props:bar.props");
  }

  @Test
  public void javaTestJavaRuntimeAttribute() throws Exception {
    scratch.file(
        "a/BUILD",
        "java_runtime(",
        "    name = 'jvm17',",
        "    java = 'java17_home/bin/java',",
        "    version = 17,",
        ")",
        "java_runtime(",
        "    name = 'jvm21',",
        "    java = 'java21_home/bin/java',",
        "    version = 21,",
        ")",
        "toolchain(",
        "    name = 'java_runtime_toolchain17',",
        "    toolchain = ':jvm17',",
        "    toolchain_type = '" + TOOLS_REPOSITORY + "//tools/jdk:runtime_toolchain_type',",
        ")",
        "java_test(",
        "    name = 'test',",
        "    srcs = ['FooTest.java'],",
        "    test_class = 'FooTest',",
        "    # in the absence of java_runtime, should use current toolchain",
        ")",
        "java_test(",
        "    name = 'test_jvm21',",
        "    srcs = ['FooTest.java'],",
        "    test_class = 'FooTest',",
        "    java_runtime = ':jvm21',",
        ")");
    useConfiguration("--extra_toolchains=//a:java_runtime_toolchain17");

    var ct = getConfiguredTarget("//a:test");
    var ct21 = getConfiguredTarget("//a:test_jvm21");

    String javaBinPath =
        JavaTestUtil.getJavaBinForJavaBinaryExecutable(
            getRuleContext(ct), getGeneratingAction(getExecutable(ct)));
    assertThat(javaBinPath).contains("java17_home");

    String javaBinPath21 =
        JavaTestUtil.getJavaBinForJavaBinaryExecutable(
            getRuleContext(ct21), getGeneratingAction(getExecutable(ct21)));
    assertThat(javaBinPath21).contains("java21_home");
  }

  @Test
  public void javaTestSetsSecurityManagerPropertyOnVersion17() throws Exception {
    scratch.file(
        "a/BUILD",
        "java_runtime(",
        "    name = 'jvm',",
        "    java = 'java_home/bin/java',",
        "    version = 17,",
        ")",
        "toolchain(",
        "    name = 'java_runtime_toolchain',",
        "    toolchain = ':jvm',",
        "    toolchain_type = '" + TOOLS_REPOSITORY + "//tools/jdk:runtime_toolchain_type',",
        ")",
        "java_test(",
        "    name = 'test',",
        "    srcs = ['FooTest.java'],",
        "    test_class = 'FooTest',",
        ")");
    useConfiguration("--extra_toolchains=//a:java_runtime_toolchain");
    var ct = getConfiguredTarget("//a:test");
    String jvmFlags =
        JavaTestUtil.getJvmFlagsForJavaBinaryExecutable(
            getRuleContext(ct), getGeneratingAction(getExecutable(ct)));
    assertThat(jvmFlags).contains("-Djava.security.manager=allow");
  }

  // regression test for https://github.com/bazelbuild/bazel/issues/20378
  @Test
  public void javaTestInvalidTestClassAtRootPackage() throws Exception {
    scratch.file("BUILD", "java_test(name = 'some_test', srcs = ['SomeTest.java'])");
    invalidatePackages();

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//:some_test"));

    assertThat(error).hasMessageThat().contains("cannot determine test class");
  }
}
