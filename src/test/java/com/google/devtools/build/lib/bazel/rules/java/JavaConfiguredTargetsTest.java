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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.java.JavaTestUtil;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of bazel java rules. */
@RunWith(JUnit4.class)
public final class JavaConfiguredTargetsTest extends BuildViewTestCase {

  @Test
  public void testResourceStripPrefix() throws Exception {
    scratch.file(
        "a/BUILD",
        "java_binary(",
        "   name = 'bin',",
        "   srcs = ['Foo.java'],",
        "   resources = ['path/to/strip/bar.props'],",
        "   main_class = 'Foo',",
        "   resource_strip_prefix = 'a/path/to/strip'",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//a:bin");

    assertThat(target).isNotNull();
    String resourceJarArgs =
        Joiner.on(" ").join(getGeneratingSpawnActionArgs(getBinArtifact("bin.jar", target)));
    assertThat(resourceJarArgs).contains("--resources a/path/to/strip/bar.props:bar.props");
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
        JavaTestUtil.getJvmFlagsForJavaBinaryExecutable(getGeneratingAction(getExecutable(ct)));
    assertThat(jvmFlags).contains("-Djava.security.manager=allow");
  }

  // regression test for https://github.com/bazelbuild/bazel/issues/20378
  @Test
  public void javaTestInvalidTestClassAtRootPackage() throws Exception {
    scratch.file("BUILD", "java_test(name = 'some_test', srcs = ['SomeTest.java'])");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//:some_test"));

    assertThat(error).hasMessageThat().contains("cannot determine test class");
  }
}
