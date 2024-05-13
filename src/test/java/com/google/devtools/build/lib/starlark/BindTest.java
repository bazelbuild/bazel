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
package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.function.Function;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for using bind() with Starlark rules. */
@RunWith(JUnit4.class)
public class BindTest extends BuildViewTestCase {

  @Before
  public final void createFiles() throws Exception {
    analysisMock.javaSupport().setupRulesJava(mockToolsConfig, Function.identity());
    setupStarlarkRules(scratch);
    scratch.file(
        "test/BUILD",
        """
        load("//rules:java_rules_skylark.bzl", "java_library")

        java_library(
            name = "giraffe",
            srcs = ["Giraffe.java"],
        )

        java_library(
            name = "safari",
            srcs = ["Safari.java"],
            deps = ["//external:long-horse"],
        )
        """);

    // We need to overwrite the Jdk BUILD file because the Starlark rules also depend on having a
    // jar target here, which the built-in rules don't need, and which therefore isn't part of the
    // mock Java setup.
    scratch.overwriteFile(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/jdk/BUILD",
        "load(':java_toolchain_alias.bzl', 'java_runtime_alias')",
        "package(default_visibility = ['//visibility:public'])",
        "java_runtime_alias(name = 'current_java_runtime')",
        "toolchain_type(name = 'runtime_toolchain_type')",
        "toolchain(",
        "   name = 'java_runtime_toolchain',",
        "   toolchain_type = ':runtime_toolchain_type',",
        "   toolchain = ':jdk',",
        ")",
        "filegroup(name = 'java', srcs = ['bin/java'])",
        "filegroup(name = 'jar', srcs = ['bin/jar'])",
        "filegroup(name = 'javac', srcs = ['bin/javac'])",
        "alias(name='host_jdk', actual=':jdk')",
        "java_runtime(name = 'jdk', srcs = ['k8/empty', 'k8/empty2'], java_home = 'k8')",
        "filegroup(name='toolchain', srcs=[])");

    scratch.appendFile(
        "WORKSPACE",
        "register_toolchains('" + TestConstants.TOOLS_REPOSITORY + "//tools/jdk:all')",
        "bind(",
        "    name = 'long-horse',",
        "    actual = '//test:giraffe',",
        ")");
  }

  @Test
  public void testFilesToBuild() throws Exception {
    invalidatePackages();
    ConfiguredTarget giraffeTarget = getConfiguredTarget("//test:giraffe");
    Artifact giraffeArtifact =
        ActionsTestUtil.getFirstArtifactEndingWith(getFilesToBuild(giraffeTarget), "giraffe.jar");
    ConfiguredTarget safariTarget = getConfiguredTarget("//test:safari");
    Action safariAction =
        getGeneratingAction(
            ActionsTestUtil.getFirstArtifactEndingWith(
                getFilesToBuild(safariTarget), "safari.jar"));
    assertThat(safariAction.getInputs().toList()).contains(giraffeArtifact);
  }

  private static void setupStarlarkRules(Scratch scratch) throws IOException {
    Runfiles runfiles = Runfiles.create();
    scratch.file("rules/BUILD");
    String rulesSourcePath =
        runfiles.rlocation(TestConstants.BUILD_RULES_DATA_PATH + "java_rules_skylark.bzl");
    Path rulesDestinationPath = scratch.resolve("rules/java_rules_skylark.bzl");
    scratch.file(
        rulesDestinationPath.getPathString(),
        Files.asCharSource(new File(rulesSourcePath), Charset.defaultCharset()).read());
  }
}
