// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests JavaInfo API for Skylark.
 */
@RunWith(JUnit4.class)
public class JavaInfoSkylarkApiTest extends BuildViewTestCase {

  private static final String HOST_JAVA_RUNTIME_LABEL =
      TestConstants.TOOLS_REPOSITORY + "//tools/jdk:current_host_java_runtime";

  @Test
  public void buildHelperCreateJavaInfoWithOutputJarOnly() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(",
        "  name = 'my_skylark_rule',",
        "  output_jar = 'my_skylark_rule_lib.jar',",
        "  source_jars = ['my_skylark_rule_src.jar']",
        ")");
    assertNoEvents();

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        fetchJavaInfo().getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoWithOutputJarAndUseIJar() throws Exception {
   
    ruleBuilder()
        .withIJar()
        .build();

    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(",
        "  name = 'my_skylark_rule',",
        "  output_jar = 'my_skylark_rule_lib.jar',",
        "  source_jars = ['my_skylark_rule_src.jar']",
        ")");
    assertNoEvents();

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        fetchJavaInfo().getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib-ijar.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib-ijar.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoJavaRuleOutputJarsProviderSourseJarOutputJarAndUseIJar()
      throws Exception {
    ruleBuilder()
        .withIJar()
        .build();

    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(",
        "  name = 'my_skylark_rule',",
        "  output_jar = 'my_skylark_rule_lib.jar',",
        "  source_jars = ['my_skylark_rule_src.jar'],",
        ")");
    assertNoEvents();

    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider =
        fetchJavaInfo().getProvider(JavaRuleOutputJarsProvider.class);

    assertThat(prettyArtifactNames(javaRuleOutputJarsProvider.getAllSrcOutputJars()))
        .containsExactly("foo/my_skylark_rule_src.jar");

    assertThat(prettyArtifactNames(javaRuleOutputJarsProvider.getAllClassOutputJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");

    assertThat(javaRuleOutputJarsProvider.getOutputJars())
        .hasSize(1);
    OutputJar outputJar = javaRuleOutputJarsProvider.getOutputJars().get(0);

    assertThat(outputJar.getIJar().prettyPrint())
        .isEqualTo("foo/my_skylark_rule_lib-ijar.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoWithDeps() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_direct', srcs = ['java/A.java'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar'],",
        "        dep = [':my_java_lib_direct']",
        ")");
    assertNoEvents();

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        fetchJavaInfo().getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_direct.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_direct.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_direct-hjar.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoWithRunTimeDeps() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_direct', srcs = ['java/A.java'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar'],",
        "        dep_runtime = [':my_java_lib_direct']",
        ")");
    assertNoEvents();

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        fetchJavaInfo().getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_direct.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoWithDepsAndNeverLink() throws Exception {
    ruleBuilder()
        .withNeverLink()
        .build();
    
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_direct', srcs = ['java/A.java'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar'],",
        "        dep = [':my_java_lib_direct']",
        ")");
    assertNoEvents();

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        fetchJavaInfo().getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .isEmpty();
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .isEmpty();
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_direct.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_direct-hjar.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoSourceJarsProviderWithSourceJars() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar']",
        ")");
    assertNoEvents();

    JavaSourceJarsProvider sourceJarsProvider =
        fetchJavaInfo().getProvider(JavaSourceJarsProvider.class);

    assertThat(prettyArtifactNames(sourceJarsProvider.getSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar");

    assertThat(prettyArtifactNames(sourceJarsProvider.getTransitiveSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoWithSourcesFiles() throws Exception {
    ruleBuilder().withSourceFiles().build();

    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(",
        "  name = 'my_skylark_rule',",
        "  output_jar = 'my_skylark_rule_lib.jar',",
        "  sources = ['ClassA.java', 'ClassB.java', 'ClassC.java', 'ClassD.java'],",
        ")");
    assertNoEvents();

    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider =
        fetchJavaInfo().getProvider(JavaRuleOutputJarsProvider.class);

    assertThat(prettyArtifactNames(javaRuleOutputJarsProvider.getAllSrcOutputJars()))
        .containsExactly("foo/my_skylark_rule_lib-src.jar");

    JavaSourceJarsProvider sourceJarsProvider =
        fetchJavaInfo().getProvider(JavaSourceJarsProvider.class);

    assertThat(prettyArtifactNames(sourceJarsProvider.getSourceJars()))
        .containsExactly("foo/my_skylark_rule_lib-src.jar");

    assertThat(prettyArtifactNames(sourceJarsProvider.getTransitiveSourceJars()))
        .containsExactly("foo/my_skylark_rule_lib-src.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoWithSourcesFilesAndSourcesJars() throws Exception {
    ruleBuilder().withSourceFiles().build();

    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(",
        "  name = 'my_skylark_rule',",
        "  output_jar = 'my_skylark_rule_lib.jar',",
        "  sources = ['ClassA.java', 'ClassB.java', 'ClassC.java', 'ClassD.java'],",
        "  source_jars = ['my_skylark_rule_src-A.jar']",
        ")");
    assertNoEvents();

    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider =
        fetchJavaInfo().getProvider(JavaRuleOutputJarsProvider.class);

    assertThat(prettyArtifactNames(javaRuleOutputJarsProvider.getAllSrcOutputJars()))
        .containsExactly("foo/my_skylark_rule_lib-src.jar");

    JavaSourceJarsProvider sourceJarsProvider =
        fetchJavaInfo().getProvider(JavaSourceJarsProvider.class);

    assertThat(prettyArtifactNames(sourceJarsProvider.getSourceJars()))
        .containsExactly("foo/my_skylark_rule_lib-src.jar");

    assertThat(prettyArtifactNames(sourceJarsProvider.getTransitiveSourceJars()))
        .containsExactly("foo/my_skylark_rule_lib-src.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoSourceJarsProviderWithDeps() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_direct', srcs = ['java/A.java'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar'],",
        "        dep = [':my_java_lib_direct']",
        ")");
    assertNoEvents();

    JavaSourceJarsProvider sourceJarsProvider =
        fetchJavaInfo().getProvider(JavaSourceJarsProvider.class);

    assertThat(prettyArtifactNames(sourceJarsProvider.getSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar");

    assertThat(prettyArtifactNames(sourceJarsProvider.getTransitiveSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar", "foo/libmy_java_lib_direct-src.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoJavaSourceJarsProviderAndRuntimeDeps() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_direct', srcs = ['java/A.java'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar'],",
        "        dep_runtime = [':my_java_lib_direct']",
        ")");
    assertNoEvents();

    JavaSourceJarsProvider sourceJarsProvider =
        fetchJavaInfo().getProvider(JavaSourceJarsProvider.class);

    assertThat(prettyArtifactNames(sourceJarsProvider.getSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar");

    assertThat(prettyArtifactNames(sourceJarsProvider.getTransitiveSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar", "foo/libmy_java_lib_direct-src.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoJavaSourceJarsProviderAndTransitiveDeps() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_transitive', srcs = ['java/B.java'])",
        "java_library(name = 'my_java_lib_direct',",
        "             srcs = ['java/A.java'],",
        "             deps = [':my_java_lib_transitive'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar'],",
        "        dep = [':my_java_lib_direct']",
        ")");
    assertNoEvents();

    JavaSourceJarsProvider sourceJarsProvider =
        fetchJavaInfo().getProvider(JavaSourceJarsProvider.class);

    assertThat(prettyArtifactNames(sourceJarsProvider.getSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar");

    assertThat(prettyArtifactNames(sourceJarsProvider.getTransitiveSourceJars()))
        .containsExactly(
            "foo/my_skylark_rule_src.jar",
            "foo/libmy_java_lib_direct-src.jar",
            "foo/libmy_java_lib_transitive-src.jar");
  }

  @Test
  public void buildHelperCreateJavaInfoJavaSourceJarsProviderAndTransitiveRuntimeDeps()
      throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_transitive', srcs = ['java/B.java'])",
        "java_library(name = 'my_java_lib_direct',",
        "             srcs = ['java/A.java'],",
        "             deps = [':my_java_lib_transitive'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar'],",
        "        dep = [':my_java_lib_direct']",
        ")");
    assertNoEvents();

    JavaSourceJarsProvider sourceJarsProvider =
        fetchJavaInfo().getProvider(JavaSourceJarsProvider.class);

    assertThat(prettyArtifactNames(sourceJarsProvider.getSourceJars()))
        .containsExactly("foo/my_skylark_rule_src.jar");

    assertThat(prettyArtifactNames(sourceJarsProvider.getTransitiveSourceJars()))
        .containsExactly(
            "foo/my_skylark_rule_src.jar",
            "foo/libmy_java_lib_direct-src.jar",
            "foo/libmy_java_lib_transitive-src.jar");
  }

  /**
   * Tests that JavaExportsProvider is empty by default.
   */
  @Test
  public void buildHelperCreateJavaInfoExportIsEmpty() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        source_jars = ['my_skylark_rule_src.jar']",
        ")");
    assertNoEvents();

    JavaExportsProvider exportsProvider = fetchJavaInfo().getProvider(JavaExportsProvider.class);

    assertThat(exportsProvider.getTransitiveExports()).isEmpty();
  }

  /** Test exports adds dependencies to JavaCompilationArgsProvider. */
  @Test
  public void buildHelperCreateJavaInfoExportProviderExportsDepsAdded() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_exports', srcs = ['java/A.java'])",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        dep_exports = [':my_java_lib_exports']",
        ")");
    assertNoEvents();

    JavaInfo javaInfo = fetchJavaInfo();

    JavaExportsProvider exportsProvider = javaInfo.getProvider(JavaExportsProvider.class);

    assertThat(exportsProvider.getTransitiveExports()).isEmpty();

    JavaSourceJarsProvider javaSourceJarsProvider =
        javaInfo.getProvider(JavaSourceJarsProvider.class);

    assertThat(javaSourceJarsProvider.getSourceJars()).isEmpty();

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        javaInfo.getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_exports.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_exports.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_exports-hjar.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_exports.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_exports.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly("foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_exports-hjar.jar");
  }

  /**
   * Test exports adds itself and recursive dependencies to JavaCompilationArgsProvider
   * and JavaExportsProvider populated.
   */
  @Test
  public void buildHelperCreateJavaInfoExportProvider() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_c', srcs = ['java/C.java'])",
        "java_library(name = 'my_java_lib_b', srcs = ['java/B.java'])",
        "java_library(name = 'my_java_lib_a', srcs = ['java/A.java'],",
        "             deps = [':my_java_lib_b', ':my_java_lib_c'],",
        "             exports = [':my_java_lib_b']",
        "            )",
        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        dep_exports = [':my_java_lib_a']",
        ")");
    assertNoEvents();

    JavaInfo javaInfo = fetchJavaInfo();

    JavaExportsProvider exportsProvider = javaInfo.getProvider(JavaExportsProvider.class);

    assertThat(
        exportsProvider.getTransitiveExports())
        .containsExactly(Label.parseAbsolute("//foo:my_java_lib_b"));

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        javaInfo.getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_a.jar", "foo/libmy_java_lib_b.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_a.jar", "foo/libmy_java_lib_b.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a-hjar.jar",
            "foo/libmy_java_lib_b-hjar.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a.jar",
            "foo/libmy_java_lib_b.jar",
            "foo/libmy_java_lib_c.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a.jar",
            "foo/libmy_java_lib_b.jar",
            "foo/libmy_java_lib_c.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a-hjar.jar",
            "foo/libmy_java_lib_b-hjar.jar",
            "foo/libmy_java_lib_c-hjar.jar");
  }


  /**
   * Tests case:
   *  my_lib
   *  //   \
   *  a    c
   * //    \\
   * b      d
   *
   * where single line is normal dependency and double is exports dependency.
   */
  @Test
  public void buildHelperCreateJavaInfoExportProvider001() throws Exception {
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'my_java_lib_b', srcs = ['java/B.java'])",
        "java_library(name = 'my_java_lib_a', srcs = ['java/A.java'],",
        "             deps = [':my_java_lib_b'],",
        "             exports = [':my_java_lib_b']",
        "            )",
        "java_library(name = 'my_java_lib_d', srcs = ['java/D.java'])",
        "java_library(name = 'my_java_lib_c', srcs = ['java/C.java'],",
        "             deps = [':my_java_lib_d'],",
        "             exports = [':my_java_lib_d']",
        "            )",

        "my_rule(name = 'my_skylark_rule',",
        "        output_jar = 'my_skylark_rule_lib.jar',",
        "        dep = [':my_java_lib_a', ':my_java_lib_c'],",
        "        dep_exports = [':my_java_lib_a']",
        ")");
    assertNoEvents();

    JavaInfo javaInfo = fetchJavaInfo();

    JavaExportsProvider exportsProvider = javaInfo.getProvider(JavaExportsProvider.class);

    assertThat(
        exportsProvider.getTransitiveExports())
        .containsExactly(Label.parseAbsolute("//foo:my_java_lib_b"));

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        javaInfo.getProvider(JavaCompilationArgsProvider.class);

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_a.jar", "foo/libmy_java_lib_b.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getFullCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar", "foo/libmy_java_lib_a.jar", "foo/libmy_java_lib_b.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a-hjar.jar",
            "foo/libmy_java_lib_b-hjar.jar");

    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a.jar",
            "foo/libmy_java_lib_b.jar",
            "foo/libmy_java_lib_c.jar",
            "foo/libmy_java_lib_d.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider
                    .getRecursiveJavaCompilationArgs()
                    .getFullCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a.jar",
            "foo/libmy_java_lib_b.jar",
            "foo/libmy_java_lib_c.jar",
            "foo/libmy_java_lib_d.jar");
    assertThat(
            prettyArtifactNames(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getCompileTimeJars()))
        .containsExactly(
            "foo/my_skylark_rule_lib.jar",
            "foo/libmy_java_lib_a-hjar.jar",
            "foo/libmy_java_lib_b-hjar.jar",
            "foo/libmy_java_lib_c-hjar.jar",
            "foo/libmy_java_lib_d-hjar.jar");
  }


  private RuleBuilder ruleBuilder(){
    return new RuleBuilder();
  }

  private class RuleBuilder{
    private boolean useIJar = false;
    private boolean neverLink = false;
    private boolean sourceFiles = false;

    private RuleBuilder withIJar() {
      useIJar = true;
      return this;
    }

    private RuleBuilder withNeverLink() {
      neverLink = true;
      return this;
    }

    private RuleBuilder withSourceFiles() {
      sourceFiles = true;
      return this;
    }

    private void build() throws Exception {
      if (useIJar || sourceFiles) {
        writeBuildFileForJavaToolchain();
      }

      String[] lines = {
        "result = provider()",
        "def _impl(ctx):",
        "  ctx.actions.write(ctx.outputs.output_jar, 'JavaInfo API Test', is_executable=False) ",
        "  dp = [dep[java_common.provider] for dep in ctx.attr.dep]",
        "  dp_runtime = [dep[java_common.provider] for dep in ctx.attr.dep_runtime]",
        "  dp_exports = [dep[java_common.provider] for dep in ctx.attr.dep_exports]",
        "  javaInfo = JavaInfo(",
        "    output_jar = ctx.outputs.output_jar, ",
        useIJar ? "    use_ijar = True," : "    use_ijar = False,",
        neverLink ? "    neverlink = True," : "",
        "    source_jars = ctx.files.source_jars,",
        "    sources = ctx.files.sources,",
        "    deps = dp,",
        "    runtime_deps = dp_runtime,",
        "    exports = dp_exports,",
        useIJar || sourceFiles ? "    actions = ctx.actions," : "",
        useIJar || sourceFiles ? "    java_toolchain = ctx.attr._toolchain," : "",
        sourceFiles ? "    host_javabase = ctx.attr._host_javabase," : "",
        "  )",
        "  return [result(property = javaInfo)]",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep' : attr.label_list(),",
        "    'dep_runtime' : attr.label_list(),",
        "    'dep_exports' : attr.label_list(),",
        "    'output_jar' : attr.output(default=None, mandatory=True),",
        "    'source_jars' : attr.label_list(allow_files=['.jar']),",
        "    'sources' : attr.label_list(allow_files=['.java']),",
        useIJar || sourceFiles
            ? "    '_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),"
            : "",
        sourceFiles
            ? "    '_host_javabase': attr.label(default = Label('"
                + HOST_JAVA_RUNTIME_LABEL
                + "')),"
            : "",
        "  }",
        ")"
      };

      scratch.file("foo/extension.bzl", lines);
    }
  }

  private JavaInfo fetchJavaInfo() throws Exception {
    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:my_skylark_rule");
    Info info =
        myRuleTarget.get(new SkylarkKey(Label.parseAbsolute("//foo:extension.bzl"), "result"));

    @SuppressWarnings("unchecked")
    JavaInfo javaInfo = (JavaInfo) info.getValue("property");
    return javaInfo;
  }
}
