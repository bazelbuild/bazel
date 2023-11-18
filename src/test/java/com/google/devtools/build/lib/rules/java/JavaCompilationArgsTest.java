// Copyright 2021 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.rules.java.JavaCommon.collectJavaCompilationArgs;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.io.IOException;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link JavaCompilationArgsProvider} and {@link JavaCommon#collectJavaCompilationArgs}.
 */
@RunWith(JUnit4.class)
public class JavaCompilationArgsTest extends BuildViewTestCase {

  private static final String EXPORTS_JAR_PRETTY_NAME = "test/libto_be_exported.jar";
  private static final String EXPORTS_HJAR_PRETTY_NAME = "test/libto_be_exported-hjar.jar";
  private static final String TRANSITIVE_EXPORTS_JAR_PRETTY_NAME = "test/libdep_to_be_exported.jar";
  private static final String TRANSITIVE_EXPORTS_HJAR_PRETTY_NAME =
      "test/libdep_to_be_exported-hjar.jar";
  private static final String DEP_JAR_PRETTY_NAME = "test2/libdirect_dep.jar";
  private static final String DEP_HJAR_PRETTY_NAME = "test2/libdirect_dep-hjar.jar";
  private static final String TRANSITIVE_DEP_JAR_PRETTY_NAME = "test2/libtransitive_dep.jar";
  private static final String TRANSITIVE_DEP_HJAR_PRETTY_NAME = "test2/libtransitive_dep-hjar.jar";

  private Artifact compileTimeJar;
  private Artifact runtimeJar;

  @Before
  public void createArtifacts() {
    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(
            scratch.getFileSystem().getPath("/exec"), RootType.Output, "root");
    runtimeJar = ActionsTestUtil.createArtifact(root, "runtime.jar");
    compileTimeJar = ActionsTestUtil.createArtifact(root, "compiletime.jar");
  }

  @Test
  public void getJavaCompilationArgsWithCompilationArtifacts() throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ false,
            getJavaCompilationArtifacts(),
            /* deps= */ ImmutableList.of(),
            /* runtimeDeps= */ ImmutableList.of(),
            /* exports= */ ImmutableList.of());

    assertThat(javaCompilationArgs.getDirectCompileTimeJars().toList())
        .containsExactly(compileTimeJar);
    assertThat(javaCompilationArgs.getDirectFullCompileTimeJars().toList())
        .containsExactly(compileTimeJar);
    assertThat(javaCompilationArgs.getRuntimeJars().toList()).containsExactly(runtimeJar);
  }

  @Test
  public void getJavaCompilationArgsWithCompilationArtifactsWithExports() throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ false,
            /* compilationArtifacts= */ getJavaCompilationArtifacts(),
            /* deps= */ ImmutableList.of(),
            /* runtimeDeps= */ ImmutableList.of(),
            /* exports= */ getExports());
    assertThat(prettyArtifactNames(javaCompilationArgs.getDirectCompileTimeJars()))
        .containsExactly(compileTimeJar.prettyPrint(), EXPORTS_HJAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getDirectFullCompileTimeJars()))
        .containsExactly(compileTimeJar.prettyPrint(), EXPORTS_JAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(),
            EXPORTS_HJAR_PRETTY_NAME,
            TRANSITIVE_EXPORTS_HJAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveFullCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(),
            EXPORTS_JAR_PRETTY_NAME,
            TRANSITIVE_EXPORTS_JAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getRuntimeJars()))
        .containsExactly(
            runtimeJar.prettyPrint(), EXPORTS_JAR_PRETTY_NAME, TRANSITIVE_EXPORTS_JAR_PRETTY_NAME);
  }

  @Test
  public void getJavaCompilationArgsNeverlinkWithCompilationArtifactsWithExports()
      throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ true,
            /* compilationArtifacts= */ getJavaCompilationArtifacts(),
            /* deps= */ ImmutableList.of(),
            /* runtimeDeps= */ ImmutableList.of(),
            /* exports= */ getExports());

    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(),
            EXPORTS_HJAR_PRETTY_NAME,
            TRANSITIVE_EXPORTS_HJAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveFullCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(),
            EXPORTS_JAR_PRETTY_NAME,
            TRANSITIVE_EXPORTS_JAR_PRETTY_NAME);
    assertThat(javaCompilationArgs.getRuntimeJars().toList()).isEmpty();
  }

  @Test
  public void getJavaCompilationArgsWithCompilationArtifactsAndExportsAndDeps() throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ false,
            /* compilationArtifacts= */ getJavaCompilationArtifacts(),
            /* deps= */ getDeps(),
            /* runtimeDeps= */ ImmutableList.of(),
            /* exports= */ getExports());

    assertThat(prettyArtifactNames(javaCompilationArgs.getDirectCompileTimeJars()))
        .containsExactly(compileTimeJar.prettyPrint(), EXPORTS_HJAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getDirectFullCompileTimeJars()))
        .containsExactly(compileTimeJar.prettyPrint(), EXPORTS_JAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getRuntimeJars()))
        .containsExactly(
            runtimeJar.prettyPrint(),
            EXPORTS_JAR_PRETTY_NAME,
            DEP_JAR_PRETTY_NAME,
            TRANSITIVE_EXPORTS_JAR_PRETTY_NAME,
            TRANSITIVE_DEP_JAR_PRETTY_NAME);
  }

  @Test
  public void getJavaCompilationArgsWithCompilationArtifactsAndDeps() throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ false,
            /* compilationArtifacts= */ getJavaCompilationArtifacts(),
            /* deps= */ getDeps(),
            /* runtimeDeps= */ ImmutableList.of(),
            /* exports= */ ImmutableList.of());

    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(), DEP_HJAR_PRETTY_NAME, TRANSITIVE_DEP_HJAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveFullCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(), DEP_JAR_PRETTY_NAME, TRANSITIVE_DEP_JAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getRuntimeJars()))
        .containsExactly(
            runtimeJar.prettyPrint(), DEP_JAR_PRETTY_NAME, TRANSITIVE_DEP_JAR_PRETTY_NAME);
  }

  @Test
  public void getJavaCompilationArgsNeverlinkWithCompilationArtifactsAndDeps() throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ true,
            /* compilationArtifacts= */ getJavaCompilationArtifacts(),
            /* deps= */ getDeps(),
            /* runtimeDeps= */ ImmutableList.of(),
            /* exports= */ ImmutableList.of());

    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(), DEP_HJAR_PRETTY_NAME, TRANSITIVE_DEP_HJAR_PRETTY_NAME);
    assertThat(prettyArtifactNames(javaCompilationArgs.getTransitiveFullCompileTimeJars()))
        .containsExactly(
            compileTimeJar.prettyPrint(), DEP_JAR_PRETTY_NAME, TRANSITIVE_DEP_JAR_PRETTY_NAME);
    assertThat(javaCompilationArgs.getRuntimeJars().toList()).isEmpty();
  }

  @Test
  public void getJavaCompilationArgsWithCompilationArtifactsAndRuntimeDeps() throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ false,
            /* compilationArtifacts= */ getJavaCompilationArtifacts(),
            /* deps= */ ImmutableList.of(),
            /* runtimeDeps= */ getDeps(),
            /* exports= */ ImmutableList.of());

    assertThat(javaCompilationArgs.getTransitiveCompileTimeJars().toList())
        .containsExactly(compileTimeJar);
    assertThat(javaCompilationArgs.getTransitiveFullCompileTimeJars().toList())
        .containsExactly(compileTimeJar);
    assertThat(prettyArtifactNames(javaCompilationArgs.getRuntimeJars()))
        .containsExactly(
            runtimeJar.prettyPrint(), DEP_JAR_PRETTY_NAME, TRANSITIVE_DEP_JAR_PRETTY_NAME);
  }

  @Test
  public void getJavaCompilationArgsNeverlinkWithCompilationArtifactsAndRuntimeDeps()
      throws Exception {
    JavaCompilationArgsProvider javaCompilationArgs =
        collectJavaCompilationArgs(
            /* isNeverLink= */ true,
            /* compilationArtifacts= */ getJavaCompilationArtifacts(),
            /* deps= */ ImmutableList.of(),
            /* runtimeDeps= */ getDeps(),
            /* exports= */ ImmutableList.of());

    assertThat(javaCompilationArgs.getTransitiveCompileTimeJars().toList())
        .containsExactly(compileTimeJar);
    assertThat(javaCompilationArgs.getTransitiveFullCompileTimeJars().toList())
        .containsExactly(compileTimeJar);
    assertThat(prettyArtifactNames(javaCompilationArgs.getRuntimeJars()))
        .containsExactly(DEP_JAR_PRETTY_NAME, TRANSITIVE_DEP_JAR_PRETTY_NAME);
  }

  private JavaCompilationArtifacts getJavaCompilationArtifacts() {
    return new JavaCompilationArtifacts.Builder()
        .addRuntimeJar(runtimeJar)
        .addCompileTimeJarAsFullJar(compileTimeJar)
        .build();
  }

  private List<JavaCompilationArgsProvider> getExports() throws Exception {
    createExportsTarget();
    return ImmutableList.of(
        JavaInfo.getProvider(
            JavaCompilationArgsProvider.class, getConfiguredTarget("//test:to_be_exported")));
  }

  private void createExportsTarget() throws IOException {
    scratch.file(
        "test/BUILD",
        "java_library(name = 'to_be_exported', srcs = ['A.java'], ",
        "    deps = [':dep_to_be_exported'])",
        "java_library(name= 'dep_to_be_exported', srcs = ['B.java'])");
  }

  private List<JavaCompilationArgsProvider> getDeps() throws Exception {
    createDepsTarget();
    return ImmutableList.of(
        JavaInfo.getProvider(
            JavaCompilationArgsProvider.class, getConfiguredTarget("//test2:direct_dep")));
  }

  private void createDepsTarget() throws IOException {
    scratch.file(
        "test2/BUILD",
        "java_library(name = 'direct_dep', srcs = ['A.java'], deps = [':transitive_dep'])",
        "java_library(name= 'transitive_dep', srcs = ['B.java'])");
  }
}
