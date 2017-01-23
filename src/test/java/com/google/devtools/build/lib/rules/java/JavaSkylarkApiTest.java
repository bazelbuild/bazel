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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.SkylarkProviders;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkClassObjectConstructor.SkylarkKey;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests Skylark API for Java rules.
 */
@RunWith(JUnit4.class)
public class JavaSkylarkApiTest extends BuildViewTestCase {
  @Test
  public void testJavaPlugin() throws Exception {
    scratch.file(
      "java/test/extension.bzl",
      "result = provider()",
      "def impl(ctx):",
      "   depj = ctx.attr.dep.java",
      "   return [result(",
      "             processor_classpath = depj.annotation_processing.processor_classpath,",
      "             processor_classnames = depj.annotation_processing.processor_classnames,",
      "          )]",
      "my_rule = rule(impl, attrs = { 'dep' : attr.label() })"
    );
    scratch.file(
      "java/test/BUILD",
      "load(':extension.bzl', 'my_rule')",
      "java_library(name = 'plugin_dep',",
      "    srcs = [ 'ProcessorDep.java'])",
      "java_plugin(name = 'plugin',",
      "    srcs = ['AnnotationProcessor.java'],",
      "    processor_class = 'com.google.process.stuff',",
      "    deps = [ ':plugin_dep' ])",
      "java_library(name = 'to_be_processed',",
      "    plugins = [':plugin'],",
      "    srcs = ['ToBeProcessed.java'])",
      "my_rule(name = 'my', dep = ':to_be_processed')");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//java/test:my");
    SkylarkProviders provider = configuredTarget.getProvider(SkylarkProviders.class);
    SkylarkClassObject skylarkClassObject = provider
      .getDeclaredProvider(
          new SkylarkKey(Label.parseAbsolute("//java/test:extension.bzl"), "result"));

    assertThat(Iterables.transform((List<?>) skylarkClassObject.getValue("processor_classpath"),
        new Function<Object, String>() {
          @Nullable
          @Override
          public String apply(@Nullable Object o) {
            return ((Artifact) o).getFilename();
          }
        }))
        .containsExactly("libplugin.jar", "libplugin_dep.jar");

    assertThat((List<?>) skylarkClassObject.getValue("processor_classnames"))
        .containsExactly("com.google.process.stuff");
  }

  @Test
  public void cannotConstructJavaProvider() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "my_provider = provider()",
        "def _impl(ctx):",
        "  java_p = java_common.provider",
        "  dep_params = java_p()",
        "  return [my_provider(p = dep_params)]",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file("foo/BUILD", "load(':extension.bzl', 'my_rule')", "my_rule(name = 'r')");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//foo:r")).isNull();
    assertContainsEvent("'java_common.provider' cannot be constructed from Skylark");
  }

  @Test
  public void javaProviderExposedOnJavaLibrary() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "my_provider = provider()",
        "def _impl(ctx):",
        "  dep_params = ctx.attr.dep[java_common.provider]",
        "  return [my_provider(p = dep_params)]",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'jl', srcs = ['java/A.java'])",
        "my_rule(name = 'r', dep = ':jl')");

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:r");
    ConfiguredTarget javaLibraryTarget = getConfiguredTarget("//foo:jl");
    SkylarkKey myProviderKey =
        new SkylarkKey(Label.parseAbsolute("//foo:extension.bzl"), "my_provider");
    SkylarkClassObject declaredProvider =
        myRuleTarget.getProvider(SkylarkProviders.class).getDeclaredProvider(myProviderKey);
    Object javaProvider = declaredProvider.getValue("p");
    assertThat(javaProvider).isInstanceOf(JavaProvider.class);
    assertThat(javaLibraryTarget.getProvider(JavaProvider.class)).isEqualTo(javaProvider);
  }

  @Test
  public void javaProviderPropagation() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _impl(ctx):",
        "  dep_params = ctx.attr.dep[java_common.provider]",
        "  return [dep_params]",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'jl', srcs = ['java/A.java'])",
        "my_rule(name = 'r', dep = ':jl')",
        "java_library(name = 'jl_top', srcs = ['java/C.java'], deps = [':r'])");

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:r");
    ConfiguredTarget javaLibraryTarget = getConfiguredTarget("//foo:jl");
    ConfiguredTarget topJavaLibraryTarget = getConfiguredTarget("//foo:jl_top");

    Object javaProvider = myRuleTarget.get(JavaProvider.JAVA_PROVIDER.getKey());
    assertThat(javaProvider).isInstanceOf(JavaProvider.class);

    JavaProvider jlJavaProvider = javaLibraryTarget.getProvider(JavaProvider.class);

    assertThat(jlJavaProvider == javaProvider).isTrue();

    JavaProvider jlTopJavaProvider = topJavaLibraryTarget.getProvider(JavaProvider.class);

    javaCompilationArgsHaveTheSameParent(
        jlJavaProvider.getProvider(JavaCompilationArgsProvider.class).getJavaCompilationArgs(),
        jlTopJavaProvider.getProvider(JavaCompilationArgsProvider.class).getJavaCompilationArgs());
  }

  private static boolean javaCompilationArgsHaveTheSameParent(
      JavaCompilationArgs args, JavaCompilationArgs otherArgs) {
    if (!nestedSetsOfArtifactHaveTheSameParent(
        args.getCompileTimeJars(), otherArgs.getCompileTimeJars())) {
      return false;
    }
    if (!nestedSetsOfArtifactHaveTheSameParent(
        args.getInstrumentationMetadata(), otherArgs.getInstrumentationMetadata())) {
      return false;
    }
    if (!nestedSetsOfArtifactHaveTheSameParent(args.getRuntimeJars(), otherArgs.getRuntimeJars())) {
      return false;
    }
    return true;
  }

  private static boolean nestedSetsOfArtifactHaveTheSameParent(
      NestedSet<Artifact> artifacts, NestedSet<Artifact> otherArtifacts) {
    Iterator<Artifact> iterator = artifacts.iterator();
    Iterator<Artifact> otherIterator = otherArtifacts.iterator();
    while (iterator.hasNext() && otherIterator.hasNext()) {
      Artifact artifact = (Artifact) iterator.next();
      Artifact otherArtifact = (Artifact) otherIterator.next();
      if (!artifact
          .getPath()
          .getParentDirectory()
          .equals(otherArtifact.getPath().getParentDirectory())) {
        return false;
      }
    }
    if (iterator.hasNext() || otherIterator.hasNext()) {
      return false;
    }
    return true;
  }
}