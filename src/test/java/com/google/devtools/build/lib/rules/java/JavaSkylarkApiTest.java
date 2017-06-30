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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyJarNames;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkClassObjectConstructor.SkylarkKey;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests Skylark API for Java rules.
 */
@RunWith(JUnit4.class)
public class JavaSkylarkApiTest extends BuildViewTestCase {

  @Test
  public void testExposesJavaSkylarkApiProvider() throws Exception {
    scratch.file(
        "java/test/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(",
        "  name = 'dep',",
        "  srcs = [ 'Dep.java'],",
        ")",
        "my_rule(",
        "  name = 'my',",
        "  dep = ':dep',",
        ")");
    scratch.file(
        "java/test/extension.bzl",
        "result = provider()",
        "def impl(ctx):",
        "   depj = ctx.attr.dep.java",
        "   return [result(",
        "             source_jars = depj.source_jars,",
        "             transitive_deps = depj.transitive_deps,",
        "             transitive_runtime_deps = depj.transitive_runtime_deps,",
        "             transitive_source_jars = depj.transitive_source_jars,",
        "             outputs = depj.outputs.jars,",
        "          )]",
        "my_rule = rule(impl, attrs = { 'dep' : attr.label() })");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//java/test:my");
    SkylarkClassObject skylarkClassObject =
        configuredTarget.get(
            new SkylarkKey(Label.parseAbsolute("//java/test:extension.bzl"), "result"));

    SkylarkNestedSet sourceJars = ((SkylarkNestedSet) skylarkClassObject.getValue("source_jars"));
    SkylarkNestedSet transitiveDeps =
        ((SkylarkNestedSet) skylarkClassObject.getValue("transitive_deps"));
    SkylarkNestedSet transitiveRuntimeDeps =
        ((SkylarkNestedSet) skylarkClassObject.getValue("transitive_runtime_deps"));
    SkylarkNestedSet transitiveSourceJars =
        ((SkylarkNestedSet) skylarkClassObject.getValue("transitive_source_jars"));
    SkylarkList<OutputJar> outputJars =
        ((SkylarkList<OutputJar>) skylarkClassObject.getValue("outputs"));

    assertThat(artifactFilesNames(sourceJars.toCollection(Artifact.class)))
        .containsExactly("libdep-src.jar");
    assertThat(artifactFilesNames(transitiveDeps.toCollection(Artifact.class)))
        .containsExactly("libdep-hjar.jar");
    assertThat(artifactFilesNames(transitiveRuntimeDeps.toCollection(Artifact.class)))
        .containsExactly("libdep.jar");
    assertThat(artifactFilesNames(transitiveSourceJars.toCollection(Artifact.class)))
        .containsExactly("libdep-src.jar");
    assertThat(outputJars).hasSize(1);
    assertThat(outputJars.get(0).getClassJar().getFilename()).isEqualTo("libdep.jar");
  }

  private static Collection<String> artifactFilesNames(Collection<Artifact> artifacts) {
    List<String> result = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      result.add(artifact.getFilename());
    }
    return result;
  }

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
    SkylarkClassObject skylarkClassObject = configuredTarget.get(
          new SkylarkKey(Label.parseAbsolute("//java/test:extension.bzl"), "result"));

    assertThat((List<?>) skylarkClassObject.getValue("processor_classnames"))
        .containsExactly("com.google.process.stuff");
    assertThat(
            Iterables.transform(
                ((SkylarkNestedSet) skylarkClassObject.getValue("processor_classpath"))
                    .toCollection(),
                new Function<Object, String>() {
                  @Override
                  public String apply(Object o) {
                    return ((Artifact) o).getFilename();
                  }
                }))
        .containsExactly("libplugin.jar", "libplugin_dep.jar");

  }

  @Test
  public void testJavaProviderFieldsAreSkylarkAccessible() throws Exception {
    // The Skylark evaluation itself will test that compile_jars and
    // transitive_runtime_jars are returning a list readable by Skylark with
    // the expected number of entries.
    scratch.file(
        "java/test/extension.bzl",
        "result = provider()",
        "def impl(ctx):",
        "   java_provider = ctx.attr.dep[java_common.provider]",
        "   jp_cjar_cnt = len(java_provider.compile_jars)",
        "   jp_rjar_cnt = len(java_provider.transitive_runtime_jars)",
        "   if(jp_cjar_cnt != ctx.attr.cnt_cjar):",
        "     fail('#compile_jars is %d, not %d' % (jp_cjar_cnt, ctx.attr.cnt_cjar))",
        "   if(jp_rjar_cnt != ctx.attr.cnt_rjar):",
        "     fail('#transitive_runtime_jars is %d, not %d' % (jp_rjar_cnt, ctx.attr.cnt_rjar))",
        "   return [result(",
        "             compile_jars = java_provider.compile_jars,",
        "             transitive_runtime_jars = java_provider.transitive_runtime_jars,",
        "          )]",
        "my_rule = rule(impl, attrs = { ",
        "  'dep' : attr.label(), ",
        "  'cnt_cjar' : attr.int(), ",
        "  'cnt_rjar' : attr.int(), ",
        "})");
    scratch.file(
        "java/test/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'parent',",
        "    srcs = [ 'Parent.java'])",
        "java_library(name = 'jl',",
        "    srcs = ['Jl.java'],",
        "    deps = [ ':parent' ])",
        "my_rule(name = 'my', dep = ':jl', cnt_cjar = 1, cnt_rjar = 2)");
    // Now, get that information and ensure it is equal to what the jl java_library
    // was presenting
    ConfiguredTarget myConfiguredTarget = getConfiguredTarget("//java/test:my");
    ConfiguredTarget javaLibraryTarget = getConfiguredTarget("//java/test:jl");

    // Extract out the information from skylark rule
    SkylarkClassObject skylarkClassObject =
        myConfiguredTarget.get(
            new SkylarkKey(Label.parseAbsolute("//java/test:extension.bzl"), "result"));

    SkylarkNestedSet rawMyCompileJars =
        (SkylarkNestedSet) (skylarkClassObject.getValue("compile_jars"));
    SkylarkNestedSet rawMyTransitiveRuntimeJars =
        (SkylarkNestedSet) (skylarkClassObject.getValue("transitive_runtime_jars"));

    NestedSet<Artifact> myCompileJars = rawMyCompileJars.getSet(Artifact.class);
    NestedSet<Artifact> myTransitiveRuntimeJars = rawMyTransitiveRuntimeJars.getSet(Artifact.class);

    // Extract out information from native rule
    JavaCompilationArgsProvider jlJavaCompilationArgsProvider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, javaLibraryTarget);
    NestedSet<Artifact> jlCompileJars =
        jlJavaCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars();
    NestedSet<Artifact> jlTransitiveRuntimeJars =
        jlJavaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars();

    // Using reference equality since should be precisely identical
    assertThat(myCompileJars == jlCompileJars).isTrue();
    assertThat(myTransitiveRuntimeJars == jlTransitiveRuntimeJars).isTrue();
  }

  @Test
  public void constructJavaProvider() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _impl(ctx):",
        "  my_provider = java_common.create_provider(",
        "        compile_time_jars = depset(ctx.files.compile_time_jars),",
        "        runtime_jars = depset(ctx.files.runtime_jars),",
        "        transitive_compile_time_jars = depset(ctx.files.transitive_compile_time_jars),",
        "        transitive_runtime_jars = depset(ctx.files.transitive_runtime_jars),",
        "        source_jars = depset(ctx.files.source_jars))",
        "  return [my_provider]",
        "my_rule = rule(_impl, ",
        "    attrs = { ",
        "        'compile_time_jars' : attr.label_list(allow_files=['.jar']),",
        "        'runtime_jars': attr.label_list(allow_files=['.jar']),",
        "        'transitive_compile_time_jars': attr.label_list(allow_files=['.jar']),",
        "        'transitive_runtime_jars': attr.label_list(allow_files=['.jar']),",
        "        'source_jars': attr.label_list(allow_files=['.jar'])",
        "})");
    scratch.file("foo/liba.jar");
    scratch.file("foo/libb.jar");
    scratch.file("foo/liba-src.jar");
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'myrule',",
        "    compile_time_jars = ['liba.jar'],",
        "    runtime_jars = ['libb.jar'],",
        "    source_jars = ['liba-src.jar'],",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//foo:myrule");
    JavaCompilationArgsProvider provider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, target);
    assertThat(provider).isNotNull();
    List<String> compileTimeJars =
        prettyJarNames(provider.getJavaCompilationArgs().getCompileTimeJars());
    assertThat(compileTimeJars).containsExactly("foo/liba.jar");

    List<String> runtimeJars =
        prettyJarNames(provider.getJavaCompilationArgs().getRuntimeJars());
    assertThat(runtimeJars).containsExactly("foo/libb.jar");
    JavaSourceJarsProvider sourcesProvider =
        JavaProvider.getProvider(JavaSourceJarsProvider.class, target);
    List<String> sourceJars = prettyJarNames(sourcesProvider.getSourceJars());
    assertThat(sourceJars).containsExactly("foo/liba-src.jar");
  }

  @Test
  public void constructJavaProviderWithAnotherJavaProvider() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _impl(ctx):",
        "  transitive_provider = java_common.merge(",
        "      [dep[java_common.provider] for dep in ctx.attr.deps])",
        "  my_provider = java_common.create_provider(",
        "        compile_time_jars = depset(ctx.files.compile_time_jars),",
        "        runtime_jars = depset(ctx.files.runtime_jars))",
        "  return [java_common.merge([my_provider, transitive_provider])]",
        "my_rule = rule(_impl, ",
        "    attrs = { ",
        "        'compile_time_jars' : attr.label_list(allow_files=['.jar']),",
        "        'runtime_jars': attr.label_list(allow_files=['.jar']),",
        "        'deps': attr.label_list()",
        "})");
    scratch.file("foo/liba.jar");
    scratch.file("foo/libb.jar");
    scratch.file("foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'java_dep',",
        "    srcs = ['A.java'])",
        "my_rule(name = 'myrule',",
        "    compile_time_jars = ['liba.jar'],",
        "    runtime_jars = ['libb.jar'],",
        "    deps = [':java_dep']",
        ")"
    );
    ConfiguredTarget target = getConfiguredTarget("//foo:myrule");
    JavaCompilationArgsProvider provider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, target);
    assertThat(provider).isNotNull();
    List<String> compileTimeJars =
        prettyJarNames(provider.getJavaCompilationArgs().getCompileTimeJars());
    assertThat(compileTimeJars).containsExactly("foo/liba.jar", "foo/libjava_dep-hjar.jar");

    List<String> runtimeJars = prettyJarNames(
        provider.getJavaCompilationArgs().getRuntimeJars());
    assertThat(runtimeJars).containsExactly("foo/libb.jar", "foo/libjava_dep.jar");
  }

  @Test
  public void constructJavaProviderJavaLibrary() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _impl(ctx):",
        "  my_provider = java_common.create_provider(",
        "        transitive_compile_time_jars = depset(ctx.files.transitive_compile_time_jars),",
        "        transitive_runtime_jars = depset(ctx.files.transitive_runtime_jars))",
        "  return [my_provider]",
        "my_rule = rule(_impl, ",
        "    attrs = { ",
        "        'transitive_compile_time_jars' : attr.label_list(allow_files=['.jar']),",
        "        'transitive_runtime_jars': attr.label_list(allow_files=['.jar'])",
        "})");
    scratch.file("foo/liba.jar");
    scratch.file("foo/libb.jar");
    scratch.file("foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'myrule',",
        "    transitive_compile_time_jars = ['liba.jar'],",
        "    transitive_runtime_jars = ['libb.jar']",
        ")",
        "java_library(name = 'java_lib',",
        "    srcs = ['C.java'],",
        "    deps = [':myrule']",
        ")"
    );
    ConfiguredTarget target = getConfiguredTarget("//foo:java_lib");
    JavaCompilationArgsProvider provider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, target);
    List<String> compileTimeJars = prettyJarNames(
        provider.getRecursiveJavaCompilationArgs().getCompileTimeJars());
    assertThat(compileTimeJars).containsExactly("foo/libjava_lib-hjar.jar", "foo/liba.jar");

    List<String> runtimeJars = prettyJarNames(
        provider.getRecursiveJavaCompilationArgs().getRuntimeJars());
    assertThat(runtimeJars).containsExactly("foo/libjava_lib.jar", "foo/libb.jar");
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
        myRuleTarget.get(myProviderKey);
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

  @Test
  public void skylarkJavaToJavaLibraryAttributes() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _impl(ctx):",
        "  dep_params = ctx.attr.dep[java_common.provider]",
        "  return struct(providers = [dep_params])",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'jl_bottom_for_deps', srcs = ['java/A.java'])",
        "java_library(name = 'jl_bottom_for_exports', srcs = ['java/A2.java'])",
        "java_library(name = 'jl_bottom_for_runtime_deps', srcs = ['java/A2.java'])",
        "my_rule(name = 'mya', dep = ':jl_bottom_for_deps')",
        "my_rule(name = 'myb', dep = ':jl_bottom_for_exports')",
        "my_rule(name = 'myc', dep = ':jl_bottom_for_runtime_deps')",
        "java_library(name = 'lib_exports', srcs = ['java/B.java'], deps = [':mya'],",
        "  exports = [':myb'], runtime_deps = [':myc'])",
        "java_library(name = 'lib_interm', srcs = ['java/C.java'], deps = [':lib_exports'])",
        "java_library(name = 'lib_top', srcs = ['java/D.java'], deps = [':lib_interm'])");
    assertNoEvents();

    // Test that all bottom jars are on the runtime classpath of lib_exports.
    ConfiguredTarget jlExports = getConfiguredTarget("//foo:lib_exports");
    JavaCompilationArgsProvider jlExportsProvider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, jlExports);
    assertThat(prettyJarNames(jlExportsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsAllOf(
            "foo/libjl_bottom_for_deps.jar",
            "foo/libjl_bottom_for_runtime_deps.jar",
            "foo/libjl_bottom_for_exports.jar");

    // Test that libjl_bottom_for_exports.jar is in the recursive java compilation args of lib_top.
    ConfiguredTarget jlTop = getConfiguredTarget("//foo:lib_interm");
    JavaCompilationArgsProvider jlTopProvider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, jlTop);
    assertThat(prettyJarNames(jlTopProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .contains("foo/libjl_bottom_for_exports.jar");
  }

  @Test
  public void skylarkJavaToJavaBinaryAttributes() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _impl(ctx):",
        "  dep_params = ctx.attr.dep[java_common.provider]",
        "  return struct(providers = [dep_params])",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'jl_bottom_for_deps', srcs = ['java/A.java'])",
        "java_library(name = 'jl_bottom_for_runtime_deps', srcs = ['java/A2.java'])",
        "my_rule(name = 'mya', dep = ':jl_bottom_for_deps')",
        "my_rule(name = 'myb', dep = ':jl_bottom_for_runtime_deps')",
        "java_binary(name = 'binary', srcs = ['java/B.java'], main_class = 'foo.A',",
        "  deps = [':mya'], runtime_deps = [':myb'])");
    assertNoEvents();

    // Test that all bottom jars are on the runtime classpath.
    ConfiguredTarget binary = getConfiguredTarget("//foo:binary");
    assertThat(prettyJarNames(
        binary.getProvider(JavaRuntimeClasspathProvider.class).getRuntimeClasspath()))
            .containsAllOf(
                "foo/libjl_bottom_for_deps.jar", "foo/libjl_bottom_for_runtime_deps.jar");
  }

  @Test
  public void skylarkJavaToJavaImportAttributes() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _impl(ctx):",
        "  dep_params = ctx.attr.dep[java_common.provider]",
        "  return struct(providers = [dep_params])",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "java_library(name = 'jl_bottom_for_deps', srcs = ['java/A.java'])",
        "java_library(name = 'jl_bottom_for_runtime_deps', srcs = ['java/A2.java'])",
        "my_rule(name = 'mya', dep = ':jl_bottom_for_deps')",
        "my_rule(name = 'myb', dep = ':jl_bottom_for_runtime_deps')",
        "java_import(name = 'import', jars = ['B.jar'], deps = [':mya'], runtime_deps = [':myb'])");
    assertNoEvents();

    // Test that all bottom jars are on the runtime classpath.
    ConfiguredTarget importTarget = getConfiguredTarget("//foo:import");
    JavaCompilationArgsProvider compilationProvider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, importTarget);
    assertThat(prettyJarNames(
        compilationProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()))
        .containsAllOf(
            "foo/libjl_bottom_for_deps.jar", "foo/libjl_bottom_for_runtime_deps.jar");
  }

  @Test
  public void strictDepsEnabled() throws Exception {
    scratch.file(
        "foo/custom_library.bzl",
        "def _impl(ctx):",
        "  java_provider = java_common.merge([dep[java_common.provider] for dep in ctx.attr.deps])",
        "  if not ctx.attr.strict_deps:",
        "    java_provider = java_common.make_non_strict(java_provider)",
        "  return [java_provider]",
        "custom_library = rule(",
        "  attrs = {",
        "    'deps': attr.label_list(),",
        "    'strict_deps': attr.bool()",
        "  },",
        "  implementation = _impl",
        ")"
    );
    scratch.file(
        "foo/BUILD",
        "load(':custom_library.bzl', 'custom_library')",
        "custom_library(name = 'custom', deps = [':a'], strict_deps = True)",
        "java_library(name = 'a', srcs = ['java/A.java'], deps = [':b'])",
        "java_library(name = 'b', srcs = ['java/B.java'])"
    );

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:custom");
    JavaCompilationArgsProvider javaCompilationArgsProvider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, myRuleTarget);
    List<String> directJars = prettyJarNames(
        javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars());
    assertThat(directJars).containsExactly("foo/liba.jar");
  }

  @Test
  public void strictDepsDisabled() throws Exception {
    scratch.file(
        "foo/custom_library.bzl",
        "def _impl(ctx):",
        "  java_provider = java_common.merge([dep[java_common.provider] for dep in ctx.attr.deps])",
        "  if not ctx.attr.strict_deps:",
        "    java_provider = java_common.make_non_strict(java_provider)",
        "  return [java_provider]",
        "custom_library = rule(",
        "  attrs = {",
        "    'deps': attr.label_list(),",
        "    'strict_deps': attr.bool()",
        "  },",
        "  implementation = _impl",
        ")"
    );
    scratch.file(
        "foo/BUILD",
        "load(':custom_library.bzl', 'custom_library')",
        "custom_library(name = 'custom', deps = [':a'], strict_deps = False)",
        "java_library(name = 'a', srcs = ['java/A.java'], deps = [':b'])",
        "java_library(name = 'b', srcs = ['java/B.java'])"
    );

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:custom");
    JavaCompilationArgsProvider javaCompilationArgsProvider =
        JavaProvider.getProvider(JavaCompilationArgsProvider.class, myRuleTarget);
    List<String> directJars = prettyJarNames(
        javaCompilationArgsProvider.getJavaCompilationArgs().getRuntimeJars());
    assertThat(directJars).containsExactly("foo/liba.jar", "foo/libb.jar");
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
