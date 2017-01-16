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

package com.google.devtools.build.lib.rules.java.proto;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.Artifact.ROOT_RELATIVE_PATH_STRING;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyJarNames;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimaps;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.SkylarkProviders;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import com.google.devtools.build.lib.rules.java.JavaProvider;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.runtime.Runfiles;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the Skylark version of java_lite_proto_library rule.
 */
@RunWith(JUnit4.class)
public class SkylarkJavaLiteProtoLibraryTest extends BuildViewTestCase {
  private static final String RULE_DIRECTORY = "tools/build_rules/java_lite_proto_library";
  private ActionsTestUtil actionsTestUtil;

  @Before
  public final void setUpMocks() throws Exception {
    scratch.file(
        "java/com/google/io/protocol/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_import(name = 'protocol',",
        "            jars = [ 'protocol.jar' ])");
    scratch.file(
        "java/com/google/io/protocol2/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_import(name = 'protocol2',",
        "            jars = [ 'protocol2.jar' ])");

    scratch.file("net/proto/BUILD", "exports_files(['sawzall_message_set.proto'])");
    scratch.file("net/proto2/compiler/public/BUILD", "exports_files(['protocol_compiler'])");

    mockToolchains();

    actionsTestUtil = actionsTestUtil();
  }

  @Before
  public final void setupSkylarkRule() throws Exception {
    File[] files = Runfiles.location(RULE_DIRECTORY).listFiles();
    for (File file : files) {
      scratch.file(RULE_DIRECTORY + "/" + file.getName(), Files.readAllBytes(file.toPath()));
    }
    scratch.file(RULE_DIRECTORY + "/BUILD", "exports_files(['java_lite_proto_library.bzl'])");
    invalidatePackages();
  }

  private void mockToolchains() throws IOException {
    mockRuntimes();

    scratch.file(
        "tools/proto/toolchains/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "proto_lang_toolchain(",
        "    name = 'javalite',",
        "    command_line = '--java_out=lite,immutable,no_enforce_api_compatibility:$(OUT)',",
        "    runtime = '//protobuf:javalite_runtime',",
        ")");
  }

  private void mockRuntimes() throws IOException {
    mockToolsConfig.overwrite(
        "protobuf/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(name = 'javalite_runtime', srcs = ['javalite_runtime.java'])");
  }

  /** Tests that java_binaries which depend on proto_libraries depend on the right set of files. */
  @Test
  public void testBinaryDeps() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//" + RULE_DIRECTORY + ":java_lite_proto_library.bzl', ",
        "'java_lite_proto_library')",
        "java_lite_proto_library(name = 'lite_pb2', deps = [':foo'])",
        "proto_library(name = 'foo', srcs = ['foo.proto', 'bar.proto'], deps = [':baz'])",
        "proto_library(name = 'baz', srcs = ['baz.proto'])");

    ConfiguredTarget target = getConfiguredTarget("//x:lite_pb2");
    NestedSet<Artifact> filesToBuild = getFilesToBuild(target);
    Iterable<String> deps = prettyArtifactNames(actionsTestUtil.artifactClosureOf(filesToBuild));

    // Should depend on compiler and Java proto1 API.
    assertThat(deps).contains("net/proto2/compiler/public/protocol_compiler");

    // Also should not depend on RPC APIs.
    assertThat(deps).doesNotContain("apps/xplat/rpc/codegen/protoc-gen-rpc");

    // Should depend on Java outputs.
    assertThat(deps).contains("x/foo-lite-src.jar");
    assertThat(deps).contains("x/baz-lite-src.jar");

    // Should depend on Java libraries.
    assertThat(deps).contains("x/libfoo-lite.jar");
    assertThat(deps).contains("x/libbaz-lite.jar");
    assertThat(deps).contains("protobuf/libjavalite_runtime-hjar.jar");
  }

  /** Tests that we pass the correct arguments to the protocol compiler. */
  @Test
  public void testJavaProto2CompilerArgs() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//" + RULE_DIRECTORY + ":java_lite_proto_library.bzl',",
        "'java_lite_proto_library')",
        "java_lite_proto_library(name = 'lite_pb2', deps = [':protolib'])",
        "proto_library(name = 'protolib', srcs = ['file.proto'])");

    String genfilesDir = targetConfig.getGenfilesFragment().getPathString();

    List<String> args =
        getGeneratingSpawnAction(getConfiguredTarget("//x:lite_pb2"), "x/protolib-lite-src.jar")
            .getRemainingArguments();

    assertThat(args)
        .contains(
            "--java_out=lite,immutable,no_enforce_api_compatibility:"
                + genfilesDir
                + "/x/protolib-lite-src.jar");

    MoreAsserts.assertContainsSublist(args, "-Ix/file.proto=x/file.proto", "x/file.proto");
  }

  @Test
  public void testProtoLibraryBuildsCompiledJar() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "java",
            "lite_pb2",
            "load('//tools/build_rules/java_lite_proto_library:java_lite_proto_library.bzl',",
            "'java_lite_proto_library')",
            "java_lite_proto_library(name = 'lite_pb2', deps = [':compiled'])",
            "proto_library(name = 'compiled',",
            "              srcs = [ 'ok.proto' ])");

    Artifact compiledJar =
        ActionsTestUtil.getFirstArtifactEndingWith(
            getFilesToBuild(target), "/libcompiled-lite.jar");
    assertThat(compiledJar).isNotNull();
  }

  @Test
  public void testEmptySrcsForJavaApi() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "notbad",
            "lite_pb2",
            "load('//tools/build_rules/java_lite_proto_library:java_lite_proto_library.bzl',",
            "'java_lite_proto_library')",
            "java_lite_proto_library(name = 'lite_pb2', deps = [':null_lib'])",
            "proto_library(name = 'null_lib')");
    JavaCompilationArgsProvider provider = getJavaCompilationArgsProvider(target);
    assertThat(provider).isNotNull();
    assertThat(provider.getJavaCompilationArgs()).isNotNull();
  }

  @Test
  public void testSameVersionCompilerArguments() throws Exception {
    scratch.file(
        "cross/BUILD",
        "load('//tools/build_rules/java_lite_proto_library:java_lite_proto_library.bzl',",
        "'java_lite_proto_library')",
        "java_lite_proto_library(name = 'lite_pb2', deps = ['bravo'], strict_deps = 0)",
        "proto_library(name = 'bravo', srcs = ['bravo.proto'], deps = [':alpha'])",
        "proto_library(name = 'alpha')");

    String genfilesDir = targetConfig.getGenfilesFragment().getPathString();

    ConfiguredTarget litepb2 = getConfiguredTarget("//cross:lite_pb2");

    List<String> args =
        getGeneratingSpawnAction(litepb2, "cross/bravo-lite-src.jar").getRemainingArguments();
    assertThat(args)
        .contains(
            "--java_out=lite,immutable,no_enforce_api_compatibility:"
                + genfilesDir
                + "/cross/bravo-lite-src.jar");
    MoreAsserts.assertContainsSublist(
        args, "-Icross/bravo.proto=cross/bravo.proto", "cross/bravo.proto");

    List<String> directJars =
        prettyJarNames(
            getJavaCompilationArgsProvider(litepb2).getJavaCompilationArgs().getRuntimeJars());
    assertThat(directJars).containsExactly("cross/libbravo-lite.jar");
  }

  /** Protobufs should always be compiled with the default and proto javacopts. */
  @Test
  public void testJavacOpts() throws Exception {
    ConfiguredTarget rule =
        scratchConfiguredTarget(
            "x",
            "lite_pb2",
            "load('//tools/build_rules/java_lite_proto_library:java_lite_proto_library.bzl',",
            "'java_lite_proto_library')",
            "java_lite_proto_library(name = 'lite_pb2', deps = [':proto_lib'])",
            "proto_library(name = 'proto_lib',",
            "              srcs = ['input1.proto', 'input2.proto'])");
    JavaCompilationArgs compilationArgs =
        getJavaCompilationArgsProvider(rule).getJavaCompilationArgs();
    assertThat(compilationArgs.getInstrumentationMetadata()).isEmpty();

    ImmutableListMultimap<String, Artifact> runtimeJars =
        Multimaps.index(compilationArgs.getRuntimeJars(), ROOT_RELATIVE_PATH_STRING);

    Artifact jar = Iterables.getOnlyElement(runtimeJars.get("x/libproto_lib-lite.jar"));
    JavaCompileAction action = (JavaCompileAction) getGeneratingAction(jar);

    List<String> commandLine = ImmutableList.copyOf(action.buildCommandLine());
    assertThat(commandLine).contains("-protoMarkerForTest");
  }

  /**
   * Verify that a java_lite_proto_library exposes Skylark providers for the Java code it generates.
   */
  @Test
  public void testJavaProtosExposeSkylarkProviders() throws Exception {
    scratch.file(
        "proto/extensions.bzl",
        "def _impl(ctx):",
        "  print (ctx.attr.dep[java_common.provider])",
        "custom_rule = rule(",
        "  implementation=_impl,",
        "  attrs={",
        "    'dep': attr.label()",
        "  },",
        ")");
    scratch.file(
        "proto/BUILD",
        "load('/proto/extensions', 'custom_rule')",
        "load('//tools/build_rules/java_lite_proto_library:java_lite_proto_library.bzl',",
        "'java_lite_proto_library')",
        "proto_library(",
        "    name = 'proto',",
        "    srcs = [ 'file.proto' ],",
        ")",
        "java_lite_proto_library(name = 'lite_pb2', deps = [':proto'])",
        "custom_rule(name = 'custom', dep = ':lite_pb2')");
    update(
        ImmutableList.of("//proto:custom"),
        false /* keepGoing */,
        1 /* loadingPhaseThreads */,
        true /* doAnalysis */,
        new EventBus());
    // Implicitly check that `update()` above didn't throw an exception. This implicitly checks that
    // ctx.attr.dep.java.{transitive_deps, outputs}, above, is defined.
  }

  @Test
  public void testProtoLibraryInterop() throws Exception {
    scratch.file(
        "proto/BUILD",
        "load('//tools/build_rules/java_lite_proto_library:java_lite_proto_library.bzl',",
        "'java_lite_proto_library')",
        "proto_library(",
        "    name = 'proto',",
        "    srcs = [ 'file.proto' ],",
        "    java_api_version = 2,",
        ")",
        "java_lite_proto_library(name = 'lite_pb2', deps = [':proto'])");
    update(
        ImmutableList.of("//proto:lite_pb2"),
        false /* keepGoing */,
        1 /* loadingPhaseThreads */,
        true /* doAnalysis */,
        new EventBus());
  }

  /**
   * Tests that a java_lite_proto_library only provides direct jars corresponding on the
   * proto_library rules it directly depends on, excluding anything that the proto_library rules
   * depends on themselves. This does not concern strict-deps in the compilation of the generated
   * Java code itself, only compilation of regular code in java_library/java_binary and similar
   * rules.
   */
  @Test
  public void jplCorrectlyDefinesDirectJars_strictDepsEnabled() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//tools/build_rules/java_lite_proto_library:java_lite_proto_library.bzl',",
        "'java_lite_proto_library')",
        "java_lite_proto_library(name = 'foo_lite_pb2', deps = [':foo'], strict_deps = 1)",
        "proto_library(",
        "    name = 'foo',",
        "    srcs = [ 'foo.proto' ],",
        "    deps = [ ':bar' ],",
        ")",
        "java_lite_proto_library(name = 'bar_lite_pb2', deps = [':bar'])",
        "proto_library(",
        "    name = 'bar',",
        "    srcs = [ 'bar.proto' ],",
        "    deps = [ ':baz' ],",
        ")",
        "proto_library(",
        "    name = 'baz',",
        "    srcs = [ 'baz.proto' ],",
        ")");

    {
      JavaCompilationArgsProvider compilationArgsProvider =
          getJavaCompilationArgsProvider(getConfiguredTarget("//x:foo_lite_pb2"));

      Iterable<String> directJars =
          prettyJarNames(compilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars());

      assertThat(directJars).containsExactly("x/libfoo-lite-hjar.jar");
    }

    {
      JavaCompilationArgsProvider compilationArgsProvider =
          getJavaCompilationArgsProvider(getConfiguredTarget("//x:bar_lite_pb2"));

      Iterable<String> directJars =
          prettyJarNames(compilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars());

      assertThat(directJars).containsExactly("x/libbar-lite-hjar.jar");
    }
  }

  private static JavaCompilationArgsProvider getJavaCompilationArgsProvider(
      ConfiguredTarget target) {
    SkylarkProviders skylarkProviders = target.getProvider(SkylarkProviders.class);
    JavaProvider javaProvider =
        (JavaProvider) skylarkProviders.getDeclaredProvider(JavaProvider.JAVA_PROVIDER.getKey());
    return javaProvider.getJavaCompilationArgsProvider();
  }
}
