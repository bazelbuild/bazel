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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getProcessorNames;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getProcessorPath;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;
import static java.util.Arrays.stream;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import com.google.testing.junit.testparameterinjector.TestParameters.TestParametersValues;
import com.google.testing.junit.testparameterinjector.TestParametersValuesProvider;
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
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

  private StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(
            keyForBuild(Label.parseCanonical("//myinfo:myinfo.bzl")), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  private static ImmutableList<String> artifactFilesNames(NestedSet<Artifact> artifacts) {
    return artifactFilesNames(artifacts.toList());
  }

  private static ImmutableList<String> artifactFilesNames(Iterable<Artifact> artifacts) {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (Artifact artifact : artifacts) {
      result.add(artifact.getFilename());
    }
    return result.build();
  }

  @Test
  public void javaInfoConstructorWithNeverlink() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "java/test/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(name = "somedep")
        """);
    scratch.file(
        "java/test/custom_rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "JavaInfo")
        def _impl(ctx):
            output_jar = ctx.actions.declare_file("lib" + ctx.label.name + ".jar")
            ctx.actions.write(output_jar, "")
            java_info = JavaInfo(
                output_jar = output_jar,
                compile_jar = None,
                neverlink = True,
            )
            return [
                java_info,
            ]

        java_custom_library = rule(
            implementation = _impl,
            fragments = ["java"],
            provides = [JavaInfo],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//java/test:somedep");

    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    assertThat(javaInfo.isNeverlink()).isTrue();
  }

  @Test
  public void javaCommonMergeWithNeverlink() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "java/test/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(name = "somedep")
        """);
    scratch.file(
        "java/test/custom_rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "JavaInfo", "java_common")
        def _impl(ctx):
            output_jar = ctx.actions.declare_file("lib" + ctx.label.name + ".jar")
            ctx.actions.write(output_jar, "")
            java_info_with_neverlink = JavaInfo(
                output_jar = output_jar,
                compile_jar = None,
                neverlink = True,
            )
            java_info_without_neverlink = JavaInfo(
                output_jar = output_jar,
                compile_jar = None,
            )
            java_info = java_common.merge([java_info_with_neverlink, java_info_without_neverlink])
            return [
                java_info,
            ]

        java_custom_library = rule(
            implementation = _impl,
            fragments = ["java"],
            provides = [JavaInfo],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//java/test:somedep");

    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    assertThat(javaInfo.isNeverlink()).isTrue();
  }

  @Test
  public void javaCommonCompileWithNeverlink() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "java/test/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(
            name = "somedep",
            srcs = ["Dependency.java"],
        )
        """);
    scratch.file(
        "java/test/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common', 'JavaInfo')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib' + ctx.label.name + '.jar')",
        "  java_info = java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "    neverlink = True,",
        "  )",
        "  return [",
        "      java_info",
        "  ]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=['.java']),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java'],",
        "  provides = [JavaInfo],",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//java/test:somedep");

    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    assertThat(javaInfo.isNeverlink()).isTrue();
  }

  /**
   * Tests that java_common.compile propagates native libraries from deps, runtime_deps, and
   * exports.
   */
  @Test
  public void javaCommonCompile_nativeLibrariesPropagate() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "java/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(
            name = "custom",
            srcs = ["A.java"],
            exports = [":lib_exports"],
            runtime_deps = [":lib_runtime_deps"],
            deps = [":lib_deps"],
        )

        java_library(
            name = "lib_deps",
            srcs = ["B.java"],
            deps = [":native_deps1.so"],
        )

        cc_library(
            name = "native_deps1.so",
            srcs = ["a.cc"],
        )

        java_library(
            name = "lib_runtime_deps",
            srcs = ["C.java"],
            deps = [":native_rdeps1.so"],
        )

        cc_library(
            name = "native_rdeps1.so",
            srcs = ["c.cc"],
        )

        java_library(
            name = "lib_exports",
            srcs = ["D.java"],
            deps = [":native_exports1.so"],
        )

        cc_library(
            name = "native_exports1.so",
            srcs = ["e.cc"],
        )
        """);
    scratch.file(
        "java/test/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common', 'JavaInfo')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib' + ctx.label.name + '.jar')",
        "  compilation_provider = java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    deps = [dep[JavaInfo] for dep in ctx.attr.deps if JavaInfo in dep],",
        "    runtime_deps = [dep[JavaInfo] for dep in ctx.attr.runtime_deps if JavaInfo in dep],",
        "    exports = [dep[JavaInfo] for dep in ctx.attr.exports if JavaInfo in dep],",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "  )",
        "  return [",
        "      compilation_provider",
        "  ]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  outputs = {",
        "    'my_output': 'lib%{name}.jar'",
        "  },",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True),",
        "    'deps': attr.label_list(),",
        "    'runtime_deps': attr.label_list(),",
        "    'exports': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java']",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//java/test:custom");

    JavaInfo info = JavaInfo.getJavaInfo(configuredTarget);
    NestedSet<LibraryToLink> nativeLibraries = info.getTransitiveNativeLibraries();
    assertThat(nativeLibraries.toList().stream().map(lib -> lib.getStaticLibrary().prettyPrint()))
        .containsExactly(
            "java/test/libnative_rdeps1.so.a",
            "java/test/libnative_exports1.so.a",
            "java/test/libnative_deps1.so.a")
        .inOrder();
  }

  /**
   * Tests that java_common.compile propagates native libraries passed by native_libraries argument.
   */
  @Test
  public void javaCommonCompile_directNativeLibraries() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "java/test/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(
            name = "custom",
            srcs = ["A.java"],
            ccdeps = [":native.so"],
        )

        cc_library(
            name = "native.so",
            srcs = ["a.cc"],
        )
        """);
    scratch.file(
        "java/test/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib' + ctx.label.name + '.jar')",
        "  compilation_provider = java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    native_libraries = [dep[CcInfo] for dep in ctx.attr.ccdeps if CcInfo in dep],",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "  )",
        "  return [",
        "      compilation_provider",
        "  ]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  outputs = {",
        "    'my_output': 'lib%{name}.jar'",
        "  },",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True),",
        "    'ccdeps': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java']",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//java/test:custom");

    JavaInfo info = JavaInfo.getJavaInfo(configuredTarget);
    NestedSet<LibraryToLink> nativeLibraries = info.getTransitiveNativeLibraries();
    assertThat(nativeLibraries.toList().stream().map(lib -> lib.getStaticLibrary().prettyPrint()))
        .containsExactly("java/test/libnative.so.a")
        .inOrder();
  }

  @Test
  public void strictDepsEnabled() throws Exception {
    scratch.file(
        "foo/custom_library.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common")
        load("@rules_java//java/common:java_info.bzl", "JavaInfo")
        def _impl(ctx):
            java_provider = java_common.merge([dep[JavaInfo] for dep in ctx.attr.deps])
            if not ctx.attr.strict_deps:
                java_provider = java_common.make_non_strict(java_provider)
            return [java_provider]

        custom_library = rule(
            attrs = {
                "deps": attr.label_list(),
                "strict_deps": attr.bool(),
            },
            implementation = _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":custom_library.bzl", "custom_library")

        custom_library(
            name = "custom",
            strict_deps = True,
            deps = [":a"],
        )

        java_library(
            name = "a",
            srcs = ["java/A.java"],
            deps = [":b"],
        )

        java_library(
            name = "b",
            srcs = ["java/B.java"],
        )
        """);

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:custom");
    JavaCompilationArgsProvider javaCompilationArgsProvider =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, myRuleTarget);
    assertThat(prettyArtifactNames(javaCompilationArgsProvider.directCompileTimeJars()))
        .containsExactly("foo/liba-hjar.jar");
    assertThat(prettyArtifactNames(javaCompilationArgsProvider.directFullCompileTimeJars()))
        .containsExactly("foo/liba.jar");
    assertThat(prettyArtifactNames(javaCompilationArgsProvider.directHeaderCompilationJars()))
        .containsExactly("foo/liba-tjar.jar");
  }

  @Test
  public void strictDepsDisabled() throws Exception {
    scratch.file(
        "foo/custom_library.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common")
        load("@rules_java//java/common:java_info.bzl", "JavaInfo")
        def _impl(ctx):
            java_provider = java_common.merge([dep[JavaInfo] for dep in ctx.attr.deps])
            if not ctx.attr.strict_deps:
                java_provider = java_common.make_non_strict(java_provider)
            return [java_provider]

        custom_library = rule(
            attrs = {
                "deps": attr.label_list(),
                "strict_deps": attr.bool(),
            },
            implementation = _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":custom_library.bzl", "custom_library")

        custom_library(
            name = "custom",
            strict_deps = False,
            deps = [":a"],
        )

        java_library(
            name = "a",
            srcs = ["java/A.java"],
            deps = [":b"],
        )

        java_library(
            name = "b",
            srcs = ["java/B.java"],
        )
        """);

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:custom");
    JavaCompilationArgsProvider javaCompilationArgsProvider =
        JavaInfo.getProvider(JavaCompilationArgsProvider.class, myRuleTarget);
    assertThat(prettyArtifactNames(javaCompilationArgsProvider.runtimeJars()))
        .containsExactly("foo/liba.jar", "foo/libb.jar");
    assertThat(prettyArtifactNames(javaCompilationArgsProvider.directCompileTimeJars()))
        .containsExactly("foo/liba-hjar.jar", "foo/libb-hjar.jar");
    assertThat(prettyArtifactNames(javaCompilationArgsProvider.directFullCompileTimeJars()))
        .containsExactly("foo/liba.jar", "foo/libb.jar");
    assertThat(prettyArtifactNames(javaCompilationArgsProvider.directHeaderCompilationJars()))
        .containsExactly("foo/liba-hjar.jar", "foo/libb-hjar.jar");
  }

  @Test
  public void strictJavaDepsFlagExposed_default() throws Exception {
    scratch.file(
        "foo/rule.bzl",
        """
        result = provider()

        def _impl(ctx):
            return [result(strict_java_deps = ctx.fragments.java.strict_java_deps)]

        myrule = rule(
            implementation = _impl,
            fragments = ["java"],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":rule.bzl", "myrule")

        myrule(name = "myrule")
        """);
    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:myrule");
    StructImpl info =
        (StructImpl)
            configuredTarget.get(
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//foo:rule.bzl")), "result"));
    assertThat(((String) info.getValue("strict_java_deps"))).isEqualTo("default");
  }

  @Test
  public void strictJavaDepsFlagExposed_error() throws Exception {
    scratch.file(
        "foo/rule.bzl",
        """
        result = provider()

        def _impl(ctx):
            return [result(strict_java_deps = ctx.fragments.java.strict_java_deps)]

        myrule = rule(
            implementation = _impl,
            fragments = ["java"],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":rule.bzl", "myrule")

        myrule(name = "myrule")
        """);
    useConfiguration("--strict_java_deps=ERROR");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:myrule");
    StructImpl info =
        (StructImpl)
            configuredTarget.get(
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//foo:rule.bzl")), "result"));
    assertThat(((String) info.getValue("strict_java_deps"))).isEqualTo("error");
  }

  @Test
  public void mergeRuntimeOutputJarsTest() throws Exception {
    scratch.file(
        "foo/custom_library.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common")
        load("@rules_java//java/common:java_info.bzl", "JavaInfo")
        def _impl(ctx):
            java_provider = java_common.merge([dep[JavaInfo] for dep in ctx.attr.deps])
            return [java_provider]

        custom_library = rule(
            attrs = {
                "deps": attr.label_list(),
                "strict_deps": attr.bool(),
            },
            implementation = _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_library.bzl", "custom_library")
        load("@rules_java//java:defs.bzl", "java_library")

        custom_library(
            name = "custom",
            deps = [
                ":a",
                ":b",
            ],
        )

        java_library(
            name = "a",
            srcs = ["java/A.java"],
        )

        java_library(
            name = "b",
            srcs = ["java/B.java"],
        )
        """);

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:custom");
    JavaInfo javaInfo = JavaInfo.getJavaInfo(myRuleTarget);
    List<String> directJars = prettyArtifactNames(javaInfo.getRuntimeOutputJars());
    assertThat(directJars).containsExactly("foo/liba.jar", "foo/libb.jar");
  }

  @Test
  public void javaToolchainFlag_default() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/rule.bzl",
        """
        result = provider()

        def _impl(ctx):
            return [result(java_toolchain_label = ctx.attr._java_toolchain)]

        myrule = rule(
            implementation = _impl,
            fragments = ["java"],
            attrs = {"_java_toolchain": attr.label(default = Label("//foo:alias"))},
        )
        """);
    scratch.file(
        "foo/BUILD",
        "load(':rule.bzl', 'myrule')",
        "load('"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:java_toolchain_alias.bzl', 'java_toolchain_alias')",
        "java_toolchain_alias(name='alias')",
        "myrule(name='myrule')");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:myrule");
    StructImpl info =
        (StructImpl)
            configuredTarget.get(
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//foo:rule.bzl")), "result"));
    JavaToolchainProvider javaToolchainProvider =
        JavaToolchainProvider.from((ConfiguredTarget) info.getValue("java_toolchain_label"));
    Label javaToolchainLabel = javaToolchainProvider.getToolchainLabel();
    assertWithMessage(javaToolchainLabel.toString())
        .that(
            javaToolchainLabel.toString().endsWith("jdk:remote_toolchain")
                || javaToolchainLabel.toString().endsWith("jdk:toolchain")
                || javaToolchainLabel.toString().endsWith("jdk:toolchain_host"))
        .isTrue();
  }

  @Test
  public void javaToolchainFlag_set() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/rule.bzl",
        """
        result = provider()

        def _impl(ctx):
            return [result(java_toolchain_label = ctx.attr._java_toolchain)]

        myrule = rule(
            implementation = _impl,
            fragments = ["java"],
            attrs = {"_java_toolchain": attr.label(default = Label("//foo:alias"))},
        )
        """);
    scratch.file(
        "foo/BUILD",
        "load(':rule.bzl', 'myrule')",
        "load('"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:java_toolchain_alias.bzl', 'java_toolchain_alias')",
        "java_toolchain_alias(name='alias')",
        "myrule(name='myrule')");
    useConfiguration("--extra_toolchains=//java/com/google/test:all");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:myrule");
    StructImpl info =
        (StructImpl)
            configuredTarget.get(
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//foo:rule.bzl")), "result"));
    JavaToolchainProvider javaToolchainProvider =
        JavaToolchainProvider.from((ConfiguredTarget) info.getValue("java_toolchain_label"));
    Label javaToolchainLabel = javaToolchainProvider.getToolchainLabel();
    assertThat(javaToolchainLabel.toString()).isEqualTo("//java/com/google/test:toolchain");
  }

  @Test
  public void testCompileOutputJarHasManifestProto() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/java_custom_library.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib%s.jar' % ctx.label.name)",
        "  compilation_provider = java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "  )",
        "  return [compilation_provider]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=['.java']),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java'],",
        ")");
    scratch.file(
        "foo/BUILD",
        """
        load(":java_custom_library.bzl", "java_custom_library")

        java_custom_library(
            name = "b",
            srcs = ["java/B.java"],
        )
        """);

    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:b");
    JavaInfo info = JavaInfo.getJavaInfo(configuredTarget);
    ImmutableList<JavaOutput> javaOutputs = info.getJavaOutputs();
    assertThat(javaOutputs).hasSize(1);
    JavaOutput output = javaOutputs.get(0);
    assertThat(output.manifestProto().getFilename()).isEqualTo("libb.jar_manifest_proto");
  }

  @Test
  public void testCompileWithNeverlinkDeps() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/java_custom_library.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common', 'JavaInfo')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib%s.jar' % ctx.label.name)",
        "  deps = [deps[JavaInfo] for deps in ctx.attr.deps]",
        "  compilation_provider = java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    deps = deps,",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "  )",
        "  return [compilation_provider]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=['.java']),",
        "    'deps': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java'],",
        ")");
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":java_custom_library.bzl", "java_custom_library")

        java_library(
            name = "b",
            srcs = ["java/B.java"],
            neverlink = 1,
        )

        java_custom_library(
            name = "a",
            srcs = ["java/A.java"],
            deps = [":b"],
        )
        """);

    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:a");
    JavaInfo info = JavaInfo.getJavaInfo(configuredTarget);
    assertThat(artifactFilesNames(info.getTransitiveRuntimeJars().toList(Artifact.class)))
        .containsExactly("liba.jar");
    assertThat(artifactFilesNames(info.getTransitiveSourceJars().getSet(Artifact.class)))
        .containsExactly("liba-src.jar", "libb-src.jar");
    assertThat(artifactFilesNames(info.getTransitiveCompileTimeJars().toList(Artifact.class)))
        .containsExactly("liba-hjar.jar", "libb-hjar.jar");
  }

  @Test
  public void testCompileOutputJarNotInRuntimePathWithoutAnySourcesDefined() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/java_custom_library.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common', 'JavaInfo')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib%s.jar' % ctx.label.name)",
        "  exports = [export[JavaInfo] for export in ctx.attr.exports]",
        "  compilation_provider = java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    exports = exports,",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "  )",
        "  return [compilation_provider]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=['.java']),",
        "    'exports': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java'],",
        ")");
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":java_custom_library.bzl", "java_custom_library")

        java_library(
            name = "b",
            srcs = ["java/B.java"],
        )

        java_custom_library(
            name = "c",
            srcs = [],
            exports = [":b"],
        )
        """);

    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:c");
    JavaInfo info = JavaInfo.getJavaInfo(configuredTarget);
    assertThat(artifactFilesNames(info.getTransitiveRuntimeJars().toList(Artifact.class)))
        .containsExactly("libb.jar");
    assertThat(artifactFilesNames(info.getTransitiveCompileTimeJars().toList(Artifact.class)))
        .containsExactly("libb-hjar.jar");
    ImmutableList<JavaOutput> javaOutputs = info.getJavaOutputs();
    assertThat(javaOutputs).hasSize(1);
    JavaOutput output = javaOutputs.get(0);
    assertThat(output.classJar().getFilename()).isEqualTo("libc.jar");
    assertThat(output.compileJar()).isNull();
  }

  @Test
  public void testConfiguredTargetToolchain() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);

    scratch.file(
        "a/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_runtime")
        load(":rule.bzl", "jrule")

        java_runtime(
            name = "jvm",
            srcs = [],
            java_home = "/foo/bar",
        )

        jrule(
            name = "r",
            srcs = ["S.java"],
        )
        """);

    scratch.file(
        "a/rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib' + ctx.label.name + '.jar')",
        "  java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    java_toolchain = ctx.attr._java_toolchain[platform_common.ToolchainInfo],",
        "  )",
        "  return []",
        "jrule = rule(",
        "  implementation = _impl,",
        "  outputs = {",
        "    'my_output': 'lib%{name}.jar'",
        "  },",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=['.java']),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java'])");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:r");
    assertContainsEvent("got element of type ToolchainInfo, want JavaToolchainInfo");
  }

  @Test
  public void defaultJavacOpts() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "a/rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common")
        load("//myinfo:myinfo.bzl", "MyInfo")

        def _impl(ctx):
            return MyInfo(
                javac_opts = java_common.default_javac_opts(
                    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
                ),
            )

        get_javac_opts = rule(
            _impl,
            attrs = {
                "_java_toolchain": attr.label(default = Label("//java/com/google/test:toolchain")),
            },
        )
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":rule.bzl", "get_javac_opts")

        get_javac_opts(name = "r")
        """);

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked") // Use an extra variable in order to suppress the warning.
    Sequence<String> javacopts = (Sequence<String>) getMyInfoFromTarget(r).getValue("javac_opts");
    assertThat(String.join(" ", javacopts)).contains("-source 6 -target 6");
  }

  @Test
  public void defaultJavacOpts_asDepset() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "a/rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common")
        load("//myinfo:myinfo.bzl", "MyInfo")

        def _impl(ctx):
            return MyInfo(
                javac_opts = java_common.default_javac_opts_depset(
                    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
                ),
            )

        get_javac_opts = rule(
            _impl,
            attrs = {
                "_java_toolchain": attr.label(default = Label("//java/com/google/test:toolchain")),
            },
        )
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":rule.bzl", "get_javac_opts")

        get_javac_opts(name = "r")
        """);

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    NestedSet<String> javacopts =
        Depset.cast(getMyInfoFromTarget(r).getValue("javac_opts"), String.class, "javac_opts");

    assertThat(String.join(" ", javacopts.toList())).contains("-source 6 -target 6");
  }

  @Test
  public void defaultJavacOpts_toolchainProvider() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "a/rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common")
        load("//myinfo:myinfo.bzl", "MyInfo")

        def _impl(ctx):
            return MyInfo(
                javac_opts = java_common.default_javac_opts(
                    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
                ),
            )

        get_javac_opts = rule(
            _impl,
            attrs = {
                "_java_toolchain": attr.label(default = Label("//java/com/google/test:toolchain")),
            },
        )
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":rule.bzl", "get_javac_opts")

        get_javac_opts(name = "r")
        """);

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked") // Use an extra variable in order to suppress the warning.
    Sequence<String> javacopts = (Sequence<String>) getMyInfoFromTarget(r).getValue("javac_opts");
    assertThat(String.join(" ", javacopts)).contains("-source 6 -target 6");
  }

  @Test
  public void testJavaRuntimeProviderFiles() throws Exception {
    scratch.file("a/a.txt", "hello");
    scratch.file(
        "a/BUILD",
        "load('@rules_java//java:defs.bzl', 'java_runtime')",
        "load(':rule.bzl', 'jrule')",
        "load('"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:java_toolchain_alias.bzl', 'java_runtime_alias')",
        "java_runtime(name='jvm', srcs=['a.txt'], java_home='foo/bar')",
        "java_runtime_alias(name='alias')",
        "jrule(name='r')",
        "toolchain(",
        "    name = 'java_runtime_toolchain',",
        "    toolchain = ':jvm',",
        "    toolchain_type = '"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:runtime_toolchain_type',",
        ")");

    scratch.file(
        "a/rule.bzl",
        """
        load("@rules_java//java/common:java_common.bzl", "java_common")
        def _impl(ctx):
            provider = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
            return DefaultInfo(
                files = provider.files,
            )

        jrule = rule(_impl, attrs = {"_java_runtime": attr.label(default = Label("//a:alias"))})
        """);

    useConfiguration("--extra_toolchains=//a:all");
    ConfiguredTarget ct = getConfiguredTarget("//a:r");
    Depset files = (Depset) ct.get("files");
    assertThat(prettyArtifactNames(files.toList(Artifact.class))).containsExactly("a/a.txt");
  }

  @Test
  public void testJavaLibraryCollectsCoverageDependenciesFromResources() throws Exception {
    useConfiguration("--collect_code_coverage");

    scratch.file(
        "java/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "lib",
            resources = [":libjni.so"],
        )

        cc_binary(
            name = "libjni.so",
            srcs = ["jni.cc"],
            linkshared = 1,
        )
        """);

    InstrumentedFilesInfo target = getInstrumentedFilesProvider("//java:lib");

    assertThat(prettyArtifactNames(target.getInstrumentedFiles())).containsExactly("java/jni.cc");
    assertThat(prettyArtifactNames(target.getInstrumentationMetadataFiles()))
        .containsExactly(
            "java/libjni.soruntime_objects_list.txt",
            "java/libjni.so",
            "java/_objs/libjni.so/jni.gcno");
  }

  @Test
  public void testSkipAnnotationProcessing() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common', 'JavaInfo',"
            + " 'JavaPluginInfo')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file('lib' + ctx.label.name + '.jar')",
        "  compilation_provider = java_common.compile(",
        "    ctx,",
        "    source_files = ctx.files.srcs,",
        "    output = output_jar,",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "    deps = [p[JavaInfo] for p in ctx.attr.deps],",
        "    plugins = [p[JavaPluginInfo] for p in ctx.attr.plugins],",
        "    enable_annotation_processing = False,",
        "  )",
        "  return [DefaultInfo(files = depset([output_jar])), compilation_provider]",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  outputs = {",
        "    'my_output': 'lib%{name}.jar'",
        "  },",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True),",
        "    'deps': attr.label_list(providers=[JavaInfo]),",
        "    'plugins': attr.label_list(providers=[JavaPluginInfo]),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java']",
        ")");
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")
        load("@rules_java//java:defs.bzl", "java_library", "java_plugin")

        java_plugin(
            name = "processor",
            srcs = ["processor.java"],
            data = ["processor_data.txt"],
            generates_api = 1,  # so Turbine would normally run it
            processor_class = "Foo",
        )

        java_library(
            name = "exports_processor",
            exported_plugins = [":processor"],
        )

        java_custom_library(
            name = "custom",
            srcs = ["custom.java"],
            plugins = [":processor"],
            deps = [":exports_processor"],
        )

        java_custom_library(
            name = "custom_noproc",
            srcs = ["custom.java"],
        )
        """);

    ConfiguredTarget custom = getConfiguredTarget("//foo:custom");
    ConfiguredTarget customNoproc = getConfiguredTarget("//foo:custom_noproc");
    assertNoEvents();

    JavaCompileAction javacAction =
        (JavaCompileAction) getGeneratingActionForLabel("//foo:libcustom.jar");
    assertThat(javacAction.getMnemonic()).isEqualTo("Javac");
    assertThat(getProcessorNames(javacAction)).isEmpty();
    assertThat(getProcessorPath(javacAction)).isNotEmpty();
    assertThat(artifactFilesNames(javacAction.getInputs())).contains("processor_data.txt");

    JavaCompileAction turbineAction =
        (JavaCompileAction) getGeneratingAction(getBinArtifact("libcustom-hjar.jar", custom));
    assertThat(turbineAction.getMnemonic()).isEqualTo("JavacTurbine");
    ImmutableList<String> args = turbineAction.getArguments();
    assertThat(args).doesNotContain("--processors");

    // enable_annotation_processing=False shouldn't disable direct classpaths if there are no
    // annotation processors that need to be disabled
    SpawnAction turbineActionNoProc =
        (SpawnAction)
            getGeneratingAction(getBinArtifact("libcustom_noproc-hjar.jar", customNoproc));
    assertThat(turbineActionNoProc.getMnemonic()).isEqualTo("Turbine");
    assertThat(turbineActionNoProc.getArguments()).doesNotContain("--processors");
  }

  @Test
  public void testCompileWithDisablingCompileJarIsPrivateApi() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_common.compile(",
        "    ctx,",
        "    output = ctx.actions.declare_file('output.jar'),",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "    enable_compile_jar_action = False,",
        "  )",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
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

    assertContainsEvent("got unexpected keyword argument: enable_compile_jar_action");
  }

  @Test
  public void testCompileWithClasspathResourcesIsPrivateApi() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file("foo/resource.txt", "Totally real resource content");
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_common.compile(",
        "    ctx,",
        "    output = ctx.actions.declare_file('output.jar'),",
        "    classpath_resources = ctx.files.classpath_resources,",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "  )",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(),",
        "    'classpath_resources': attr.label_list(allow_files = True),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        "  fragments = ['java']",
        ")");
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(
            name = "custom",
            classpath_resources = ["resource.txt"],
        )
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:custom");

    assertContainsEvent("got unexpected keyword argument: classpath_resources");
  }

  @Test
  public void testInjectingRuleKindIsPrivateApi() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_common.compile(",
        "    ctx,",
        "    output = ctx.actions.declare_file('output.jar'),",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "    injecting_rule_kind = 'example_rule_kind',",
        "  )",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
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

    assertContainsEvent("got unexpected keyword argument: injecting_rule_kind");
  }

  @Test
  public void testEnableJSpecifyIsPrivateApi() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_common.compile(",
        "    ctx,",
        "    output = ctx.actions.declare_file('output.jar'),",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "    enable_jspecify = False,",
        "  )",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
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

    assertContainsEvent("got unexpected keyword argument: enable_jspecify");
  }

  @Test
  public void testMergeJavaOutputsIsPrivateApi() throws Exception {
    scratch.file(
        "foo/custom_rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common", "JavaInfo")
        def _impl(ctx):
            output_jar = ctx.actions.declare_file("lib.jar")
            java_info = JavaInfo(output_jar = output_jar, compile_jar = None)
            java_common.merge(
                [java_info],
                merge_java_outputs = False,
            )
            return []

        java_custom_library = rule(
            implementation = _impl,
            fragments = ["java"],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(name = "custom")
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:custom");

    assertContainsEvent("got unexpected keyword argument: merge_java_outputs");
  }

  @Test
  public void testMergeSourceJarsIsPrivateApi() throws Exception {
    scratch.file(
        "foo/custom_rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common", "JavaInfo")
        def _impl(ctx):
            output_jar = ctx.actions.declare_file("lib.jar")
            java_info = JavaInfo(output_jar = output_jar, compile_jar = None)
            java_common.merge(
                [java_info],
                merge_source_jars = False,
            )
            return []

        java_custom_library = rule(
            implementation = _impl,
            fragments = ["java"],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "java_custom_library")

        java_custom_library(name = "custom")
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:custom");

    assertContainsEvent("got unexpected keyword argument: merge_source_jars");
  }

  @Test
  public void testCompileIncludeCompilationInfoIsPrivateApi() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_common.compile(",
        "    ctx,",
        "    output = ctx.actions.declare_file('output.jar'),",
        "    java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],",
        "    include_compilation_info = False,",
        "  )",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
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

    assertContainsEvent("got unexpected keyword argument: include_compilation_info");
  }

  @Test
  public void testCompileWithResourceJarsIsPrivateApi() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo]",
        "  java_common.compile(",
        "    ctx,",
        "    output = ctx.actions.declare_file('output.jar'),",
        "    java_toolchain = java_toolchain,",
        "    resource_jars = ['foo.jar'],",
        "  )",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(),",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
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

    assertContainsEvent("got unexpected keyword argument: resource_jars");
  }

  @Test
  public void testRunIjarWithOutputParameterIsPrivateApi() throws Exception {
    JavaTestUtil.writeBuildFileForJavaToolchain(scratch);
    scratch.file(
        "foo/custom_rule.bzl",
        "load('@rules_java//java:defs.bzl', 'java_common')",
        "def _impl(ctx):",
        "  java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo]",
        "  java_common.run_ijar(",
        "    ctx.actions,",
        "    jar = ctx.actions.declare_file('input.jar'),",
        "    output = ctx.actions.declare_file('output.jar'),",
        "    java_toolchain = java_toolchain,",
        "  )",
        "  return []",
        "java_custom_library = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    '_java_toolchain': attr.label(default = Label('//java/com/google/test:toolchain')),",
        "  },",
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

    assertContainsEvent("got unexpected keyword argument: output");
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

  @Test
  public void mergeAddExports() throws Exception {
    scratch.file(
        "foo/custom_library.bzl",
        """
        load("@rules_java//java:defs.bzl", "java_common", "JavaInfo")
        def _impl(ctx):
            java_provider = java_common.merge([dep[JavaInfo] for dep in ctx.attr.deps])
            return [java_provider]

        custom_library = rule(
            attrs = {
                "deps": attr.label_list(),
            },
            implementation = _impl,
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":custom_library.bzl", "custom_library")

        custom_library(
            name = "custom",
            deps = [":a"],
        )

        java_library(
            name = "a",
            srcs = ["java/A.java"],
            add_exports = ["java.base/java.lang"],
        )
        """);

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:custom");
    JavaModuleFlagsProvider provider =
        JavaInfo.getProvider(JavaModuleFlagsProvider.class, myRuleTarget);
    assertThat(provider.toFlags()).containsExactly("--add-exports=java.base/java.lang=ALL-UNNAMED");
  }

  private InstrumentedFilesInfo getInstrumentedFilesProvider(String label) throws Exception {
    return getConfiguredTarget(label).get(InstrumentedFilesInfo.STARLARK_CONSTRUCTOR);
  }

  @Test
  public void hermeticStaticLibs() throws Exception {
    scratch.file("a/libStatic.a");
    scratch.file(
        "a/BUILD",
        "load('@rules_java//java:defs.bzl', 'java_runtime')",
        "load(':rule.bzl', 'jrule')",
        "load('"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:java_toolchain_alias.bzl', 'java_runtime_alias')",
        "genrule(name='gen', cmd='', outs=['foo/bar/bin/java'])",
        "cc_import(name='libs', static_library = 'libStatic.a')",
        "cc_library(name = 'jdk_static_libs00', data = ['libStatic.a'], linkstatic = 1)",
        "java_runtime(name='jvm', srcs=[], java='foo/bar/bin/java', lib_modules='lib/modules', "
            + "hermetic_srcs = ['lib/hermetic.properties'], hermetic_static_libs = ['libs'])",
        "java_runtime_alias(name='alias')",
        "jrule(name='r')",
        "toolchain(",
        "    name = 'java_runtime_toolchain',",
        "    toolchain = ':jvm',",
        "    toolchain_type = '"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:runtime_toolchain_type',",
        ")");
    scratch.file(
        "a/rule.bzl",
        """
        load("//myinfo:myinfo.bzl", "MyInfo")
        load("@rules_java//java/common:java_common.bzl", "java_common")

        def _impl(ctx):
            provider = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
            return MyInfo(
                hermetic_static_libs = provider.hermetic_static_libs,
            )

        jrule = rule(_impl, attrs = {"_java_runtime": attr.label(default = Label("//a:alias"))})
        """);

    useConfiguration("--extra_toolchains=//a:all");
    ConfiguredTarget ct = getConfiguredTarget("//a:r");
    StructImpl myInfo = getMyInfoFromTarget(ct);
    @SuppressWarnings("unchecked")
    Sequence<CcInfo> hermeticStaticLibs =
        (Sequence<CcInfo>) myInfo.getValue("hermetic_static_libs");
    assertThat(hermeticStaticLibs).hasSize(1);
    assertThat(
            hermeticStaticLibs.get(0).getCcLinkingContext().getLibraries().toList().stream()
                .map(lib -> lib.getStaticLibrary().prettyPrint()))
        .containsExactly("a/libStatic.a");
  }

  @Test
  public void implicitLibCtSym() throws Exception {
    scratch.file("a/libStatic.a");
    scratch.file(
        "a/BUILD",
        "load('@rules_java//java:defs.bzl', 'java_runtime')",
        "load(':rule.bzl', 'jrule')",
        "load('"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:java_toolchain_alias.bzl', 'java_runtime_alias')",
        "java_runtime(",
        "    name='jvm',",
        "    srcs=[",
        "        'foo/bar/bin/java',",
        "        'foo/bar/lib/ct.sym',",
        "    ],",
        "    java='foo/bar/bin/java',",
        ")",
        "java_runtime_alias(name='alias')",
        "jrule(name='r')",
        "toolchain(",
        "    name = 'java_runtime_toolchain',",
        "    toolchain = ':jvm',",
        "    toolchain_type = '"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:runtime_toolchain_type',",
        ")");
    scratch.file(
        "a/rule.bzl",
        """
        load("//myinfo:myinfo.bzl", "MyInfo")
        load("@rules_java//java/common:java_common.bzl", "java_common")

        def _impl(ctx):
            provider = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
            return MyInfo(
                lib_ct_sym = provider.lib_ct_sym,
            )

        jrule = rule(_impl, attrs = {"_java_runtime": attr.label(default = Label("//a:alias"))})
        """);

    useConfiguration("--extra_toolchains=//a:all");
    ConfiguredTarget ct = getConfiguredTarget("//a:r");
    StructImpl myInfo = getMyInfoFromTarget(ct);
    Artifact libCtSym = myInfo.getValue("lib_ct_sym", Artifact.class);
    assertThat(libCtSym).isNotNull();
    assertThat(libCtSym.getExecPathString()).isEqualTo("a/foo/bar/lib/ct.sym");
  }

  @Test
  public void explicitLibCtSym() throws Exception {
    scratch.file("a/libStatic.a");
    scratch.file(
        "a/BUILD",
        "load('@rules_java//java:defs.bzl', 'java_runtime')",
        "load(':rule.bzl', 'jrule')",
        "load('"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:java_toolchain_alias.bzl', 'java_runtime_alias')",
        "java_runtime(",
        "    name='jvm',",
        "    srcs=[",
        "        'foo/bar/bin/java',",
        "        'foo/bar/lib/ct.sym',",
        "    ],",
        "    java='foo/bar/bin/java',",
        "    lib_ct_sym='lib/ct.sym',",
        ")",
        "java_runtime_alias(name='alias')",
        "jrule(name='r')",
        "toolchain(",
        "    name = 'java_runtime_toolchain',",
        "    toolchain = ':jvm',",
        "    toolchain_type = '"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/jdk:runtime_toolchain_type',",
        ")");
    scratch.file(
        "a/rule.bzl",
        """
        load("//myinfo:myinfo.bzl", "MyInfo")
        load("@rules_java//java/common:java_common.bzl", "java_common")

        def _impl(ctx):
            provider = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
            return MyInfo(
                lib_ct_sym = provider.lib_ct_sym,
            )

        jrule = rule(_impl, attrs = {"_java_runtime": attr.label(default = Label("//a:alias"))})
        """);

    useConfiguration("--extra_toolchains=//a:all");
    ConfiguredTarget ct = getConfiguredTarget("//a:r");
    StructImpl myInfo = getMyInfoFromTarget(ct);
    Artifact libCtSym = myInfo.getValue("lib_ct_sym", Artifact.class);
    assertThat(libCtSym).isNotNull();
    assertThat(libCtSym.getExecPathString()).isEqualTo("a/lib/ct.sym");
  }

  @Test
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

  @Test
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

  @Test
  public void testProviderValidationPrintsProviderName() throws Exception {
    scratch.file(
        "foo/rule.bzl",
        """
        load("@rules_java//java/common:java_info.bzl", "JavaInfo")
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

  @Test
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
