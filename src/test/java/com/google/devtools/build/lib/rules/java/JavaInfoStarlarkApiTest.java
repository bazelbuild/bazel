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
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Map;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests JavaInfo API for Starlark. */
@RunWith(JUnit4.class)
public class JavaInfoStarlarkApiTest extends BuildViewTestCase {

  @Test
  public void buildHelperCreateJavaInfoWithModuleFlags() throws Exception {
    setBuildLanguageOptions("--noincompatible_java_info_merge_runtime_module_flags");
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":extension.bzl", "my_rule")

        java_library(
            name = "my_java_lib_direct",
            srcs = ["java/A.java"],
            add_exports = ["java.base/java.lang"],
            add_opens = ["java.base/java.lang"],
        )

        java_library(
            name = "my_java_lib_runtime",
            srcs = ["java/A.java"],
            add_opens = ["java.base/java.util"],
        )

        java_library(
            name = "my_java_lib_exports",
            srcs = ["java/A.java"],
            add_opens = ["java.base/java.math"],
        )

        my_rule(
            name = "my_starlark_rule",
            add_exports = ["java.base/java.lang.invoke"],
            dep = [":my_java_lib_direct"],
            dep_exports = [":my_java_lib_exports"],
            dep_runtime = [":my_java_lib_runtime"],
            output_jar = "my_starlark_rule_lib.jar",
        )
        """);
    assertNoEvents();

    JavaModuleFlagsProvider ruleOutputs =
        fetchJavaInfo().getProvider(JavaModuleFlagsProvider.class);

    if (analysisMock.isThisBazel()) {
      assertThat(ruleOutputs.toFlags())
          .containsExactly(
              "--add-exports=java.base/java.lang=ALL-UNNAMED",
              "--add-exports=java.base/java.lang.invoke=ALL-UNNAMED",
              // NB: no java.base/java.util as the JavaInfo constructor doesn't
              // look at runtime_deps for module flags.
              "--add-opens=java.base/java.lang=ALL-UNNAMED",
              "--add-opens=java.base/java.math=ALL-UNNAMED")
          .inOrder();
    } else {
      // add_exports/add_opens ignored in JavaInfo constructor in #newJavaInfo below
      assertThat(ruleOutputs.toFlags())
          .containsExactly(
              "--add-exports=java.base/java.lang=ALL-UNNAMED",
              // NB: no java.base/java.util as the JavaInfo constructor doesn't
              // look at runtime_deps for module flags.
              "--add-opens=java.base/java.lang=ALL-UNNAMED",
              "--add-opens=java.base/java.math=ALL-UNNAMED")
          .inOrder();
    }
  }

  @Test
  public void buildHelperCreateJavaInfoWithModuleFlagsIncompatibleMergeRuntime() throws Exception {
    setBuildLanguageOptions("--incompatible_java_info_merge_runtime_module_flags");
    ruleBuilder().build();
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load(":extension.bzl", "my_rule")

        java_library(
            name = "my_java_lib_direct",
            srcs = ["java/A.java"],
            add_exports = ["java.base/java.lang"],
            add_opens = ["java.base/java.lang"],
        )

        java_library(
            name = "my_java_lib_runtime",
            srcs = ["java/A.java"],
            add_opens = ["java.base/java.util"],
        )

        java_library(
            name = "my_java_lib_exports",
            srcs = ["java/A.java"],
            add_opens = ["java.base/java.math"],
        )

        my_rule(
            name = "my_starlark_rule",
            add_exports = ["java.base/java.lang.invoke"],
            dep = [":my_java_lib_direct"],
            dep_exports = [":my_java_lib_exports"],
            dep_runtime = [":my_java_lib_runtime"],
            output_jar = "my_starlark_rule_lib.jar",
        )
        """);
    assertNoEvents();

    JavaModuleFlagsProvider ruleOutputs =
        fetchJavaInfo().getProvider(JavaModuleFlagsProvider.class);

    if (analysisMock.isThisBazel()) {
      assertThat(ruleOutputs.toFlags())
          .containsExactly(
              "--add-exports=java.base/java.lang=ALL-UNNAMED",
              "--add-exports=java.base/java.lang.invoke=ALL-UNNAMED",
              "--add-opens=java.base/java.util=ALL-UNNAMED",
              "--add-opens=java.base/java.math=ALL-UNNAMED",
              "--add-opens=java.base/java.lang=ALL-UNNAMED")
          .inOrder();
    } else {
      // add_exports/add_opens ignored in JavaInfo constructor in #newJavaInfo below
      assertThat(ruleOutputs.toFlags())
          .containsExactly(
              "--add-exports=java.base/java.lang=ALL-UNNAMED",
              "--add-opens=java.base/java.util=ALL-UNNAMED",
              "--add-opens=java.base/java.math=ALL-UNNAMED",
              "--add-opens=java.base/java.lang=ALL-UNNAMED")
          .inOrder();
    }
  }

  @Test
  public void starlarkJavaOutputsCanBeAddedToJavaPluginInfo() throws Exception {
    Artifact classJar = createArtifact("foo.jar");
    StarlarkInfo starlarkJavaOutput =
        makeStruct(ImmutableMap.of("source_jars", Starlark.NONE, "class_jar", classJar));
    StarlarkInfo starlarkPluginInfo =
        makeStruct(
            ImmutableMap.of(
                "java_outputs", StarlarkList.immutableOf(starlarkJavaOutput),
                "plugins", JavaPluginData.empty(),
                "api_generating_plugins", JavaPluginData.empty()));

    JavaPluginInfo pluginInfo = JavaPluginInfo.wrap(starlarkPluginInfo);

    assertThat(pluginInfo).isNotNull();
    assertThat(pluginInfo.getJavaOutputs()).hasSize(1);
    assertThat(pluginInfo.getJavaOutputs().get(0).classJar()).isEqualTo(classJar);
  }

  @Test
  public void nativeAndStarlarkJavaOutputsCanBeAddedToADepset() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        """
        load("@rules_java//java/common:java_info.bzl", "JavaInfo")
        def _impl(ctx):
            f = ctx.actions.declare_file(ctx.label.name + ".jar")
            ctx.actions.write(f, "")
            return [JavaInfo(output_jar = f, compile_jar = None)]

        my_rule = rule(implementation = _impl)
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":extension.bzl", "my_rule")

        my_rule(name = "my_starlark_rule")
        """);
    JavaOutput nativeOutput =
        JavaOutput.builder().setClassJar(createArtifact("native.jar")).build();
    ImmutableList<JavaOutput> starlarkOutputs =
        JavaInfo.getJavaInfo(getConfiguredTarget("//foo:my_starlark_rule")).getJavaOutputs();

    Depset depset =
        Depset.fromDirectAndTransitive(
            Order.STABLE_ORDER,
            /* direct= */ ImmutableList.builder().add(nativeOutput).addAll(starlarkOutputs).build(),
            /* transitive= */ ImmutableList.of(),
            /* strict= */ true);

    assertThat(depset).isNotNull();
    assertThat(depset.toList()).hasSize(2);
  }

  @Test
  public void testNeverlinkIsStoredAsABoolean() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        """
        load("@rules_java//java/common:java_info.bzl", "JavaInfo")
        def _impl(ctx):
            f = ctx.actions.declare_file(ctx.label.name + ".jar")
            ctx.actions.write(f, "")
            return [JavaInfo(output_jar = f, compile_jar = None, neverlink = 1)]

        my_rule = rule(implementation = _impl)
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":extension.bzl", "my_rule")

        my_rule(name = "my_starlark_rule")
        """);

    JavaInfo javaInfo = JavaInfo.getJavaInfo(getConfiguredTarget("//foo:my_starlark_rule"));

    assertThat(javaInfo).isNotNull();
    assertThat(javaInfo.isNeverlink()).isTrue();
  }

  @Test
  public void translateStarlarkJavaInfo_minimal() throws Exception {
    ImmutableMap<String, Object> fields = getBuilderWithMandataryFields().buildOrThrow();
    StarlarkInfo starlarkInfo = makeStruct(fields);

    JavaInfo javaInfo = JavaInfo.wrap(starlarkInfo);

    assertThat(javaInfo).isNotNull();
    assertThat(javaInfo.getProvider(JavaCompilationArgsProvider.class)).isNotNull();
    assertThat(javaInfo.getCompilationInfoProvider()).isNull();
    assertThat(javaInfo.getJavaModuleFlagsInfo()).isEqualTo(JavaModuleFlagsProvider.EMPTY);
    assertThat(javaInfo.getJavaPluginInfo())
        .isEqualTo(JavaPluginInfo.empty(JavaPluginInfo.PROVIDER));
  }

  @Test
  public void translateStarlarkJavaInfo_binariesDoNotContainCompilationArgs() throws Exception {
    ImmutableMap<String, Object> fields =
        getBuilderWithMandataryFields().put("_is_binary", true).buildOrThrow();
    StarlarkInfo starlarkInfo = makeStruct(fields);

    JavaInfo javaInfo = JavaInfo.wrap(starlarkInfo);

    assertThat(javaInfo).isNotNull();
    assertThat(javaInfo.getProvider(JavaCompilationArgsProvider.class)).isNull();
  }

  @Test
  public void translateStarlarkJavaInfo_compilationInfo() throws Exception {
    ImmutableMap<String, Object> fields =
        getBuilderWithMandataryFields()
            .put(
                "compilation_info",
                makeStruct(
                    ImmutableMap.of(
                        "javac_options",
                            Depset.of(
                                String.class,
                                NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, "opt1", "opt2")),
                        "boot_classpath", StarlarkList.immutableOf(createArtifact("cp.jar")))))
            .buildOrThrow();
    StarlarkInfo starlarkInfo = makeStruct(fields);

    JavaInfo javaInfo = JavaInfo.wrap(starlarkInfo);

    assertThat(javaInfo).isNotNull();
    assertThat(javaInfo.getCompilationInfoProvider()).isNotNull();
    assertThat(javaInfo.getCompilationInfoProvider().getJavacOptsList())
        .containsExactly("opt1", "opt2");
    assertThat(javaInfo.getCompilationInfoProvider().getBootClasspathList()).hasSize(1);
    assertThat(prettyArtifactNames(javaInfo.getCompilationInfoProvider().getBootClasspathList()))
        .containsExactly("cp.jar");
  }

  @Test
  public void translatedStarlarkCompilationInfoEqualsNativeInstance() throws Exception {
    Artifact bootClasspathArtifact = createArtifact("boot.jar");
    NestedSet<Artifact> compilationClasspath =
        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, createArtifact("compile.jar"));
    NestedSet<Artifact> runtimeClasspath =
        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, createArtifact("runtime.jar"));
    StarlarkInfo starlarkInfo =
        makeStruct(
            ImmutableMap.of(
                "compilation_classpath", Depset.of(Artifact.class, compilationClasspath),
                "runtime_classpath", Depset.of(Artifact.class, runtimeClasspath),
                "javac_options",
                    Depset.of(
                        String.class,
                        NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, "opt1", "opt2")),
                "boot_classpath", StarlarkList.immutableOf(bootClasspathArtifact)));
    JavaCompilationInfoProvider nativeCompilationInfo =
        new JavaCompilationInfoProvider.Builder()
            .setCompilationClasspath(compilationClasspath)
            .setRuntimeClasspath(runtimeClasspath)
            .setJavacOpts(NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, "opt1", "opt2"))
            .setBootClasspath(
                NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, bootClasspathArtifact))
            .build();

    JavaCompilationInfoProvider starlarkCompilationInfo =
        JavaCompilationInfoProvider.fromStarlarkCompilationInfo(starlarkInfo);

    assertThat(starlarkCompilationInfo).isNotNull();
    assertThat(starlarkCompilationInfo).isEqualTo(nativeCompilationInfo);
  }

  @Test
  public void translateStarlarkJavaInfo_moduleFlagsInfo() throws Exception {
    ImmutableMap<String, Object> fields =
        getBuilderWithMandataryFields()
            .put(
                "module_flags_info",
                makeStruct(
                    ImmutableMap.of(
                        "add_exports", makeDepset(String.class, "export1", "export2"),
                        "add_opens", makeDepset(String.class, "open1", "open2"))))
            .buildOrThrow();
    StarlarkInfo starlarkInfo = makeStruct(fields);

    JavaInfo javaInfo = JavaInfo.wrap(starlarkInfo);

    assertThat(javaInfo).isNotNull();
    assertThat(javaInfo.getJavaModuleFlagsInfo()).isNotNull();
    assertThat(javaInfo.getJavaModuleFlagsInfo().getAddExports().toList())
        .containsExactly("export1", "export2");
    assertThat(javaInfo.getJavaModuleFlagsInfo().getAddOpens().toList())
        .containsExactly("open1", "open2");
  }

  @Test
  public void translateStarlarkJavaInfo_pluginInfo() throws Exception {
    ImmutableMap<String, Object> fields =
        getBuilderWithMandataryFields()
            .put(
                "plugins",
                JavaPluginData.create(
                    NestedSetBuilder.create(Order.STABLE_ORDER, "c1", "c2", "c3"),
                    NestedSetBuilder.create(Order.STABLE_ORDER, createArtifact("f1")),
                    NestedSetBuilder.emptySet(Order.STABLE_ORDER)))
            .buildKeepingLast();
    StarlarkInfo starlarkInfo = makeStruct(fields);

    JavaInfo javaInfo = JavaInfo.wrap(starlarkInfo);

    assertThat(javaInfo).isNotNull();
    assertThat(javaInfo.plugins()).isNotNull();
    assertThat(javaInfo.plugins().processorClasses().toList()).containsExactly("c1", "c2", "c3");
    assertThat(prettyArtifactNames(javaInfo.plugins().processorClasspath())).containsExactly("f1");
  }

  private static ImmutableMap.Builder<String, Object> getBuilderWithMandataryFields() {
    Depset emptyDepset = Depset.of(Artifact.class, NestedSetBuilder.create(Order.STABLE_ORDER));
    return ImmutableMap.<String, Object>builder()
        .put("transitive_native_libraries", emptyDepset)
        .put("compile_jars", emptyDepset)
        .put("full_compile_jars", emptyDepset)
        .put("transitive_compile_time_jars", emptyDepset)
        .put("transitive_runtime_jars", emptyDepset)
        .put("_transitive_full_compile_time_jars", emptyDepset)
        .put("_compile_time_java_dependencies", emptyDepset)
        .put("header_compilation_direct_deps", emptyDepset)
        .put("plugins", JavaPluginData.empty())
        .put("api_generating_plugins", JavaPluginData.empty())
        .put("java_outputs", StarlarkList.empty())
        .put("transitive_source_jars", emptyDepset)
        .put("source_jars", StarlarkList.empty())
        .put("runtime_output_jars", StarlarkList.empty());
  }

  private Artifact createArtifact(String path) throws IOException {
    Path execRoot = scratch.dir("/");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "fake-root");
    return ActionsTestUtil.createArtifact(root, path);
  }

  private static <T> Depset makeDepset(Class<T> clazz, T... elems) {
    return Depset.of(clazz, NestedSetBuilder.create(Order.STABLE_ORDER, elems));
  }

  private static StarlarkInfo makeStruct(Map<String, Object> struct) {
    return StructProvider.STRUCT.create(struct, "");
  }

  private RuleBuilder ruleBuilder() {
    return new RuleBuilder();
  }

  private class RuleBuilder {
    private String[] newJavaInfo() {
      ImmutableList.Builder<String> lines = ImmutableList.builder();
      lines.add(
          "load('@rules_java//java:defs.bzl', 'java_common', 'JavaInfo',"
              + " 'JavaPluginInfo')",
          "result = provider()",
          "def _impl(ctx):",
          "  ctx.actions.write(ctx.outputs.output_jar, 'JavaInfo API Test', is_executable=False) ",
          "  dp = [dep[java_common.provider] for dep in ctx.attr.dep]",
          "  dp_runtime = [dep[java_common.provider] for dep in ctx.attr.dep_runtime]",
          "  dp_exports = [dep[java_common.provider] for dep in ctx.attr.dep_exports]",
          "  dp_exported_plugins = [dep[JavaPluginInfo] for dep in ctx.attr.dep_exported_plugins]",
          "  dp_libs = [dep[CcInfo] for dep in ctx.attr.cc_dep]",
          "  compile_jar = ctx.outputs.output_jar",
          "  if ctx.files.source_jars:",
          "    source_jar = list(ctx.files.source_jars)[0]",
          "  else:",
          "    source_jar = None",
          "  javaInfo = JavaInfo(",
          "    output_jar = ctx.outputs.output_jar,",
          "    compile_jar = compile_jar,",
          "    source_jar = source_jar,",
          "    deps = dp,",
          "    runtime_deps = dp_runtime,",
          "    exports = dp_exports,",
          "    exported_plugins = dp_exported_plugins,",
          "    jdeps = ctx.file.jdeps,",
          "    compile_jdeps = ctx.file.compile_jdeps,",
          "    generated_class_jar = ctx.file.generated_class_jar,",
          "    generated_source_jar = ctx.file.generated_source_jar,",
          "    native_headers_jar = ctx.file.native_headers_jar,",
          "    manifest_proto = ctx.file.manifest_proto,",
          "    native_libraries = dp_libs,");
      if (analysisMock.isThisBazel()) {
        lines.add(
            "    add_exports = ctx.attr.add_exports,", //
            "    add_opens = ctx.attr.add_opens,");
      }
      lines.add(
          "  )", //
          "  return [result(property = javaInfo)]");
      return lines.build().toArray(new String[] {});
    }

    private void build() throws Exception {
      ImmutableList.Builder<String> lines = ImmutableList.builder();
      lines.add(newJavaInfo());
      lines.add(
          "my_rule = rule(",
          "  implementation = _impl,",
          "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
          "  attrs = {",
          "    'dep' : attr.label_list(),",
          "    'cc_dep' : attr.label_list(),",
          "    'dep_runtime' : attr.label_list(),",
          "    'dep_exports' : attr.label_list(),",
          "    'dep_exported_plugins' : attr.label_list(),",
          "    'output_jar' : attr.output(mandatory=True),",
          "    'source_jars' : attr.label_list(allow_files=['.jar']),",
          "    'sources' : attr.label_list(allow_files=['.java']),",
          "    'jdeps' : attr.label(allow_single_file=True),",
          "    'compile_jdeps' : attr.label(allow_single_file=True),",
          "    'generated_class_jar' : attr.label(allow_single_file=True),",
          "    'generated_source_jar' : attr.label(allow_single_file=True),",
          "    'native_headers_jar' : attr.label(allow_single_file=True),",
          "    'manifest_proto' : attr.label(allow_single_file=True),",
          "    'add_exports' : attr.string_list(),",
          "    'add_opens' : attr.string_list(),",
          "  }",
          ")");

      scratch.file("foo/extension.bzl", lines.build().toArray(new String[] {}));
    }
  }

  private JavaInfo fetchJavaInfo() throws Exception {
    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:my_starlark_rule");
    StructImpl info =
        (StructImpl)
            myRuleTarget.get(
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//foo:extension.bzl")), "result"));

    return JavaInfo.wrap(info.getValue("property", Info.class));
  }
}
