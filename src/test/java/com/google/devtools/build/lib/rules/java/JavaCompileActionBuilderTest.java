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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.collect.MoreCollectors.onlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getDirectJars;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getJavacArguments;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.devtools.build.lib.view.proto.Deps.Dependency.Kind;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link JavaCompileActionBuilder}. */
@RunWith(JUnit4.class)
public final class JavaCompileActionBuilderTest extends BuildViewTestCase {

  @Test
  public void testClassdirIsInBlazeOut() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_binary")
        java_binary(
            name = "a",
            srcs = ["a.java"],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:a.jar");
    List<String> command = new ArrayList<>();
    command.addAll(getJavacArguments(action));
    MoreAsserts.assertContainsSublist(
        command,
        "--output",
        targetConfig
            .getBinFragment(RepositoryName.MAIN)
            .getRelative("java/com/google/test/a.jar")
            .getPathString());
  }

  @Test
  public void progressMessage() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = [
                "a.java",
                "b.java",
            ],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getProgressMessage())
        .isEqualTo("Building java/com/google/test/liba.jar (2 source files)");
  }

  @Test
  public void progressMessageWithSourceJars() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = [
                "a.java",
                "archive.srcjar",
                "b.java",
            ],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getProgressMessage())
        .isEqualTo("Building java/com/google/test/liba.jar (2 source files, 1 source jar)");
  }

  @Test
  public void progressMessageAnnotationProcessors() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library", "java_plugin")
        java_plugin(
            name = "foo",
            srcs = ["Foo.java"],
            processor_class = "Foo",
        )

        java_plugin(
            name = "bar",
            srcs = ["Bar.java"],
            processor_class = "com.google.Bar",
        )

        java_library(
            name = "a",
            srcs = [
                "a.java",
                "archive.srcjar",
                "b.java",
            ],
            plugins = [
                ":foo",
                ":bar",
            ],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getProgressMessage())
        .isEqualTo(
            "Building java/com/google/test/liba.jar (2 source files, 1 source jar)"
                + " and running annotation processors (Foo, Bar)");
  }

  @Test
  public void testLocale() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getIncompleteEnvironmentForTesting())
        .containsEntry("LC_CTYPE", analysisMock.isThisBazel() ? "C.UTF-8" : "en_US.UTF-8");
  }

  @Test
  public void testClasspathReduction() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
            deps = [":b"],
        )

        java_library(
            name = "b",
            srcs = ["B.java"],
            deps = [
                ":c",
                ":d",
            ],
        )

        java_library(
            name = "c",
            srcs = ["C.java"],
        )

        java_library(
            name = "d",
            srcs = ["D.java"],
        )
        """);
    Artifact bJdeps =
        getBinArtifact("libb-hjar.jdeps", getConfiguredTarget("//java/com/google/test:b"));
    Artifact cHjar =
        getBinArtifact("libc-hjar.jar", getConfiguredTarget("//java/com/google/test:libc.jar"));
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    JavaCompileActionContext context = new JavaCompileActionContext();
    Deps.Dependency dep =
        Deps.Dependency.newBuilder()
            .setKind(Kind.EXPLICIT)
            .setPath(cHjar.getExecPathString())
            .build();
    context.insertDependencies(bJdeps, Deps.Dependencies.newBuilder().addDependency(dep).build());
    assertThat(
            artifactsToStrings(
                action.getReducedClasspath(new ActionExecutionContextBuilder().build(), context)))
        .containsExactly(
            "bin java/com/google/test/libb-hjar.jar", "bin java/com/google/test/libc-hjar.jar");
  }

  @Test
  public void testTurbineCpuReservation() throws Exception {
    useConfiguration("--java_header_compilation=true", "--experimental_turbine_cpu_reservation=2");
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
            deps = [":b"],
        )
        java_library(
            name = "b",
            srcs = ["b.java"],
        )
        """);
    JavaCompileAction compileAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    Action action = getTurbineAction(compileAction);

    if (TestConstants.PRODUCT_NAME.equals("bazel")) {
      assertThat(paramFileArgsForAction(action)).contains("-XDnoParallel");
    } else {
      assertThat(paramFileArgsForAction(action)).doesNotContain("-XDnoParallel");
    }
    assertThat(action.getExecutionInfo().keySet().stream().filter(k -> k.startsWith("cpu:")))
        .containsExactly("cpu:2");
  }

  @Test
  public void testNoTurbineCpuReservation() throws Exception {
    useConfiguration("--java_header_compilation=true");
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
            deps = [":b"],
        )
        java_library(
            name = "b",
            srcs = ["b.java"],
        )
        """);
    JavaCompileAction compileAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    Action action = getTurbineAction(compileAction);

    if (TestConstants.PRODUCT_NAME.equals("bazel")) {
      assertThat(paramFileArgsForAction(action)).contains("-XDnoParallel");
    } else {
      assertThat(paramFileArgsForAction(action)).doesNotContain("-XDnoParallel");
    }
    assertThat(action.getExecutionInfo().keySet().stream().filter(k -> k.startsWith("cpu:")))
        .isEmpty();
  }

  private CommandAction getTurbineAction(JavaCompileAction compileAction) throws Exception {
    return (CommandAction)
        getGeneratingAction(getBinArtifacts(compileAction).collect(onlyElement()));
  }

  private static Stream<Artifact> getBinArtifacts(JavaCompileAction compileAction)
      throws Exception {
    return getInputs(compileAction, getDirectJars(compileAction)).stream()
        .filter(a -> a.getFilename().endsWith("-hjar.jar"));
  }

  @Test
  public void testUnusedDepsVerifyFlags() throws Exception {
    scratch.file("third_party/bazel_rules/rules_java/BUILD");
    scratch.file(
        "third_party/bazel_rules/rules_java/rule.bzl",
        """
        load("@rules_java//java:defs.bzl", "JavaInfo", "JavaPluginInfo", rules_java_common = "java_common")

        def _my_rule_impl(ctx):
            output = ctx.outputs.jar
            manifest = ctx.actions.declare_file(ctx.label.name + ".manifest")
            internal_common = java_common.internal_DO_NOT_USE()
            dep = ctx.attr.deps[0]
            compile_jar = dep[JavaInfo].compile_jars.to_list()[0]
            internal_common.create_compilation_action(
                ctx,
                ctx.attr._java_toolchain[rules_java_common.JavaToolchainInfo],
                output,
                manifest,
                JavaPluginInfo(runtime_deps = []),
                depset([compile_jar]),
                depset([compile_jar]),
                depset(),
                depset(),
                depset(),
                depset(),
                "ERROR",
                ctx.label,
                direct_dep_jars_to_verify = [struct(jar = compile_jar, label = str(dep.label))],
            )
            return [DefaultInfo(files = depset([output]))]

        my_rule = rule(
            implementation = _my_rule_impl,
            outputs = {
                "jar": "%{name}.jar",
            },
            attrs = {
                "deps": attr.label_list(),
                "_java_toolchain": attr.label(default = "@bazel_tools//tools/jdk:current_java_toolchain"),
            },
            fragments = ["java"],
            toolchains = ["@bazel_tools//tools/jdk:toolchain_type"],
        )
        """);
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        load("//third_party/bazel_rules/rules_java:rule.bzl", "my_rule")

        java_library(
            name = "dep",
            srcs = ["Dep.java"],
        )

        my_rule(
            name = "a",
            deps = [":dep"],
        )
        """);

    JavaCompileAction compileAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:a.jar");
    List<String> command = getJavacArguments(compileAction);
    assertThat(command).containsAtLeast("--direct_dep_jar", "--direct_dep_label");
    int jarIdx = command.indexOf("--direct_dep_jar");
    int labelIdx = command.indexOf("--direct_dep_label");
    assertThat(command.get(jarIdx + 1)).contains("libdep");
    assertThat(command.get(labelIdx + 1)).endsWith("//java/com/google/test:dep");
  }
}
