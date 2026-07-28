// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.lang.String.format;

import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PathMappers}. */
@RunWith(JUnit4.class)
public class PathMappersTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    useConfiguration("--experimental_output_paths=strip");
  }

  @Test
  public void javaLibraryWithJavacopts() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        genrule(
            name = 'gen_b',
            outs = ['B.java'],
            cmd = '<some command>',
        )
        genrule(
            name = 'gen_c',
            outs = ['C.java'],
            cmd = '<some command>',
        )
        java_library(
            name = 'a',
            javacopts = [
                '-XepOpt:foo:bar=$(location B.java)',
                '-XepOpt:baz=$(location C.java),$(location B.java)',
            ],
            srcs = [
                'A.java',
                'B.java',
                'C.java',
            ],
        )
        """);

    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    PathMapper pathMapper =
        PathMappers.create(
            action,
            PathMappers.getOutputPathsMode(targetConfig),
            /* isStarlarkAction= */ false,
            /* inputMetadataProvider= */ null);

    assertThat(pathMapper.isNoop()).isFalse();
    String outDir = analysisMock.getProductName() + "-out";
    assertThat(
            action.getCommandLines().allArguments(pathMapper).stream()
                .filter(arg -> arg.contains("java/com/google/test/"))
                .collect(toImmutableList()))
        .containsExactly(
            "java/com/google/test/A.java",
            format("%s/cfg/bin/java/com/google/test/B.java", outDir),
            format("%s/cfg/bin/java/com/google/test/C.java", outDir),
            format("%s/cfg/bin/java/com/google/test/liba.jar", outDir),
            format("%s/cfg/bin/java/com/google/test/liba-native-header.jar", outDir),
            format("%s/cfg/bin/java/com/google/test/liba.jar_manifest_proto", outDir),
            format("%s/cfg/bin/java/com/google/test/liba.jdeps", outDir),
            format("-XepOpt:foo:bar=%s/cfg/bin/java/com/google/test/B.java", outDir),
            format(
                "-XepOpt:baz=%s/cfg/bin/java/com/google/test/C.java,%s/cfg/bin/java/com/google/test/B.java",
                outDir, outDir));
  }

  private void addStarlarkRule(Dict<String, String> executionRequirements) throws IOException {
    scratch.file("defs/BUILD");
    scratch.file(
        "defs/defs.bzl",
        "def _map_each(file):",
        "    return '{}:{}:{}:{}'.format(file.short_path, file.path, file.root.path, file.dirname)",
        "def _my_rule_impl(ctx):",
        "    args = ctx.actions.args()",
        "    args.add(ctx.outputs.out)",
        "    args.add_all(",
        "        depset(ctx.files.srcs),",
        "        before_each = '-source',",
        "        format_each = '<%s>',",
        "        map_each = _map_each,",
        "    )",
        "    ctx.actions.run(",
        "        outputs = [ctx.outputs.out],",
        "        inputs = ctx.files.srcs,",
        "        executable = ctx.executable._tool,",
        "        arguments = [args],",
        "        mnemonic = 'MyRuleAction',",
        format(
            "        execution_requirements = %s,",
            Starlark.repr(executionRequirements, StarlarkSemantics.DEFAULT)),
        "    )",
        "    return [DefaultInfo(files = depset([ctx.outputs.out]))]",
        "my_rule = rule(",
        "    implementation = _my_rule_impl,",
        "    attrs = {",
        "        'srcs': attr.label_list(allow_files = True),",
        "        'out': attr.output(mandatory = True),",
        "        '_tool': attr.label(",
        "            default = '//tool',",
        "            executable = True,",
        "            cfg = 'exec',",
        "        ),",
        "    },",
        ")");
    scratch.file(
        "pkg/BUILD",
        """
        load('//defs:defs.bzl', 'my_rule')
        genrule(
            name = 'gen_src',
            outs = ['gen_src.txt'],
            cmd = '<some command>',
        )
        my_rule(
            name = 'my_rule',
            out = 'out.bin',
            srcs = [
                ':gen_src',
                'source.txt',
            ],
        )
        """);
    scratch.file(
        "tool/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = 'tool',
            srcs = ['tool.sh'],
            visibility = ['//visibility:public'],
        )
        """);
  }

  @Test
  public void starlarkRule_optedInViaExecutionRequirements() throws Exception {
    addStarlarkRule(
        Dict.<String, String>builder().put("supports-path-mapping", "1").buildImmutable());

    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//pkg:my_rule");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    assertThat(spawn.getPathMapper().isNoop()).isFalse();
    String outDir = analysisMock.getProductName() + "-out";
    assertThat(spawn.getArguments().stream().collect(toImmutableList()))
        .containsExactly(
            format("%s/cfg/bin/tool/tool", outDir),
            format("%s/cfg/bin/pkg/out.bin", outDir),
            "-source",
            format(
                "<pkg/gen_src.txt:%1$s/cfg/bin/pkg/gen_src.txt:%1$s/cfg/bin:%1$s/cfg/bin/pkg>",
                outDir),
            "-source",
            "<pkg/source.txt:pkg/source.txt::pkg>")
        .inOrder();
  }

  @Test
  public void starlarkRule_optedInViaModifyExecutionInfo() throws Exception {
    useConfiguration(
        "--experimental_output_paths=strip",
        "--modify_execution_info=MyRuleAction=+supports-path-mapping");
    addStarlarkRule(Dict.empty());

    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//pkg:my_rule");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    assertThat(spawn.getPathMapper().isNoop()).isFalse();
    String outDir = analysisMock.getProductName() + "-out";
    assertThat(spawn.getArguments().stream().collect(toImmutableList()))
        .containsExactly(
            format("%s/cfg/bin/tool/tool", outDir),
            format("%s/cfg/bin/pkg/out.bin", outDir),
            "-source",
            format(
                "<pkg/gen_src.txt:%1$s/cfg/bin/pkg/gen_src.txt:%1$s/cfg/bin:%1$s/cfg/bin/pkg>",
                outDir),
            "-source",
            "<pkg/source.txt:pkg/source.txt::pkg>")
        .inOrder();
  }

  @Test
  public void starlarkRule_stringExecutablePath() throws Exception {
    scratch.file("defs/BUILD");
    scratch.file(
        "defs/defs.bzl",
        """
        def my_rule_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name)
            ctx.actions.run(
                executable = ctx.executable.tool.path,
                arguments = [ctx.actions.args().add(out)],
                outputs = [out],
                tools = [ctx.executable.tool],
                execution_requirements = {"supports-path-mapping": "1"},
            )
            return DefaultInfo(files = depset([out]))
        my_rule = rule(
            implementation = my_rule_impl,
            attrs = {
                "tool": attr.label(
                    default = "//foo:script",
                    cfg = "exec",
                    executable = True,
                ),
            },
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = 'script',
            srcs = ['script.sh'],
            visibility = ['//visibility:public'],
        )
        """);
    scratch.file(
        "BUILD",
        """
        load("//defs:defs.bzl", "my_rule")
        my_rule(name = "my_rule")
        """);

    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//:my_rule");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    assertThat(spawn.getPathMapper().isNoop()).isFalse();
    String outDir = analysisMock.getProductName() + "-out";
    assertThat(spawn.getArguments())
        .containsExactly(
            "%s/cfg/bin/foo/script".formatted(outDir), "%s/cfg/bin/my_rule".formatted(outDir))
        .inOrder();
  }

  @Test
  public void forActionKey() {
    var pathMapper = PathMapper.forActionKey(CoreOptions.OutputPathsMode.STRIP);
    assertThat(pathMapper.isNoop()).isFalse();
    assertThat(pathMapper.map(PathFragment.create("pkg/file")))
        .isEqualTo(PathFragment.create("pkg/file"));
    assertThat(pathMapper.map(PathFragment.create("bazel-out/k8-fastbuild-ST-12345/bin/pkg/file")))
        .isEqualTo(PathFragment.create("bazel-out/pm-k8-fastbuild-ST-12345/bin/pkg/file"));
  }

  @Test
  public void starlarkRule_archivedTreePaths() throws Exception {
    String outDir = analysisMock.getProductName() + "-out";
    scratch.file("defs/BUILD");
    scratch.file(
        "defs/defs.bzl",
        """
        def my_rule_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name)
            args = ctx.actions.args()
            args.add(out)
            args.add("--input")
            args.add("%1$s/k8-fastbuild/bin/pkg/standard.js")
            args.add("--input")
            args.add("%1$s/:archived_tree_artifacts/k8-fastbuild/bin/pkg/tree.zip")
            ctx.actions.run(
                executable = ctx.executable.tool.path,
                arguments = [args],
                outputs = [out],
                tools = [ctx.executable.tool],
                mnemonic = "Android",  # Using a supported mnemonic to enable path-stripping.
                execution_requirements = {"supports-path-mapping": "1"},
            )
            return DefaultInfo(files = depset([out]))
        my_rule = rule(
            implementation = my_rule_impl,
            attrs = {
                "tool": attr.label(
                    default = "//foo:script",
                    cfg = "exec",
                    executable = True,
                ),
            },
        )
        """
            .formatted(outDir));
    scratch.file(
        "foo/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = 'script',
            srcs = ['script.sh'],
            visibility = ['//visibility:public'],
        )
        """);
    scratch.file(
        "BUILD",
        """
        load("//defs:defs.bzl", "my_rule")
        my_rule(name = "my_rule")
        """);

    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//:my_rule");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    assertThat(spawn.getPathMapper().isNoop()).isFalse();

    assertThat(spawn.getArguments())
        .containsExactly(
            "%s/cfg/bin/foo/script".formatted(outDir),
            "%s/cfg/bin/my_rule".formatted(outDir),
            "--input",
            "%s/cfg/bin/pkg/standard.js".formatted(outDir),
            "--input",
            "%s/:archived_tree_artifacts/cfg/bin/pkg/tree.zip".formatted(outDir))
        .inOrder();
  }

  @Test
  public void starlarkRule_inputsOutputsCollision() throws Exception {
    scratch.file(
        "defs/defs.bzl",
        """
        def _flag_impl(ctx):
            return []

        bool_flag = rule(implementation = _flag_impl, build_setting = config.bool(flag = True))

        def _transition_impl(settings, attr):
            return {"//defs:transitioned": True}

        _transitioned = transition(
            implementation = _transition_impl,
            inputs = [],
            outputs = ["//defs:transitioned"],
        )

        def _my_rule_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name + ".out")
            args = ctx.actions.args()
            args.add(out)
            args.add_all(ctx.files.dep)
            ctx.actions.run(
                outputs = [out],
                inputs = ctx.files.dep,
                executable = ctx.executable._tool,
                arguments = [args],
                execution_requirements = {"supports-path-mapping": "1"},
            )
            return [DefaultInfo(files = depset([out]))]

        my_rule = rule(
            implementation = _my_rule_impl,
            attrs = {
                "dep": attr.label_list(cfg = _transitioned, allow_files = True),
                "_tool": attr.label(default = "//tool", executable = True, cfg = "exec"),
            },
        )
        """);
    scratch.file(
        "defs/BUILD",
        """
        load("//defs:defs.bzl", "bool_flag")

        bool_flag(
            name = "transitioned",
            build_setting_default = False,
            visibility = ["//visibility:public"],
        )

        config_setting(
            name = "is_transitioned",
            flag_values = {":transitioned": "True"},
            visibility = ["//visibility:public"],
        )
        """);
    scratch.file(
        "collide/BUILD",
        """
        load("//defs:defs.bzl", "my_rule")

        my_rule(
            name = "a",
            dep = select({
                "//defs:is_transitioned": [],
                "//conditions:default": [":a"],
            }),
        )
        """);
    scratch.file(
        "tool/BUILD",
        """
        load('//test_defs:foo_binary.bzl', 'foo_binary')
        foo_binary(
            name = 'tool',
            srcs = ['tool.sh'],
            visibility = ['//visibility:public'],
        )
        """);

    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//collide:a");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    assertThat(spawn.getPathMapper().isNoop()).isTrue();
    String outDir = analysisMock.getProductName() + "-out";
    assertThat(spawn.getArguments()).doesNotContain(format("%s/cfg/bin/collide/a.out", outDir));
    assertThat(spawn.getArguments().stream().anyMatch(arg -> arg.endsWith("/bin/collide/a.out")))
        .isTrue();
  }
}
