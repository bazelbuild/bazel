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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParamFileActionInput;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions.OutputPathsMode;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.rules.java.JavaCompileAction;
import com.google.devtools.build.lib.util.Fingerprint;
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
    addStarlarkRule(executionRequirements, "MyRuleAction");
  }

  private void addStarlarkRule(Dict<String, String> executionRequirements, String mnemonic)
      throws IOException {
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
        format("        mnemonic = '%s',", mnemonic),
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
  public void starlarkRule_packagePathMapping() throws Exception {
    addStarlarkRule(
        Dict.<String, String>builder()
            .put("supports-path-mapping", "output-package")
            .buildImmutable());

    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//pkg:my_rule");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    PathMapper pathMapper = spawn.getPathMapper();
    assertThat(pathMapper.isNoop()).isFalse();
    String outDir = analysisMock.getProductName() + "-out";
    // Only the outputs of the action lose the package. Inputs, even those in the same package, and
    // the tool are only config-stripped. short_path is never mapped.
    assertThat(spawn.getArguments().stream().collect(toImmutableList()))
        .containsExactly(
            format("%s/cfg/bin/tool/tool", outDir),
            format("%s/out/bin/out.bin", outDir),
            "-source",
            format(
                "<pkg/gen_src.txt:%1$s/cfg/bin/pkg/gen_src.txt:%1$s/cfg/bin:%1$s/cfg/bin/pkg>",
                outDir),
            "-source",
            "<pkg/source.txt:pkg/source.txt::pkg>")
        .inOrder();

    Artifact genSrc = getBinArtifact("gen_src.txt", getConfiguredTarget("//pkg:gen_src"));
    Artifact out = getBinArtifact("out.bin", getConfiguredTarget("//pkg:my_rule"));
    for (Artifact artifact : ImmutableList.of(genSrc, out)) {
      assertThat(pathMapper.computeExecPathLengthDiff((Artifact.DerivedArtifact) artifact))
          .isEqualTo(
              artifact.getExecPathString().length()
                  - pathMapper.getMappedExecPathString(artifact).length());
    }
    assertThat(pathMapper.map(out.getExecPath()))
        .isEqualTo(PathFragment.create(outDir + "/out/bin/out.bin"));
    assertThat(pathMapper.map(genSrc.getExecPath()))
        .isEqualTo(PathFragment.create(outDir + "/cfg/bin/pkg/gen_src.txt"));
    assertThat(pathMapper.mapRoot(out).getExecPathString()).isEqualTo(outDir + "/out/bin");
    assertThat(pathMapper.mapRoot(genSrc).getExecPathString()).isEqualTo(outDir + "/cfg/bin");
  }

  @Test
  public void starlarkRule_packagePathMapping_identicalAcrossPackages() throws Exception {
    // Sets up //tool.
    addStarlarkRule(Dict.empty());
    scratch.file(
        "defs/merge.bzl",
        """
        def _dirname(file):
            return file.dirname

        def _merge_impl(ctx):
            # The output names don't depend on the target name.
            out = ctx.actions.declare_file("merged.bin")
            out_tree = ctx.actions.declare_directory("merged_tree")
            args = ctx.actions.args()
            args.add("--output", out)
            args.add_all([out], before_each = "--output_dir", map_each = _dirname)
            args.add_all([out_tree], before_each = "--output_tree", expand_directories = False)
            args.add_all(ctx.files.srcs, before_each = "--input")
            args.add_all(ctx.files.srcs, before_each = "--dir", map_each = _dirname)
            ctx.actions.run(
                outputs = [out, out_tree],
                inputs = ctx.files.srcs,
                executable = ctx.executable._tool,
                arguments = [args],
                execution_requirements = {"supports-path-mapping": "output-package"},
            )
            return [DefaultInfo(files = depset([out, out_tree]))]

        merge = rule(
            implementation = _merge_impl,
            attrs = {
                "srcs": attr.label_list(allow_files = True),
                "_tool": attr.label(default = "//tool", executable = True, cfg = "exec"),
            },
        )
        """);
    scratch.file(
        "shared/BUILD",
        """
        exports_files(['shared.txt'])
        genrule(
            name = 'gen_src',
            outs = ['gen_src.txt'],
            cmd = '<some command>',
            visibility = ['//visibility:public'],
        )
        """);
    for (String pkg : ImmutableList.of("app1", "app2/nested")) {
      scratch.file(
          pkg + "/BUILD",
          """
          load('//defs:merge.bzl', 'merge')
          merge(
              name = 'merge',
              srcs = [
                  '//shared:gen_src',
                  '//shared:shared.txt',
              ],
          )
          """);
    }

    Spawn app1Spawn = getMergeSpawn("//app1:merge");
    Spawn app2Spawn = getMergeSpawn("//app2/nested:merge");

    String outDir = analysisMock.getProductName() + "-out";
    assertThat(app1Spawn.getArguments())
        .containsExactly(
            format("%s/cfg/bin/tool/tool", outDir),
            "--output",
            format("%s/out/bin/merged.bin", outDir),
            "--output_dir",
            format("%s/out/bin", outDir),
            "--output_tree",
            format("%s/out/bin/merged_tree", outDir),
            "--input",
            format("%s/cfg/bin/shared/gen_src.txt", outDir),
            "--input",
            "shared/shared.txt",
            "--dir",
            format("%s/cfg/bin/shared", outDir),
            "--dir",
            "shared")
        .inOrder();
    assertThat(app2Spawn.getArguments()).isEqualTo(app1Spawn.getArguments());
    assertThat(mappedPaths(app1Spawn.getInputFiles().flatten(), app1Spawn.getPathMapper()))
        .isEqualTo(mappedPaths(app2Spawn.getInputFiles().flatten(), app2Spawn.getPathMapper()));
    assertThat(mappedPaths(app1Spawn.getOutputFiles(), app1Spawn.getPathMapper()))
        .isEqualTo(mappedPaths(app2Spawn.getOutputFiles(), app2Spawn.getPathMapper()));
  }

  @Test
  public void starlarkRule_outputPackagePathMapping_nestedOutputsAndParamFile() throws Exception {
    // Sets up //tool.
    addStarlarkRule(Dict.empty());
    scratch.file(
        "defs/nested.bzl",
        """
        def _nested_impl(ctx):
            foo = ctx.actions.declare_file("foo.txt")
            lib_foo = ctx.actions.declare_file("lib/foo.txt")
            args = ctx.actions.args()
            args.use_param_file("@%s", use_always = True)
            args.add_all([foo, lib_foo], before_each = "--output")
            ctx.actions.run(
                outputs = [foo, lib_foo],
                executable = ctx.executable._tool,
                arguments = [args],
                execution_requirements = {"supports-path-mapping": "output-package"},
            )
            return [DefaultInfo(files = depset([foo, lib_foo]))]

        nested = rule(
            implementation = _nested_impl,
            attrs = {
                "_tool": attr.label(default = "//tool", executable = True, cfg = "exec"),
            },
        )
        """);
    scratch.file("app1/BUILD", "load('//defs:nested.bzl', 'nested')", "nested(name = 'a')");
    scratch.file("app2/nested/BUILD", "load('//defs:nested.bzl', 'nested')", "nested(name = 'b')");

    String outDir = analysisMock.getProductName() + "-out";
    for (String label : ImmutableList.of("//app1:a", "//app2/nested:b")) {
      ConfiguredTarget target = getConfiguredTarget(label);
      SpawnAction action = (SpawnAction) getGeneratingAction(getBinArtifact("foo.txt", target));
      Spawn spawn =
          action.getSpawn(
              new ActionExecutionContextBuilder()
                  .setMetadataProvider(new FakeActionInputFileCache())
                  .build());

      // The outputs lose the package, but keep their path relative to it. The same applies to the
      // param file, which is named after the primary output.
      assertThat(mappedPaths(spawn.getOutputFiles(), spawn.getPathMapper()))
          .containsExactly(
              format("%s/out/bin/foo.txt", outDir), format("%s/out/bin/lib/foo.txt", outDir));
      assertThat(spawn.getArguments())
          .containsExactly(
              format("%s/cfg/bin/tool/tool", outDir),
              format("@%s/out/bin/foo.txt-0.params", outDir))
          .inOrder();
      ParamFileActionInput paramFile =
          Iterables.getOnlyElement(
              Iterables.filter(spawn.getInputFiles().flatten(), ParamFileActionInput.class));
      assertThat(spawn.getPathMapper().map(paramFile.getExecPath()))
          .isEqualTo(PathFragment.create(outDir + "/out/bin/foo.txt-0.params"));
      assertThat(paramFile.getArguments())
          .containsExactly(
              "--output",
              format("%s/out/bin/foo.txt", outDir),
              "--output",
              format("%s/out/bin/lib/foo.txt", outDir))
          .inOrder();
    }
  }

  private Spawn getMergeSpawn(String label) throws Exception {
    SpawnAction action =
        (SpawnAction) getGeneratingAction(getBinArtifact("merged.bin", getConfiguredTarget(label)));
    return action.getSpawn(
        new ActionExecutionContextBuilder()
            .setMetadataProvider(new FakeActionInputFileCache())
            .build());
  }

  private static ImmutableList<String> mappedPaths(
      Iterable<? extends ActionInput> inputs, PathMapper pathMapper) {
    return ImmutableList.sortedCopyOf(
        Streams.stream(inputs)
            .map(input -> pathMapper.map(input.getExecPath()).getPathString())
            .collect(toImmutableList()));
  }

  @Test
  public void starlarkRule_packagePathMapping_ignoredWithHeuristicPathMapping() throws Exception {
    useConfiguration(
        "--experimental_output_paths=strip",
        "--modify_execution_info=MyRuleAction=+supports-heuristic-path-mapping");
    addStarlarkRule(
        Dict.<String, String>builder()
            .put("supports-path-mapping", "output-package")
            .buildImmutable());

    assertOutputOfMyRuleOnlyConfigStripped();
  }

  @Test
  public void starlarkRule_packagePathMapping_ignoredForLegacyArgsHandling() throws Exception {
    addStarlarkRule(
        Dict.<String, String>builder()
            .put("supports-path-mapping", "output-package")
            .buildImmutable(),
        "AndroidMyRuleAction");

    assertOutputOfMyRuleOnlyConfigStripped();
  }

  private void assertOutputOfMyRuleOnlyConfigStripped() throws Exception {
    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//pkg:my_rule");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    PathMapper pathMapper = spawn.getPathMapper();
    assertThat(pathMapper.isNoop()).isFalse();
    String outDir = analysisMock.getProductName() + "-out";
    Artifact out = getBinArtifact("out.bin", getConfiguredTarget("//pkg:my_rule"));
    assertThat(pathMapper.getMappedExecPathString(out))
        .isEqualTo(format("%s/cfg/bin/pkg/out.bin", outDir));
  }

  @Test
  public void starlarkRule_heuristicPathMappingNotAllowedViaExecutionRequirements()
      throws Exception {
    useConfiguration("--experimental_output_paths=strip");
    addStarlarkRule(
        Dict.<String, String>builder()
            .put("supports-heuristic-path-mapping", "1")
            .buildImmutable());

    checkError(
        "//pkg:my_rule",
        "execution requirement 'supports-heuristic-path-mapping' cannot be set directly; it can"
            + " only be enabled via --modify_execution_info");
  }

  @Test
  public void starlarkRule_heuristicPathMappingAllowedViaModifyExecutionInfo() throws Exception {
    useConfiguration(
        "--experimental_output_paths=strip",
        "--modify_execution_info=MyRuleAction=+supports-heuristic-path-mapping");
    addStarlarkRule(Dict.empty());

    SpawnAction action = (SpawnAction) getGeneratingActionForLabel("//pkg:my_rule");
    Spawn spawn =
        action.getSpawn(
            new ActionExecutionContextBuilder()
                .setMetadataProvider(new FakeActionInputFileCache())
                .build());

    assertThat(spawn.getPathMapper().isNoop()).isFalse();
  }

  @Test
  public void customCommandLine_heuristicPathMapping_stripsPrefixedAndFormattedArgs()
      throws Exception {
    ActionsTestUtil.MockAction heuristicAction =
        new ActionsTestUtil.MockAction(
            ImmutableList.of(), ImmutableSet.of(getBinArtifactWithNoOwner("pkg/out.bin"))) {
          @Override
          public ImmutableMap<String, String> getExecutionInfo() {
            return ImmutableMap.of(ExecutionRequirements.SUPPORTS_HEURISTIC_PATH_MAPPING, "");
          }
        };
    PathMapper pathMapper =
        PathMappers.create(
            heuristicAction,
            OutputPathsMode.STRIP,
            /* isStarlarkAction= */ false,
            /* inputMetadataProvider= */ null);

    String outDir = analysisMock.getProductName() + "-out";
    CustomCommandLine commandLine =
        CustomCommandLine.builder()
            .addPrefixed("-I", format("%s/k8-fastbuild/bin/pkg/foo.h", outDir))
            .addFormatted("-DROOT=%s", format("%s/k8-fastbuild/bin", outDir))
            .build();

    assertThat(commandLine.arguments(/* inputMetadataProvider= */ null, pathMapper))
        .containsExactly(
            format("-I%s/cfg/bin/pkg/foo.h", outDir), format("-DROOT=%s/cfg/bin", outDir))
        .inOrder();
  }

  @Test
  public void addToFingerprint_differentiatesHeuristicPathMapping() throws Exception {
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    NestedSet<Artifact> emptyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);

    Fingerprint fpOff = new Fingerprint();
    PathMappers.addToFingerprint(
        "Mnemonic",
        ImmutableMap.of(),
        emptyArtifacts,
        actionKeyContext,
        OutputPathsMode.STRIP,
        fpOff);

    Fingerprint fpStructured = new Fingerprint();
    PathMappers.addToFingerprint(
        "Mnemonic",
        ImmutableMap.of(ExecutionRequirements.SUPPORTS_PATH_MAPPING, ""),
        emptyArtifacts,
        actionKeyContext,
        OutputPathsMode.STRIP,
        fpStructured);

    Fingerprint fpHeuristic = new Fingerprint();
    PathMappers.addToFingerprint(
        "Mnemonic",
        ImmutableMap.of(ExecutionRequirements.SUPPORTS_HEURISTIC_PATH_MAPPING, ""),
        emptyArtifacts,
        actionKeyContext,
        OutputPathsMode.STRIP,
        fpHeuristic);

    String digestOff = fpOff.hexDigestAndReset();
    String digestStructured = fpStructured.hexDigestAndReset();
    String digestHeuristic = fpHeuristic.hexDigestAndReset();

    assertThat(digestOff).isNotEqualTo(digestStructured);
    assertThat(digestStructured).isNotEqualTo(digestHeuristic);
    assertThat(digestOff).isNotEqualTo(digestHeuristic);
  }

  @Test
  public void addToFingerprint_differentiatesPackagePathMapping() throws Exception {
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    NestedSet<Artifact> emptyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);

    Fingerprint fpStructured = new Fingerprint();
    PathMappers.addToFingerprint(
        "Mnemonic",
        ImmutableMap.of(ExecutionRequirements.SUPPORTS_PATH_MAPPING, ""),
        emptyArtifacts,
        actionKeyContext,
        OutputPathsMode.STRIP,
        fpStructured);

    Fingerprint fpPackage = new Fingerprint();
    PathMappers.addToFingerprint(
        "Mnemonic",
        ImmutableMap.of(
            ExecutionRequirements.SUPPORTS_PATH_MAPPING,
            ExecutionRequirements.PATH_MAPPING_OUTPUT_PACKAGE),
        emptyArtifacts,
        actionKeyContext,
        OutputPathsMode.STRIP,
        fpPackage);

    Fingerprint fpPackageOff = new Fingerprint();
    PathMappers.addToFingerprint(
        "Mnemonic",
        ImmutableMap.of(
            ExecutionRequirements.SUPPORTS_PATH_MAPPING,
            ExecutionRequirements.PATH_MAPPING_OUTPUT_PACKAGE),
        emptyArtifacts,
        actionKeyContext,
        OutputPathsMode.OFF,
        fpPackageOff);

    assertThat(fpStructured.hexDigestAndReset()).isNotEqualTo(fpPackage.hexDigestAndReset());
    assertThat(fpPackageOff.hexDigestAndReset()).isEqualTo(new Fingerprint().hexDigestAndReset());
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
