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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.truth.Correspondence;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.OS;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit test for proto_common module. */
@RunWith(TestParameterInjector.class)
public class BazelProtoCommonTest extends BuildViewTestCase {
  private static final Correspondence<String, String> MATCHES_REGEX =
      Correspondence.from((a, b) -> Pattern.matches(b, a), "matches");

  private static final StarlarkProviderIdentifier boolProviderId =
      StarlarkProviderIdentifier.forKey(
          new StarlarkProvider.Key(
              Label.parseCanonicalUnchecked("//foo:should_generate.bzl"), "BoolProvider"));

  @Before
  public final void setup() throws Exception {
    MockProtoSupport.setupWorkspace(scratch);
    invalidatePackages();

    MockProtoSupport.setup(mockToolsConfig);

    scratch.file(
        "third_party/x/BUILD",
        "licenses(['unencumbered'])",
        "cc_binary(name = 'plugin', srcs = ['plugin.cc'])",
        "cc_library(name = 'runtime', srcs = ['runtime.cc'])",
        "filegroup(name = 'descriptors', srcs = ['metadata.proto', 'descriptor.proto'])",
        "filegroup(name = 'any', srcs = ['any.proto'])",
        "filegroup(name = 'something', srcs = ['something.proto'])",
        "proto_library(name = 'mixed', srcs = [':descriptors', ':something'])",
        "proto_library(name = 'denied', srcs = [':descriptors', ':any'])");
    scratch.file(
        "foo/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "proto_lang_toolchain(",
        "    name = 'toolchain',",
        "    command_line = '--java_out=param1,param2:$(OUT)',",
        "    plugin_format_flag = '--plugin=%s',",
        "    plugin = '//third_party/x:plugin',",
        "    runtime = '//third_party/x:runtime',",
        "    blacklisted_protos = ['//third_party/x:denied'],",
        "    progress_message = 'Progress Message %{label}',",
        "    mnemonic = 'MyMnemonic',",
        ")",
        "proto_lang_toolchain(",
        "    name = 'toolchain_noplugin',",
        "    command_line = '--java_out=param1,param2:$(OUT)',",
        "    runtime = '//third_party/x:runtime',",
        "    blacklisted_protos = ['//third_party/x:denied'],",
        "    progress_message = 'Progress Message %{label}',",
        "    mnemonic = 'MyMnemonic',",
        ")");

    scratch.file(
        "foo/generate.bzl",
        "def _resource_set_callback(os, inputs_size):",
        "   return {'memory': 25 + 0.15 * inputs_size, 'cpu': 1}",
        "def _impl(ctx):",
        "  outfile = ctx.actions.declare_file('out')",
        "  kwargs = {}",
        "  if ctx.attr.plugin_output:",
        "    kwargs['plugin_output'] = ctx.attr.plugin_output",
        "  if ctx.attr.additional_args:",
        "    additional_args = ctx.actions.args()",
        "    additional_args.add_all(ctx.attr.additional_args)",
        "    kwargs['additional_args'] = additional_args",
        "  if ctx.files.additional_tools:",
        "    kwargs['additional_tools'] = ctx.files.additional_tools",
        "  if ctx.files.additional_inputs:",
        "    kwargs['additional_inputs'] = depset(ctx.files.additional_inputs)",
        "  if ctx.attr.use_resource_set:",
        "    kwargs['resource_set'] = _resource_set_callback",
        "  if ctx.attr.progress_message:",
        "    kwargs['experimental_progress_message'] = ctx.attr.progress_message",
        "  proto_common_do_not_use.compile(",
        "    ctx.actions,",
        "    ctx.attr.proto_dep[ProtoInfo],",
        "    ctx.attr.toolchain[proto_common_do_not_use.ProtoLangToolchainInfo],",
        "    [outfile],",
        "    **kwargs)",
        "  return [DefaultInfo(files = depset([outfile]))]",
        "generate_rule = rule(_impl,",
        "  attrs = {",
        "     'proto_dep': attr.label(),",
        "     'plugin_output': attr.string(),",
        "     'toolchain': attr.label(default = '//foo:toolchain'),",
        "     'additional_args': attr.string_list(),",
        "     'additional_tools': attr.label_list(cfg = 'exec'),",
        "     'additional_inputs': attr.label_list(allow_files = True),",
        "     'use_resource_set': attr.bool(),",
        "     'progress_message': attr.string(),",
        "  })");

    scratch.file(
        "foo/should_generate.bzl",
        "BoolProvider = provider()",
        "def _impl(ctx):",
        "  result = proto_common_do_not_use.experimental_should_generate_code(",
        "    ctx.attr.proto_dep[ProtoInfo],",
        "    ctx.attr.toolchain[proto_common_do_not_use.ProtoLangToolchainInfo],",
        "    'MyRule',",
        "    ctx.attr.proto_dep.label)",
        "  return [BoolProvider(value = result)]",
        "should_generate_rule = rule(_impl,",
        "  attrs = {",
        "     'proto_dep': attr.label(),",
        "     'toolchain': attr.label(default = '//foo:toolchain'),",
        "  })");

    scratch.file(
        "foo/declare_generated_files.bzl",
        "def _impl(ctx):",
        "  files = proto_common_do_not_use.declare_generated_files(",
        "    ctx.actions,",
        "    ctx.attr.proto_dep[ProtoInfo],",
        "    ctx.attr.extension,",
        "    (lambda s: s.replace('-','_').replace('.','/')) if ctx.attr.python_names else None)",
        "  for f in files:",
        "    ctx.actions.write(f, '')",
        "  return [DefaultInfo(files = depset(files))]",
        "declare_generated_files = rule(_impl,",
        "  attrs = {",
        "     'proto_dep': attr.label(),",
        "     'extension': attr.string(),",
        "     'python_names': attr.bool(default = False),",
        "  })");
  }

  /** Verifies basic usage of <code>proto_common.generate_code</code>. */
  @Test
  public void generateCode_basic() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    List<String> cmdLine = spawnAction.getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            "-Ibar/A.proto=bar/A.proto",
            "bar/A.proto")
        .inOrder();
    assertThat(spawnAction.getMnemonic()).isEqualTo("MyMnemonic");
    assertThat(spawnAction.getProgressMessage()).isEqualTo("Progress Message //bar:simple");
  }

  /** Verifies usage of proto_common.generate_code with no plugin specified by toolchain. */
  @Test
  public void generateCode_noPlugin() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto',",
        "  toolchain = '//foo:toolchain_noplugin')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    List<String> cmdLine =
        getGeneratingSpawnAction(getBinArtifact("out", target)).getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly("-Ibar/A.proto=bar/A.proto", "bar/A.proto")
        .inOrder();
  }

  /**
   * Verifies usage of <code>proto_common.generate_code</code> with <code>plugin_output</code>
   * parameter.
   */
  @Test
  public void generateCode_withPluginOutput() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto', plugin_output = 'foo.srcjar')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    List<String> cmdLine =
        getGeneratingSpawnAction(getBinArtifact("out", target)).getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--java_out=param1,param2:foo.srcjar",
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            "-Ibar/A.proto=bar/A.proto",
            "bar/A.proto")
        .inOrder();
  }

  /**
   * Verifies usage of <code>proto_common.generate_code</code> with <code>additional_args</code>
   * parameter.
   */
  @Test
  public void generateCode_additionalArgs() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto', additional_args = ['--a', '--b'])");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    List<String> cmdLine =
        getGeneratingSpawnAction(getBinArtifact("out", target)).getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--a",
            "--b",
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            "-Ibar/A.proto=bar/A.proto",
            "bar/A.proto")
        .inOrder();
  }

  /**
   * Verifies usage of <code>proto_common.generate_code</code> with <code>additional_tools</code>
   * parameter.
   */
  @Test
  public void generateCode_additionalTools() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "cc_binary(name = 'tool1', srcs = ['tool1.cc'])",
        "cc_binary(name = 'tool2', srcs = ['tool2.cc'])",
        "generate_rule(name = 'simple', proto_dep = ':proto',",
        "  additional_tools = [':tool1', ':tool2'])");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    assertThat(prettyArtifactNames(spawnAction.getTools()))
        .containsAtLeast("bar/tool1", "bar/tool2", "third_party/x/plugin");
  }

  /**
   * Verifies usage of <code>proto_common.generate_code</code> with <code>additional_tools</code>
   * parameter and no plugin on the toolchain.
   */
  @Test
  public void generateCode_additionalToolsNoPlugin() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "cc_binary(name = 'tool1', srcs = ['tool1.cc'])",
        "cc_binary(name = 'tool2', srcs = ['tool2.cc'])",
        "generate_rule(name = 'simple',",
        "  proto_dep = ':proto',",
        "  additional_tools = [':tool1', ':tool2'],",
        "  toolchain = '//foo:toolchain_noplugin',",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    assertThat(prettyArtifactNames(spawnAction.getTools()))
        .containsAtLeast("bar/tool1", "bar/tool2");
  }

  /**
   * Verifies usage of <code>proto_common.generate_code</code> with <code>additional_inputs</code>
   * parameter.
   */
  @Test
  public void generateCode_additionalInputs() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto',",
        "  additional_inputs = [':input1.txt', ':input2.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    assertThat(prettyArtifactNames(spawnAction.getInputs()))
        .containsAtLeast("bar/input1.txt", "bar/input2.txt");
  }

  /**
   * Verifies usage of <code>proto_common.generate_code</code> with <code>resource_set</code>
   * parameter.
   */
  @Test
  public void generateCode_resourceSet() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto', use_resource_set = True)");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    assertThat(spawnAction.getResourceSetOrBuilder().buildResourceSet(OS.DARWIN, 0))
        .isEqualTo(ResourceSet.createWithRamCpu(25, 1));
    assertThat(spawnAction.getResourceSetOrBuilder().buildResourceSet(OS.LINUX, 2))
        .isEqualTo(ResourceSet.createWithRamCpu(25.3, 1));
  }

  /** Verifies <code>--protocopts</code> are passed to command line. */
  @Test
  public void generateCode_protocOpts() throws Exception {
    useConfiguration("--protocopt=--foo", "--protocopt=--bar");
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    List<String> cmdLine = spawnAction.getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            "--foo",
            "--bar",
            "-Ibar/A.proto=bar/A.proto",
            "bar/A.proto")
        .inOrder();
  }

  /**
   * Verifies <code>proto_common.generate_code</code> correctly handles direct generated <code>
   * .proto</code> files.
   */
  @Test
  public void generateCode_directGeneratedProtos() throws Exception {
    useConfiguration("--noincompatible_generated_protos_in_virtual_imports");
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "genrule(name = 'generate', srcs = ['A.txt'], cmd = '', outs = ['G.proto'])",
        "proto_library(name = 'proto', srcs = ['A.proto', 'G.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    List<String> cmdLine = spawnAction.getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            "--proto_path=bl?azel?-out/k8-fastbuild/bin",
            "-Ibar/A.proto=bar/A.proto",
            "-Ibar/G.proto=bl?azel?-out/k8-fastbuild/bin/bar/G.proto",
            "bar/A.proto",
            "bl?azel?-out/k8-fastbuild/bin/bar/G.proto")
        .inOrder();
  }

  /**
   * Verifies <code>proto_common.generate_code</code> correctly handles in-direct generated <code>
   * .proto</code> files.
   */
  @Test
  public void generateCode_inDirectGeneratedProtos() throws Exception {
    useConfiguration("--noincompatible_generated_protos_in_virtual_imports");
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "genrule(name = 'generate', srcs = ['A.txt'], cmd = '', outs = ['G.proto'])",
        "proto_library(name = 'generated', srcs = ['G.proto'])",
        "proto_library(name = 'proto', srcs = ['A.proto'], deps = [':generated'])",
        "generate_rule(name = 'simple', proto_dep = ':proto')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    List<String> cmdLine = spawnAction.getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            "--proto_path=bl?azel?-out/k8-fastbuild/bin",
            "-Ibar/A.proto=bar/A.proto",
            "-Ibar/G.proto=bl?azel?-out/k8-fastbuild/bin/bar/G.proto",
            "bar/A.proto")
        .inOrder();
  }

  /**
   * Verifies <code>proto_common.generate_code</code> correctly handles external <code>proto_library
   * </code>-es.
   */
  @Test
  @TestParameters({
    "{virtual: false, sibling: false, generated: false, expectedFlags:"
        + " ['--proto_path=external/foo','-Ie/E.proto=external/foo/e/E.proto']}",
    "{virtual: false, sibling: false, generated: true, expectedFlags:"
        + " ['--proto_path=bl?azel?-out/k8-fastbuild/bin/external/foo',"
        + " '-Ie/E.proto=bl?azel?-out/k8-fastbuild/bin/external/foo/e/E.proto']}",
    "{virtual: true, sibling: false, generated: false,expectedFlags:"
        + " ['--proto_path=external/foo','-Ie/E.proto=external/foo/e/E.proto']}",
    "{virtual: true, sibling: false, generated: true, expectedFlags:"
        + " ['--proto_path=bl?azel?-out/k8-fastbuild/bin/external/foo/e/_virtual_imports/e',"
        + " '-Ie/E.proto=bl?azel?-out/k8-fastbuild/bin/external/foo/e/_virtual_imports/e/e/E.proto']}",
    "{virtual: true, sibling: true, generated: false,expectedFlags:"
        + " ['--proto_path=../foo','-Ie/E.proto=../foo/e/E.proto']}",
    "{virtual: true, sibling: true, generated: true, expectedFlags:"
        + " ['--proto_path=bl?azel?-out/foo/k8-fastbuild/bin/e/_virtual_imports/e',"
        + " '-Ie/E.proto=bl?azel?-out/foo/k8-fastbuild/bin/e/_virtual_imports/e/e/E.proto']}",
    "{virtual: false, sibling: true, generated: false,expectedFlags:"
        + " ['--proto_path=../foo','-Ie/E.proto=../foo/e/E.proto']}",
    "{virtual: false, sibling: true, generated: true, expectedFlags:"
        + " ['--proto_path=bl?azel?-out/foo/k8-fastbuild/bin','-Ie/E.proto=bl?azel?-out/foo/k8-fastbuild/bin/e/E.proto']}",
  })
  public void generateCode_externalProtoLibrary(
      boolean virtual, boolean sibling, boolean generated, List<String> expectedFlags)
      throws Exception {
    if (virtual) {
      useConfiguration("--incompatible_generated_protos_in_virtual_imports");
    } else {
      useConfiguration("--noincompatible_generated_protos_in_virtual_imports");
    }
    if (sibling) {
      setBuildLanguageOptions("--experimental_sibling_repository_layout");
    }
    scratch.appendFile("WORKSPACE", "local_repository(name = 'foo', path = '/foo')");
    invalidatePackages();
    scratch.file("/foo/WORKSPACE");
    scratch.file(
        "/foo/e/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "proto_library(name='e', srcs=['E.proto'])",
        generated
            ? "genrule(name = 'generate', srcs = ['A.txt'], cmd = '', outs = ['E.proto'])"
            : "");
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'], deps = ['@foo//e:e'])",
        "generate_rule(name = 'simple', proto_dep = ':proto')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    List<String> cmdLine = spawnAction.getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            expectedFlags.get(0),
            "-Ibar/A.proto=bar/A.proto",
            expectedFlags.get(1),
            "bar/A.proto")
        .inOrder();
  }

  /** Verifies <code>experimental_progress_message</code> parameters. */
  @Test
  public void generateCode_overrideProgressMessage() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:generate.bzl', 'generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "generate_rule(name = 'simple', proto_dep = ':proto', progress_message = 'My %{label}')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    SpawnAction spawnAction = getGeneratingSpawnAction(getBinArtifact("out", target));
    List<String> cmdLine = spawnAction.getRemainingArguments();
    assertThat(cmdLine)
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "--plugin=bl?azel?-out/[^/]*-exec-[^/]*/bin/third_party/x/plugin",
            "-Ibar/A.proto=bar/A.proto",
            "bar/A.proto")
        .inOrder();
    assertThat(spawnAction.getMnemonic()).isEqualTo("MyMnemonic");
    assertThat(spawnAction.getProgressMessage()).isEqualTo("My //bar:simple");
  }

  /** Verifies <code>proto_common.should_generate_code</code> call. */
  @Test
  public void shouldGenerateCode_basic() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:should_generate.bzl', 'should_generate_rule')",
        "proto_library(name = 'proto', srcs = ['A.proto'])",
        "should_generate_rule(name = 'simple', proto_dep = ':proto')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    StarlarkInfo boolProvider = (StarlarkInfo) target.get(boolProviderId);
    assertThat(boolProvider.getValue("value", Boolean.class)).isTrue();
  }

  /** Verifies <code>proto_common.should_generate_code</code> call. */
  @Test
  public void shouldGenerateCode_dontGenerate() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:should_generate.bzl', 'should_generate_rule')",
        "should_generate_rule(name = 'simple', proto_dep = '//third_party/x:denied')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    StarlarkInfo boolProvider = (StarlarkInfo) target.get(boolProviderId);
    assertThat(boolProvider.getValue("value", Boolean.class)).isFalse();
  }

  /** Verifies <code>proto_common.should_generate_code</code> call. */
  @Test
  public void shouldGenerateCode_mixed() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:should_generate.bzl', 'should_generate_rule')",
        "should_generate_rule(name = 'simple', proto_dep = '//third_party/x:mixed')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//bar:simple");

    assertContainsEvent(
        "The 'srcs' attribute of '@//third_party/x:mixed' contains protos for which 'MyRule'"
            + " shouldn't generate code (third_party/x/metadata.proto,"
            + " third_party/x/descriptor.proto), in addition to protos for which it should"
            + " (third_party/x/something.proto).\n"
            + "Separate '@//third_party/x:mixed' into 2 proto_library rules.");
  }

  /** Verifies <code>proto_common.declare_generated_files</code> call. */
  @Test
  public void declareGenerateFiles_basic() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:declare_generated_files.bzl', 'declare_generated_files')",
        "proto_library(name = 'proto', srcs = ['A.proto', 'b/B.proto'])",
        "declare_generated_files(name = 'simple', proto_dep = ':proto', extension = '.cc')");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    assertThat(prettyArtifactNames(target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("bar/A.cc", "bar/b/B.cc");
  }

  /** Verifies <code>proto_common.declare_generated_files</code> call for Python. */
  @Test
  public void declareGenerateFiles_pythonc() throws Exception {
    scratch.file(
        "bar/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//foo:declare_generated_files.bzl', 'declare_generated_files')",
        "proto_library(name = 'proto', srcs = ['my-proto.gen.proto'])",
        "declare_generated_files(name = 'simple', proto_dep = ':proto', extension = '_pb2.py',",
        "  python_names = True)");

    ConfiguredTarget target = getConfiguredTarget("//bar:simple");

    assertThat(prettyArtifactNames(target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("bar/my_proto/gen_pb2.py");
  }
}
