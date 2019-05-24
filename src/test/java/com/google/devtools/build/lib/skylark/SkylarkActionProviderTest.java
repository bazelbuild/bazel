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
package com.google.devtools.build.lib.skylark;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the Skylark-accessible actions provider on rule configured targets. */
@RunWith(JUnit4.class)
public class SkylarkActionProviderTest extends AnalysisTestCase {

  @Test
  public void aspectGetsActionProviderForNativeRule() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "foo = provider()",
        "def _impl(target, ctx):",
        "   return [foo(actions = target.actions)]",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file(
        "test/BUILD",
        "genrule(",
        "   name = 'xxx',",
        "   cmd = 'echo \"hello\" > $@',",
        "   outs = ['mygen.out']",
        ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");

    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspects()).getConfiguredAspect();

    SkylarkKey fooKey =
        new SkylarkKey(Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "foo");

    StructImpl fooProvider = (StructImpl) configuredAspect.get(fooKey);
    assertThat(fooProvider.getValue("actions")).isNotNull();
    @SuppressWarnings("unchecked")
    SkylarkList<ActionAnalysisMetadata> actions =
        (SkylarkList<ActionAnalysisMetadata>) fooProvider.getValue("actions");
    assertThat(actions).isNotEmpty();

    ActionAnalysisMetadata action = actions.get(0);
    assertThat(action.getMnemonic()).isEqualTo("Genrule");
  }

  @Test
  @SuppressWarnings("unchecked")
  public void aspectGetsActionProviderForSkylarkRule() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "foo = provider()",
        "def _impl(target, ctx):",
        "   mnemonics = [a.mnemonic for a in target.actions]",
        "   envs = [a.env for a in target.actions]",
        "   inputs = [a.inputs.to_list() for a in target.actions]",
        "   outputs = [a.outputs.to_list() for a in target.actions]",
        "   argv = [a.argv for a in target.actions]",
        "   return [foo(",
        "       actions = target.actions,",
        "       mnemonics = mnemonics,",
        "       envs = envs,",
        "       inputs = inputs,",
        "       outputs = outputs,",
        "       argv = argv",
        "    )]",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file(
        "test/rule.bzl",
        "def impl(ctx):",
        "  output_file0 = ctx.actions.declare_file('myfile0')",
        "  output_file1 = ctx.actions.declare_file('myfile1')",
        "  executable = ctx.actions.declare_file('executable')",
        "  ctx.actions.run(outputs=[output_file0], executable=executable,",
        "      mnemonic='MyAction0', env={'foo':'bar', 'pet':'puppy'})",
        "  ctx.actions.run_shell(outputs=[executable, output_file1],",
        "      command='fakecmd', mnemonic='MyAction1', env={'pet':'bunny'})",
        "  return None",
        "my_rule = rule(impl)");
    scratch.file(
        "test/BUILD", "load('//test:rule.bzl', 'my_rule')", "my_rule(", "   name = 'xxx',", ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");

    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspects()).getConfiguredAspect();

    SkylarkKey fooKey =
        new SkylarkKey(Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "foo");
    StructImpl fooProvider = (StructImpl) configuredAspect.get(fooKey);
    assertThat(fooProvider.getValue("actions")).isNotNull();

    SkylarkList<ActionAnalysisMetadata> actions =
        (SkylarkList<ActionAnalysisMetadata>) fooProvider.getValue("actions");
    assertThat(actions).hasSize(2);

    SkylarkList<String> mnemonics =
        (SkylarkList<String>) fooProvider.getValue("mnemonics");
    assertThat(mnemonics).containsExactly("MyAction0", "MyAction1");

    SkylarkList<SkylarkDict<String, String>> envs =
        (SkylarkList<SkylarkDict<String, String>>) fooProvider.getValue("envs");
    assertThat(envs).containsExactly(
        SkylarkDict.of(null, "foo", "bar", "pet", "puppy"),
        SkylarkDict.of(null, "pet", "bunny"));

    SkylarkList<SkylarkList<Artifact>> inputs =
        (SkylarkList<SkylarkList<Artifact>>) fooProvider.getValue("inputs");
    assertThat(flattenArtifactNames(inputs)).containsExactly("executable");

    SkylarkList<SkylarkList<Artifact>> outputs =
        (SkylarkList<SkylarkList<Artifact>>) fooProvider.getValue("outputs");
    assertThat(flattenArtifactNames(outputs)).containsExactly("myfile0", "executable", "myfile1");

    SkylarkList<SkylarkList<String>> argv =
        (SkylarkList<SkylarkList<String>>) fooProvider.getValue("argv");
    assertThat(argv.get(0)).hasSize(1);
    assertThat(argv.get(0).get(0)).endsWith("executable");
    assertThat(argv.get(1)).contains("fakecmd");
  }

  private static List<String> flattenArtifactNames(
      SkylarkList<SkylarkList<Artifact>> artifactLists) {
    return artifactLists.stream()
        .flatMap(artifacts -> artifacts.stream())
        .map(artifact -> artifact.getFilename())
        .collect(Collectors.toList());
  }
}
