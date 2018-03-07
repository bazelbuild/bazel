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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.syntax.SkylarkList;
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

    SkylarkKey fooKey = new SkylarkKey(Label.parseAbsolute("//test:aspect.bzl"), "foo");

    assertThat(configuredAspect.get(fooKey).getValue("actions")).isNotNull();
    @SuppressWarnings("unchecked")
    SkylarkList<ActionAnalysisMetadata> actions =
        (SkylarkList<ActionAnalysisMetadata>) configuredAspect.get(fooKey).getValue("actions");
    assertThat(actions).isNotEmpty();

    ActionAnalysisMetadata action = actions.get(0);
    assertThat(action.getMnemonic()).isEqualTo("Genrule");
  }

  @Test
  public void aspectGetsActionProviderForSkylarkRule() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "foo = provider()",
        "def _impl(target, ctx):",
        "   return [foo(actions = target.actions)]",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file(
        "test/rule.bzl",
        "def impl(ctx):",
        "  output_file0 = ctx.new_file('myfile0')",
        "  output_file1 = ctx.new_file('myfile1')",
        "  ctx.action(outputs=[output_file0], command='fakecmd0', mnemonic='MyAction0')",
        "  ctx.action(outputs=[output_file1], command='fakecmd1', mnemonic='MyAction1')",
        "  return None",
        "my_rule = rule(impl)");
    scratch.file(
        "test/BUILD", "load('//test:rule.bzl', 'my_rule')", "my_rule(", "   name = 'xxx',", ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");

    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspects()).getConfiguredAspect();

    SkylarkKey fooKey = new SkylarkKey(Label.parseAbsolute("//test:aspect.bzl"), "foo");

    assertThat(configuredAspect.get(fooKey).getValue("actions")).isNotNull();
    @SuppressWarnings("unchecked")
    SkylarkList<ActionAnalysisMetadata> actions =
        (SkylarkList<ActionAnalysisMetadata>) configuredAspect.get(fooKey).getValue("actions");
    assertThat(actions).hasSize(2);

    assertThat(actions.stream().map(action -> action.getMnemonic()).collect(Collectors.toList()))
        .containsExactly("MyAction0", "MyAction1");
  }
}
