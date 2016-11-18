// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.Iterables.transform;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.analysis.OutputGroupProvider.INTERNAL_SUFFIX;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.SkylarkProviders;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.Arrays;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Skylark aspects
 */
@RunWith(JUnit4.class)
public class SkylarkAspectsTest extends AnalysisTestCase {
  protected boolean keepGoing() {
    return false;
  }

  private static final String LINE_SEPARATOR = System.lineSeparator();

  @Test
  public void testAspect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");
  }

  private Iterable<String> getAspectDescriptions(AnalysisResult analysisResult) {
    return transform(
        analysisResult.getAspects(),
        new Function<AspectValue, String>() {
          @Nullable
          @Override
          public String apply(AspectValue aspectValue) {
            return String.format(
                "%s(%s)",
                aspectValue.getConfiguredAspect().getName(),
                aspectValue.getLabel().toString());
          }
        });
  }

  @Test
  public void testAspectCommandLineLabel() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("//test:aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");
  }

  @Test
  public void testAspectCommandLineRepoLabel() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        scratch.readFile("WORKSPACE"),
        "local_repository(name='local', path='local/repo')"
    );
    scratch.file(
        "local/repo/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("local/repo/BUILD");

    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("@local//:aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("@local//:aspect.bzl%MyAspect(//test:xxx)");
  }

  private Iterable<String> getLabelsToBuild(AnalysisResult analysisResult) {
    return transform(
        analysisResult.getTargetsToBuild(),
        new Function<ConfiguredTarget, String>() {
          @Nullable
          @Override
          public String apply(ConfiguredTarget configuredTarget) {
            return configuredTarget.getLabel().toString();
          }
        });
  }


  @Test
  public void testAspectAllowsFragmentsToBeSpecified() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl, fragments=['jvm'], host_fragments=['cpp'])");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    AspectValue aspectValue = Iterables.getOnlyElement(analysisResult.getAspects());
    AspectDefinition aspectDefinition = aspectValue.getAspect().getDefinition();
    assertThat(
        aspectDefinition.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(Jvm.class, ConfigurationTransition.NONE))
        .isTrue();
    assertThat(
        aspectDefinition.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(Jvm.class, ConfigurationTransition.HOST))
        .isFalse();
    assertThat(
        aspectDefinition.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(CppConfiguration.class, ConfigurationTransition.NONE))
        .isFalse();
    assertThat(
        aspectDefinition.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(CppConfiguration.class, ConfigurationTransition.HOST))
        .isTrue();
  }

  @Test
  public void testAspectPropagating() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   s = set([target.label])",
        "   c = set([ctx.rule.kind])",
        "   for i in ctx.rule.attr.deps:",
        "       s += i.target_labels",
        "       c += i.rule_kinds",
        "   return struct(target_labels = s, rule_kinds = c)",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['deps'],",
        ")");
    scratch.file(
        "test/BUILD",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "java_library(",
        "     name = 'xxx',",
        "     srcs = ['A.java'],",
        "     deps = [':yyy'],",
        ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    AspectValue aspectValue = analysisResult.getAspects().iterator().next();
    SkylarkProviders skylarkProviders =
        aspectValue.getConfiguredAspect().getProvider(SkylarkProviders.class);
    assertThat(skylarkProviders).isNotNull();
    Object names = skylarkProviders.getValue("target_labels");
    assertThat(names).isInstanceOf(SkylarkNestedSet.class);
    assertThat(
            transform(
                (SkylarkNestedSet) names,
                new Function<Object, String>() {
                  @Nullable
                  @Override
                  public String apply(Object o) {
                    assertThat(o).isInstanceOf(Label.class);
                    return o.toString();
                  }
                }))
        .containsExactly("//test:xxx", "//test:yyy");
    Object ruleKinds = skylarkProviders.getValue("rule_kinds");
    assertThat(ruleKinds).isInstanceOf(SkylarkNestedSet.class);
    assertThat((SkylarkNestedSet) ruleKinds).containsExactly("java_library");
  }

  @Test
  public void aspectsPropagatingForDefaultAndImplicit() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   s = set([target.label])",
        "   c = set([ctx.rule.kind])",
        "   a = ctx.rule.attr",
        "   if hasattr(a, '_stl') and a._stl:",
        "       s += a._stl.target_labels",
        "       c += a._stl.rule_kinds",
        "   if hasattr(a, '_stl_default') and a._stl_default:",
        "       s += a._stl_default.target_labels",
        "       c += a._stl_default.rule_kinds",
        "   return struct(target_labels = s, rule_kinds = c)",
        "",
        "def _rule_impl(ctx):",
        "   pass",
        "",
        "my_rule = rule(implementation = _rule_impl,",
        "   attrs = { '_stl' : attr.label(default = Label('//test:xxx')) },",
        ")",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['_stl', '_stl_default'],",
        ")");
    scratch.file(
        "test/BUILD",
        "load('/test/aspect', 'my_rule')",
        "cc_library(",
        "     name = 'xxx',",
        ")",
        "my_rule(",
        "     name = 'yyy',",
        ")"
    );
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:yyy");
    AspectValue aspectValue = analysisResult.getAspects().iterator().next();
    SkylarkProviders skylarkProviders =
        aspectValue.getConfiguredAspect().getProvider(SkylarkProviders.class);
    assertThat(skylarkProviders).isNotNull();
    Object names = skylarkProviders.getValue("target_labels");
    assertThat(names).isInstanceOf(SkylarkNestedSet.class);
    assertThat(
        transform(
            (SkylarkNestedSet) names,
            new Function<Object, String>() {
              @Nullable
              @Override
              public String apply(Object o) {
                assertThat(o).isInstanceOf(Label.class);
                return ((Label) o).getName();
              }
            }))
        .containsExactly("stl", "xxx", "yyy");
  }

  @Test
  public void testAspectWithOutputGroups() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   f = target.output_group('_hidden_top_level" + INTERNAL_SUFFIX + "')",
        "   return struct(output_groups = { 'my_result' : f })",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        ")");
    scratch.file(
        "test/BUILD",
        "java_library(",
        "     name = 'xxx',",
        "     srcs = ['A.java'],",
        ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    AspectValue aspectValue = analysisResult.getAspects().iterator().next();
    OutputGroupProvider outputGroupProvider =
        aspectValue.getConfiguredAspect().getProvider(OutputGroupProvider.class);
    assertThat(outputGroupProvider).isNotNull();
    NestedSet<Artifact> names = outputGroupProvider.getOutputGroup("my_result");
    assertThat(names).isNotEmpty();
    NestedSet<Artifact> expectedSet = getConfiguredTarget("//test:xxx")
        .getProvider(OutputGroupProvider.class)
        .getOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL);
    assertThat(names).containsExactlyElementsIn(expectedSet);
  }

  @Test
  public void testAspectWithOutputGroupsAsList() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   g = target.output_group('_hidden_top_level" + INTERNAL_SUFFIX + "')",
        "   return struct(output_groups = { 'my_result' : [ f for f in g] })",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        ")");
    scratch.file(
        "test/BUILD",
        "java_library(",
        "     name = 'xxx',",
        "     srcs = ['A.java'],",
        ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(
        transform(
            analysisResult.getTargetsToBuild(),
            new Function<ConfiguredTarget, String>() {
              @Nullable
              @Override
              public String apply(ConfiguredTarget configuredTarget) {
                return configuredTarget.getLabel().toString();
              }
            }))
        .containsExactly("//test:xxx");
    AspectValue aspectValue = analysisResult.getAspects().iterator().next();
    OutputGroupProvider outputGroupProvider =
        aspectValue.getConfiguredAspect().getProvider(OutputGroupProvider.class);
    assertThat(outputGroupProvider).isNotNull();
    NestedSet<Artifact> names = outputGroupProvider.getOutputGroup("my_result");
    assertThat(names).isNotEmpty();
    NestedSet<Artifact> expectedSet = getConfiguredTarget("//test:xxx")
        .getProvider(OutputGroupProvider.class)
        .getOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL);
    assertThat(names).containsExactlyElementsIn(expectedSet);
  }


  @Test
  public void testAspectsFromSkylarkRules() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   s = set([target.label])",
        "   for i in ctx.rule.attr.deps:",
        "       s += i.target_labels",
        "   return struct(target_labels = s)",
        "",
        "def _rule_impl(ctx):",
        "   s = set([])",
        "   for i in ctx.attr.attr:",
        "       s += i.target_labels",
        "   return struct(rule_deps = s)",
        "",
        "MyAspect = aspect(",
        "   implementation=_aspect_impl,",
        "   attr_aspects=['deps'],",
        ")",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : ",
        "             attr.label_list(mandatory=True, allow_files=True, aspects = [MyAspect]) },",
        ")");

    scratch.file(
        "test/BUILD",
        "load('/test/aspect', 'my_rule')",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = [':yyy'],",
        ")");

    AnalysisResult analysisResult = update("//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    ConfiguredTarget target = analysisResult.getTargetsToBuild().iterator().next();
    SkylarkProviders skylarkProviders = target.getProvider(SkylarkProviders.class);
    assertThat(skylarkProviders).isNotNull();
    Object names = skylarkProviders.getValue("rule_deps");
    assertThat(names).isInstanceOf(SkylarkNestedSet.class);
    assertThat(
        transform(
            (SkylarkNestedSet) names,
            new Function<Object, String>() {
              @Nullable
              @Override
              public String apply(Object o) {
                assertThat(o).isInstanceOf(Label.class);
                return o.toString();
              }
            }))
        .containsExactly("//test:yyy");
  }

  @Test
  public void testAspectOnLabelAttr() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   return struct(aspect_data='foo')",
        "",
        "def _rule_impl(ctx):",
        "   return struct(data=ctx.attr.attr.aspect_data)",
        "",
        "MyAspect = aspect(",
        "   implementation=_aspect_impl,",
        ")",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : ",
        "             attr.label(aspects = [MyAspect]) },",
        ")");

    scratch.file(
        "test/BUILD",
        "load('/test/aspect', 'my_rule')",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = ':yyy',",
        ")");

    AnalysisResult analysisResult = update("//test:xxx");
    ConfiguredTarget target = analysisResult.getTargetsToBuild().iterator().next();
    SkylarkProviders skylarkProviders = target.getProvider(SkylarkProviders.class);
    Object value = skylarkProviders.getValue("data");
    assertThat(value).isEqualTo("foo");
  }

  @Test
  public void testAspectsDoNotAttachToFiles() throws Exception {
    FileSystemUtils.appendIsoLatin1(scratch.resolve("WORKSPACE"),
        "bind(name = 'yyy', actual = '//test:zzz.jar')");
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['deps'],",
        ")");
    scratch.file("test/zzz.jar");
    scratch.file(
        "test/BUILD",
        "exports_files(['zzz.jar'])",
        "java_library(",
        "     name = 'xxx',",
        "     srcs = ['A.java'],",
        "     deps = ['//external:yyy'],",
        ")");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException expected) {
      assertThat(expected.getMessage())
          .contains("Analysis of aspect '/test/aspect%MyAspect of //test:xxx' failed");
    }
    assertContainsEvent("//test:aspect.bzl%MyAspect is attached to source file zzz.jar but "
        + "aspects must be attached to rules");
  }

  @Test
  public void testAspectFailingExecution() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return 1/0",
        "",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:1: in "
            + "//test:aspect.bzl%MyAspect aspect on java_library rule //test:xxx: \n"
            + "Traceback (most recent call last):"
            + LINE_SEPARATOR
            + "\tFile \"/workspace/test/BUILD\", line 1"
            + LINE_SEPARATOR
            + "\t\t//test:aspect.bzl%MyAspect(...)"
            + LINE_SEPARATOR
            + "\tFile \"/workspace/test/aspect.bzl\", line 2, in _impl"
            + LINE_SEPARATOR
            + "\t\t1 / 0"
            + LINE_SEPARATOR
            + "integer division by zero");
  }

  @Test
  public void testAspectFailingReturnsNotAStruct() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return 0",
        "",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("Aspect implementation doesn't return a struct");
  }

  @Test
  public void testAspectFailingReturnsUnsafeObject() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def foo():",
        "   return 0",
        "def _impl(target, ctx):",
        "   return struct(x = foo)",
        "",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:1: in //test:aspect.bzl%MyAspect aspect on java_library rule"
        + " //test:xxx: \n"
        + "\n"
        + "\n"
        + "/workspace/test/aspect.bzl:4:11: Value of provider 'x' is of an illegal type: function");
  }

  @Test
  public void testAspectFailingOrphanArtifacts() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "  ctx.new_file('missing_in_action.txt')",
        "  return struct()",
        "",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:1: in "
            + "//test:aspect.bzl%MyAspect aspect on java_library rule //test:xxx: \n"
            + "\n"
            + "\n"
            + "The following files have no generating action:\n"
            + "test/missing_in_action.txt\n");
  }

  @Test
  public void topLevelAspectIsNotAnAspect() throws Exception {
    scratch.file("test/aspect.bzl", "MyAspect = 4");
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("MyAspect from //test:aspect.bzl is not an aspect");
  }


  @Test
  public void duplicateOutputGroups() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "  f = ctx.new_file('f.txt')",
        "  ctx.file_action(f, 'f')",
        "  return struct(output_groups = { 'duplicate' : set([f]) })",
        "",
        "MyAspect = aspect(implementation=_impl)",
        "def _rule_impl(ctx):",
        "  g = ctx.new_file('g.txt')",
        "  ctx.file_action(g, 'g')",
        "  return struct(output_groups = { 'duplicate' : set([g]) })",
        "my_rule = rule(_rule_impl)",
        "def _noop(ctx):",
        "  pass",
        "rbase = rule(_noop, attrs = { 'dep' : attr.label(aspects = [MyAspect]) })"
    );
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule', 'rbase')",
        "my_rule(name = 'xxx')",
        "rbase(name = 'yyy', dep = ':xxx')"
    );

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update("//test:yyy");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("ERROR /workspace/test/BUILD:3:1: Output group duplicate provided twice");
  }

  @Test
  public void outputGroupsFromTwoAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target, ctx):",
        "  f = ctx.new_file(target.label.name + '_a1.txt')",
        "  ctx.file_action(f, 'f')",
        "  return struct(output_groups = { 'a1_group' : set([f]) })",
        "",
        "a1 = aspect(implementation=_a1_impl, attr_aspects = ['dep'])",
        "def _rule_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct()",
        "  og = {k:ctx.attr.dep.output_groups[k] for k in ctx.attr.dep.output_groups}",
        "  return struct(output_groups = og)",
        "my_rule1 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a1]) })",
        "def _a2_impl(target, ctx):",
        "  g = ctx.new_file(target.label.name + '_a2.txt')",
        "  ctx.file_action(g, 'f')",
        "  return struct(output_groups = { 'a2_group' : set([g]) })",
        "",
        "a2 = aspect(implementation=_a2_impl, attr_aspects = ['dep'])",
        "my_rule2 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a2]) })"
        );
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule1', 'my_rule2')",
        "my_rule1(name = 'base')",
        "my_rule1(name = 'xxx', dep = ':base')",
        "my_rule2(name = 'yyy', dep = ':xxx')"
    );


    AnalysisResult analysisResult = update("//test:yyy");
    OutputGroupProvider outputGroupProvider =
        Iterables
            .getOnlyElement(analysisResult.getTargetsToBuild())
            .getProvider(OutputGroupProvider.class);
    assertThat(getOutputGroupContents(outputGroupProvider, "a1_group"))
        .containsExactly("test/base_a1.txt");
    assertThat(getOutputGroupContents(outputGroupProvider, "a2_group"))
        .containsExactly("test/xxx_a2.txt");
  }

  @Test
  public void duplicateOutputGroupsFromTwoAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target, ctx):",
        "  f = ctx.new_file(target.label.name + '_a1.txt')",
        "  ctx.file_action(f, 'f')",
        "  return struct(output_groups = { 'a1_group' : set([f]) })",
        "",
        "a1 = aspect(implementation=_a1_impl, attr_aspects = ['dep'])",
        "def _rule_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct()",
        "  og = {k:ctx.attr.dep.output_groups[k] for k in ctx.attr.dep.output_groups}",
        "  return struct(output_groups = og)",
        "my_rule1 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a1]) })",
        "def _a2_impl(target, ctx):",
        "  g = ctx.new_file(target.label.name + '_a2.txt')",
        "  ctx.file_action(g, 'f')",
        "  return struct(output_groups = { 'a1_group' : set([g]) })",
        "",
        "a2 = aspect(implementation=_a2_impl, attr_aspects = ['dep'])",
        "my_rule2 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a2]) })"
    );
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule1', 'my_rule2')",
        "my_rule1(name = 'base')",
        "my_rule1(name = 'xxx', dep = ':base')",
        "my_rule2(name = 'yyy', dep = ':xxx')"
    );


    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult analysisResult = update("//test:yyy");
      assertThat(analysisResult.hasError()).isTrue();
      assertThat(keepGoing()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expected.
    }
    assertContainsEvent("ERROR /workspace/test/BUILD:3:1: Output group a1_group provided twice");
  }


  private static Iterable<String> getOutputGroupContents(OutputGroupProvider outputGroupProvider,
      String groupName) {
    return Iterables.transform(outputGroupProvider.getOutputGroup(groupName),
        new Function<Artifact, String>() {
          @Override
          public String apply(Artifact artifact) {
             return artifact.getRootRelativePathString();
          }
        });
  }


  @Test
  public void duplicateSkylarkProviders() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "  return struct(duplicate = 'x')",
        "",
        "MyAspect = aspect(implementation=_impl)",
        "def _rule_impl(ctx):",
        "  return struct(duplicate = 'y')",
        "my_rule = rule(_rule_impl)",
        "def _noop(ctx):",
        "  pass",
        "rbase = rule(_noop, attrs = { 'dep' : attr.label(aspects = [MyAspect]) })"
    );
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule', 'rbase')",
        "my_rule(name = 'xxx')",
        "rbase(name = 'yyy', dep = ':xxx')"
    );

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update("//test:yyy");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("ERROR /workspace/test/BUILD:3:1: Provider duplicate provided twice");
  }


  @Test
  public void topLevelAspectDoesNotExist() throws Exception {
    scratch.file("test/aspect.bzl", "");
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("MyAspect from //test:aspect.bzl is not an aspect");
  }

  @Test
  public void topLevelAspectDoesNotExist2() throws Exception {
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "Extension file not found. Unable to load file '//test:aspect.bzl': "
        + "file doesn't exist or isn't a file");
  }

  @Test
  public void topLevelAspectDoesNotExistNoBuildFile() throws Exception {
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("foo/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "Every .bzl file must have a corresponding package, but 'foo' does not have one. "
        + "Please create a BUILD file in the same or any parent directory. "
        + "Note that this BUILD file does not need to do anything except exist.");
  }

  @Test
  public void testAspectParametersUncovered() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectUncovered = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectUncovered]) },",
        ")");
    scratch.file("test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent(//"ERROR /workspace/test/aspect.bzl:9:11: "
        "Aspect //test:aspect.bzl%MyAspectUncovered requires rule my_rule to specify attribute "
        + "'my_attr' with type string.");
  }

  @Test
  public void testAspectParametersTypeMismatch() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectMismatch = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectMismatch]),",
        "              'my_attr' : attr.int() },",
       ")");
    scratch.file("test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr = 4)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent(
        "Aspect //test:aspect.bzl%MyAspectMismatch requires rule my_rule to specify attribute "
        + "'my_attr' with type string.");
  }

  @Test
  public void testAspectParametersBadDefault() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectBadDefault = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['a'], default='b') },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectBadDefault]) },",
        ")");
    scratch.file("test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent("ERROR /workspace/test/aspect.bzl:5:22: "
        + "Aspect parameter attribute 'my_attr' has a bad default value: has to be one of 'a' "
        + "instead of 'b'");
  }

  @Test
  public void testAspectParametersBadValue() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectBadValue = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['a']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectBadValue]),",
        "              'my_attr' : attr.string() },",
        ")");
    scratch.file("test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr='b')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent("ERROR /workspace/test/BUILD:2:1: //test:xxx: invalid value in 'my_attr' "
        + "attribute: has to be one of 'a' instead of 'b'");
  }

  @Test
  public void testAspectParameters() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspect = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspect]),",
        "              'my_attr' : attr.string() },",
        ")");
    scratch.file("test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr = 'aaa')");

    AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testAspectParametersOptional() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectOptParam = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa'], default='aaa') },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectOptParam]) },",
        ")");
    scratch.file("test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx')");

    AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testAspectParametersOptionalOverride() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   if (ctx.attr.my_attr == 'a'):",
        "       fail('Rule is not overriding default, still has value ' + ctx.attr.my_attr)",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectOptOverride = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['a', 'b'], default='a') },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectOptOverride]),",
        "              'my_attr' : attr.string() },",
        ")");
    scratch.file("test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr = 'b')");

    AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void multipleExecutablesInTarget() throws Exception {
    scratch.file("foo/extension.bzl",
        "def _aspect_impl(target, ctx):",
        "   return struct()",
        "my_aspect = aspect(_aspect_impl)",
        "def _main_rule_impl(ctx):",
        "   pass",
        "my_rule = rule(_main_rule_impl,",
        "   attrs = { ",
        "      'exe1' : attr.label(executable = True, allow_files = True),",
        "      'exe2' : attr.label(executable = True, allow_files = True),",
        "   },",
        ")"
    );

    scratch.file("foo/tool.sh", "#!/bin/bash");
    scratch.file("foo/BUILD",
        "load('extension',  'my_rule')",
        "my_rule(name = 'main', exe1 = ':tool.sh', exe2 = ':tool.sh')"
    );
    AnalysisResult analysisResultOfRule =
        update(ImmutableList.<String>of(), "//foo:main");
    assertThat(analysisResultOfRule.hasError()).isFalse();

    AnalysisResult analysisResultOfAspect =
        update(ImmutableList.<String>of("/foo/extension.bzl%my_aspect"), "//foo:main");
    assertThat(analysisResultOfAspect.hasError()).isFalse();
  }


  @Test
  public void testAspectFragmentAccessSuccess() throws Exception {
    getConfiguredTargetForAspectFragment(
        "ctx.fragments.cpp.compiler", "'cpp'", "", "", "");
    assertNoEvents();
  }

  @Test
  public void testAspectHostFragmentAccessSuccess() throws Exception {
    getConfiguredTargetForAspectFragment(
        "ctx.host_fragments.cpp.compiler", "", "'cpp'", "", "");
    assertNoEvents();
  }

  @Test
  public void testAspectFragmentAccessError() throws Exception {
    reporter.removeHandler(failFastHandler);
    try {
      getConfiguredTargetForAspectFragment(
          "ctx.fragments.cpp.compiler", "'java'", "'cpp'", "'cpp'", "");
      fail("update() should have failed");
    } catch (ViewCreationFailedException e) {
      // expected
    }
    assertContainsEvent(
        "//test:aspect.bzl%MyAspect aspect on my_rule has to declare 'cpp' as a "
            + "required fragment in target configuration in order to access it. Please update the "
            + "'fragments' argument of the rule definition "
            + "(for example: fragments = [\"cpp\"])");
  }

  @Test
  public void testAspectHostFragmentAccessError() throws Exception {
    reporter.removeHandler(failFastHandler);
    try {
      getConfiguredTargetForAspectFragment(
          "ctx.host_fragments.cpp.compiler", "'cpp'", "'java'", "", "'cpp'");
      fail("update() should have failed");
    } catch (ViewCreationFailedException e) {
      // expected
    }
    assertContainsEvent(
        "//test:aspect.bzl%MyAspect aspect on my_rule has to declare 'cpp' as a "
            + "required fragment in host configuration in order to access it. Please update the "
            + "'host_fragments' argument of the rule definition "
            + "(for example: host_fragments = [\"cpp\"])");
  }

  private ConfiguredTarget getConfiguredTargetForAspectFragment(
      String fullFieldName,
      String fragments,
      String hostFragments,
      String ruleFragments,
      String ruleHostFragments)
      throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   return struct(result = str(" + fullFieldName + "))",
        "",
        "def _rule_impl(ctx):",
        "   return struct(stuff = '...')",
        "",
        "MyAspect = aspect(",
        "   implementation=_aspect_impl,",
        "   attr_aspects=['deps'],",
        "   fragments=[" + fragments + "],",
        "   host_fragments=[" + hostFragments + "],",
        ")",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : ",
        "             attr.label_list(mandatory=True, allow_files=True, aspects = [MyAspect]) },",
        "   fragments=[" + ruleFragments + "],",
        "   host_fragments=[" + ruleHostFragments + "],",
        ")");
    scratch.file(
        "test/BUILD",
        "load('/test/aspect', 'my_rule')",
        "exports_files(['zzz'])",
        "my_rule(",
        "     name = 'yyy',",
        "     attr = ['zzz'],",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = ['yyy'],",
        ")");

    AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    if (result.hasError()) {
      assertThat(keepGoing()).isTrue();
      throw new ViewCreationFailedException("Analysis failed");
    }

    return getConfiguredTarget("//test:xxx");
  }

  @Test
  public void invalidateAspectOnBzlFileChange() throws Exception {
    scratch.file("test/build_defs.bzl", aspectBzlFile("'deps'"));
    scratch.file(
        "test/BUILD",
        "load('build_defs', 'repro', 'repro_no_aspect')",
        "repro_no_aspect(name = 'r0')",
        "repro_no_aspect(name = 'r1', deps = [':r0'])",
        "repro(name = 'r2', deps = [':r1'])");
    buildTargetAndCheckRuleInfo("//test:r0", "//test:r1");

    // Make aspect propagation list empty.
    scratch.overwriteFile("test/build_defs.bzl", aspectBzlFile(""));

    // The aspect should not propagate to //test:r0 anymore.
    buildTargetAndCheckRuleInfo("//test:r1");
  }

  private void buildTargetAndCheckRuleInfo(String... expectedLabels) throws Exception {
    AnalysisResult result = update(ImmutableList.<String>of(), "//test:r2");
    ConfiguredTarget configuredTarget = result.getTargetsToBuild().iterator().next();
    SkylarkNestedSet ruleInfoValue =
        (SkylarkNestedSet)
            configuredTarget.getProvider(SkylarkProviders.class).getValue("rule_info");
    assertThat(ruleInfoValue.getSet(String.class))
        .containsExactlyElementsIn(Arrays.asList(expectedLabels));
  }

  private String[] aspectBzlFile(String attrAspects) {
    return new String[] {
        "def _repro_aspect_impl(target, ctx):",
        "    s = set([str(target.label)])",
        "    for d in ctx.rule.attr.deps:",
        "       if hasattr(d, 'aspect_info'):",
        "         s = s | d.aspect_info",
        "    return struct(aspect_info = s)",
        "",
        "_repro_aspect = aspect(",
        "    _repro_aspect_impl,",
        "    attr_aspects = [" + attrAspects + "],",
        ")",
        "",
        "def repro_impl(ctx):",
        "    s = set()",
        "    for d in ctx.attr.deps:",
        "       if hasattr(d, 'aspect_info'):",
        "         s = s | d.aspect_info",
        "    return struct(rule_info = s)",
        "",
        "def repro_no_aspect_impl(ctx):",
        "    pass",
        "",
        "repro_no_aspect = rule(implementation = repro_no_aspect_impl,",
        "             attrs = {",
        "                       'deps': attr.label_list(",
        "                             allow_files = True,",
        "                       )",
        "                      },",
        ")",
        "",
        "repro = rule(implementation = repro_impl,",
        "             attrs = {",
        "                       'deps': attr.label_list(",
        "                             allow_files = True,",
        "                             aspects = [_repro_aspect],",
        "                       )",
        "                      },",
        ")"
    };
  }


  @Test
  public void aspectOutputsToBinDirectory() throws Exception {
    scratch.file("foo/extension.bzl",
        "def _aspect_impl(target, ctx):",
        "   file = ctx.new_file('aspect-output-' + target.label.name)",
        "   ctx.file_action(file, 'data')",
        "   return struct(aspect_file = file)",
        "my_aspect = aspect(_aspect_impl)",
        "def _rule_impl(ctx):",
        "   pass",
        "rule_bin_out = rule(_rule_impl, output_to_genfiles=False)",
        "rule_gen_out = rule(_rule_impl, output_to_genfiles=True)",
        "def _main_rule_impl(ctx):",
        "   s = set()",
        "   for d in ctx.attr.deps:",
        "       s = s | set([d.aspect_file])",
        "   return struct(aspect_files = s)",
        "main_rule = rule(_main_rule_impl,",
        "   attrs = { 'deps' : attr.label_list(aspects = [my_aspect]) },",
        ")"
    );

    scratch.file("foo/BUILD",
        "load('extension', 'rule_bin_out', 'rule_gen_out', 'main_rule')",
        "rule_bin_out(name = 'rbin')",
        "rule_gen_out(name = 'rgen')",
        "main_rule(name = 'main', deps = [':rbin', ':rgen'])"
    );
    AnalysisResult analysisResult = update(ImmutableList.<String>of(), "//foo:main");
    ConfiguredTarget target = analysisResult.getTargetsToBuild().iterator().next();
    NestedSet<Artifact> aspectFiles =
        ((SkylarkNestedSet) target.getProvider(SkylarkProviders.class).getValue("aspect_files"))
            .getSet(Artifact.class);
    assertThat(transform(aspectFiles, new Function<Artifact, String>() {
      @Override
      public String apply(Artifact artifact) {
        return artifact.getFilename();
      }
    })).containsExactly("aspect-output-rbin", "aspect-output-rgen");
    for (Artifact aspectFile : aspectFiles) {
      String rootPath = aspectFile.getRoot().getExecPath().toString();
      assertWithMessage("Artifact %s should not be in genfiles", aspectFile)
          .that(rootPath).doesNotContain("genfiles");
      assertWithMessage("Artifact %s should be in bin", aspectFile)
          .that(rootPath).endsWith("bin");
    }
  }

  @Test
  public void toplevelAspectOnFile() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "exports_files(['file.txt'])");
    scratch.file("test/file.txt", "");
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:file.txt");
    assertThat(analysisResult.hasError()).isFalse();
    assertThat(analysisResult.getAspects()).isEmpty();
  }

  @Test
  public void sharedAttributeDefintionWithAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target,ctx):",
        "  return struct()",
        "my_aspect = aspect(implementation = _aspect_impl)",
        "_ATTR = { 'deps' : attr.label_list(aspects = [my_aspect]) }",
        "def _dummy_impl(ctx):",
        "  pass",
        "r1 = rule(_dummy_impl, attrs =  _ATTR)",
        "r2 = rule(_dummy_impl, attrs =  _ATTR)"
    );

    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 't1')",
        "r2(name = 't2', deps = [':t1'])"
    );
    AnalysisResult analysisResult = update("//test:t2");
    assertThat(analysisResult.hasError()).isFalse();
  }

  @Test
  public void multipleAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target,ctx):",
        "  return struct()",
        "my_aspect = aspect(implementation = _aspect_impl)",
        "def _dummy_impl(ctx):",
        "  pass",
        "r1 = rule(_dummy_impl, ",
        "          attrs = { 'deps' : attr.label_list(aspects = [my_aspect, my_aspect]) })"
    );

    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1')",
        "r1(name = 't1')"
    );
    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update("//test:r1");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (TargetParsingException | ViewCreationFailedException expected) {
      // expected.
    }
    assertContainsEvent("Aspect //test:aspect.bzl%my_aspect added more than once");
  }

  @Test
  public void topLevelAspectsAndExtraActions() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target,ctx):",
        "  f = ctx.new_file('dummy.txt')",
        "  ctx.action(outputs = [f], command='echo xxx > $(location f)', mnemonic='AspectAction')",
        "  return struct()",
        "my_aspect = aspect(implementation = _aspect_impl)"
    );
    scratch.file(
        "test/BUILD",
        "extra_action(",
        "    name = 'xa',",
        "    cmd = 'echo $(EXTRA_ACTION_FILE) > $(output file.xa)',",
        "    out_templates = ['file.xa'],",
        ")",
        "action_listener(",
        "    name = 'al',",
        "    mnemonics = [ 'AspectAction' ],",
        "    extra_actions = [ ':xa' ])",
        "java_library(name = 'xxx')"
    );
    useConfiguration("--experimental_action_listener=//test:al");
    AnalysisResult analysisResult = update(
        ImmutableList.<String>of("test/aspect.bzl%my_aspect"),
        "//test:xxx");
    assertThat(Iterables.transform(analysisResult.getAdditionalArtifactsToBuild(),
        new Function<Artifact, String>() {
          @Override
          public String apply(Artifact artifact) {
            return artifact.getFilename();
          }
        })).contains("file.xa");
  }

  @Test
  public void aspectsPropagatingToAllAttributes() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   s = set([target.label])",
        "   if hasattr(ctx.rule.attr, 'runtime_deps'):",
        "     for i in ctx.rule.attr.runtime_deps:",
        "       s += i.target_labels",
        "   return struct(target_labels = s)",
        "",
        "MyAspect = aspect(",
        "    implementation=_impl,",
        "    attrs = { '_tool' : attr.label(default = Label('//test:tool')) },",
        "    attr_aspects=['*'],",
        ")");
    scratch.file(
        "test/BUILD",
        "java_library(",
        "    name = 'tool',",
        ")",
        "java_library(",
        "     name = 'bar',",
        "     runtime_deps = [':tool'],",
        ")",
        "java_library(",
        "     name = 'foo',",
        "     runtime_deps = [':bar'],",
        ")");
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:foo");
    AspectValue aspectValue = analysisResult.getAspects().iterator().next();
    SkylarkProviders skylarkProviders =
        aspectValue.getConfiguredAspect().getProvider(SkylarkProviders.class);
    assertThat(skylarkProviders).isNotNull();
    Object names = skylarkProviders.getValue("target_labels");
    assertThat(names).isInstanceOf(SkylarkNestedSet.class);
    assertThat(
        transform(
              (SkylarkNestedSet) names,
              new Function<Object, String>() {
                @Nullable
                @Override
                public String apply(Object o) {
                  assertThat(o).isInstanceOf(Label.class);
                  return ((Label) o).getName();
                }
              }))
        .containsExactly("foo", "bar", "tool");
  }

  /**
   * Simple straightforward linear aspects-on-aspects.
   */
  @Test
  public void aspectOnAspectLinear() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = a1p(text = 'random'))",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "  value = []",
        "  if hasattr(ctx.rule.attr.dep, 'a2p'):",
        "     value += ctx.rule.attr.dep.a2p.value",
        "  if hasattr(target, 'a1p'):",
        "     value.append(str(target.label) + '=yes')",
        "  else:",
        "     value.append(str(target.label) + '=no')",
        "  return struct(a2p = a2p(value = value))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})"
    );
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 'r0')",
        "r1(name = 'r1', dep = ':r0')",
        "r2(name = 'r2', dep = ':r1')"
    );
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    SkylarkList result = (SkylarkList) target.get("result");

    // "yes" means that aspect a2 sees a1's providers.
    assertThat(result).containsExactly("//test:r0=yes", "//test:r1=no");
  }

  /**
   * Diamond case.
   * rule r1 depends or r0 with aspect a1.
   * rule r2 depends or r0 with aspect a2.
   * rule rcollect depends on r1, r2 with aspect a3.
   *
   * Aspect a3 should be applied twice to target r0: once in [a1, a3] sequence
   * and once in [a2, a3] sequence.
   */
  @Test
  public void aspectOnAspectDiamond() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = 'text from a1')",
        "a1 = aspect(_a1_impl, attr_aspects = ['deps'])",
        "",
        "def _a2_impl(target,ctx):",
        "  return struct(a2p = 'text from a2')",
        "a2 = aspect(_a2_impl, attr_aspects = ['deps'])",
        "",
        "def _a3_impl(target,ctx):",
        "  value = []",
        "  f = ctx.new_file('a3.out')",
        "  ctx.file_action(f, 'text')",
        "  for dep in ctx.rule.attr.deps:",
        "     if hasattr(dep, 'a3p'):",
        "         value += dep.a3p",
        "  s = str(target.label) + '='",
        "  if hasattr(target, 'a1p'):",
        "     s += 'a1p'",
        "  if hasattr(target, 'a2p'):",
        "     s += 'a2p'",
        "  value.append(s)",
        "  return struct(a3p = value)",
        "a3 = aspect(_a3_impl, attr_aspects = ['deps'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _rcollect_impl(ctx):",
        "  value = []",
        "  for dep in ctx.attr.deps:",
        "     if hasattr(dep, 'a3p'):",
        "         value += dep.a3p",
        "  return struct(result = value)",
        "r1 = rule(_r1_impl, attrs = { 'deps' : attr.label_list(aspects = [a1])})",
        "r2 = rule(_r1_impl, attrs = { 'deps' : attr.label_list(aspects = [a2])})",
        "rcollect = rule(_rcollect_impl, attrs = { 'deps' : attr.label_list(aspects = [a3])})"
    );
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2', 'rcollect')",
        "r1(name = 'r0')",
        "r1(name = 'r1', deps = [':r0'])",
        "r2(name = 'r2', deps = [':r0'])",
        "rcollect(name = 'rcollect', deps = [':r1', ':r2'])"
    );
    AnalysisResult analysisResult = update("//test:rcollect");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    SkylarkList result = (SkylarkList) target.get("result");
    assertThat(result).containsExactly(
        "//test:r0=a1p", "//test:r1=", "//test:r0=a2p", "//test:r2=");
  }

  /**
   * Linear with duplicates.
   * r2_1 depends on r0 with aspect a2.
   * r1 depends on r2_1 with aspect a1.
   * r2 depends on r1 with aspect a2.
   *
   * There should be just one instance of aspect a2 on r0, and is should *not* see a1.
   */
  @Test
  public void aspectOnAspectLinearDuplicates() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = 'a1p')",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "  value = []",
        "  if hasattr(ctx.rule.attr.dep, 'a2p'):",
        "     value += ctx.rule.attr.dep.a2p.value",
        "  if hasattr(target, 'a1p'):",
        "     value.append(str(target.label) + '=yes')",
        "  else:",
        "     value.append(str(target.label) + '=no')",
        "  return struct(a2p = a2p(value = value))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})"
    );
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 'r0')",
        "r2(name = 'r2_1', dep = ':r0')",
        "r1(name = 'r1', dep = ':r2_1')",
        "r2(name = 'r2', dep = ':r1')"
    );
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    SkylarkList result = (SkylarkList) target.get("result");
    // "yes" means that aspect a2 sees a1's providers.
    assertThat(result).containsExactly("//test:r0=no", "//test:r1=no", "//test:r2_1=yes");
  }



  @RunWith(JUnit4.class)
  public static final class WithKeepGoing extends SkylarkAspectsTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return new FlagBuilder().with(Flag.KEEP_GOING);
    }

    @Override
    protected boolean keepGoing() {
      return true;
    }
  }
}
