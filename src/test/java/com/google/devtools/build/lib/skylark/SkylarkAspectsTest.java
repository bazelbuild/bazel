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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import javax.annotation.Nullable;

/**
 * Tests for Skylark aspects
 */
@RunWith(JUnit4.class)
public class SkylarkAspectsTest extends AnalysisTestCase {
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
    assertThat(
            transform(
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
                }))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");
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
    AspectKey aspectKey = aspectValue.getKey();
    AspectDefinition aspectDefinition = aspectKey.getAspect().getDefinition();
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
  public void testAspectWithOutputGroups() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   f = target.output_group('_hidden_top_level')",
        "   return struct(output_groups = { 'my_result' : f })",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['deps'],",
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
      update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:1: in java_library rule //test:xxx: \n"
            + "Traceback (most recent call last):\n"
            + "\tFile \"/workspace/test/BUILD\", line 1\n"
            + "\t\t//test:aspect.bzl%MyAspect(...)\n"
            + "\tFile \"/workspace/test/aspect.bzl\", line 2, in _impl\n"
            + "\t\t1 / 0\n"
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
      update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
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
      update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:1: in java_library rule //test:xxx: \n"
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
      update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:1: in java_library rule //test:xxx: \n"
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
      update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("MyAspect from //test:aspect.bzl is not an aspect");
  }

  @Test
  public void topLevelAspectDoesNotExist() throws Exception {
    scratch.file("test/aspect.bzl", "");
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
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
      update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
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
      update(ImmutableList.of("foo/aspect.bzl%MyAspect"), "//test:xxx");
      fail();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "Every .bzl file must have a corresponding package, but 'foo' does not have one. "
        + "Please create a BUILD file in the same or any parent directory. "
        + "Note that this BUILD file does not need to do anything except exist.");
  }
}
