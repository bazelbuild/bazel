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

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.collect.Iterables.transform;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseArtifactNames;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;

import junit.framework.AssertionFailedError;

import java.util.LinkedList;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Integration tests for Skylark.
 */
public class SkylarkIntegrationTest extends BuildViewTestCase {
  private static final Joiner LINE_JOINER = Joiner.on("\n");

  @Override
  public void setUp() throws Exception {
    super.setUp();
  }

  public void testSameMethodNames() throws Exception {
    // The alias feature of load() may hide the fact that two methods in the stack trace have the
    // same name. This is perfectly legal as long as these two methods are actually distinct.
    // Consequently, no "Recursion was detected" error must be thrown.
    scratch.file(
        "test/skylark/extension.bzl",
        "load('/test/skylark/other', other_impl = 'impl')",
        "def impl(ctx):",
        "  other_impl(ctx)",
        "empty = rule(implementation = impl)");
    scratch.file("test/skylark/other.bzl", "def impl(ctx):", "  print('This rule does nothing')");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'empty')",
        "empty(name = 'test_target')");

    getConfiguredTarget("//test/skylark:test_target");
  }

  public void testRecursionDetection() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  _impl(ctx)",
        "empty = rule(implementation = _impl)");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'empty')",
        "empty(name = 'test_target')");

    getConfiguredTarget("//test/skylark:test_target");
    assertContainsEvent("Recursion was detected when calling '_impl' from '_impl'");
  }

  public void testMacroHasGeneratorAttributes() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  print('This rule does nothing')",
        "",
        "empty = rule(implementation = _impl)",
        "no_macro = rule(implementation = _impl)",
        "",
        "def macro(name, visibility=None):",
        "  empty(name = name, visibility=visibility)",
        "def native_macro(name):",
        "  native.cc_library(name = name + '_suffix')");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', macro_rule = 'macro', no_macro_rule = 'no_macro',",
        "  native_macro_rule = 'native_macro')",
        "macro_rule(name = 'macro_target')",
        "no_macro_rule(name = 'no_macro_target')",
        "native_macro_rule(name = 'native_macro_target')",
        "cc_binary(name = 'cc_target', deps = ['cc_dep'])",
        "cc_library(name = 'cc_dep')");

    AttributeContainer withMacro = getContainerForTarget("macro_target");
    assertThat(withMacro.getAttr("generator_name")).isEqualTo("macro_target");
    assertThat(withMacro.getAttr("generator_function")).isEqualTo("macro");
    assertThat(withMacro.getAttr("generator_location")).isEqualTo("test/skylark/BUILD:3");

    // Attributes are only set when the rule was created by a macro
    AttributeContainer noMacro = getContainerForTarget("no_macro_target");
    assertThat(noMacro.getAttr("generator_name")).isEqualTo("");
    assertThat(noMacro.getAttr("generator_function")).isEqualTo("");
    assertThat(noMacro.getAttr("generator_location")).isEqualTo("");

    AttributeContainer nativeMacro = getContainerForTarget("native_macro_target_suffix");
    assertThat(nativeMacro.getAttr("generator_name")).isEqualTo("native_macro_target");
    assertThat(nativeMacro.getAttr("generator_function")).isEqualTo("native_macro");
    assertThat(nativeMacro.getAttr("generator_location")).isEqualTo("test/skylark/BUILD:5");

    AttributeContainer ccTarget = getContainerForTarget("cc_target");
    assertThat(ccTarget.getAttr("generator_name")).isEqualTo("");
    assertThat(ccTarget.getAttr("generator_function")).isEqualTo("");
    assertThat(ccTarget.getAttr("generator_location")).isEqualTo("");
  }

  private AttributeContainer getContainerForTarget(String targetName) throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:" + targetName);
    return target.getTarget().getAssociatedRule().getAttributeContainer();
  }

  public void testStackTraceErrorInFunction() throws Exception {
    runStackTraceTest(
        "str",
        "\t\tstr.index(1)\n"
            + "Method string.index(sub: string, start: int, end: int or NoneType) is not "
            + "applicable for arguments (int, int, NoneType): 'sub' is int, "
            + "but should be string");
  }

  public void testStackTraceMissingMethod() throws Exception {
    runStackTraceTest("None", "\t\tNone.index(1)\n" + "Type NoneType has no function index(int)");
  }

  protected void runStackTraceTest(String object, String errorMessage) throws Exception {
    reporter.removeHandler(failFastHandler);
    String expectedTrace =
        Joiner.on("\n")
            .join(
                "Traceback (most recent call last):",
                "\tFile \"/workspace/test/skylark/BUILD\", line 3",
                "\t\tcustom_rule(name = 'cr')",
                "\tFile \"/workspace/test/skylark/extension.bzl\", line 5, in custom_rule_impl",
                "\t\tfoo()",
                "\tFile \"/workspace/test/skylark/extension.bzl\", line 8, in foo",
                "\t\tbar(2, 4)",
                "\tFile \"/workspace/test/skylark/extension.bzl\", line 10, in bar",
                "\t\tfirst(x, y, z)",
                "\tFile \"/workspace/test/skylark/functions.bzl\", line 2, in first",
                "\t\tsecond(a, b)",
                "\tFile \"/workspace/test/skylark/functions.bzl\", line 5, in second",
                "\t\tthird('legal')",
                "\tFile \"/workspace/test/skylark/functions.bzl\", line 7, in third",
                errorMessage);
    scratch.file(
        "test/skylark/extension.bzl",
        "load('/test/skylark/functions', 'first')",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  ftb = set(attr1)",
        "  foo()",
        "  return struct(provider_key = ftb)",
        "def foo():",
        "  bar(2,4)",
        "def bar(x,y,z=1):",
        "  first(x,y, z)",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)})");
    scratch.file(
        "test/skylark/functions.bzl",
        "def first(a, b, c):",
        "  second(a, b)",
        "  third(b)",
        "def second(a, b):",
        "  third('legal')",
        "def third(str):",
        "  " + object + ".index(1)");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    getConfiguredTarget("//test/skylark:cr");
    assertContainsEvent(expectedTrace);
  }

  public void testFilesToBuild() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  ftb = set(attr1)",
        "  return struct(runfiles = ctx.runfiles(), files = ftb)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertEquals("//test/skylark:cr", target.getLabel().toString());
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("a.txt");
  }

  public void testRunfiles() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  rf = ctx.runfiles(files = attr1)",
        "  return struct(runfiles = rf)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertEquals("//test/skylark:cr", target.getLabel().toString());
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(RunfilesProvider.class).getDefaultRunfiles().getAllArtifacts()))
        .containsExactly("a.txt");
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(RunfilesProvider.class).getDataRunfiles().getAllArtifacts()))
        .containsExactly("a.txt");
  }

  public void testAccessRunfiles() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  runfiles = ctx.attr.x.default_runfiles.files",
        "  return struct(files = runfiles)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'x': attr.label(allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "cc_library(name = 'lib', data = ['a.txt'])",
        "custom_rule(name = 'cr1', x = ':lib')",
        "custom_rule(name = 'cr2', x = 'b.txt')");

    scratch.file("test/skylark/a.txt");
    scratch.file("test/skylark/b.txt");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr1");
    List<String> baseArtifactNames =
        ActionsTestUtil.baseArtifactNames(target.getProvider(FileProvider.class).getFilesToBuild());
    assertThat(baseArtifactNames).containsExactly("a.txt");

    target = getConfiguredTarget("//test/skylark:cr2");
    baseArtifactNames =
        ActionsTestUtil.baseArtifactNames(target.getProvider(FileProvider.class).getFilesToBuild());
    assertThat(baseArtifactNames).isEmpty();
  }

  public void testStatefulRunfiles() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  rf1 = ctx.runfiles(files = attr1)",
        "  rf2 = ctx.runfiles()",
        "  return struct(data_runfiles = rf1, default_runfiles = rf2)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory = True, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertEquals("//test/skylark:cr", target.getLabel().toString());
    assertTrue(target.getProvider(RunfilesProvider.class).getDefaultRunfiles().isEmpty());
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(RunfilesProvider.class).getDataRunfiles().getAllArtifacts()))
        .containsExactly("a.txt");
  }

  public void testExecutableGetsInRunfilesAndFilesToBuild() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  ctx.file_action(output = ctx.outputs.executable, content = 'echo hello')",
        "  rf = ctx.runfiles(ctx.files.data)",
        "  return struct(runfiles = rf)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl, executable = True,",
        "  attrs = {'data': attr.label_list(cfg=DATA_CFG, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', data = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertEquals("//test/skylark:cr", target.getLabel().toString());
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(RunfilesProvider.class).getDefaultRunfiles().getAllArtifacts()))
        .containsExactly("a.txt", "cr")
        .inOrder();
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("cr");
  }

  public void testCannotSpecifyRunfilesWithDataOrDefaultRunfiles() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  rf = ctx.runfiles()",
        "  return struct(runfiles = rf, default_runfiles = rf)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "Cannot specify the provider 'runfiles' together with "
            + "'data_runfiles' or 'default_runfiles'",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
  }

  public void testTransitiveInfoProviders() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  ftb = set(attr1)",
        "  return struct(provider_key = ftb)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//test/skylark:cr");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                ((SkylarkNestedSet) target.get("provider_key")).getSet(Artifact.class)))
        .containsExactly("a.txt");
  }

  public void testMandatoryProviderMissing() throws Exception {
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  return struct()",
        "",
        "dependent_rule = rule(implementation = rule_impl)",
        "",
        "main_rule = rule(implementation = rule_impl,",
        "    attrs = {'dependencies': attr.label_list(providers = ['some_provider'],",
        "        allow_files=True)})");

    checkError(
        "test",
        "b",
        "in dependencies attribute of main_rule rule //test:b: "
            + "'//test:a' does not have mandatory provider 'some_provider'",
        "load('/test/skylark/extension', 'dependent_rule')",
        "load('/test/skylark/extension', 'main_rule')",
        "",
        "dependent_rule(name = 'a')",
        "main_rule(name = 'b', dependencies = [':a'])");
  }

  public void testActions() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  output = ctx.outputs.o",
        "  ctx.action(",
        "    inputs = attr1,",
        "    outputs = [output],",
        "    command = 'echo')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)},",
        "  outputs = {'o': 'o.txt'})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    getConfiguredTarget("//test/skylark:cr");

    FileConfiguredTarget target = getFileConfiguredTarget("//test/skylark:o.txt");
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                getGeneratingAction(target.getArtifact()).getInputs()))
        .containsExactly("a.txt");
  }

  public void testRuleClassImplicitOutputFunction() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  files = [ctx.outputs.o]",
        "  ctx.action(",
        "    outputs = files,",
        "    command = 'echo')",
        "  ftb = set(files)",
        "  return struct(runfiles = ctx.runfiles(), files = ftb)",
        "",
        "def output_func(attr_map):",
        "  if attr_map.attr2 != None: return {}",
        "  return {'o': attr_map.attr1 + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.string(),",
        "           'attr2': attr.label()},",
        "  outputs = output_func)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = 'bar')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("bar.txt");
  }

  public void testRuleClassImplicitOutputs() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  files = [ctx.outputs.lbl, ctx.outputs.list, ctx.outputs.str]",
        "  print('==!=!=!=')",
        "  print(files)",
        "  ctx.action(",
        "    outputs = files,",
        "    command = 'echo')",
        "  return struct(files = set(files))",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {",
        "    'attr1': attr.label(allow_files=True),",
        "    'attr2': attr.label_list(allow_files=True),",
        "    'attr3': attr.string(),",
        "  },",
        "  outputs = {",
        "    'lbl': '%{attr1}.a',",
        "    'list': '%{attr2}.b',",
        "    'str': '%{attr3}.c',",
        "})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(",
        "  name='cr',",
        "  attr1='f1.txt',",
        "  attr2=['f2.txt'],",
        "  attr3='f3.txt',",
        ")");

    scratch.file("test/skylark/f1.txt");
    scratch.file("test/skylark/f2.txt");
    scratch.file("test/skylark/f3.txt");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("f1.a", "f2.b", "f3.txt.c");
  }

  public void testRuleClassImplicitOutputFunctionAndDefaultValue() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  ctx.action(",
        "    outputs = [ctx.outputs.o],",
        "    command = 'echo')",
        "  return struct(runfiles = ctx.runfiles())",
        "",
        "def output_func(attr_map):",
        "  return {'o': attr_map.attr1 + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.string(default='bar')},",
        "  outputs = output_func)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = None)");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("bar.txt");
  }

  public void testRuleClassNonMandatoryEmptyOutputs() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(",
        "      o1=ctx.outputs.o1,",
        "      o2=ctx.outputs.o2)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'o1': attr.output(), 'o2': attr.output_list()})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertEquals(Runtime.NONE, target.get("o1"));
    assertEquals(MutableList.EMPTY, target.get("o2"));
  }

  public void testRuleClassImplicitAndExplicitOutputNamesCollide() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct()",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'o': attr.output_list()},",
        "  outputs = {'o': '%{name}.txt'})");

    checkError(
        "test/skylark",
        "cr",
        "Multiple outputs with the same key: o",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', o = [':bar.txt'])");
  }

  public void testRuleClassDefaultFilesToBuild() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  files = [ctx.outputs.o]",
        "  ctx.action(",
        "    outputs = files,",
        "    command = 'echo')",
        "  ftb = set(files)",
        "  for i in ctx.outputs.out:",
        "    ctx.file_action(output=i, content='hi there')",
        "",
        "def output_func(attr_map):",
        "  return {'o': attr_map.attr1 + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {",
        "    'attr1': attr.string(),",
        "    'out': attr.output_list()",
        "  },",
        "  outputs = output_func)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = 'bar', out=['other'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("bar.txt", "other")
        .inOrder();
  }

  public void testBadCallbackFunction() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl", "def impl(): return 0", "", "custom_rule = rule(impl)");

    checkError(
        "test/skylark",
        "cr",
        "impl() does not accept positional arguments",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
  }

  public void testRuleClassImplicitOutputFunctionBadAttr() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "def output_func(attr_map):",
        "  return {'a': attr_map.bad_attr}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.string()},",
        "  outputs = output_func)");

    checkError(
        "test/skylark",
        "cr",
        "Attribute 'bad_attr' either doesn't exist or uses a select()",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = 'bar')");
  }

  public void testHelperFunctionInRuleImplementation() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def helper_func(attr1):",
        "  return set(attr1)",
        "",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  ftb = helper_func(attr1)",
        "  return struct(runfiles = ctx.runfiles(), files = ftb)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertEquals("//test/skylark:cr", target.getLabel().toString());
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("a.txt");
  }

  public void testMultipleImportsOfSameRule() throws Exception {
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "     attrs = {'dep': attr.label_list(allow_files=True)})");

    scratch.file(
        "test/skylark1/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name = 'cr1')");

    scratch.file(
        "test/skylark2/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name = 'cr2', dep = ['//test/skylark1:cr1'])");

    getConfiguredTarget("//test/skylark2:cr2");
  }

  public void testFunctionGeneratingRules() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def impl(ctx): return None",
        "def gen(): return rule(impl)",
        "r = gen()",
        "s = gen()");

    scratch.file(
        "test/skylark/BUILD", "load('extension', 'r', 's')", "r(name = 'r')", "s(name = 's')");

    getConfiguredTarget("//test/skylark:r");
    getConfiguredTarget("//test/skylark:s");
  }

  public void testImportInSkylark() throws Exception {
    scratch.file("test/skylark/implementation.bzl", "def custom_rule_impl(ctx):", "  return None");

    scratch.file(
        "test/skylark/extension.bzl",
        "load('/test/skylark/implementation', 'custom_rule_impl')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "     attrs = {'dep': attr.label_list(allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name = 'cr')");

    getConfiguredTarget("//test/skylark:cr");
  }

  public void testRuleAliasing() throws Exception {
    scratch.file(
        "test/skylark/implementation.bzl",
        "def impl(ctx): return struct()",
        "custom_rule = rule(implementation = impl)");

    scratch.file(
        "test/skylark/ext.bzl",
        "load('/test/skylark/implementation', 'custom_rule')",
        "def impl(ctx): return struct()",
        "custom_rule1 = rule(implementation = impl)",
        "custom_rule2 = custom_rule1",
        "custom_rule3 = custom_rule");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/ext', 'custom_rule1', 'custom_rule2', 'custom_rule3')",
        "custom_rule4 = custom_rule3",
        "custom_rule1(name = 'cr1')",
        "custom_rule2(name = 'cr2')",
        "custom_rule3(name = 'cr3')",
        "custom_rule4(name = 'cr4')");

    getConfiguredTarget("//test/skylark:cr1");
    getConfiguredTarget("//test/skylark:cr2");
    getConfiguredTarget("//test/skylark:cr3");
    getConfiguredTarget("//test/skylark:cr4");
  }

  public void testRecursiveImport() throws Exception {
    scratch.file("test/skylark/ext2.bzl", "load('/test/skylark/ext1', 'symbol2')");

    scratch.file("test/skylark/ext1.bzl", "load('/test/skylark/ext2', 'symbol1')");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/ext1', 'custom_rule')",
        "genrule(name = 'rule')");

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:rule");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      // This is expected
    }
    assertContainsEvent(
        "test/skylark/BUILD: cycle in referenced extension files: \n"
            + "  * test/skylark/ext1.bzl\n"
            + "    test/skylark/ext2.bzl\n"
            + "  * test/skylark/ext1.bzl");
  }

  public void testSymbolPropagateThroughImports() throws Exception {
    scratch.file("test/skylark/implementation.bzl", "def custom_rule_impl(ctx):", "  return None");

    scratch.file(
        "test/skylark/extension2.bzl", "load('/test/skylark/implementation', 'custom_rule_impl')");

    scratch.file(
        "test/skylark/extension1.bzl",
        "load('/test/skylark/extension2', 'custom_rule_impl')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "     attrs = {'dep': attr.label_list()})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension1', 'custom_rule')",
        "custom_rule(name = 'cr')");

    getConfiguredTarget("//test/skylark:cr");
  }

  public void testAccessingJvmFragment() throws Exception {
    checkFieldAccess(
        "ctx.fragments.jvm.java_executable", "third_party/java/jdk/jdk-mock-64/bin/java");
  }

  public void testAccessingJvmHostFragment() throws Exception {
    checkFieldAccess(
        "ctx.host_fragments.jvm.java_executable", "third_party/java/jdk/jdk-mock-64/bin/java");
  }

  public void testAccessingCppFragment() throws Exception {
    checkFieldAccess("ctx.fragments.cpp.compiler", "gcc-4.4.0");
  }

  public void testAccessingCppHostFragment() throws Exception {
    checkFieldAccess("ctx.host_fragments.cpp.compiler", "gcc-4.4.0");
  }

  public void testAccessingJavaFragment() throws Exception {
    checkFieldAccess("ctx.fragments.java.default_javac_flags", "[]");
  }

  public void testAccessingJavaHostFragment() throws Exception {
    checkFieldAccess("ctx.host_fragments.java.default_javac_flags", "[]");
  }

  public void testAccessingNonExistingFragment() throws Exception {
    expectFragmentError(
        "ctx.fragments.nothing.compiler",
        "There is no configuration fragment named 'nothing' in target configuration. "
            + "Available fragments: 'cpp', 'java', 'jvm'");
  }

  public void testAccessingNonExistingHostFragment() throws Exception {
    expectFragmentError(
        "ctx.host_fragments.nothing.compiler",
        "There is no configuration fragment named 'nothing' in host configuration. "
            + "Available fragments: 'cpp', 'java', 'jvm'");
  }

  // Ensures that the legacy way of accessing fragments via configuration no longer works.
  public void testLegacyFragmentAccess() throws Exception {
    expectFragmentError("ctx.configuration.fragment(cpp).compiler", "name 'cpp' is not defined");
  }

  private void expectFragmentError(String fieldName, String expectedError) throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    try {
      checkFieldAccess(fieldName, "");
      fail("There should have been an exception in expectFragmentError()");
    } catch (Exception ex) {
      assertContainsEvent(expectedError);
    }
  }

  public void testFragmentAccessError() throws Exception {
    reporter.removeHandler(failFastHandler);
    getConfiguredTargetForFragment("ctx.fragments.cpp.compiler", "'java'", "'cpp'");
    assertContainsEvent(
        "custom_rule has to declare 'cpp' as a required fragment in target "
            + "configuration in order to access it. Please update the 'fragments' argument "
            + "of the rule definition (for example: fragments = [\"cpp\"])");
  }

  public void testHostFragmentAccessError() throws Exception {
    reporter.removeHandler(failFastHandler);
    getConfiguredTargetForFragment("ctx.host_fragments.cpp.compiler", "'cpp'", "'java'");
    assertContainsEvent(
        "custom_rule has to declare 'cpp' as a required fragment in host "
            + "configuration in order to access it. Please update the 'host_fragments' argument "
            + "of the rule definition (for example: host_fragments = [\"cpp\"])");
  }

  private void checkFieldAccess(String fullFieldName, String expectedResult) throws Exception {
    checkFieldAccess(fullFieldName, "'cpp', 'java', 'jvm'", expectedResult);
  }

  private void checkFieldAccess(String fullFieldName, String fragments, String expectedResult)
      throws Exception {
    assertThat(getConfiguredTargetForFragment(fullFieldName, fragments, fragments).get("result"))
        .isEqualTo(expectedResult);
  }

  private ConfiguredTarget getConfiguredTargetForFragment(
      String fullFieldName, String fragments, String hostFragments) throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(result = str(" + fullFieldName + ")",
        ")",
        "custom_rule = rule(implementation = custom_rule_impl,",
        String.format("     fragments = [%s],", fragments),
        String.format("     host_fragments = [%s],", hostFragments),
        ")");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension','custom_rule')",
        "custom_rule(name = 'cr')");

    return getConfiguredTarget("//test/skylark:cr");
  }

  public void testLateBoundAttribute() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  ftb = set([ctx.file._attr2])",
        "  return struct(runfiles = ctx.runfiles(), files = ftb)",
        "",
        "def attr_value(attr_map, cfg):",
        "  return attr_map.attr1",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label(allow_files=True),",
        "          '_attr2': attr.label(default=attr_value, allow_files=True, single_file=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = '//test/skylark:file')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("file");
  }

  public void testLateBoundAttributesNone() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  if ctx.attr._attr1 != None: fail('should be None')",
        "  f = set(ctx.attr._attr2)", // label_list defaults to []
        "  return struct(runfiles = ctx.runfiles(), files = f)",
        "",
        "def attr_value(attr_map, cfg):",
        "  return None",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'_attr1': attr.label(default=attr_value),",
        "           '_attr2': attr.label_list(default=attr_value)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .isEmpty();
  }

  public void testLateBoundAttributeBadType() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx): pass",
        "",
        "def attr_value(attr_map, cfg): return 5",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'_attr': attr.label(default=attr_value)})");

    checkError(
        "test/skylark",
        "cr",
        "When computing the default value of :attr, expected 'label', got 'int'",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
  }

  public void testLateBoundAttributeBadAttr() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "def attr_value(attr_map, cfg):",
        "  return attr_map.bad_attr",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label(allow_files=True),",
        "          '_attr2': attr.label(default=attr_value, allow_files=True)})");

    checkError(
        "test/skylark",
        "cr",
        "No such regular (non late-bound) attribute 'bad_attr'",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = '//test/skylark:file')");
  }

  public void testLateBoundAttributeBadName() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "def attr_value(attr_map, cfg):",
        "  return attr_map.bad_attr",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label(default=attr_value)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = '//test/skylark:file')");

    reporter.removeHandler(failFastHandler);
    try {
      // The error happens during the loading of the Skylark file so checkError don't work here
      getTarget("//test/skylark:cr");
      fail();
    } catch (Exception e) {
      assertContainsEvent(
          "When an attribute value is a function, the attribute must be private (start with '_')");
    }
  }

  public void testEmptyName() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'': attr.label()})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = '//test/skylark:file')");

    reporter.removeHandler(failFastHandler);
    try {
      // The error happens during the loading of the Skylark file so checkError don't work here
      getTarget("//test/skylark:cr");
      fail();
    } catch (Exception e) {
      assertContainsEvent("Attribute name cannot be empty");
    }
  }

  public void testValidationEnvironmentDoesNotCollide() throws Exception {
    scratch.file("test/skylark/extension1.bzl", "some_variable = 'a'");
    scratch.file("test/skylark/extension2.bzl", "some_variable = 1");
    scratch.file(
        "test/skylark/rule1.bzl",
        "load('/test/skylark/extension1', 'some_variable')",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "def attr_value(attr_map, cfg):",
        "  return attr_map.bad_attr",
        "",
        "custom_rule1 = rule(implementation = custom_rule_impl)");
    scratch.file(
        "test/skylark/rule2.bzl",
        "load('/test/skylark/extension2', 'some_variable')",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "def attr_value(attr_map, cfg):",
        "  return attr_map.bad_attr",
        "",
        "custom_rule2 = rule(implementation = custom_rule_impl)");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/rule1', 'custom_rule1')",
        "load('/test/skylark/rule2', 'custom_rule2')",
        "",
        "custom_rule1(name = 'cr1')",
        "custom_rule2(name = 'cr2')");
    getConfiguredTarget("//test/skylark:cr1");
    getConfiguredTarget("//test/skylark:cr2");
  }

  public void testImportedFunctionExecutesInItsDefinitionEnvironment() throws Exception {
    scratch.file(
        "test/skylark/helper.bzl",
        "some_constant = set(['a'])",
        "def some_func():",
        "  return set(['b'])",
        "def helper_func():",
        "  return some_constant + some_func()");

    scratch.file(
        "test/skylark/implementation.bzl",
        "load('/test/skylark/helper', 'helper_func')",
        "def custom_rule_impl(ctx):",
        "  return struct(",
        "    p = helper_func())",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/implementation', 'custom_rule')",
        "custom_rule(name = 'cr')");

    assertEquals(
        ImmutableList.of("b", "a"),
        ((SkylarkNestedSet) getConfiguredTarget("//test/skylark:cr").get("p")).toCollection());
  }

  public void testImportedFunctionValidation() throws Exception {
    scratch.file("test/skylark/helper.bzl", "def helper_func():", "  return set(['a'])");

    scratch.file(
        "test/skylark/implementation.bzl",
        "load('/test/skylark/helper', 'helper_func')",
        "def custom_rule_impl(ctx):",
        "  a = helper_func()",
        "  return struct(",
        "    p = a)",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/implementation', 'custom_rule')",
        "custom_rule(name = 'cr')");

    assertEquals(
        ImmutableList.of("a"),
        ((SkylarkNestedSet) getConfiguredTarget("//test/skylark:cr").get("p")).toCollection());
  }

  public void testExpectFailureWrongError() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx): fail('kaputt')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "kaputt",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name='cr', expect_failure='other')");
  }

  public void testExpectFailureNoError() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx): return struct()",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "Expected failure not found: other",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name='cr', expect_failure='other')");
  }

  public void testExpectFailureSuccess() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  fail('kaputt')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr1', expect_failure = 'kaputt*')",
        "custom_rule(name = 'cr2', expect_failure = 'k.*t*')");

    getConfiguredTarget("//test/skylark:cr1");
    getConfiguredTarget("//test/skylark:cr2");
  }

  public void testCallFunctionFromBuildFile() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def add_suffix(s): return s + '_suf'",
        "def custom_rule_impl(ctx): return struct()",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule', 'add_suffix')",
        "",
        "custom_rule(name = add_suffix('foo'))");

    getConfiguredTarget("//test/skylark:foo_suf");
  }

  public void testTransitiveInfoProviderValueIsNotValid() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx): return struct(a = ctx.configuration)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "Value of provider 'a' is of an illegal type: configuration",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name='cr')");
  }

  public void testTransitiveInfoProviderCompositeValueIsNotValid() throws Exception {
    reporter.addHandler(printHandler);
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx): return struct(a = struct(a=[{'key': ctx.configuration}]))",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "Value of provider 'a' is of an illegal type: configuration",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name='cr')");
  }

  private void createSimpleExtension(String path, String provides, boolean overwrite)
      throws Exception {
    String content =
        LINE_JOINER.join(
            "def custom_rule_impl(ctx):",
            "  return struct(provider=" + provides + ")",
            "",
            "custom_rule = rule(implementation = custom_rule_impl)");
    if (overwrite) {
      scratch.overwriteFile(path, content);
    } else {
      scratch.file(path, content);
    }
  }

  private void createSimpleExtension(String provides, boolean overwrite) throws Exception {
    createSimpleExtension("test/skylark/extension.bzl", provides, overwrite);
  }

  public void testSkylarkExtensionsAreReloaded() throws Exception {
    createSimpleExtension("'a'", false);
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
    assertEquals("a", getConfiguredTarget("//test/skylark:cr").get("provider"));

    createSimpleExtension("'b'", true);
    invalidatePackages();
    assertEquals("b", getConfiguredTarget("//test/skylark:cr").get("provider"));
  }

  public void testTransitiveSkylarkExtensionsAreReloaded() throws Exception {
    createSimpleExtension("'a'", false);
    scratch.file(
        "test/skylark/extension2.bzl",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule_2 = custom_rule");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension2', 'custom_rule_2')",
        "",
        "custom_rule_2(name = 'cr')");

    assertEquals("a", getConfiguredTarget("//test/skylark:cr").get("provider"));
    createSimpleExtension("'b'", true);

    invalidatePackages();
    assertEquals("b", getConfiguredTarget("//test/skylark:cr").get("provider"));
  }

  public void testSkylarkImportOverridesNativeRules() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(custom_provider = 'a')",
        "",
        "genrule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'genrule')",
        "",
        "genrule(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertEquals("a", target.get("custom_provider"));
  }

  public void testSkylarkImportOverridesNativeRulesWithAliasing() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(custom_provider = 'a')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)",
        "genrule = custom_rule");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'genrule')",
        "",
        "genrule(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertEquals("a", target.get("custom_provider"));
  }

  public void testSkylarkImportOverridesNativeRulesTwoImports() throws Exception {
    scratch.file(
        "test/skylark/extension1.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(custom_provider = 'a')",
        "",
        "genrule = rule(implementation = custom_rule_impl)");
    scratch.file(
        "test/skylark/extension2.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(custom_provider = 'b')",
        "",
        "genrule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension1', 'genrule')",
        "first_genrule = genrule",
        "load('/test/skylark/extension2', 'genrule')",
        "",
        "first_genrule(name = 'cr1')",
        "genrule(name = 'cr2')");

    ConfiguredTarget target1 = getConfiguredTarget("//test/skylark:cr1");
    assertEquals("a", target1.get("custom_provider"));
    ConfiguredTarget target2 = getConfiguredTarget("//test/skylark:cr2");
    assertEquals("b", target2.get("custom_provider"));
  }

  public void testNativeRulesWorkFromSkylarkBuildExtensions() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func(name):",
        "  native.genrule(name = name,",
        "    outs = [name + '.txt'],",
        "    tags = None,", // None values should be ignored
        "    cmd = 'echo hello >@')");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'extension_func')",
        "",
        "extension_func(name = 'rule1')");

    getConfiguredTarget("//test/skylark:rule1");
    // Check output file
    getFileConfiguredTarget("//test/skylark:rule1.txt");
  }

  public void testNativeRulesWithGlobInSkylark() throws Exception {
    // The glob is evaluated using the BUILD file's package.
    scratch.file("test/project/data1.dat", "");
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func(name):",
        "  native.genrule(name = name,",
        "    srcs = native.glob(['*.dat']),",
        "    outs = [name + '.txt'],",
        "    cmd = 'echo $(SRCS) hello >@')");

    scratch.file(
        "test/project/BUILD",
        "load('/test/skylark/extension', 'extension_func')",
        "",
        "extension_func(name = 'rule1')");

    ConfiguredTarget target = getConfiguredTarget("//test/project:rule1");
    Action action = getGeneratingAction(Iterables.getOnlyElement(getFilesToBuild(target)));
    assertThat(baseArtifactNames(action.getInputs())).contains("data1.dat");
  }

  public void testNativeRulesWithGlobFromBuildFile() throws Exception {
    scratch.file("test/BUILD");
    scratch.file("test/project/data1.dat", "");
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func(name, srcs):",
        "  native.genrule(name = name,",
        "    srcs = srcs,",
        "    outs = [name + '.txt'],",
        "    cmd = 'echo $(SRCS) hello >@')");

    scratch.file(
        "test/project/BUILD",
        "load('/test/skylark/extension', 'extension_func')",
        "",
        "extension_func(name = 'rule1', srcs = glob(['*.dat']))");

    ConfiguredTarget target = getConfiguredTarget("//test/project:rule1");
    Action action = getGeneratingAction(Iterables.getOnlyElement(getFilesToBuild(target)));
    assertThat(baseArtifactNames(action.getInputs())).contains("data1.dat");
  }

  public void testGlobWithExcludes() throws Exception {
    scratch.file("test/project/a.dat", "");
    scratch.file("test/project/b.dat", "");
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func(name, includes, excludes):",
        "  native.genrule(name = name,",
        "    srcs = native.glob(includes, excludes),",
        "    outs = [name + '.txt'],",
        "    cmd = 'echo $(SRCS) hello >@')");

    scratch.file(
        "test/project/BUILD",
        "load('/test/skylark/extension', 'extension_func')",
        "",
        "extension_func(name = 'rule1', includes = ['*.dat'], excludes = ['b.dat'])");

    ConfiguredTarget target = getConfiguredTarget("//test/project:rule1");
    Action action = getGeneratingAction(Iterables.getOnlyElement(getFilesToBuild(target)));
    assertThat(baseArtifactNames(action.getInputs())).contains("a.dat");
    assertThat(baseArtifactNames(action.getInputs())).containsNoneIn(ImmutableList.of("b.dat"));
  }

  public void testFailInSkylarkExtensions() throws Exception {
    scratch.file("test/skylark/extension.bzl", "def func(name):", "  fail('not implemented')");

    checkError(
        "test/skylark",
        "cr2",
        "not implemented",
        "load('/test/skylark/extension', 'func')",
        "func(name = 'cr')",
        "genrule(name = 'cr2')");
  }

  public void testUseNativeFromTopLevel() throws Exception {
    scratch.file("test/skylark/extension.bzl", "native.cc_library(name = 'cr2')", "a = 0");

    scratch.file(
        "test/skylark/BUILD", "cc_library(name = 'cr')", "load('/test/skylark/extension', 'a')");

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:cr");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      assertContainsEvent("The native module cannot be accessed from here");
    }
  }

  private void checkSymbolIsNotAccessibleInRuleImplementationPhase(
      String symbolName, String ruleImplementationLines) throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        ruleImplementationLines,
        "  return struct()",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        symbolName + "() can only be called during the loading phase",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name = 'cr')");
  }

  public void testNativeModuleIsNotAccessibleInRuleImplementationPhase() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  native.genrule(name = 'some_genrule',",
        "    outs = ['out.txt'],",
        "    cmd = 'echo hello >@')",
        "  return struct()",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "genrule() can only be called during the loading phase",
        "load('/test/skylark/extension', 'custom_rule')",
        "custom_rule(name = 'cr')");
  }

  public void testNativeModulePackageFunction() throws Exception {
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func():",
        "  native.package(features = ['foo'])");

    scratch.file(
        "test/project/BUILD",
        "load('/test/skylark/extension', 'extension_func')",
        "",
        "extension_func()",
        "genrule(name = 'a', cmd = '', outs = ['b'])");

    Package pkg = getConfiguredTarget("//test/project:a").getTarget().getPackage();
    assertEquals(Iterables.getOnlyElement(pkg.getFeatures()), "foo");
  }

  public void testAttrModuleIsNotAccessibleInRuleImplementationPhase() throws Exception {
    checkSymbolIsNotAccessibleInRuleImplementationPhase("attr.label", "  attr.label()");
  }

  public void testRuleFunctionIsNotAccessibleInRuleImplementationPhase() throws Exception {
    checkSymbolIsNotAccessibleInRuleImplementationPhase(
        "rule", "  rule(implementation = custom_rule_impl)");
  }

  public void testPreludeFile() throws Exception {
    createSimpleExtension("test/skylark/extension.bzl", "'a'", false);
    scratch.overwriteFile(
        "tools/build_rules/prelude_blaze", "load('/test/skylark/extension', 'custom_rule')");
    scratch.file("test/skylark/BUILD", "custom_rule(name = 'cr')");
    invalidatePackages();
    simulateLoadingPhase();

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertEquals("a", target.get("provider"));
  }

  public void testPreludeFileCaching() throws Exception {
    createSimpleExtension("test/skylark/extension1.bzl", "'a'", false);
    scratch.overwriteFile(
        "tools/build_rules/prelude_blaze", "load('/test/skylark/extension1', 'custom_rule')");
    scratch.file("test/skylark/BUILD", "custom_rule(name = 'cr')");
    invalidatePackages();
    simulateLoadingPhase();

    ConfiguredTarget target1 = getConfiguredTarget("//test/skylark:cr");
    assertEquals("a", target1.get("provider"));

    createSimpleExtension("test/skylark/extension2.bzl", "'b'", false);
    scratch.overwriteFile(
        "tools/build_rules/prelude_blaze", "load('/test/skylark/extension2', 'custom_rule')");
    invalidatePackages();
    simulateLoadingPhase();

    ConfiguredTarget target2 = getConfiguredTarget("//test/skylark:cr");
    assertEquals("b", target2.get("provider"));
  }

  public void testPreludeFileParsedAsBuildFile() throws Exception {
    createSimpleExtension("test/skylark/extension.bzl", "'a'", false);
    scratch.overwriteFile(
        "tools/build_rules/prelude_blaze",
        "load('/test/skylark/extension', 'custom_rule')",
        "def func(): return None");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule_test')",
        "",
        "custom_rule_test(name = 'cr')");

    invalidatePackages();

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:cr");
      fail();
    } catch (NoSuchTargetException e) {
      assertContainsEvent("syntax error at 'def': This is not supported in BUILD files");
    }
  }

  public void testTestRule() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  executable = ctx.outputs.executable",
        "  ctx.file_action(",
        "    output = executable,",
        "    executable = True,",
        "    content = 'echo Hello')",
        "",
        "custom_rule_test = rule(implementation = rule_impl, test = True)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule_test')",
        "",
        "custom_rule_test(name = 'cr')");

    getConfiguredTarget("//test/skylark:cr");
  }

  public void testTestRuleWithoutRunfiles() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx): pass",
        "",
        "custom_rule_test = rule(implementation = rule_impl, test = True)");

    checkError(
        "test/skylark",
        "cr",
        "The following files have no generating action",
        "load('/test/skylark/extension', 'custom_rule_test')",
        "",
        "custom_rule_test(name = 'cr')");
  }

  public void testTestRuleBadRuleClassName() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  return None",
        "",
        "custom_rule = rule(implementation = rule_impl, test = True)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    reporter.removeHandler(failFastHandler);
    try {
      // The error happens during the loading of the Skylark file so checkError don't work here
      getTarget("//test/skylark:cr");
      fail();
    } catch (Exception e) {
      assertContainsEvent(
          "Invalid rule class name 'custom_rule', test rule class names must "
              + "end with '_test' and other rule classes must not");
    }
  }

  public void testPrivateRule() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  return struct(p = 5)",
        "",
        "_custom_rule = rule(implementation = rule_impl)",
        "def macro(name):",
        "  _custom_rule(name = name)");

    scratch.file(
        "test/skylark/BUILD", "load('/test/skylark/extension', 'macro')", "", "macro(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertThat(target.get("p")).isEqualTo(5);
  }

  public void testExecutableRule() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  executable = ctx.outputs.executable",
        "  ctx.file_action(",
        "    output = executable,",
        "    executable = True,",
        "    content = 'echo Hello')",
        "  default_runfiles = ctx.runfiles([executable])",
        "  return struct(runfiles = default_runfiles)",
        "",
        "custom_rule = rule(implementation = rule_impl, executable = True)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    FilesToRunProvider provider = target.getProvider(FilesToRunProvider.class);
    assertNotNull(provider.getRunfilesSupport());
    assertNotNull(provider.getExecutable());
    assertSame(provider.getExecutable(), provider.getRunfilesSupport().getExecutable());
  }

  public void testLabelMustBeExecutable() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  return None",
        "",
        "custom_rule = rule(implementation = rule_impl,",
        "  attrs = {'exe': attr.label(executable=True)})");

    checkError(
        "test/skylark",
        "cr",
        "does not refer to a valid executable target",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "genrule(name = 'g',",
        "  cmd = '',",
        "  outs = ['out'])",
        "custom_rule(name = 'cr', exe='g')");
  }

  public void testExecutableTool() throws Exception {
    scratch.file(
        "test/skylark/tool.bzl",
        "def rule_impl(ctx):",
        "  executable = ctx.outputs.executable",
        "  ctx.file_action(",
        "    output = executable,",
        "    executable = True,",
        "    content = 'echo Hello')",
        "",
        "tool_rule = rule(implementation = rule_impl, executable = True)");

    scratch.file(
        "test/skylark/main.bzl",
        "def rule_impl(ctx):",
        "  ctx.action(",
        "    outputs = [ctx.outputs.out],",
        "    executable = ctx.executable.tool,",
        "    arguments = ['--flag'])",
        "",
        "main_rule = rule(implementation = rule_impl,",
        "    outputs = {'out': '%{name}.sh'},",
        "    attrs = {'tool': attr.label(executable = True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/main', 'main_rule')",
        "load('/test/skylark/tool', 'tool_rule')",
        "",
        "tool_rule(name = 'mytool')",
        "main_rule(name = 'myrule', tool = ':mytool')");

    ConfiguredTarget tool = getConfiguredTarget("//test/skylark:mytool");
    // Check the tool has runfiles
    assertThat(baseArtifactNames(getRunfilesSupport(tool).getRunfiles().getAllArtifacts()))
        .contains("mytool");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:myrule");
    SpawnAction action =
        (SpawnAction) getGeneratingAction(Iterables.getOnlyElement(getFilesToBuild(target)));
    // the first argument i.e. the executable is mytool
    MoreAsserts.assertEndsWith("test/skylark/mytool", action.getArguments().get(0));
    // the runfiles of the tool are among the action inputs
    assertThat(baseArtifactNames(action.getInputs())).contains("test_Sskylark_Smytool-runfiles");
    assertTrue(
        ActionsTestUtil.getFirstArtifactEndingWith(
            action.getInputs(), "test_Sskylark_Smytool-runfiles")
            .isMiddlemanArtifact());
  }

  public void testTryingToImportNonexistingExtension() throws Exception {
    scratch.file(
        "test/BUILD",
        "load('/test/skylark/non_existing_extension', 'custom_rule_test')",
        "",
        "custom_rule_test(name = 'cr')");
    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test:cr");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      assertContainsEvent("Extension file not found: 'test/skylark/non_existing_extension.bzl'");
    }
  }

  public void testTryingToImportBrokenExtension() throws Exception {
    scratch.file("test/ext.bzl", "var = 2", "var = 3");
    scratch.file("test/BUILD", "load('/test/ext', 'var')", "", "custom_rule_test(name = 'cr')");
    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test:cr");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      assertContainsEvent("Variable var is read only");
      assertContainsEvent("Extension 'test/ext.bzl' has errors");
    }
  }

  private void writeConfigFile() throws Exception {
    scratch.file(
        "conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'})",
        "config_setting(",
        "    name = 'b',",
        "    values = {'test_arg': 'b'})");
  }

  public void testUseConfigurableAttributeValueInSkylarkRule() throws Exception {
    writeConfigFile();
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(provider = ctx.attr.a)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'a': attr.string()})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', a = select({'//conditions:a' : 'x', '//conditions:b' : 'y'}))");

    useConfiguration("--test_arg=a");
    assertEquals("x", getConfiguredTarget("//test/skylark:cr").get("provider"));
    useConfiguration("--test_arg=b");
    assertEquals("y", getConfiguredTarget("//test/skylark:cr").get("provider"));
  }

  private String helloCcDepTargets() {
    return LINE_JOINER.join(
        "  native.cc_library(",
        "    name = 'adep',",
        "    srcs = ['adep.cc'])",
        "  native.cc_library(",
        "    name = 'bdep',",
        "    srcs = ['bdep.cc'])");
  }

  public void testDefineConfigurableAttributeInSkylarkExtension() throws Exception {
    writeConfigFile();
    scratch.file(
        "test/skylark/extension.bzl",
        "def build_extension(name):",
        "  native.cc_binary(",
        "    name = name,",
        "    srcs = ['hello.cc'],",
        "    deps = select({",
        "        '//conditions:a': [':adep'],",
        "        '//conditions:b': [':bdep'],",
        "    }))",
        "",
        helloCcDepTargets());

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'build_extension')",
        "",
        "build_extension('hello')");

    checkConfigurableHelloTargets();
  }

  public void testPassConfigurableAttributeToSkylarkExtension() throws Exception {
    writeConfigFile();
    scratch.file(
        "test/skylark/extension.bzl",
        "def build_extension(name, deps):",
        "  native.cc_binary(",
        "    name = name,",
        "    srcs = ['hello.cc'],",
        "    deps = deps)",
        "",
        helloCcDepTargets());

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'build_extension')",
        "",
        "build_extension('hello',",
        "    select({",
        "        '//conditions:a': [':adep'],",
        "        '//conditions:b': [':bdep'],",
        "    }))");

    checkConfigurableHelloTargets();
  }

  public void testConfigurableAttributesConcatenateSkylarkAndNativeLists() throws Exception {
    writeConfigFile();
    scratch.file(
        "test/skylark/extension.bzl",
        "def return_select():",
        "    return select({'//conditions:a': ['foo.in']}) + ['skylark.list']");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'return_select')",
        "genrule(",
        "    name = 'gen',",
        "    srcs = return_select() + ['native.list'],",
        "    outs = ['gen.out'],",
        "    cmd = 'echo hi > $@')");
    useConfiguration("--test_arg=a");
    getConfiguredTarget("//test/skylark:gen");
  }

  public void testConfigurableAttributesNotAllowedInImplicitOutputs() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx): pass",
        "def _get_outputs(attr_map):",
        "    return {'default': attr_map.x + '.out'}",
        "simple = rule(",
        "    implementation = _impl,",
        "    attrs = {'x': attr.string()},",
        "    outputs = _get_outputs)");
    checkError(
        "test/skylark",
        "simple",
        "Attribute 'x' either doesn't exist or uses a select()",
        "load('/test/skylark/extension', 'simple')",
        "simple(",
        "    name = 'simple',",
        "    x = select({'//conditions:default': 'foo'}))");
  }

  public void testArtifactsWithoutGeneratingActions() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  ctx.new_file(ctx.configuration.bin_dir, ctx.outputs.o2, 'param')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'o1': attr.output_list()},",
        "  outputs = {'o2': '%{name}.txt'})");

    checkError(
        "test/skylark",
        "cr",
        "The following files have no generating action:\n"
            + "test/skylark/cr.txt\n"
            + "test/skylark/cr.txtparam\n"
            + "test/skylark/output.txt",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', o1 = [':output.txt'])");
  }

  public void testSkylarkListWorksInBuildFile_LoadReturnsCorrectListType() throws Exception {
    scratch.file("test/skylark/extension.bzl", "l1 = ['a']");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'l1')",
        "",
        "l2 = l1 + ['b']",
        "genrule(name = l2[0], cmd = 'echo hi >@', outs = ['a.txt'])",
        "genrule(name = l2[1], cmd = 'echo hi >@', outs = ['b.txt'])");
    getConfiguredTarget("//test/skylark:a");
    getConfiguredTarget("//test/skylark:b");
  }

  public void testSkylarkListWorksInBuildFile_FunctionReturnsCorrectListType() throws Exception {
    scratch.file("test/skylark/extension.bzl", "def gen_list():", "  return ['a']");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'gen_list')",
        "",
        "l2 = gen_list() + ['b']",
        "genrule(name = l2[0], cmd = 'echo hi >@', outs = ['a.txt'])",
        "genrule(name = l2[1], cmd = 'echo hi >@', outs = ['b.txt'])");
    getConfiguredTarget("//test/skylark:a");
    getConfiguredTarget("//test/skylark:b");
  }

  public void testSkylarkListWorksInBuildFile_LoadReturnsMutableCopy() throws Exception {
    scratch.file("test/skylark/extension.bzl", "l = ['a']");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'l')",
        "",
        "l.append('b')",
        "genrule(name = ''.join(l), cmd = 'echo hi >@', outs = ['a.txt'])");
    getConfiguredTarget("//test/skylark:ab");

    // Check that the Skylark value isn't modified (i.e. there is no 'b' value).
    scratch.file(
        "test/skylark2/BUILD",
        "load('/test/skylark/extension', 'l')",
        "",
        "l.append('c')",
        "genrule(name = ''.join(l), cmd = 'echo hi >@', outs = ['a.txt'])");
    getConfiguredTarget("//test/skylark2:ac");
  }

  public void testBuildFileListWorksInSkylark() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func(l):",
        "  l = l + ['c']",
        "  for i in l:",
        "    native.genrule(name = i, cmd = 'echo hi >@', outs = [i + '.txt'])");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'extension_func')",
        "",
        "extension_func(['a', 'b'])");
    getConfiguredTarget("//test/skylark:a");
    getConfiguredTarget("//test/skylark:b");
    getConfiguredTarget("//test/skylark:c");
  }

  public void testSkylarkLoadedListCanBeConcatenated() throws Exception {
    scratch.file("test/skylark/extension.bzl", "l1 = ('a',)", "l2 = ['a']");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'l1')",
        "",
        "m = l1 + ('b',) # tuple",
        "genrule(name = ''.join(m), cmd = 'echo hi >@', outs = ['a.txt'])");
    getConfiguredTarget("//test/skylark:ab");

    scratch.file(
        "test/skylark2/BUILD",
        "load('/test/skylark/extension', 'l2')",
        "",
        "m = l2 + ['b']",
        "genrule(name = ''.join(m), cmd = 'echo hi >@', outs = ['b.txt'])");
    getConfiguredTarget("//test/skylark2:ab");
  }

  public void testSkylarkLoadedListBadType() throws Exception {
    scratch.file("test/skylark/extension.bzl", "l1 = ['a']");
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'l1')",
        "",
        "m = l1 + ('b',)",
        "genrule(name = ''.join(m), cmd = 'echo hi >@', outs = ['a.txt'])");

    try {
      getConfiguredTarget("//test/skylark:ab");
      fail();
    } catch (AssertionFailedError ex) {
      assertThat(ex.getMessage()).contains("can only concatenate List (not \"Tuple\") to List");
    }
  }

  public void testPackageNameIsPresentInBuildExtensions() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func(name):",
        "  native.genrule(",
        "    name = name, cmd = 'echo ' + PACKAGE_NAME + ' >@', outs = [name + '.txt'])");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'extension_func')",
        "",
        "extension_func('a')");
    SpawnAction action =
        (SpawnAction)
            getGeneratingAction(
                Iterables.getOnlyElement(getFilesToBuild(getConfiguredTarget("//test/skylark:a"))));
    String cmd = action.getArguments().get(2);
    assertThat(cmd).contains("echo test/skylark >@");
  }

  public void testStringDictAttribute() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def impl(ctx):",
        "  return struct(p = ctx.attr.stringdict['a'])",
        "r = rule(implementation=impl, attrs={'stringdict': attr.string_dict()})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'r')",
        "",
        "r(name='r1', stringdict={'a': 'b'})");
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:r1");
    assertEquals("b", target.get("p"));
  }

  public void testPackageNameIsNotPresentWhenSkylarkFileExecutes() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def extension_func():",
        // TODO(bazel-team): clean up the behavior of PACKAGE_NAME.
        "  if not PACKAGE_NAME:",
        "    fail(\"name 'PACKAGE_NAME' is not defined\")",
        "  return PACKAGE_NAME",
        "v = extension_func()");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'v')",
        "",
        "genrule(name='a', cmd='echo hi >$@', outs=['a.txt'])");

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:a");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      assertContainsEvent("name 'PACKAGE_NAME' is not defined");
    }
  }

  public void testNonMandatoryAttributesWithSingleFile() throws Exception {
    scratch.file("test/skylark/file.txt", "");
    scratch.file(
        "test/skylark/extension.bzl",
        "def impl(ctx):",
        "  lbl = ctx.attr.a.label if ctx.attr.a else None",
        "  if hasattr(ctx.file, 'a'):",
        "    return struct(f=ctx.file.a, l=lbl)",
        "  else:",
        "    return struct(l=lbl)",
        "",
        "r = rule(implementation=impl,",
        "    attrs={'a': attr.label(allow_files=True, single_file=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'r')",
        "",
        "r(name='r1', a='file.txt')",
        "r(name='r2')");
    ConfiguredTarget target1 = getConfiguredTarget("//test/skylark:r1");
    assertEquals("file.txt", ((Label) target1.get("l")).getName());
    assertEquals("file.txt", ((Artifact) target1.get("f")).getFilename());
    ConfiguredTarget target2 = getConfiguredTarget("//test/skylark:r2");
    assertEquals(Runtime.NONE, target2.get("l"));
    assertEquals(Runtime.NONE, target2.get("f"));
  }

  public void testWorkspaceStatusArtifacts() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def impl(ctx):",
        "  ctx.file_action(ctx.outputs.o, '')",
        // info_file is SpecialArtifact, check if Skylark is unaware of this
        "  f = ctx.info_file",
        "  f = ctx.outputs.o",
        "  l1 = [ctx.info_file, ctx.outputs.o]",
        "  l2 = [ctx.outputs.o, ctx.info_file]",
        "  s1 = set([ctx.info_file]) + [ctx.outputs.o]",
        "  s2 = set([ctx.outputs.o]) + [ctx.info_file]",
        "  return struct(",
        "     info=ctx.info_file, version=ctx.version_file)",
        "",
        "r = rule(implementation=impl, outputs={'o': '%{name}.txt'})");

    scratch.file("test/skylark/BUILD", "load('/test/skylark/extension', 'r')", "", "r(name='r')");
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:r");
    assertEquals("build-info.txt", ((Artifact) target.get("info")).getFilename());
    assertEquals("build-changelist.txt", ((Artifact) target.get("version")).getFilename());
  }

  public void testPrintFunction() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def impl(ctx):",
        "  print('analysis phase hello')",
        "",
        "print('loading phase hello')",
        "r = rule(implementation = impl)");

    scratch.file("test/skylark/BUILD", "load('/test/skylark/extension', 'r')", "", "r(name='r')");
    getConfiguredTarget("//test/skylark:r");
    assertContainsEventsInOrder("loading phase hello", "analysis phase hello");
  }

  public void testPrintFunctionInMacro() throws Exception {
    scratch.file("test/skylark/extension.bzl", "def macro():", "  print('macro hello')");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'macro')",
        "",
        "macro()",
        "genrule(name = 'r', outs = ['r.txt'], cmd = 'echo hi >@')");
    getConfiguredTarget("//test/skylark:r");
    assertContainsEventsInOrder("macro hello");
  }

  public void testInvalidLabel() throws Exception {
    scratch.file("test/skylark/iproute2_src_pkg", "");
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  Label(ctx.attr.locations[0])",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'locations': attr.string_list()})");

    checkError(
        "test/skylark",
        "cr",
        "Illegal absolute label syntax: invalid_label",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', locations = ['invalid_label'])");
  }

  public void testAddingToNestedSetComingFromNativeRuleDoesNotWork() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  ctx.attr.deps[0].files + ctx.attr.deps[1].files",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'deps': attr.label_list()})",
        "",
        "def macro(name):",
        "  native.genrule(name = name + '1', cmd = 'echo hi >$@', outs = [name + '1.txt'])",
        "  native.genrule(name = name + '2', cmd = 'echo hi >$@', outs = [name + '2.txt'])",
        "  custom_rule(name = name, deps = [':' + name + '1', ':' + name + '2'])");

    checkError(
        "test/skylark",
        "cr",
        "Cannot add more elements to this set. Sets created in "
            + "native rules cannot be left side operands of the + operator.",
        "load('/test/skylark/extension', 'macro')",
        "",
        "macro(name = 'cr')");
  }

  public void testFilesIsNotSetOfFiles() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  return struct(files = set(['a']))",
        "",
        "custom_rule = rule(implementation = rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "expected set of Files for 'files' but got set of strings instead: set([\"a\"])",
        "load('/test/skylark/extension', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
  }

  public void testPackageGroup() throws Exception {
    scratch.file("test/pkg1/BUILD");
    scratch.file("test/pkg2/BUILD");
    scratch.file(
        "test/skylark/extension.bzl",
        "def macro(name):",
        "  native.package_group(name = 'default_pg', packages = ['//test/pkg2'])",
        "  native.package_group(name = name,",
        "                       packages = ['//test/pkg1'], includes = [':default_pg'])");

    scratch.file("test/skylark/BUILD", "load('extension', 'macro')", "macro('my_pg')");
    PackageGroup target = (PackageGroup) getTarget("//test/skylark:my_pg");
    assertThat(target.getContainedPackages()).containsExactly("test/pkg1");
    assertThat(target.getIncludes())
        .containsExactly(Label.parseAbsolute("//test/skylark:default_pg"));
  }

  public void testExportsFiles() throws Exception {
    scratch.file("test/skylark/a.txt");
    scratch.file(
        "test/skylark/extension.bzl",
        "def macro():",
        "  native.exports_files(['a.txt'], visibility = ['//visibility:private'])");

    scratch.file("test/skylark/BUILD", "load('extension', 'macro')", "macro()");
    // Without the exports_files in the extension this would throws a NoSuchTargetException
    InputFile target = (InputFile) getTarget("//test/skylark:a.txt");
    assertThat(target.getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseAbsolute("//visibility:private"));
  }

  public void testErrorPropagation() throws Exception {
    scratch.file("test/skylark/extension.bzl", "fail('my error')", "a = 'my_target'");
    scratch.file("test/skylark/BUILD", "load('extension', 'a')", "cc_library(name = a)");

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:my_target");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      assertThat(e.getMessage()).contains("Extension file 'test/skylark/extension.bzl' has errors");
    }
    assertContainsEvent("my error");
  }

  public void testErrorPropagation2() throws Exception {
    scratch.file("test/skylark/extension.bzl", "a = 'my_target'", "fail('my error')");
    scratch.file("test/skylark/BUILD", "load('extension', 'a')", "cc_library(name = a)");

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:my_target");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      assertThat(e.getMessage()).contains("Extension file 'test/skylark/extension.bzl' has errors");
    }
    assertContainsEvent("my error");
  }

  private void checkConfigurableHelloTargets() throws Exception {
    useConfiguration("--test_arg=a");
    Iterable<String> inputsA = getInputsForFilesToBuildAction("//test/skylark:hello");
    assertThat(inputsA).contains("adep.pic.o");
    assertThat(inputsA).doesNotContain("bdep.pic.o");

    useConfiguration("--test_arg=b");
    Iterable<String> inputsB = getInputsForFilesToBuildAction("//test/skylark:hello");
    assertThat(inputsB).contains("bdep.pic.o");
    assertThat(inputsB).doesNotContain("adep.pic.o");
  }

  private Iterable<String> getInputsForFilesToBuildAction(String label) throws Exception {
    return ActionsTestUtil.baseArtifactNames(
        getGeneratingAction(getOnlyElement(getFilesToBuild(getConfiguredTarget(label))))
            .getInputs());
  }

  // Regression test for https://github.com/google/bazel/issues/121
  public void testAddingListAndSkylarkList() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def macro(n, input_dict):",
        "  d = ' '.join(input_dict['list'] + ['2'])",
        "  native.genrule(name = n, cmd = 'echo %s >@' % d, outs = [n + '.txt'])");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'macro')",
        "",
        "macro(n = 'a', input_dict = {'list': ['1']})");
    SpawnAction action =
        (SpawnAction)
            getGeneratingAction(
                Iterables.getOnlyElement(getFilesToBuild(getConfiguredTarget("//test/skylark:a"))));
    String cmd = action.getArguments().get(2);
    assertThat(cmd).contains("echo 1 2 >@");
  }

  public void testLoadAlias() throws Exception {
    createBzlFileForAlias();
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/alias', number = 'one')",
        "number(name = 'target')");

    checkNumber("target", 1);
  }

  public void testLoadAliasIdentity() throws Exception {
    createBzlFileForAlias();
    scratch.file(
        "test/skylark/BUILD", "load('/test/skylark/alias', one = 'one')", "one(name = 'target')");

    checkNumber("target", 1);
  }

  public void testLoadAliasUseExistingName() throws Exception {
    createBzlFileForAlias();
    scratch.file(
        "test/skylark/BUILD", "load('/test/skylark/alias', two = 'one')", "two(name = 'target')");

    checkNumber("target", 1);
  }

  public void testLoadAliasSwapDefinitions() throws Exception {
    createBzlFileForAlias();
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/alias', one = 'two', two = 'one')",
        "one(name = 'target1')",
        "two(name = 'target2')");

    checkNumber("target1", 2);
    checkNumber("target2", 1);
  }

  public void testLoadAliasMultipleImports() throws Exception {
    createBzlFileForAlias();
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/alias', 'one', two = 'one', three = 'one')",
        "one(name = 'target1')",
        "two(name = 'target2')",
        "three(name = 'target3')");

    checkNumber("target1", 1);
    checkNumber("target2", 1);
    checkNumber("target3", 1);
  }

  public void testLoadAliasConflict() throws Exception {
    createBzlFileForAlias();
    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/alias', one = 'two', 'one')",
        "one(name = 'target')");

    reporter.removeHandler(failFastHandler);
    checkLoadAliasError("Symbol 'one' has already been loaded");
  }

  public void testLoadAliasNotFound() throws Exception {
    createBzlFileForAlias();
    scratch.file(
        "test/skylark/BUILD", "load('/test/skylark/alias', one = 'four')", "one(name = 'target')");

    reporter.removeHandler(failFastHandler);
    checkLoadAliasError("no such variable: four");
  }

  private void checkLoadAliasError(String errorMessage) throws Exception {
    try {
      getTarget("//test/skylark:target");
      fail("Expected NoSuchTargetException");
    } catch (NoSuchTargetException nste) {
      assertContainsEvent(errorMessage);
    }
  }

  private void createBzlFileForAlias() throws Exception {
    scratch.file(
        "test/skylark/alias.bzl",
        "def one_impl(ctx):",
        "  return struct(num = 1)",
        "",
        "def two_impl(ctx):",
        "  return struct(num = 2)",
        "",
        "def three_impl(ctx):",
        "  return struct(num = 3)",
        "",
        "one = rule(implementation = one_impl)",
        "two = rule(implementation = two_impl)",
        "three = rule(implementation = three_impl)");
  }

  private void checkNumber(String targetName, int expectedValue) throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:" + targetName);
    assertThat(target.get("num")).isEqualTo(expectedValue);
  }

  public void testInclusionOfRunfilesOfInputs() throws Exception {
    scratch.file("run/file/test/secret_script.sh", "#!/bin/bash");
    scratch.file("run/file/test/important_data", "secret");
    scratch.file("run/file/test/useless_data", "useless!");
    scratch.file(
        "run/file/test/myrule.bzl",
        "def _impl(ctx):",
        "  ctx.action(command = '%s > %s' % (ctx.executable.script.path, ctx.outputs.out.path), ",
        "  inputs = [ctx.executable.script], ",
        "  outputs = [ctx.outputs.out])",
        "my_rule = rule(implementation = _impl, ",
        "  attrs = {",
        "    'script' : attr.label(allow_files=True,executable=True,cfg=HOST_CFG),",
        "    'junk' : attr.label(allow_files=True,executable=True),",
        "  }",
        "  ,",
        "  outputs = {'out': '%{name}_out'}  ",
        ")");
    scratch.file(
        "run/file/test/BUILD",
        "load('/run/file/test/myrule', 'my_rule')",
        "",
        "sh_binary(",
        "  name = 'important_binary', ",
        "  srcs = ['secret_script.sh'],",
        "  data = ['important_data'],",
        ")",
        "sh_binary(",
        "  name = 'useless_binary', ",
        "  srcs = ['secret_script.sh'],",
        "  data = ['useless_data'],",
        ")",
        "my_rule(",
        "  name = 'main',",
        "  script = ':important_binary',",
        "  junk = ':useless_binary',",
        ")");

    Iterable<String> runfiles = extractDataRunfiles(getConfiguredTarget("//run/file/test:main"));
    assertThat(runfiles).contains("run/file/test/secret_script.sh");
    assertThat(runfiles).contains("run/file/test/important_data");
    // Attribute 'junk' did not have HOST_CFG, which means that useless_data must
    // not be included here
    assertThat(runfiles).doesNotContain("run/file/test/useless_data");
  }

  /**
   * Returns the data runfiles via some arcane magic. This includes:
   * a) Getting a middleman artifact from the inputs of the generating action
   * b) Expanding this artifact, as seen in ActionInputHelper#actionGraphMiddlemanExpander, by
   * collecting the inputs of the generating action
   *
   * <p>RunfilesProvider, FilesToRunProvider and RunfilesSupplier did not return any valid results.
   */
  private Iterable<String> extractDataRunfiles(ConfiguredTarget target) {
    List<String> result = new LinkedList<>();

    Artifact fileToBuild = Iterables.getOnlyElement(getFilesToBuild(target));
    Iterable<Artifact> inputs = getGeneratingAction(fileToBuild).getInputs();

    for (Artifact current : inputs) {
      if (current.isMiddlemanArtifact()) {
        Iterables.addAll(result, Artifact.toExecPaths(getGeneratingAction(current).getInputs()));
      }
    }

    return result;
  }

  public void testLoadList() throws Exception {
    scratch.file("test/skylark/extension.bzl", "li = [('abc', '11'), ('def', '22')]");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'li')",
        "[cc_library(name = lm[0]) for lm in li]");

    // Check that the targets can be loaded.
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:abc");
    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:abc");

    target = getConfiguredTarget("//test/skylark:def");
    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:def");
  }

  public void testListIsFrozenAfterLoad() throws Exception {
    scratch.file(
        "test/deflist.bzl",
        "l1 = [1, 2]",
        "l2 = l1",
        "l1.extend([3, 4])",
        "if l2 != [1, 2, 3, 4]: fail('l2=%r' % (l2,))");

    scratch.file(
        "test/uselist.bzl",
        "load('deflist', 'l1')",
        "print('loading uselist, l1=%r' % (l1,))",
        "l3 = l1 + [5, 6]",
        "if l3 != [1, 2, 3, 4, 5, 6]: fail('l3=%r' % (l3,))");

    scratch.file(
        "test/abuselist.bzl",
        "load('deflist', 'l1')",
        "l4 = []",
        "print('loading abuselist, l1=%r' % (l1,))",
        "l1.append(5)",
        "fail('l1=%r' % (l1,))");

    scratch.file(
        "test/BUILD",
        "load('uselist', 'l3')",
        "load('abuselist', 'l4')",
        "genrule(name='foo', outs=['one'], cmd='echo 1 > $@')");

    try {
      getConfiguredTarget("//test:foo");
      fail("An Exception was expected, but nothing was thrown.");
    } catch (AssertionFailedError ex) {
      assertThat(ex)
          .hasMessage("ERROR /workspace/test/abuselist.bzl:4:1: trying to mutate a frozen object");
    }
  }

  public void testLoadListIsFrozenBeforeAnalysis() throws Exception {
    scratch.file(
        "test/library.bzl",
        "l = [1, 2]",
        "def _impl(ctx): l.append(3) ; return struct()",
        "foo = rule(implementation=_impl)");

    scratch.file("test/BUILD", "load('library', 'foo')", "foo(name='foo')");

    try {
      getConfiguredTarget("//test:foo");
      fail("An Exception was expected, but nothing was thrown.");
    } catch (AssertionFailedError ex) {
      assertThat(ex)
          .hasMessage(
              LINE_JOINER.join(
                  "ERROR /workspace/test/BUILD:2:1: in foo rule //test:foo: ",
                  "Traceback (most recent call last):",
                  "\tFile \"/workspace/test/BUILD\", line 2",
                  "\t\tfoo(name = 'foo')",
                  "\tFile \"/workspace/test/library.bzl\", line 2, in _impl",
                  "\t\tl.append(3)",
                  "trying to mutate a frozen object"));
    }
  }

  public void testAspect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(
            ImmutableList.of("//test:xxx"),
            ImmutableList.<String>of("test/aspect.bzl%MyAspect"),
            false,
            LOADING_PHASE_THREADS,
            true,
            new EventBus());
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
  }

  /**
   * Skylark integration test that forces inlining.
   */
  public static class SkylarkIntegrationTestsWithInlineCalls extends SkylarkIntegrationTest {
    @Override
    public void setUp() throws Exception {
      super.setUp();
      ImmutableMap<? extends SkyFunctionName, ? extends SkyFunction> skyFunctions =
          ((InMemoryMemoizingEvaluator) getSkyframeExecutor().getEvaluatorForTesting())
              .getSkyFunctionsForTesting();
      SkylarkImportLookupFunction skylarkImportLookupFunction =
          new SkylarkImportLookupFunction(this.getRuleClassProvider(), this.getPackageFactory());
      ((PackageFunction) skyFunctions.get(SkyFunctions.PACKAGE))
          .setSkylarkImportLookupFunctionForInliningForTesting(skylarkImportLookupFunction);
    }

    @Override
    public void testRecursiveImport() throws Exception {
      scratch.file("test/skylark/ext2.bzl", "load('/test/skylark/ext1', 'symbol2')");

      scratch.file("test/skylark/ext1.bzl", "load('/test/skylark/ext2', 'symbol1')");

      scratch.file(
          "test/skylark/BUILD",
          "load('/test/skylark/ext1', 'custom_rule')",
          "genrule(name = 'rule')");

      reporter.removeHandler(failFastHandler);
      try {
        // ensureTargetsVisited() produces a different event than getTarget, and it doesn't fail
        // even though there is an error in the rule. What's going on here?
        ensureTargetsVisited("//test/skylark:rule");
        getTarget("//test/skylark:rule");
        fail();
      } catch (BuildFileContainsErrorsException e) {
        // This is expected
      }
      assertContainsEvent("cycle in referenced extension files");
      assertContainsEvent("test/skylark/ext1.bzl");
      assertContainsEvent("test/skylark/ext2.bzl");
      assertContainsEvent("Skylark import cycle");
      assertContainsEvent("Loading of target '//test/skylark:rule' failed; build aborted");
      assertThat(eventCollector).hasSize(3);
    }
  }
}
