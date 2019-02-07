// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.analysis.OutputGroupInfo.INTERNAL_SUFFIX;
import static com.google.devtools.build.lib.packages.FunctionSplitTransitionWhitelist.WHITELIST_ATTRIBUTE_NAME;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.skylark.StarlarkAttributeTransitionProvider;
import com.google.devtools.build.lib.analysis.skylark.StarlarkRuleTransitionProvider;
import com.google.devtools.build.lib.analysis.test.AnalysisFailure;
import com.google.devtools.build.lib.analysis.test.AnalysisFailureInfo;
import com.google.devtools.build.lib.analysis.test.AnalysisTestResultInfo;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkProvider;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for Skylark.
 */
@RunWith(JUnit4.class)
public class SkylarkIntegrationTest extends BuildViewTestCase {
  protected boolean keepGoing() {
    return false;
  }

  @Test
  public void testRemoteLabelAsDefaultAttributeValue() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  pass",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label_list(default=[\"@r//:t\"]) })");

    // We are only interested in whether the label string in the default value can be converted
    // to a proper Label without an exception (see GitHub issue #1442).
    // Consequently, we expect getTarget() to fail later since the repository does not exist.
    checkError(
        "test/skylark",
        "the_rule",
        "no such package '@r//': The repository could not be resolved",
        "load('//test/skylark:extension.bzl', 'my_rule')",
        "",
        "my_rule(name='the_rule')");
  }

  @Test
  public void testSameMethodNames() throws Exception {
    // The alias feature of load() may hide the fact that two methods in the stack trace have the
    // same name. This is perfectly legal as long as these two methods are actually distinct.
    // Consequently, no "Recursion was detected" error must be thrown.
    scratch.file(
        "test/skylark/extension.bzl",
        "load('//test/skylark:other.bzl', other_impl = 'impl')",
        "def impl(ctx):",
        "  other_impl(ctx)",
        "empty = rule(implementation = impl)");
    scratch.file("test/skylark/other.bzl", "def impl(ctx):", "  print('This rule does nothing')");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'empty')",
        "empty(name = 'test_target')");

    getConfiguredTarget("//test/skylark:test_target");
  }

  private AttributeContainer getContainerForTarget(String targetName) throws Exception {
    ConfiguredTargetAndData target = getConfiguredTargetAndData("//test/skylark:" + targetName);
    return target.getTarget().getAssociatedRule().getAttributeContainer();
  }

  @Test
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
        "load('//test/skylark:extension.bzl', macro_rule = 'macro', no_macro_rule = 'no_macro',",
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

  @Test
  public void sanityCheckUserDefinedTestRule() throws Exception {
    scratch.file(
        "test/skylark/test_rule.bzl",
        "def _impl(ctx):",
        "  output = ctx.outputs.out",
        "  ctx.actions.write(output = output, content = 'hello')",
        "",
        "fake_test = rule(",
        "  implementation = _impl,",
        "  test=True,",
        "  attrs = {'_xcode_config': attr.label(default = configuration_field(",
        "  fragment = 'apple', name = \"xcode_config_label\"))},",
        "  outputs = {\"out\": \"%{name}.txt\"})");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:test_rule.bzl', 'fake_test')",
        "fake_test(name = 'test_name')");
    getConfiguredTarget("//test/skylark:fake_test");
  }

  @Test
  public void testOutputGroups() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  f = ctx.attr.dep.output_group('_hidden_top_level" + INTERNAL_SUFFIX + "')",
        "  return struct(result = f, ",
        "               output_groups = { 'my_group' : f })",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label() })");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "cc_binary(name = 'lib', data = ['a.txt'])",
        "my_rule(name='my', dep = ':lib')");
    NestedSet<Artifact> hiddenTopLevelArtifacts =
        OutputGroupInfo.get(getConfiguredTarget("//test/skylark:lib"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    ConfiguredTarget myTarget = getConfiguredTarget("//test/skylark:my");
    SkylarkNestedSet result = (SkylarkNestedSet) myTarget.get("result");
    assertThat(result.getSet(Artifact.class)).containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_group"))
        .containsExactlyElementsIn(hiddenTopLevelArtifacts);
  }

  @Test
  public void testOutputGroupsDeclaredProvider() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  f = ctx.attr.dep[OutputGroupInfo]._hidden_top_level" + INTERNAL_SUFFIX,
        "  return struct(result = f, ",
        "                providers = [OutputGroupInfo(my_group = f)])",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label() })");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "cc_binary(name = 'lib', data = ['a.txt'])",
        "my_rule(name='my', dep = ':lib')");
    NestedSet<Artifact> hiddenTopLevelArtifacts =
        OutputGroupInfo.get(getConfiguredTarget("//test/skylark:lib"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    ConfiguredTarget myTarget = getConfiguredTarget("//test/skylark:my");
    SkylarkNestedSet result = (SkylarkNestedSet) myTarget.get("result");
    assertThat(result.getSet(Artifact.class)).containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_group"))
        .containsExactlyElementsIn(hiddenTopLevelArtifacts);
  }


  @Test
  public void testOutputGroupsAsDictionary() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  f = ctx.attr.dep.output_groups['_hidden_top_level" + INTERNAL_SUFFIX + "']",
        "  has_key1 = '_hidden_top_level" + INTERNAL_SUFFIX + "' in ctx.attr.dep.output_groups",
        "  has_key2 = 'foobar' in ctx.attr.dep.output_groups",
        "  all_keys = [k for k in ctx.attr.dep.output_groups]",
        "  return struct(result = f, ",
        "                has_key1 = has_key1,",
        "                has_key2 = has_key2,",
        "                all_keys = all_keys,",
        "               output_groups = { 'my_group' : f })",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label() })");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "cc_binary(name = 'lib', data = ['a.txt'])",
        "my_rule(name='my', dep = ':lib')");
    NestedSet<Artifact> hiddenTopLevelArtifacts =
        OutputGroupInfo.get(getConfiguredTarget("//test/skylark:lib"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    ConfiguredTarget myTarget = getConfiguredTarget("//test/skylark:my");
    SkylarkNestedSet result = (SkylarkNestedSet) myTarget.get("result");
    assertThat(result.getSet(Artifact.class)).containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_group"))
        .containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(myTarget.get("has_key1")).isEqualTo(Boolean.TRUE);
    assertThat(myTarget.get("has_key2")).isEqualTo(Boolean.FALSE);
    assertThat((SkylarkList) myTarget.get("all_keys"))
        .containsExactly(
            OutputGroupInfo.HIDDEN_TOP_LEVEL,
            OutputGroupInfo.COMPILATION_PREREQUISITES,
            OutputGroupInfo.FILES_TO_COMPILE,
            OutputGroupInfo.TEMP_FILES);
  }

  @Test
  public void testOutputGroupsAsDictionaryPipe() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  f = ctx.attr.dep.output_groups['_hidden_top_level" + INTERNAL_SUFFIX + "']",
        "  g = ctx.attr.dep.output_groups['_hidden_top_level" + INTERNAL_SUFFIX + "'] | depset([])",
        "  return struct(result = g, ",
        "                output_groups = { 'my_group' : g })",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label() })");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "cc_binary(name = 'lib', data = ['a.txt'])",
        "my_rule(name='my', dep = ':lib')");
    NestedSet<Artifact> hiddenTopLevelArtifacts =
        OutputGroupInfo.get(getConfiguredTarget("//test/skylark:lib"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    ConfiguredTarget myTarget = getConfiguredTarget("//test/skylark:my");
    SkylarkNestedSet result = (SkylarkNestedSet) myTarget.get("result");
    assertThat(result.getSet(Artifact.class)).containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_group"))
        .containsExactlyElementsIn(hiddenTopLevelArtifacts);
  }

  @Test
  public void testOutputGroupsWithList() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  f = ctx.attr.dep.output_group('_hidden_top_level" + INTERNAL_SUFFIX + "')",
        "  g = list(f)",
        "  return struct(result = f, ",
        "               output_groups = { 'my_group' : g, 'my_empty_group' : [] })",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label() })");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "cc_binary(name = 'lib', data = ['a.txt'])",
        "my_rule(name='my', dep = ':lib')");
    NestedSet<Artifact> hiddenTopLevelArtifacts =
        OutputGroupInfo.get(getConfiguredTarget("//test/skylark:lib"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    ConfiguredTarget myTarget = getConfiguredTarget("//test/skylark:my");
    SkylarkNestedSet result =
        (SkylarkNestedSet) myTarget.get("result");
    assertThat(result.getSet(Artifact.class)).containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_group"))
        .containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_empty_group"))
        .isEmpty();
  }

  @Test
  public void testOutputGroupsDeclaredProviderWithList() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  f = ctx.attr.dep[OutputGroupInfo]._hidden_top_level" + INTERNAL_SUFFIX,
        "  g = list(f)",
        "  return struct(result = f, ",
        "                providers = [OutputGroupInfo(my_group = g, my_empty_group = [])])",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label() })");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "cc_binary(name = 'lib', data = ['a.txt'])",
        "my_rule(name='my', dep = ':lib')");
    NestedSet<Artifact> hiddenTopLevelArtifacts =
        OutputGroupInfo.get(getConfiguredTarget("//test/skylark:lib"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    ConfiguredTarget myTarget = getConfiguredTarget("//test/skylark:my");
    SkylarkNestedSet result =
        (SkylarkNestedSet) myTarget.get("result");
    assertThat(result.getSet(Artifact.class)).containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_group"))
        .containsExactlyElementsIn(hiddenTopLevelArtifacts);
    assertThat(OutputGroupInfo.get(myTarget).getOutputGroup("my_empty_group"))
        .isEmpty();
  }

  @Test
  public void testStackTraceErrorInFunction() throws Exception {
    runStackTraceTest(
        "str",
        "\t\tstr.index(1)"
            + System.lineSeparator()
            + "expected value of type 'string' for parameter 'sub', for call to method "
            + "index(sub, start = 0, end = None) of 'string'");
  }

  @Test
  public void testStackTraceMissingMethod() throws Exception {
    runStackTraceTest(
        "None",
        "\t\tNone.index(1)" + System.lineSeparator() + "type 'NoneType' has no method index()");
  }

  protected void runStackTraceTest(String object, String errorMessage) throws Exception {
    reporter.removeHandler(failFastHandler);
    String expectedTrace =
        Joiner.on(System.lineSeparator())
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
                "\t\tthird(\"legal\")",
                "\tFile \"/workspace/test/skylark/functions.bzl\", line 7, in third",
                errorMessage);
    scratch.file(
        "test/skylark/extension.bzl",
        "load('//test/skylark:functions.bzl', 'first')",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  ftb = depset(attr1)",
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    getConfiguredTarget("//test/skylark:cr");
    assertContainsEvent(expectedTrace);
  }

  @Test
  public void testFilesToBuild() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  ftb = depset(attr1)",
        "  return struct(runfiles = ctx.runfiles(), files = ftb)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:cr");
    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("a.txt");
  }

  @Test
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:cr");
    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(RunfilesProvider.class).getDefaultRunfiles().getAllArtifacts()))
        .containsExactly("a.txt");
    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(RunfilesProvider.class).getDataRunfiles().getAllArtifacts()))
        .containsExactly("a.txt");
  }

  @Test
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
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

  @Test
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:cr");
    assertThat(target.getProvider(RunfilesProvider.class).getDefaultRunfiles().isEmpty()).isTrue();
    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(RunfilesProvider.class).getDataRunfiles().getAllArtifacts()))
        .containsExactly("a.txt");
  }

  @Test
  public void testExecutableGetsInRunfilesAndFilesToBuild() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  ctx.actions.write(output = ctx.outputs.executable, content = 'echo hello')",
        "  rf = ctx.runfiles(ctx.files.data)",
        "  return struct(runfiles = rf)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl, executable = True,",
        "  attrs = {'data': attr.label_list(allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', data = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:cr");
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

  @Test
  public void testCannotSpecifyRunfilesWithDataOrDefaultRunfiles_struct() throws Exception {
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
  }

  @Test
  public void testCannotSpecifyRunfilesWithDataOrDefaultRunfiles_defaultInfo() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  rf = ctx.runfiles()",
        "  return struct(DefaultInfo(runfiles = rf, default_runfiles = rf))",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    checkError(
        "test/skylark",
        "cr",
        "Cannot specify the provider 'runfiles' together with "
            + "'data_runfiles' or 'default_runfiles'",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
  }

  @Test
  public void testDefaultInfoWithRunfilesConstructor() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "sh_binary(name = 'tryme',",
        "          srcs = [':tryme.sh'],",
        "          visibility = ['//visibility:public'],",
        ")");

    scratch.file(
        "src/rulez.bzl",
        "def  _impl(ctx):",
        "   info = DefaultInfo(runfiles = ctx.runfiles(files=[ctx.executable.dep]))",
        "   if info.default_runfiles.files.to_list()[0] != ctx.executable.dep:",
        "       fail('expected runfile to be in info.default_runfiles')",
        "   return [info]",
        "r = rule(_impl,",
        "         attrs = {",
        "            'dep' : attr.label(executable = True, mandatory = True, cfg = 'host'),",
        "         }",
        ")");

    scratch.file(
        "src/BUILD", "load(':rulez.bzl', 'r')", "r(name = 'r_tools', dep = '//pkg:tryme')");

    assertThat(getConfiguredTarget("//src:r_tools")).isNotNull();
  }

  @Test
  public void testInstrumentedFilesProviderWithCodeCoverageDisabled() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(instrumented_files=struct(",
        "      extensions = ['txt'],",
        "      source_attributes = ['attr1'],",
        "      dependency_attributes = ['attr2']))",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {",
        "      'attr1': attr.label_list(mandatory = True, allow_files=True),",
        "      'attr2': attr.label_list(mandatory = True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "java_library(name='jl', srcs = [':A.java'])",
        "custom_rule(name = 'cr', attr1 = [':a.txt', ':a.random'], attr2 = [':jl'])");

    useConfiguration("--nocollect_code_coverage");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:cr");
    InstrumentedFilesInfo provider = target.get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR);
    assertWithMessage("InstrumentedFilesInfo should be set.").that(provider).isNotNull();
    assertThat(ActionsTestUtil.baseArtifactNames(provider.getInstrumentedFiles())).isEmpty();
  }

  @Test
  public void testInstrumentedFilesProviderWithCodeCoverageEnabled() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(instrumented_files=struct(",
        "      extensions = ['txt'],",
        "      source_attributes = ['attr1'],",
        "      dependency_attributes = ['attr2']))",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {",
        "      'attr1': attr.label_list(mandatory = True, allow_files=True),",
        "      'attr2': attr.label_list(mandatory = True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "java_library(name='jl', srcs = [':A.java'])",
        "custom_rule(name = 'cr', attr1 = [':a.txt', ':a.random'], attr2 = [':jl'])");

    useConfiguration("--collect_code_coverage");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:cr");
    InstrumentedFilesInfo provider = target.get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR);
    assertWithMessage("InstrumentedFilesInfo should be set.").that(provider).isNotNull();
    assertThat(ActionsTestUtil.baseArtifactNames(provider.getInstrumentedFiles()))
        .containsExactly("a.txt", "A.java");
  }

  @Test
  public void testInstrumentedFilesInfo_coverageDisabled() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct(instrumented_files=struct(",
        "      extensions = ['txt'],",
        "      source_attributes = ['attr1'],",
        "      dependency_attributes = ['attr2']))",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {",
        "      'attr1': attr.label_list(mandatory = True, allow_files=True),",
        "      'attr2': attr.label_list(mandatory = True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "java_library(name='jl', srcs = [':A.java'])",
        "custom_rule(name = 'cr', attr1 = [':a.txt', ':a.random'], attr2 = [':jl'])");

    useConfiguration("--nocollect_code_coverage");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    InstrumentedFilesInfo provider = target.get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR);
    assertWithMessage("InstrumentedFilesInfo should be set.").that(provider).isNotNull();
    assertThat(ActionsTestUtil.baseArtifactNames(provider.getInstrumentedFiles())).isEmpty();
  }

  @Test
  public void testInstrumentedFilesInfo_coverageEnabled() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return [coverage_common.instrumented_files_info(ctx,",
        "      extensions = ['txt'],",
        "      source_attributes = ['attr1'],",
        "      dependency_attributes = ['attr2'])]",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {",
        "      'attr1': attr.label_list(mandatory = True, allow_files=True),",
        "      'attr2': attr.label_list(mandatory = True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "java_library(name='jl', srcs = [':A.java'])",
        "custom_rule(name = 'cr', attr1 = [':a.txt', ':a.random'], attr2 = [':jl'])");

    useConfiguration("--collect_code_coverage");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    InstrumentedFilesInfo provider = target.get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR);
    assertWithMessage("InstrumentedFilesInfo should be set.").that(provider).isNotNull();
    assertThat(ActionsTestUtil.baseArtifactNames(provider.getInstrumentedFiles()))
        .containsExactly("a.txt", "A.java");
  }

  @Test
  public void testTransitiveInfoProviders() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  ftb = depset(attr1)",
        "  return struct(provider_key = ftb)",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//test/skylark:cr");

    assertThat(
        ActionsTestUtil.baseArtifactNames(
            ((SkylarkNestedSet) target.get("provider_key")).getSet(Artifact.class)))
        .containsExactly("a.txt");
  }

  @Test
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
            + "'//test:a' does not have mandatory providers: 'some_provider'",
        "load('//test/skylark:extension.bzl', 'dependent_rule')",
        "load('//test/skylark:extension.bzl', 'main_rule')",
        "",
        "dependent_rule(name = 'a')",
        "main_rule(name = 'b', dependencies = [':a'])");
  }

  @Test
  public void testSpecialMandatoryProviderMissing() throws Exception {
    // Test that rules satisfy `providers = [...]` condition if a special provider that always
    // exists for all rules is requested. Also check external rules.

    FileSystemUtils.appendIsoLatin1(scratch.resolve("WORKSPACE"),
        "bind(name = 'bar', actual = '//test/ext:bar')");
    scratch.file(
        "test/ext/BUILD",
        "load('//test/skylark:extension.bzl', 'foobar')",
        "",
        "foobar(name = 'bar', visibility = ['//visibility:public'],)");
    scratch.file(
        "test/skylark/extension.bzl",
        "def rule_impl(ctx):",
        "  pass",
        "",
        "foobar = rule(implementation = rule_impl)",
        "main_rule = rule(implementation = rule_impl, attrs = {",
        "    'deps': attr.label_list(providers = [",
        "        'files', 'data_runfiles', 'default_runfiles',",
        "        'files_to_run', 'output_groups',",
        "    ])",
        "})");
    scratch.file(
        "test/skylark/BUILD",
        "load(':extension.bzl', 'foobar', 'main_rule')",
        "",
        "foobar(name = 'foo')",
        "main_rule(name = 'main', deps = [':foo', '//external:bar'])");

    invalidatePackages();
    getConfiguredTarget("//test/skylark:main");
  }

  @Test
  public void testActions() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  attr1 = ctx.files.attr1",
        "  output = ctx.outputs.o",
        "  ctx.actions.run_shell(",
        "    inputs = attr1,",
        "    outputs = [output],",
        "    command = 'echo')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.label_list(mandatory=True, allow_files=True)},",
        "  outputs = {'o': 'o.txt'})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    getConfiguredTarget("//test/skylark:cr");

    FileConfiguredTarget target = getFileConfiguredTarget("//test/skylark:o.txt");
    assertThat(
        ActionsTestUtil.baseArtifactNames(
            getGeneratingAction(target.getArtifact()).getInputs()))
        .containsExactly("a.txt");
  }

  @Test
  public void testRuleClassImplicitOutputFunction() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  files = [ctx.outputs.o]",
        "  ctx.actions.run_shell(",
        "    outputs = files,",
        "    command = 'echo')",
        "  ftb = depset(files)",
        "  return struct(runfiles = ctx.runfiles(), files = ftb)",
        "",
        "def output_func(name, public_attr, _private_attr):",
        "  if _private_attr != None: return {}",
        "  return {'o': name + '-' + public_attr + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'public_attr': attr.string(),",
        "           '_private_attr': attr.label()},",
        "  outputs = output_func)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', public_attr = 'bar')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
            ActionsTestUtil.baseArtifactNames(
                target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("cr-bar.txt");
  }

  @Test
  public void testRuleClassImplicitOutputFunctionDependingOnComputedAttribute() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  files = [ctx.outputs.o]",
        "  ctx.actions.run_shell(",
        "    outputs = files,",
        "    command = 'echo')",
        "  ftb = depset(files)",
        "  return struct(runfiles = ctx.runfiles(), files = ftb)",
        "",
        "def attr_func(public_attr):",
        "  return public_attr",
        "",
        "def output_func(_private_attr):",
        "  return {'o': _private_attr.name + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'public_attr': attr.label(),",
        "           '_private_attr': attr.label(default = attr_func)},",
        "  outputs = output_func)",
        "",
        "def empty_rule_impl(ctx):",
        "  pass",
        "",
        "empty_rule = rule(implementation = empty_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule', 'empty_rule')",
        "",
        "empty_rule(name = 'foo')",
        "custom_rule(name = 'cr', public_attr = '//test/skylark:foo')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("foo.txt");
  }

  @Test
  public void testRuleClassImplicitOutputs() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  files = [ctx.outputs.lbl, ctx.outputs.list, ctx.outputs.str]",
        "  print('==!=!=!=')",
        "  print(files)",
        "  ctx.actions.run_shell(",
        "    outputs = files,",
        "    command = 'echo')",
        "  return struct(files = depset(files))",
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
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

  @Test
  public void testRuleClassImplicitOutputFunctionAndDefaultValue() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  ctx.actions.run_shell(",
        "    outputs = [ctx.outputs.o],",
        "    command = 'echo')",
        "  return struct(runfiles = ctx.runfiles())",
        "",
        "def output_func(attr1):",
        "  return {'o': attr1 + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.string(default='bar')},",
        "  outputs = output_func)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = None)");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("bar.txt");
  }

  @Test
  public void testPrintProviderCollection() throws Exception {
    scratch.file(
        "test/skylark/rules.bzl",
        "",
        "FooInfo = provider()",
        "BarInfo = provider()",
        "",
        "def _top_level_rule_impl(ctx):",
        "  print('My Dep Providers:', ctx.attr.my_dep)",
        "",
        "def _dep_rule_impl(name):",
        "  providers = [",
        "      FooInfo(),",
        "      BarInfo(),",
        "  ]",
        "  return providers",
        "",
        "top_level_rule = rule(",
        "    implementation=_top_level_rule_impl,",
        "    attrs={'my_dep':attr.label()}",
        ")",
        "",
        "dep_rule = rule(",
        "    implementation=_dep_rule_impl,",
        ")");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:rules.bzl', 'top_level_rule', 'dep_rule')",
        "",
        "top_level_rule(name = 'tl', my_dep=':d')",
        "",
        "dep_rule(name = 'd')");

    getConfiguredTarget("//test/skylark:tl");
    assertContainsEvent(
        "My Dep Providers: <target //test/skylark:d, keys:[FooInfo, BarInfo, OutputGroupInfo]>");
  }

  @Test
  public void testRuleClassImplicitOutputFunctionPrints() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  print('implementation', ctx.label)",
        "  files = [ctx.outputs.o]",
        "  ctx.actions.run_shell(",
        "    outputs = files,",
        "    command = 'echo')",
        "",
        "def output_func(name):",
        "  print('output function', name)",
        "  return {'o': name + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  outputs = output_func)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    getConfiguredTarget("//test/skylark:cr");
    assertContainsEvent("output function cr");
    assertContainsEvent("implementation //test/skylark:cr");
  }

  @Test
  public void testNoOutputAttrDefault() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_output_attr_default=true");

    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  out_file = ctx.actions.declare_file(ctx.attr._o1.name)",
        "  ctx.actions.write(output=out_file, content='hi')",
        "  return struct(o1=ctx.attr._o1)",
        "",
        "def output_fn():",
        "  return Label('//test/skylark:foo.txt')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'_o1': attr.output(default = output_fn)})");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'r')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:r");
    assertContainsEvent(
        "parameter 'default' is deprecated and will be removed soon. It may be "
            + "temporarily re-enabled by setting --incompatible_no_output_attr_default=false");
  }

  @Test
  public void testTransitionGuardedByExperimentalFlag() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=false");

    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return []",
        "def transition_func(settings):",
        "  return settings",
        "my_transition = transition(implementation = transition_func, inputs = [], outputs = [])",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'r')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:r");
    assertContainsEvent("transition() is experimental and disabled by default");
  }

  @Test
  public void testNoOutputListAttrDefault() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_output_attr_default=true");

    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return []",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'outs': attr.output_list(default = [])})");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'r')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:r");
    assertContainsEvent(
        "parameter 'default' is deprecated and will be removed soon. It may be "
            + "temporarily re-enabled by setting --incompatible_no_output_attr_default=false");
  }

  @Test
  public void testLegacyOutputAttrDefault() throws Exception {
    // Note that use of the "default" parameter of attr.output and attr.output_label is deprecated
    // and barely functional. This test simply serves as proof-of-concept verification that the
    // legacy behavior remains intact.
    setSkylarkSemanticsOptions("--incompatible_no_output_attr_default=false");

    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  out_file = ctx.actions.declare_file(ctx.attr._o1.name)",
        "  ctx.actions.write(output=out_file, content='hi')",
        "  return struct(o1=ctx.attr._o1,",
        "                o2=ctx.attr.o2)",
        "",
        "def output_fn():",
        "  return Label('//test/skylark:foo.txt')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'_o1': attr.output(default = output_fn),",
        "           'o2': attr.output_list(default = [])})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertThat(target.get("o1")).isEqualTo(Label.parseAbsoluteUnchecked("//test/skylark:foo.txt"));
    assertThat(target.get("o2")).isEqualTo(MutableList.empty());
  }

  @Test
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");
    assertThat(target.get("o1")).isEqualTo(Runtime.NONE);
    assertThat(target.get("o2")).isEqualTo(MutableList.empty());
  }

  @Test
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', o = [':bar.txt'])");
  }

  @Test
  public void testRuleClassDefaultFilesToBuild() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  files = [ctx.outputs.o]",
        "  ctx.actions.run_shell(",
        "    outputs = files,",
        "    command = 'echo')",
        "  ftb = depset(files)",
        "  for i in ctx.outputs.out:",
        "    ctx.actions.write(output=i, content='hi there')",
        "",
        "def output_func(attr1):",
        "  return {'o': attr1 + '.txt'}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {",
        "    'attr1': attr.string(),",
        "    'out': attr.output_list()",
        "  },",
        "  outputs = output_func)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = 'bar', out=['other'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("bar.txt", "other")
        .inOrder();
  }

  @Test
  public void rulesReturningDeclaredProviders() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "my_provider = provider()",
        "def _impl(ctx):",
        "   return [my_provider(x = 1)]",
        "my_rule = rule(_impl)"
    );
    scratch.file(
        "test/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'r')"
    );

    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:r");
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.create(configuredTarget.getLabel().getPackageIdentifier(), "extension.bzl"),
            "my_provider");
    StructImpl declaredProvider = (StructImpl) configuredTarget.get(key);
    assertThat(declaredProvider).isNotNull();
    assertThat(declaredProvider.getProvider().getKey()).isEqualTo(key);
    assertThat(declaredProvider.getValue("x")).isEqualTo(1);
  }

  @Test
  public void rulesReturningDeclaredProvidersCompatMode() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "my_provider = provider()",
        "def _impl(ctx):",
        "   return struct(providers = [my_provider(x = 1)])",
        "my_rule = rule(_impl)"
    );
    scratch.file(
        "test/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'r')"
    );

    ConfiguredTarget configuredTarget  = getConfiguredTarget("//test:r");
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.create(configuredTarget.getLabel().getPackageIdentifier(), "extension.bzl"),
            "my_provider");
    StructImpl declaredProvider = (StructImpl) configuredTarget.get(key);
    assertThat(declaredProvider).isNotNull();
    assertThat(declaredProvider.getProvider().getKey()).isEqualTo(key);
    assertThat(declaredProvider.getValue("x")).isEqualTo(1);
  }

  @Test
  public void testRuleReturningUnwrappedDeclaredProvider() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "my_provider = provider()",
        "def _impl(ctx):",
        "   return my_provider(x = 1)",
        "my_rule = rule(_impl)"
    );
    scratch.file(
        "test/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'r')"
    );

    ConfiguredTarget configuredTarget  = getConfiguredTarget("//test:r");
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.create(configuredTarget.getLabel().getPackageIdentifier(), "extension.bzl"),
            "my_provider");
    StructImpl declaredProvider = (StructImpl) configuredTarget.get(key);
    assertThat(declaredProvider).isNotNull();
    assertThat(declaredProvider.getProvider().getKey()).isEqualTo(key);
    assertThat(declaredProvider.getValue("x")).isEqualTo(1);
  }

  @Test
  public void testConflictingProviderKeys_fromStruct_disallowed() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "my_provider = provider()",
        "other_provider = provider()",
        "def _impl(ctx):",
        "   return struct(providers = [my_provider(x = 1), other_provider(), my_provider(x = 2)])",
        "my_rule = rule(_impl)"
    );

    checkError(
        "test",
        "r",
        "Multiple conflicting returned providers with key my_provider",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'r')");
  }

  @Test
  public void testConflictingProviderKeys_fromIterable_disallowed() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "my_provider = provider()",
        "other_provider = provider()",
        "def _impl(ctx):",
        "   return [my_provider(x = 1), other_provider(), my_provider(x = 2)]",
        "my_rule = rule(_impl)"
    );

    checkError(
        "test",
        "r",
        "Multiple conflicting returned providers with key my_provider",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'r')");
  }

  @Test
  public void testRecursionDetection() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  _impl(ctx)",
        "empty = rule(implementation = _impl)");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'empty')",
        "empty(name = 'test_target')");

    getConfiguredTarget("//test/skylark:test_target");
    assertContainsEvent("Recursion was detected when calling '_impl' from '_impl'");
  }

  @Test
  public void testBadCallbackFunction() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl", "def impl(): return 0", "", "custom_rule = rule(impl)");

    checkError(
        "test/skylark",
        "cr",
        "impl() does not accept positional arguments",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");
  }

  @Test
  public void testRuleClassImplicitOutputFunctionBadAttr() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return None",
        "",
        "def output_func(bad_attr):",
        "  return {'a': bad_attr}",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "  attrs = {'attr1': attr.string()},",
        "  outputs = output_func)");

    checkError(
        "test/skylark",
        "cr",
        "Attribute 'bad_attr' either doesn't exist or uses a select()",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = 'bar')");
  }

  @Test
  public void testHelperFunctionInRuleImplementation() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def helper_func(attr1):",
        "  return depset(attr1)",
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr', attr1 = [':a.txt'])");

    ConfiguredTarget target = getConfiguredTarget("//test/skylark:cr");

    assertThat(target.getLabel().toString()).isEqualTo("//test/skylark:cr");
    assertThat(
        ActionsTestUtil.baseArtifactNames(
            target.getProvider(FileProvider.class).getFilesToBuild()))
        .containsExactly("a.txt");
  }

  @Test
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
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "custom_rule(name = 'cr1')");

    scratch.file(
        "test/skylark2/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "custom_rule(name = 'cr2', dep = ['//test/skylark1:cr1'])");

    getConfiguredTarget("//test/skylark2:cr2");
  }

  @Test
  public void testFunctionGeneratingRules() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "def impl(ctx): return None",
        "def gen(): return rule(impl)",
        "r = gen()",
        "s = gen()");

    scratch.file(
        "test/skylark/BUILD", "load(':extension.bzl', 'r', 's')", "r(name = 'r')", "s(name = 's')");

    getConfiguredTarget("//test/skylark:r");
    getConfiguredTarget("//test/skylark:s");
  }

  @Test
  public void testImportInSkylark() throws Exception {
    scratch.file("test/skylark/implementation.bzl", "def custom_rule_impl(ctx):", "  return None");

    scratch.file(
        "test/skylark/extension.bzl",
        "load('//test/skylark:implementation.bzl', 'custom_rule_impl')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl,",
        "     attrs = {'dep': attr.label_list(allow_files=True)})");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "custom_rule(name = 'cr')");

    getConfiguredTarget("//test/skylark:cr");
  }

  @Test
  public void testRuleAliasing() throws Exception {
    scratch.file(
        "test/skylark/implementation.bzl",
        "def impl(ctx): return struct()",
        "custom_rule = rule(implementation = impl)");

    scratch.file(
        "test/skylark/ext.bzl",
        "load('//test/skylark:implementation.bzl', 'custom_rule')",
        "def impl(ctx): return struct()",
        "custom_rule1 = rule(implementation = impl)",
        "custom_rule2 = custom_rule1",
        "custom_rule3 = custom_rule");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:ext.bzl', 'custom_rule1', 'custom_rule2', 'custom_rule3')",
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

  @Test
  public void testRecursiveImport() throws Exception {
    scratch.file("test/skylark/ext2.bzl", "load('//test/skylark:ext1.bzl', 'symbol2')");

    scratch.file("test/skylark/ext1.bzl", "load('//test/skylark:ext2.bzl', 'symbol1')");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:ext1.bzl', 'custom_rule')",
        "genrule(name = 'rule')");

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:rule");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      // This is expected
    }
    assertContainsEvent(
        "cycle detected in extension files: \n"
            + "    test/skylark/BUILD\n"
            + ".-> //test/skylark:ext1.bzl\n"
            + "|   //test/skylark:ext2.bzl\n"
            + "`-- //test/skylark:ext1.bzl");
  }

  @Test
  public void testRecursiveImport2() throws Exception {
    scratch.file("test/skylark/ext1.bzl", "load('//test/skylark:ext2.bzl', 'symbol2')");
    scratch.file("test/skylark/ext2.bzl", "load('//test/skylark:ext3.bzl', 'symbol3')");
    scratch.file("test/skylark/ext3.bzl", "load('//test/skylark:ext4.bzl', 'symbol4')");
    scratch.file("test/skylark/ext4.bzl", "load('//test/skylark:ext2.bzl', 'symbol2')");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:ext1.bzl', 'custom_rule')",
        "genrule(name = 'rule')");

    reporter.removeHandler(failFastHandler);
    try {
      getTarget("//test/skylark:rule");
      fail();
    } catch (BuildFileContainsErrorsException e) {
      // This is expected
    }
    assertContainsEvent(
        "cycle detected in extension files: \n"
            + "    //test/skylark:ext1.bzl\n"
            + ".-> //test/skylark:ext2.bzl\n"
            + "|   //test/skylark:ext3.bzl\n"
            + "|   //test/skylark:ext4.bzl\n"
            + "`-- //test/skylark:ext2.bzl");
  }

  @Test
  public void testSymbolPropagateThroughImports() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_transitive_loads=false");
    scratch.file("test/skylark/implementation.bzl", "def custom_rule_impl(ctx):", "  return None");

    scratch.file(
        "test/skylark/extension2.bzl",
        "load('//test/skylark:implementation.bzl', 'custom_rule_impl')");

    scratch.file(
        "test/skylark/extension1.bzl",
        "load('//test/skylark:extension2.bzl', 'custom_rule_impl')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension1.bzl', 'custom_rule')",
        "custom_rule(name = 'cr')");

    getConfiguredTarget("//test/skylark:cr");
  }

  @Test
  public void testSymbolDoNotPropagateThroughImports() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_transitive_loads=true");
    scratch.file("test/skylark/implementation.bzl", "def custom_rule_impl(ctx):", "  return None");

    scratch.file(
        "test/skylark/extension2.bzl",
        "load('//test/skylark:implementation.bzl', 'custom_rule_impl')");

    scratch.file(
        "test/skylark/extension1.bzl",
        "load('//test/skylark:extension2.bzl', 'custom_rule_impl')",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension1.bzl', 'custom_rule')",
        "custom_rule(name = 'cr')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:cr");
    assertContainsEvent("does not contain symbol 'custom_rule_impl'");
  }

  @Test
  public void testLoadSymbolTypo() throws Exception {
    scratch.file("test/skylark/ext1.bzl", "myvariable = 2");

    scratch.file("test/skylark/BUILD", "load('//test/skylark:ext1.bzl', 'myvariables')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:test_target");
    assertContainsEvent(
        "file '//test/skylark:ext1.bzl' does not contain symbol 'myvariables' "
            + "(did you mean 'myvariable'?)");
  }

  @Test
  public void testLoadSucceedsDespiteSyntaxError() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "test/skylark/macro.bzl",
        "x = 5");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:macro.bzl', 'x')",
        "pass", // syntax error
        "print(1 // (5 - x)"); // division by 0

    // Make sure that evaluation continues and load() succeeds, despite a syntax
    // error in the file.
    // We can get the division by 0 only if x was correctly loaded.
    getConfiguredTarget("//test/skylark:a");
    assertContainsEvent("syntax error");
    assertContainsEvent("integer division by zero");
  }

  @Test
  public void testOutputsObjectOrphanExecutableReportError() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "   o = ctx.outputs.executable",
        "   return [DefaultInfo(executable = o)]",
        "my_rule = rule(_impl, executable = True)"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "my_rule(name = 'xxx')"
    );

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:xxx");
    assertContainsEvent("ERROR /workspace/test/BUILD:2:1: in my_rule rule //test:xxx: ");
    assertContainsEvent("The following files have no generating action:");
    assertContainsEvent("test/xxx");
  }

  @Test
  public void testCustomExecutableUsed() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "   o = ctx.actions.declare_file('x.sh')",
        "   ctx.actions.write(o, 'echo Stuff', is_executable = True)",
        "   return [DefaultInfo(executable = o)]",
        "my_rule = rule(_impl, executable = True)"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "my_rule(name = 'xxx')"
    );

    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:xxx");
    Artifact executable = configuredTarget.getProvider(FilesToRunProvider.class).getExecutable();
    assertThat(executable.getRootRelativePathString()).isEqualTo("test/x.sh");
  }

  @Test
  public void testCustomAndDefaultExecutableReportsError() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "   e = ctx.outputs.executable",
        "   o = ctx.actions.declare_file('x.sh')",
        "   ctx.actions.write(o, 'echo Stuff', is_executable = True)",
        "   return [DefaultInfo(executable = o)]",
        "my_rule = rule(_impl, executable = True)"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "my_rule(name = 'xxx')"
    );
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:xxx");
    assertContainsEvent("ERROR /workspace/test/BUILD:2:1: in my_rule rule //test:xxx: ");
    assertContainsEvent("/workspace/test/rule.bzl:5:12: The rule 'my_rule' both accesses "
        + "'ctx.outputs.executable' and provides a different executable 'test/x.sh'. "
        + "Do not use 'ctx.output.executable'.");
  }


  @Test
  public void testCustomExecutableStrNoEffect() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "   o = ctx.actions.declare_file('x.sh')",
        "   ctx.actions.write(o, 'echo Stuff', is_executable = True)",
        "   print(str(ctx.outputs))",
        "   return [DefaultInfo(executable = o)]",
        "my_rule = rule(_impl, executable = True)"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "my_rule(name = 'xxx')"
    );

    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:xxx");
    Artifact executable = configuredTarget.getProvider(FilesToRunProvider.class).getExecutable();
    assertThat(executable.getRootRelativePathString()).isEqualTo("test/x.sh");
  }

  @Test
  public void testCustomExecutableDirNoEffect() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "   o = ctx.actions.declare_file('x.sh')",
        "   ctx.actions.write(o, 'echo Stuff', is_executable = True)",
        "   print(dir(ctx.outputs))",
        "   return [DefaultInfo(executable = o)]",
        "my_rule = rule(_impl, executable = True)"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "my_rule(name = 'xxx')"
    );

    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:xxx");
    Artifact executable = configuredTarget.getProvider(FilesToRunProvider.class).getExecutable();
    assertThat(executable.getRootRelativePathString()).isEqualTo("test/x.sh");
  }

  @Test
  public void testOutputsObjectInDifferentRuleInaccessible() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "PInfo = provider(fields = ['outputs'])",
        "def _impl(ctx):",
        "   o = ctx.actions.declare_file('x.sh')",
        "   ctx.actions.write(o, 'echo Stuff', is_executable = True)",
        "   return [PInfo(outputs = ctx.outputs), DefaultInfo(executable = o)]",
        "my_rule = rule(_impl, executable = True)",
        "def _dep_impl(ctx):",
        "   o = ctx.attr.dep[PInfo].outputs.executable",
        "   pass",
        "my_dep_rule = rule(_dep_impl, attrs = { 'dep' : attr.label() })"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule', 'my_dep_rule')",
        "my_rule(name = 'xxx')",
        "my_dep_rule(name = 'yyy', dep = ':xxx')"
    );

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:yyy");
    assertContainsEvent("ERROR /workspace/test/BUILD:3:1: in my_dep_rule rule //test:yyy: ");
    assertContainsEvent("File \"/workspace/test/rule.bzl\", line 8, in _dep_impl");
    assertContainsEvent("ctx.attr.dep[PInfo].outputs.executable");
    assertContainsEvent("cannot access outputs of rule '//test:xxx' outside "
        + "of its own rule implementation function");
  }

  @Test
  public void testOutputsObjectStringRepresentation() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "PInfo = provider(fields = ['outputs', 's'])",
        "def _impl(ctx):",
        "   ctx.actions.write(ctx.outputs.executable, 'echo Stuff', is_executable = True)",
        "   ctx.actions.write(ctx.outputs.other, 'Other')",
        "   return [PInfo(outputs = ctx.outputs, s = str(ctx.outputs))]",
        "my_rule = rule(_impl, executable = True, outputs = { 'other' : '%{name}.other' })",
        "def _dep_impl(ctx):",
        "   return [PInfo(s = str(ctx.attr.dep[PInfo].outputs))]",
        "my_dep_rule = rule(_dep_impl, attrs = { 'dep' : attr.label() })"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule', 'my_dep_rule')",
        "my_rule(name = 'xxx')",
        "my_dep_rule(name = 'yyy', dep = ':xxx')"
    );

    SkylarkKey pInfoKey =
        new SkylarkKey(Label.parseAbsolute("//test:rule.bzl", ImmutableMap.of()), "PInfo");

    ConfiguredTarget targetXXX = getConfiguredTarget("//test:xxx");
    StructImpl structXXX = (StructImpl) targetXXX.get(pInfoKey);

    assertThat(structXXX.getValue("s"))
        .isEqualTo(
            "ctx.outputs(executable = <generated file test/xxx>, "
                + "other = <generated file test/xxx.other>)");

    ConfiguredTarget targetYYY = getConfiguredTarget("//test:yyy");
    StructImpl structYYY = (StructImpl) targetYYY.get(pInfoKey);
    assertThat(structYYY.getValue("s"))
        .isEqualTo("ctx.outputs(for //test:xxx)");
  }

  @Test
  public void testExecutableRuleWithNoExecutableReportsError() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "   pass",
        "my_rule = rule(_impl, executable = True)"
    );

    scratch.file(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "my_rule(name = 'xxx')"
    );

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:xxx");
    assertContainsEvent("ERROR /workspace/test/BUILD:2:1: in my_rule rule //test:xxx: ");
    assertContainsEvent("/rule.bzl:1:5: The rule 'my_rule' is executable. "
        + "It needs to create an executable File and pass it as the 'executable' "
        + "parameter to the DefaultInfo it returns.");
  }

  @Test
  public void testExecutableFromDifferentRuleIsForbidden() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "sh_binary(name = 'tryme',",
        "          srcs = [':tryme.sh'],",
        "          visibility = ['//visibility:public'],",
        ")");

    scratch.file(
        "src/rulez.bzl",
        "def  _impl(ctx):",
        "   return [DefaultInfo(executable = ctx.executable.runme,",
        "                       files = depset([ctx.executable.runme]),",
        "          )]",
        "r = rule(_impl,",
        "         executable = True,",
        "         attrs = {",
        "            'runme' : attr.label(executable = True, mandatory = True, cfg = 'host'),",
        "         }",
        ")");

    scratch.file(
        "src/BUILD", "load(':rulez.bzl', 'r')", "r(name = 'r_tools', runme = '//pkg:tryme')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//src:r_tools");
    assertContainsEvent(
        "/workspace/src/rulez.bzl:2:12: 'executable' provided by an executable"
            + " rule 'r' should be created by the same rule.");
  }

  @Test
  public void testFileAndDirectory() throws Exception {
    scratch.file(
        "ext.bzl",
        "def _extrule(ctx):",
        "  dir = ctx.actions.declare_directory('foo/bar/baz')",
        "  ctx.actions.run_shell(",
        "      outputs = [dir],",
        "      command = 'mkdir -p ' + dir.path + ' && echo wtf > ' + dir.path + '/wtf.txt')",
        "",
        "extrule = rule(",
        "    _extrule,",
        "    outputs = {",
        "      'out': 'foo/bar/baz',",
        "    },",
        ")");
    scratch.file("BUILD", "load(':ext.bzl', 'extrule')", "", "extrule(", "    name = 'test'", ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//:test");
    assertContainsEvent("ERROR /workspace/BUILD:3:1: in extrule rule //:test:");
    assertContainsEvent("he following directories were also declared as files:");
    assertContainsEvent("foo/bar/baz");
  }

  @Test
  public void testEnvironmentConstraintsFromSkylarkRule() throws Exception {
    scratch.file(
        "buildenv/foo/BUILD",
        "environment_group(name = 'env_group',",
        "    defaults = [':default'],",
        "    environments = ['default', 'other'])",
        "environment(name = 'default')",
        "environment(name = 'other')");
    // The example skylark rule explicitly provides the MyProvider provider as a regression test
    // for a bug where a skylark rule with unsatisfied constraints but explicit providers would
    // result in Bazel throwing a null pointer exception.
    scratch.file(
        "test/skylark/extension.bzl",
        "MyProvider = provider()",
        "",
        "def _impl(ctx):",
        "  return struct(providers = [MyProvider(foo = 'bar')])",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'deps' : attr.label_list() },",
        "    provides = [MyProvider])");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "java_library(name = 'dep', srcs = ['a.java'], restricted_to = ['//buildenv/foo:other'])",
        "my_rule(name='my', deps = [':dep'])");

    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//test/skylark:my")).isNull();
    assertContainsEvent(
        "//test/skylark:dep doesn't support expected environment: //buildenv/foo:default");
  }

  @Test
  public void testNoTargetOutputGroup() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_target_output_group=true");
    scratch.file(
        "test/skylark/extension.bzl",
        "def _impl(ctx):",
        "  f = ctx.attr.dep.output_group()",
        "  return struct()",
        "my_rule = rule(implementation = _impl,",
        "    attrs = { 'dep' : attr.label() })");

    checkError(
        "test/skylark",
        "r",
        "type 'Target' has no method output_group()",
        "load('//test/skylark:extension.bzl',  'my_rule')",
        "cc_binary(name = 'lib', data = ['a.txt'])",
        "my_rule(name='r', dep = ':lib')");
  }

  @Test
  public void testAnalysisFailureInfo() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "   fail('This Is My Failure Message')",
        "   return []",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'r')");

    useConfiguration("--allow_analysis_failures=true");

    ConfiguredTarget target = getConfiguredTarget("//test:r");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.SKYLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().toList().get(0);
    assertThat(failure.getMessage()).contains("This Is My Failure Message");
    assertThat(failure.getLabel()).isEqualTo(Label.parseAbsoluteUnchecked("//test:r"));
  }

  @Test
  public void testTestResultInfo() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return [AnalysisTestResultInfo(success = True, message = 'message contents')]",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'r')");

    ConfiguredTarget target = getConfiguredTarget("//test:r");
    AnalysisTestResultInfo info =
        (AnalysisTestResultInfo) target.get(AnalysisTestResultInfo.SKYLARK_CONSTRUCTOR.getKey());
    assertThat(info.getSuccess()).isTrue();
    assertThat(info.getMessage()).isEqualTo("message contents");
  }

  @Test
  public void testAnalysisTestRuleWithActionRegistration() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  out_file = ctx.actions.declare_file('file.txt')",
        "  ctx.actions.write(output=out_file, content='hi')",
        "  return []",
        "",
        "custom_test = rule(implementation = custom_rule_impl, analysis_test = True)");

    scratch.file(
        "test/BUILD", "load('//test:extension.bzl', 'custom_test')", "", "custom_test(name = 'r')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:r");
    assertContainsEvent(
        "implementation function of a rule with analysis_test=true may not register actions");
  }

  @Test
  public void testAnalysisTestRuleWithFlag() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return [AnalysisTestResultInfo(success = True, message = 'message contents')]",
        "",
        "custom_test = rule(implementation = custom_rule_impl, analysis_test = True)");

    scratch.file(
        "test/BUILD", "load('//test:extension.bzl', 'custom_test')", "", "custom_test(name = 'r')");

    ConfiguredTarget target = getConfiguredTarget("//test:r");
    AnalysisTestResultInfo info =
        (AnalysisTestResultInfo) target.get(AnalysisTestResultInfo.SKYLARK_CONSTRUCTOR.getKey());
    assertThat(info.getSuccess()).isTrue();
    assertThat(info.getMessage()).isEqualTo("message contents");

    // TODO(cparsons): Verify implicit action registration via AnalysisTestResultInfo.
  }

  @Test
  public void testAnalysisTestTransitionOnAnalysisTest() throws Exception {
    useConfiguration("--experimental_strict_java_deps=OFF");

    scratch.file(
        "test/extension.bzl",
        "MyInfo = provider()",
        "MyDep = provider()",
        "",
        "def outer_rule_impl(ctx):",
        "  return [MyInfo(strict_java_deps = ctx.fragments.java.strict_java_deps),",
        "          MyDep(info = ctx.attr.dep[0][MyInfo]),",
        "          AnalysisTestResultInfo(success = True, message = 'message contents')]",
        "def inner_rule_impl(ctx):",
        "  return [MyInfo(strict_java_deps = ctx.fragments.java.strict_java_deps)]",
        "",
        "my_transition = analysis_test_transition(",
        "    settings = {",
        "        '//command_line_option:experimental_strict_java_deps' : 'WARN' }",
        ")",
        "inner_rule = rule(implementation = inner_rule_impl,",
        "                  fragments = ['java'])",
        "outer_rule_test = rule(",
        "  implementation = outer_rule_impl,",
        "  fragments = ['java'],",
        "  analysis_test = True,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "  })");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'inner_rule', 'outer_rule_test')",
        "",
        "inner_rule(name = 'inner')",
        "outer_rule_test(name = 'r', dep = ':inner')");

    SkylarkKey myInfoKey =
        new SkylarkKey(Label.parseAbsolute("//test:extension.bzl", ImmutableMap.of()), "MyInfo");
    SkylarkKey myDepKey =
        new SkylarkKey(Label.parseAbsolute("//test:extension.bzl", ImmutableMap.of()), "MyDep");

    ConfiguredTarget outerTarget = getConfiguredTarget("//test:r");
    StructImpl outerInfo = (StructImpl) outerTarget.get(myInfoKey);
    StructImpl outerDepInfo = (StructImpl) outerTarget.get(myDepKey);
    StructImpl innerInfo = (StructImpl) outerDepInfo.getValue("info");

    assertThat(outerInfo.getValue("strict_java_deps")).isEqualTo("off");
    assertThat(innerInfo.getValue("strict_java_deps")).isEqualTo("warn");
  }

  @Test
  public void testAnalysisTestTransitionOnNonAnalysisTest() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return []",
        "my_transition = analysis_test_transition(",
        "    settings = {",
        "        '//command_line_option:experimental_strict_java_deps' : 'WARN' }",
        ")",
        "",
        "custom_rule = rule(",
        "  implementation = custom_rule_impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "  })");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'r')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:r");
    assertContainsEvent(
        "Only rule definitions with analysis_test=True may have attributes "
            + "with analysis_test_transition transitions");
  }

  @Test
  public void testBuildSettingRule_flag() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=true");

    scratch.file("test/rules.bzl",
        "def _impl(ctx): return None",
        "build_setting_rule = rule(_impl, build_setting = config.string(flag=True))");
    scratch.file("test/BUILD",
    "load('//test:rules.bzl', 'build_setting_rule')",
    "build_setting_rule(name = 'my_build_setting', build_setting_default = 'default')");

    BuildSetting buildSetting =
        getTarget("//test:my_build_setting")
            .getAssociatedRule()
            .getRuleClassObject()
            .getBuildSetting();

    assertThat(buildSetting.getType()).isEqualTo(Type.STRING);
    assertThat(buildSetting.isFlag()).isTrue();
  }

  @Test
  public void testBuildSettingRule_settingByDefault() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=true");

    scratch.file("test/rules.bzl",
        "def _impl(ctx): return None",
        "build_setting_rule = rule(_impl, build_setting = config.string())");
    scratch.file("test/BUILD",
        "load('//test:rules.bzl', 'build_setting_rule')",
        "build_setting_rule(name = 'my_build_setting', build_setting_default = 'default')");

    BuildSetting buildSetting =
        getTarget("//test:my_build_setting")
            .getAssociatedRule()
            .getRuleClassObject()
            .getBuildSetting();

    assertThat(buildSetting.getType()).isEqualTo(Type.STRING);
    assertThat(buildSetting.isFlag()).isFalse();
  }

  @Test
  public void testBuildSettingRule_settingByFlagParameter() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=true");

    scratch.file("test/rules.bzl",
        "def _impl(ctx): return None",
        "build_setting_rule = rule(_impl, build_setting = config.string(flag=False))");
    scratch.file("test/BUILD",
        "load('//test:rules.bzl', 'build_setting_rule')",
        "build_setting_rule(name = 'my_build_setting', build_setting_default = 'default')");

    BuildSetting buildSetting =
        getTarget("//test:my_build_setting")
            .getAssociatedRule()
            .getRuleClassObject()
            .getBuildSetting();

    assertThat(buildSetting.getType()).isEqualTo(Type.STRING);
    assertThat(buildSetting.isFlag()).isFalse();
  }


  @Test
  public void testBuildSettingRule_noDefault() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=true");

    scratch.file("test/rules.bzl",
        "def _impl(ctx): return None",
        "build_setting_rule = rule(_impl, build_setting = config.string())");
    scratch.file("test/BUILD",
        "load('//test:rules.bzl', 'build_setting_rule')",
        "build_setting_rule(name = 'my_build_setting')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_build_setting");
    assertContainsEvent("missing value for mandatory attribute "
        + "'build_setting_default' in 'build_setting_rule' rule");

  }

  @Test
  public void testBuildSettingRule_errorsWithoutExperimentalFlag() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=false");

    scratch.file(
        "test/rules.bzl",
        "def _impl(ctx): return None",
        "build_setting_rule = rule(_impl, build_setting = config.string())");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'build_setting_rule')",
        "build_setting_rule(name = 'my_build_setting', build_setting_default = 'default')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_build_setting");
    assertContainsEvent(
        "parameter 'build_setting' is experimental and thus unavailable with the "
            + "current flags. It may be enabled by setting --experimental_build_setting_api");
  }

  @Test
  public void testAnalysisTestCannotDependOnAnalysisTest() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "",
        "def analysis_test_rule_impl(ctx):",
        "  return [AnalysisTestResultInfo(success = True, message = 'message contents')]",
        "def middle_rule_impl(ctx):",
        "  return []",
        "def inner_rule_impl(ctx):",
        "  return [AnalysisTestResultInfo(success = True, message = 'message contents')]",
        "",
        "my_transition = analysis_test_transition(",
        "    settings = {",
        "        '//command_line_option:experimental_strict_java_deps' : 'WARN' }",
        ")",
        "",
        "inner_rule_test = rule(",
        "  implementation = analysis_test_rule_impl,",
        "  analysis_test = True,",
        ")",
        "middle_rule = rule(",
        "  implementation = middle_rule_impl,",
        "  attrs = {'dep':  attr.label()}",
        ")",
        "outer_rule_test = rule(",
        "  implementation = analysis_test_rule_impl,",
        "  analysis_test = True,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "  })");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'outer_rule_test', 'middle_rule', 'inner_rule_test')",
        "",
        "outer_rule_test(name = 'outer', dep = ':middle')",
        "middle_rule(name = 'middle', dep = ':inner')",
        "inner_rule_test(name = 'inner')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:outer");
    assertContainsEvent(
        "analysis_test rule '//test:inner' cannot be transitively "
            + "depended on by another analysis test rule");
  }

  @Test
  public void testAnalysisTestOverDepsLimit() throws Exception {
    setupAnalysisTestDepsLimitTest(10, 12);

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:r");
    assertContainsEvent(
        "analysis test rule excedeed maximum dependency edge count. " + "Count: 13. Limit is 10.");
  }

  @Test
  public void testAnalysisTestUnderDepsLimit() throws Exception {
    setupAnalysisTestDepsLimitTest(10, 8);

    assertThat(getConfiguredTarget("//test:r")).isNotNull();
  }

  private void setupAnalysisTestDepsLimitTest(int limit, int dependencyChainSize) throws Exception {
    useConfiguration("--analysis_testing_deps_limit=" + limit);

    scratch.file(
        "test/extension.bzl",
        "",
        "def outer_rule_impl(ctx):",
        "  return [AnalysisTestResultInfo(success = True, message = 'message contents')]",
        "def dep_rule_impl(ctx):",
        "  return []",
        "",
        "my_transition = analysis_test_transition(",
        "    settings = {",
        "        '//command_line_option:experimental_strict_java_deps' : 'WARN' }",
        ")",
        "dep_rule = rule(",
        "  implementation = dep_rule_impl,",
        "  attrs = {'dep':  attr.label()}",
        ")",
        "outer_rule_test = rule(",
        "  implementation = outer_rule_impl,",
        "  fragments = ['java'],",
        "  analysis_test = True,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "  })");

    // Create a chain of targets where 'innerN' depends on 'inner{N+1}' until the max length.
    StringBuilder dependingRulesChain = new StringBuilder();
    for (int i = 0; i < dependencyChainSize; i++) {
      dependingRulesChain.append(
          String.format("dep_rule(name = 'inner%s', dep = ':inner%s')\n", i, (i + 1)));
    }
    dependingRulesChain.append(String.format("dep_rule(name = 'inner%s')", dependencyChainSize));

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'dep_rule', 'outer_rule_test')",
        "",
        "outer_rule_test(name = 'r', dep = ':inner0')",
        dependingRulesChain.toString());
  }

  @Test
  public void testBadWhitelistTransition_onNonLabelAttr() throws Exception {
    scratch.file(
        "test/rules.bzl",
        "def _impl(ctx):",
        "    return []",
        "",
        "my_rule = rule(_impl, attrs = {'"
            + WHITELIST_ATTRIBUTE_NAME
            + "':attr.string(default = 'blah')})");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'my_rule')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_rule");
    assertContainsEvent("_whitelist_function_transition attribute must be a label type");
  }

  @Test
  public void testBadWhitelistTransition_noDefaultValue() throws Exception {
    scratch.file(
        "test/rules.bzl",
        "def _impl(ctx):",
        "    return []",
        "",
        "my_rule = rule(_impl, attrs = {'" + WHITELIST_ATTRIBUTE_NAME + "':attr.label()})");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'my_rule')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_rule");
    assertContainsEvent("_whitelist_function_transition attribute must have a default value");
  }

  @Test
  public void testBadWhitelistTransition_wrongDefaultValue() throws Exception {
    scratch.file(
        "test/rules.bzl",
        "def _impl(ctx):",
        "    return []",
        "",
        "my_rule = rule(_impl, attrs = {'"
            + WHITELIST_ATTRIBUTE_NAME
            + "':attr.label(default = Label('//test:my_other_rule'))})");
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "my_rule(name = 'my_rule')",
        "my_rule(name = 'my_other_rule')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_rule");
    assertContainsEvent(
        "_whitelist_function_transition attribute does not have the expected value");
  }

  @Test
  public void testBadAnalysisTestRule_notAnalysisTest() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "",
        "def outer_rule_impl(ctx):",
        "  return [AnalysisTestResultInfo(success = True, message = 'message contents')]",
        "def dep_rule_impl(ctx):",
        "  return []",
        "",
        "my_transition = analysis_test_transition(",
        "    settings = {",
        "        '//command_line_option:experimental_strict_java_deps' : 'WARN' }",
        ")",
        "dep_rule = rule(",
        "  implementation = dep_rule_impl,",
        "  attrs = {'dep':  attr.label()}",
        ")",
        "outer_rule = rule(",
        "  implementation = outer_rule_impl,",
        "# analysis_test = True,",
        "  fragments = ['java'],",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "  })");

    scratch.file(
        "test/BUILD",
        "load('//test:extension.bzl', 'dep_rule', 'outer_rule_test')",
        "",
        "outer_rule(name = 'r', dep = ':inner')",
        "dep_rule(name = 'inner')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:outer_rule");
    assertContainsEvent(
        "Only rule definitions with analysis_test=True may have attributes with "
            + "analysis_test_transition transitions");
  }

  @Test
  public void testBadWhitelistTransition_noWhitelist() throws Exception {
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
    scratch.file(
        "test/rules.bzl",
        "def transition_func(settings):",
        "  return {'t0': {'//command_line_option:cpu': 'k8'}}",
        "my_transition = transition(implementation = transition_func, inputs = [],",
        "  outputs = ['//command_line_option:cpu'])",
        "def _my_rule_impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = _my_rule_impl,",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "#   '_whitelist_function_transition': attr.label(",
        "#       default = '//tools/whitelists/function_transition_whitelist',",
        "#   ),",
        "  })",
        "def _simple_rule_impl(ctx):",
        "  return []",
        "simple_rule = rule(_simple_rule_impl)");

    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple_rule')",
        "my_rule(name = 'my_rule', dep = ':dep')",
        "simple_rule(name = 'dep')");
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_rule");
    assertContainsEvent("Use of function-based split transition without whitelist");
  }

  @Test
  public void testPrintFromTransitionImpl() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions");
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
    scratch.file(
        "test/rules.bzl",
        "def _transition_impl(settings, attr):",
        "  print('printing from transition impl', settings['//command_line_option:test_arg'])",
        "  return {'//command_line_option:test_arg': "
            + "settings['//command_line_option:test_arg']+['meow']}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = ['//command_line_option:test_arg'],",
        "  outputs = ['//command_line_option:test_arg'],",
        ")",
        "def _rule_impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  }",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')");

    useConfiguration("--test_arg=meow");

    getConfiguredTarget("//test");
    // Test print from top level transition
    assertContainsEventWithFrequency("printing from transition impl [\"meow\"]", 1);
    // Test print from dep transition
    assertContainsEventWithFrequency("printing from transition impl [\"meow\", \"meow\"]", 1);
    // Test print from (non-top level) rule class transition
    assertContainsEventWithFrequency(
        "printing from transition impl [\"meow\", \"meow\", \"meow\"]", 1);
  }

  @Test
  public void testTransitionEquality() throws Exception {
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions");
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
    scratch.file(
        "test/rules.bzl",
        "def _transition_impl(settings, attr):",
        "  return {'//command_line_option:test_arg': ['meow']}",
        "my_transition = transition(",
        "  implementation = _transition_impl,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:test_arg'],",
        ")",
        "def _rule_impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  cfg = my_transition,",
        "  attrs = {",
        "    'dep': attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  }",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "my_rule(name = 'test', dep = ':dep')",
        "my_rule(name = 'dep')");

    useConfiguration("--test_arg=meow");

    StarlarkDefinedConfigTransition ruleTransition =
        ((StarlarkAttributeTransitionProvider)
                getTarget("//test")
                    .getAssociatedRule()
                    .getRuleClassObject()
                    .getAttributeByName("dep")
                    .getSplitTransitionProviderForTesting())
            .getStarlarkDefinedConfigTransitionForTesting();

    StarlarkDefinedConfigTransition attrTransition =
        ((StarlarkRuleTransitionProvider)
                getTarget("//test").getAssociatedRule().getRuleClassObject().getTransitionFactory())
            .getStarlarkDefinedConfigTransitionForTesting();

    assertThat(ruleTransition).isEqualTo(attrTransition);
    assertThat(attrTransition).isEqualTo(ruleTransition);
    assertThat(ruleTransition.hashCode()).isEqualTo(attrTransition.hashCode());
  }

  @Test
  public void testBadWhitelistTransition_whitelistNoCfg() throws Exception {
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
    scratch.file(
        "test/rules.bzl",
        "def _my_rule_impl(ctx): ",
        "  return []",
        "my_rule = rule(",
        "  implementation = _my_rule_impl,",
        "  attrs = {",
        "#   'dep':  attr.label(cfg = my_transition),",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist',",
        "    ),",
        "  })",
        "def _simple_rule_impl(ctx):",
        "  return []",
        "simple_rule = rule(_simple_rule_impl)");

    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'my_rule', 'simple_rule')",
        "my_rule(name = 'my_rule', dep = ':dep')",
        "simple_rule(name = 'dep')");
    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_rule");
    assertContainsEvent("Unused function-based split transition whitelist");
  }

  @Test
  public void testLicenseType() throws Exception {
    // Note that attr.license is deprecated, and thus this test is subject to imminent removal.
    // (See --incompatible_no_attr_license). However, this verifies that until the attribute
    // is removed, values of the attribute are a valid Starlark type.
    scratch.file(
        "test/rule.bzl",
        "def _my_rule_impl(ctx): ",
        "  print(ctx.attr.my_license)",
        "  return []",
        "my_rule = rule(",
        "  implementation = _my_rule_impl,",
        "  attrs = {",
        "    'my_license':  attr.license(),",
        "  })");
    scratch.file("test/BUILD", "load(':rule.bzl', 'my_rule')", "my_rule(name = 'test')");

    getConfiguredTarget("//test:test");

    assertContainsEvent("<license object>");
  }

  @Test
  public void testDisallowStructProviderSyntax() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_struct_provider_syntax=true");
    scratch.file(
        "test/skylark/extension.bzl",
        "def custom_rule_impl(ctx):",
        "  return struct()",
        "",
        "custom_rule = rule(implementation = custom_rule_impl)");
    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'custom_rule')",
        "",
        "custom_rule(name = 'cr')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test/skylark:cr");
    assertContainsEvent(
        "Returning a struct from a rule implementation function is deprecated and will be "
            + "removed soon. It may be temporarily re-enabled by setting "
            + "--incompatible_disallow_struct_provider_syntax=false");
  }

  /**
   * Skylark integration test that forces inlining.
   */
  @RunWith(JUnit4.class)
  public static class SkylarkIntegrationTestsWithInlineCalls extends SkylarkIntegrationTest {

    @Before
    public final void initializeLookupFunctions() throws Exception {
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions =
          ((InMemoryMemoizingEvaluator) getSkyframeExecutor().getEvaluatorForTesting())
              .getSkyFunctionsForTesting();
      SkylarkImportLookupFunction skylarkImportLookupFunction =
          new SkylarkImportLookupFunction(this.getRuleClassProvider(), this.getPackageFactory());
      skylarkImportLookupFunction.resetCache();
      ((PackageFunction) skyFunctions.get(SkyFunctions.PACKAGE))
          .setSkylarkImportLookupFunctionForInliningForTesting(skylarkImportLookupFunction);
    }

    @Override
    @Test
    public void testRecursiveImport() throws Exception {
      scratch.file("test/skylark/ext2.bzl", "load('//test/skylark:ext1.bzl', 'symbol2')");

      scratch.file("test/skylark/ext1.bzl", "load('//test/skylark:ext2.bzl', 'symbol1')");

      scratch.file(
          "test/skylark/BUILD",
          "load('//test/skylark:ext1.bzl', 'custom_rule')",
          "genrule(name = 'rule')");

      reporter.removeHandler(failFastHandler);
      try {
        getTarget("//test/skylark:rule");
        fail();
      } catch (BuildFileContainsErrorsException e) {
        // The reason that this is an exception and not reported to the event handler is that the
        // error is reported by the parent sky function, which we don't have here.
        assertThat(e)
            .hasMessageThat()
            .contains("Starlark import cycle: [//test/skylark:ext1.bzl, //test/skylark:ext2.bzl]");
      }
    }

    @Override
    @Test
    public void testRecursiveImport2() throws Exception {
      scratch.file("test/skylark/ext1.bzl", "load('//test/skylark:ext2.bzl', 'symbol2')");
      scratch.file("test/skylark/ext2.bzl", "load('//test/skylark:ext3.bzl', 'symbol3')");
      scratch.file("test/skylark/ext3.bzl", "load('//test/skylark:ext4.bzl', 'symbol4')");
      scratch.file("test/skylark/ext4.bzl", "load('//test/skylark:ext2.bzl', 'symbol2')");

      scratch.file(
          "test/skylark/BUILD",
          "load('//test/skylark:ext1.bzl', 'custom_rule')",
          "genrule(name = 'rule')");

      reporter.removeHandler(failFastHandler);
      try {
        getTarget("//test/skylark:rule");
        fail();
      } catch (BuildFileContainsErrorsException e) {
        // The reason that this is an exception and not reported to the event handler is that the
        // error is reported by the parent sky function, which we don't have here.
        assertThat(e)
            .hasMessageThat()
            .contains(
                "Starlark import cycle: [//test/skylark:ext2.bzl, "
                    + "//test/skylark:ext3.bzl, //test/skylark:ext4.bzl]");
      }
    }
  }
}
