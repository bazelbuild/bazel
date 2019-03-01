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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static org.junit.Assert.fail;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkInfo;
import com.google.devtools.build.lib.packages.SkylarkProvider;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.python.PyProviderUtils;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkRuleContext.
 */
@RunWith(JUnit4.class)
public class SkylarkRuleContextTest extends SkylarkTestCase {

  /** A test rule that exercises the semantics of mandatory providers. */
  private static final MockRule TESTING_RULE_FOR_MANDATORY_PROVIDERS =
      () ->
          MockRule.define(
              "testing_rule_for_mandatory_providers",
              (builder, env) ->
                  builder
                      .setUndocumented()
                      .add(attr("srcs", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
                      .add(
                          attr("deps", LABEL_LIST)
                              .legacyAllowAnyFileType()
                              .mandatoryProvidersList(
                                  ImmutableList.of(
                                      ImmutableList.of(SkylarkProviderIdentifier.forLegacy("a")),
                                      ImmutableList.of(
                                          SkylarkProviderIdentifier.forLegacy("b"),
                                          SkylarkProviderIdentifier.forLegacy("c"))))));

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(TESTING_RULE_FOR_MANDATORY_PROVIDERS);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Before
  public final void generateBuildFile() throws Exception {
    scratch.file(
        "foo/BUILD",
        "package(features = ['-f1', 'f2', 'f3'])",
        "genrule(name = 'foo',",
        "  cmd = 'dummy_cmd',",
        "  srcs = ['a.txt', 'b.img'],",
        "  tools = ['t.exe'],",
        "  outs = ['c.txt'])",
        "genrule(name = 'foo2',",
        "  cmd = 'dummy_cmd',",
        "  outs = ['e.txt'])",
        "genrule(name = 'bar',",
        "  cmd = 'dummy_cmd',",
        "  srcs = [':jl', ':gl'],",
        "  outs = ['d.txt'])",
        "java_library(name = 'jl',",
        "  srcs = ['a.java'])",
        "android_library(name = 'androidlib',",
        "  srcs = ['a.java'])",
        "java_import(name = 'asr',",
        "  jars = [ 'asr.jar' ],",
        "  srcjar = 'asr-src.jar',",
        ")",
        "genrule(name = 'gl',",
        "  cmd = 'touch $(OUTS)',",
        "  srcs = ['a.go'],",
        "  outs = [ 'gl.a', 'gl.gcgox', ],",
        "  output_to_bindir = 1,",
        ")",
        "cc_library(name = 'cc_with_features',",
        "           srcs = ['dummy.cc'],",
        "           features = ['f1', '-f3'],",
        ")"
    );
  }

  private void setUpAttributeErrorTest() throws Exception {
    scratch.file(
        "test/BUILD",
        "load('//test:macros.bzl', 'macro_native_rule', 'macro_skylark_rule', 'skylark_rule')",
        "macro_native_rule(name = 'm_native',",
        "  deps = [':jlib'])",
        "macro_skylark_rule(name = 'm_skylark',",
        "  deps = [':jlib'])",
        "java_library(name = 'jlib',",
        "  srcs = ['bla.java'])",
        "cc_library(name = 'cclib',",
        "  deps = [':jlib'])",
        "skylark_rule(name = 'skyrule',",
        "  deps = [':jlib'])");
    scratch.file("test/macros.bzl",
        "def _impl(ctx):",
        "  return",
        "skylark_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'deps': attr.label_list(providers = ['some_provider'], allow_files=True)",
        "  }",
        ")",
        "def macro_native_rule(name, deps): ",
        "  native.cc_library(name = name, deps = deps)",
        "def macro_skylark_rule(name, deps):",
        "  skylark_rule(name = name, deps = deps)");
    reporter.removeHandler(failFastHandler);
  }

  @Test
  public void hasCorrectLocationForRuleAttributeError_NativeRuleWithMacro() throws Exception {
    setUpAttributeErrorTest();
    try {
      createRuleContext("//test:m_native");
      fail("Should have failed because of invalid dependency");
    } catch (Exception ex) {
      // Macro creates native rule -> location points to the rule and the message contains details
      // about the macro.
      assertContainsEvent("misplaced here");
      // Skip the part of the error message that has details about the allowed deps since the mocks
      // for the mac tests might have different values for them.
      assertContainsEvent(". Since this "
          + "rule was created by the macro 'macro_native_rule', the error might have been caused "
          + "by the macro implementation in /workspace/test/macros.bzl:10:41");
    }
  }

  @Test
  public void hasCorrectLocationForRuleAttributeError_SkylarkRuleWithMacro() throws Exception {
    setUpAttributeErrorTest();
    try {
      createRuleContext("//test:m_skylark");
      fail("Should have failed because of invalid attribute value");
    } catch (Exception ex) {
      // Macro creates Skylark rule -> location points to the rule and the message contains details
      // about the macro.
      assertContainsEvent(
          "ERROR /workspace/test/BUILD:4:1: in deps attribute of skylark_rule rule "
              + "//test:m_skylark: '//test:jlib' does not have mandatory providers:"
              + " 'some_provider'. "
              + "Since this rule was created by the macro 'macro_skylark_rule', the error might "
              + "have been caused by the macro implementation in /workspace/test/macros.bzl:12:36");
    }
  }

  @Test
  public void hasCorrectLocationForRuleAttributeError_NativeRule() throws Exception {
    setUpAttributeErrorTest();
    try {
      createRuleContext("//test:cclib");
      fail("Should have failed because of invalid dependency");
    } catch (Exception ex) {
      // Native rule WITHOUT macro -> location points to the attribute and there is no mention of
      // 'macro' at all.
      assertContainsEvent("misplaced here");
      // Skip the part of the error message that has details about the allowed deps since the mocks
      // for the mac tests might have different values for them.
      assertDoesNotContainEvent("Since this rule was created by the macro");
    }
  }

  @Test
  public void hasCorrectLocationForRuleAttributeError_SkylarkRule() throws Exception {
    setUpAttributeErrorTest();
    try {
      createRuleContext("//test:skyrule");
      fail("Should have failed because of invalid dependency");
    } catch (Exception ex) {
      // Skylark rule WITHOUT macro -> location points to the attribute and there is no mention of
      // 'macro' at all.
      assertContainsEvent("ERROR /workspace/test/BUILD:11:10: in deps attribute of "
          + "skylark_rule rule //test:skyrule: '//test:jlib' does not have mandatory providers: "
          + "'some_provider'");
    }
  }

  @Test
  public void testMandatoryProvidersListWithSkylark() throws Exception {
    scratch.file("test/BUILD",
            "load('//test:rules.bzl', 'skylark_rule', 'my_rule', 'my_other_rule')",
            "my_rule(name = 'mylib',",
            "  srcs = ['a.py'])",
            "skylark_rule(name = 'skyrule1',",
            "  deps = [':mylib'])",
            "my_other_rule(name = 'my_other_lib',",
            "  srcs = ['a.py'])",
            "skylark_rule(name = 'skyrule2',",
            "  deps = [':my_other_lib'])");
    scratch.file("test/rules.bzl",
            "def _impl(ctx):",
            "  return",
            "skylark_rule = rule(",
            "  implementation = _impl,",
            "  attrs = {",
            "    'deps': attr.label_list(providers = [['a'], ['b', 'c']],",
            "    allow_files=True)",
            "  }",
            ")",
            "def my_rule_impl(ctx):",
            "  return struct(a = [])",
            "my_rule = rule(implementation = my_rule_impl, ",
            "  attrs = { 'srcs' : attr.label_list(allow_files=True)})",
            "def my_other_rule_impl(ctx):",
            "  return struct(b = [])",
            "my_other_rule = rule(implementation = my_other_rule_impl, ",
            "  attrs = { 'srcs' : attr.label_list(allow_files=True)})");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//test:skyrule1")).isNotNull();

    try {
      createRuleContext("//test:skyrule2");
      fail("Should have failed because of wrong mandatory providers");
    } catch (Exception ex) {
      assertContainsEvent("ERROR /workspace/test/BUILD:9:10: in deps attribute of "
              + "skylark_rule rule //test:skyrule2: '//test:my_other_lib' does not have "
              + "mandatory providers: 'a' or 'c'");
    }
  }

  @Test
  public void testMandatoryProvidersListWithNative() throws Exception {
    scratch.file("test/BUILD",
            "load('//test:rules.bzl', 'my_rule', 'my_other_rule')",
            "my_rule(name = 'mylib',",
            "  srcs = ['a.py'])",
            "testing_rule_for_mandatory_providers(name = 'skyrule1',",
            "  deps = [':mylib'])",
            "my_other_rule(name = 'my_other_lib',",
            "  srcs = ['a.py'])",
            "testing_rule_for_mandatory_providers(name = 'skyrule2',",
            "  deps = [':my_other_lib'])");
    scratch.file("test/rules.bzl",
            "def my_rule_impl(ctx):",
            "  return struct(a = [])",
            "my_rule = rule(implementation = my_rule_impl, ",
            "  attrs = { 'srcs' : attr.label_list(allow_files=True)})",
            "def my_other_rule_impl(ctx):",
            "  return struct(b = [])",
            "my_other_rule = rule(implementation = my_other_rule_impl, ",
            "  attrs = { 'srcs' : attr.label_list(allow_files=True)})");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//test:skyrule1")).isNotNull();

    try {
      createRuleContext("//test:skyrule2");
      fail("Should have failed because of wrong mandatory providers");
    } catch (Exception ex) {
      assertContainsEvent("ERROR /workspace/test/BUILD:9:10: in deps attribute of "
              + "testing_rule_for_mandatory_providers rule //test:skyrule2: '//test:my_other_lib' "
              + "does not have mandatory providers: 'a' or 'c'");
    }
  }

  /* Sharing setup code between the testPackageBoundaryError*() methods is not possible since the
   * errors already happen when loading the file. Consequently, all tests would fail at the same
   * statement. */
  @Test
  public void testPackageBoundaryError_NativeRule() throws Exception {
    scratch.file("test/BUILD", "cc_library(name = 'cclib',", "  srcs = ['sub/my_sub_lib.h'])");
    scratch.file("test/sub/BUILD", "cc_library(name = 'my_sub_lib', srcs = ['my_sub_lib.h'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:cclib");
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:2:10: Label '//test:sub/my_sub_lib.h' crosses boundary of "
            + "subpackage 'test/sub' (perhaps you meant to put the colon here: "
            + "'//test/sub:my_sub_lib.h'?)");
  }

  @Test
  public void testPackageBoundaryError_SkylarkRule() throws Exception {
    scratch.file("test/BUILD",
        "load('//test:macros.bzl', 'skylark_rule')",
        "skylark_rule(name = 'skyrule',",
        "  srcs = ['sub/my_sub_lib.h'])");
    scratch.file("test/sub/BUILD",
        "cc_library(name = 'my_sub_lib', srcs = ['my_sub_lib.h'])");
    scratch.file("test/macros.bzl",
        "def _impl(ctx):",
        "  return",
        "skylark_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True)",
        "  }",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:skyrule");
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:3:10: Label '//test:sub/my_sub_lib.h' crosses boundary of "
            + "subpackage 'test/sub' (perhaps you meant to put the colon here: "
            + "'//test/sub:my_sub_lib.h'?)");
  }

  @Test
  public void testPackageBoundaryError_SkylarkMacro() throws Exception {
    scratch.file("test/BUILD",
        "load('//test:macros.bzl', 'macro_skylark_rule')",
        "macro_skylark_rule(name = 'm_skylark',",
        "  srcs = ['sub/my_sub_lib.h'])");
    scratch.file("test/sub/BUILD",
        "cc_library(name = 'my_sub_lib', srcs = ['my_sub_lib.h'])");
    scratch.file("test/macros.bzl",
        "def _impl(ctx):",
        "  return",
        "skylark_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True)",
        "  }",
        ")",
        "def macro_skylark_rule(name, srcs=[]):",
        "  skylark_rule(name = name, srcs = srcs)");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:m_skylark");
    assertContainsEvent("ERROR /workspace/test/BUILD:2:1: Label '//test:sub/my_sub_lib.h' "
        + "crosses boundary of subpackage 'test/sub' (perhaps you meant to put the colon here: "
        + "'//test/sub:my_sub_lib.h'?)");
  }

  /* The error message for this case used to be wrong. */
  @Test
  public void testPackageBoundaryError_ExternalRepository_Boundary() throws Exception {
    scratch.file("r/WORKSPACE");
    scratch.file("r/BUILD");
    scratch.overwriteFile(
        "WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .add("local_repository(name='r', path='r')")
            .build());
    scratch.file("BUILD", "cc_library(name = 'cclib',", "  srcs = ['r/my_sub_lib.h'])");
    invalidatePackages(
        /*alsoConfigs=*/ false); // Repository shuffling messes with toolchain labels.
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//:cclib");
    assertContainsEvent(
        "/workspace/BUILD:2:10: Label '//:r/my_sub_lib.h' crosses boundary of "
            + "subpackage '@r//'");
  }

  /* The error message for this case used to be wrong. */
  @Test
  public void testPackageBoundaryError_ExternalRepository_EntirelyInside() throws Exception {
    scratch.file("/r/WORKSPACE");
    scratch.file("/r/BUILD", "cc_library(name = 'cclib',", "  srcs = ['sub/my_sub_lib.h'])");
    scratch.file("/r/sub/BUILD", "cc_library(name = 'my_sub_lib', srcs = ['my_sub_lib.h'])");
    scratch.overwriteFile("WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .add("local_repository(name='r', path='/r')")
            .build());
    invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchain labels.
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("@r//:cclib");
    assertContainsEvent(
        "/external/r/BUILD:2:10: Label '@r//:sub/my_sub_lib.h' crosses boundary of "
            + "subpackage '@r//sub' (perhaps you meant to put the colon here: "
            + "'@r//sub:my_sub_lib.h'?)");
  }

  /*
   * Making the location in BUILD file the default for "crosses boundary of subpackage" errors does
   * not work in this case since the error actually happens in the bzl file. However, because of
   * the current design, we can neither show the location in the bzl file nor display both
   * locations (BUILD + bzl).
   *
   * Since this case is less common than having such an error in a BUILD file, we can live
   * with it.
   */
  @Test
  public void testPackageBoundaryError_SkylarkMacroWithErrorInBzlFile() throws Exception {
    scratch.file("test/BUILD",
        "load('//test:macros.bzl', 'macro_skylark_rule')",
        "macro_skylark_rule(name = 'm_skylark')");
    scratch.file("test/sub/BUILD",
        "cc_library(name = 'my_sub_lib', srcs = ['my_sub_lib.h'])");
    scratch.file("test/macros.bzl",
        "def _impl(ctx):",
        "  return",
        "skylark_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True)",
        "  }",
        ")",
        "def macro_skylark_rule(name, srcs=[]):",
        "  skylark_rule(name = name, srcs = srcs + ['sub/my_sub_lib.h'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:m_skylark");
    assertContainsEvent("ERROR /workspace/test/BUILD:2:1: Label '//test:sub/my_sub_lib.h' "
        + "crosses boundary of subpackage 'test/sub' (perhaps you meant to put the colon here: "
        + "'//test/sub:my_sub_lib.h'?)");
  }

  @Test
  public void testPackageBoundaryError_NativeMacro() throws Exception {
    scratch.file("test/BUILD",
        "load('//test:macros.bzl', 'macro_native_rule')",
        "macro_native_rule(name = 'm_native',",
        "  srcs = ['sub/my_sub_lib.h'])");
    scratch.file("test/sub/BUILD",
        "cc_library(name = 'my_sub_lib', srcs = ['my_sub_lib.h'])");
    scratch.file("test/macros.bzl",
        "def macro_native_rule(name, deps=[], srcs=[]): ",
        "  native.cc_library(name = name, deps = deps, srcs = srcs)");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:m_native");
    assertContainsEvent("ERROR /workspace/test/BUILD:2:1: Label '//test:sub/my_sub_lib.h' "
        + "crosses boundary of subpackage 'test/sub' (perhaps you meant to put the colon here: "
        + "'//test/sub:my_sub_lib.h'?)");
  }

  @Test
  public void shouldGetPrerequisiteArtifacts() throws Exception {

    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.files.srcs");
    assertArtifactList(result, ImmutableList.of("a.txt", "b.img"));
  }

  private void assertArtifactList(Object result, List<String> artifacts) {
    assertThat(result).isInstanceOf(SkylarkList.class);
    SkylarkList resultList = (SkylarkList) result;
    assertThat(resultList).hasSize(artifacts.size());
    int i = 0;
    for (String artifact : artifacts) {
      assertThat(((Artifact) resultList.get(i++)).getFilename()).isEqualTo(artifact);
    }
  }

  @Test
  public void shouldGetPrerequisites() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.srcs");
    // Check for a known provider
    TransitiveInfoCollection tic1 = (TransitiveInfoCollection) ((SkylarkList) result).get(0);
    assertThat(JavaInfo.getProvider(JavaSourceJarsProvider.class, tic1)).isNotNull();
    // Check an unimplemented provider too
    assertThat(PyProviderUtils.hasLegacyProvider(tic1)).isFalse();
  }

  @Test
  public void shouldGetPrerequisite() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:asr");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.srcjar");
    TransitiveInfoCollection tic = (TransitiveInfoCollection) result;
    assertThat(tic).isInstanceOf(FileConfiguredTarget.class);
    assertThat(tic.getLabel().getName()).isEqualTo("asr-src.jar");
  }

  @Test
  public void testGetRuleAttributeListType() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertThat(result).isInstanceOf(SkylarkList.class);
  }

  @Test
  public void testGetRuleSelect() throws Exception {
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/rulestr.bzl", "def rule_dict(name):", "  return native.existing_rule(name)");

    scratch.file(
        "test/getrule/BUILD",
        "load('//test/skylark:rulestr.bzl', 'rule_dict')",
        "cc_library(name ='x', ",
        "  srcs = select({'//conditions:default': []})",
        ")",
        "rule_dict('x')");

    // Parse the BUILD file, to make sure select() makes it out of native.rule().
    createRuleContext("//test/getrule:x");
  }

  @Test
  public void testExistingRuleReturnNone() throws Exception {
    scratch.file(
        "test/rulestr.bzl",
        "def test_rule(name, x):",
        "  print(native.existing_rule(x))",
        "  if native.existing_rule(x) == None:",
        "    native.cc_library(name = name)");
    scratch.file(
        "test/BUILD",
        "load('//test:rulestr.bzl', 'test_rule')",
        "test_rule('a', 'does not exist')",
        "test_rule('b', 'BUILD')");

    assertThat(getConfiguredTarget("//test:a")).isNotNull();
    assertThat(getConfiguredTarget("//test:b")).isNotNull();
  }

  @Test
  public void existingRuleWithSelect() throws Exception {
    scratch.file(
        "test/existing_rule.bzl",
        "def macro():",
        "  s = select({'//foo': ['//bar']})",
        "  native.cc_library(name = 'x', srcs = s)",
        "  print(native.existing_rule('x')['srcs'])");
    scratch.file(
        "test/BUILD",
        "load('//test:existing_rule.bzl', 'macro')",
        "macro()",
        "cc_library(name = 'a', srcs = [])");
    getConfiguredTarget("//test:a");
    assertContainsEvent("select({Label(\"//foo:foo\"): [Label(\"//bar:bar\")]})");
  }

  @Test
  public void testGetRule() throws Exception {
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/rulestr.bzl",
        "def rule_dict(name):",
        "  return native.existing_rule(name)",
        "def rules_dict():",
        "  return native.existing_rules()",
        "def nop(ctx):",
        "  pass",
        "nop_rule = rule(attrs = {'x': attr.label()}, implementation = nop)",
        "consume_rule = rule(attrs = {'s': attr.string_list()}, implementation = nop)");

    scratch.file(
        "test/getrule/BUILD",
        "load('//test/skylark:rulestr.bzl', 'rules_dict', 'rule_dict', 'nop_rule', 'consume_rule')",
        "genrule(name = 'a', outs = ['a.txt'], ",
        "        licenses = ['notice'],",
        "        output_to_bindir = False,",
        "        tools = [ '//test:bla' ], cmd = 'touch $@')",
        "nop_rule(name = 'c', x = ':a')",
        "rlist= rules_dict()",
        "consume_rule(name = 'all_str', s = [rlist['a']['kind'], rlist['a']['name'], ",
        "                                    rlist['c']['kind'], rlist['c']['name']])",
        "adict = rule_dict('a')",
        "cdict = rule_dict('c')",
        "consume_rule(name = 'a_str', ",
        "             s = [adict['kind'], adict['name'], adict['outs'][0], adict['tools'][0]])",
        "consume_rule(name = 'genrule_attr', ",
        "             s = adict.keys())",
        "consume_rule(name = 'c_str', s = [cdict['kind'], cdict['name'], cdict['x']])");

    SkylarkRuleContext allContext = createRuleContext("//test/getrule:all_str");
    List<String> result = (List<String>) evalRuleContextCode(allContext, "ruleContext.attr.s");
    assertThat(result).containsExactly("genrule", "a", "nop_rule", "c");

    result = (List<String>) evalRuleContextCode(
        createRuleContext("//test/getrule:a_str"), "ruleContext.attr.s");
    assertThat(result).containsExactly("genrule", "a", ":a.txt", "//test:bla");

    result = (List<String>) evalRuleContextCode(
        createRuleContext("//test/getrule:c_str"), "ruleContext.attr.s");
    assertThat(result).containsExactly("nop_rule", "c", ":a");

    result = (List<String>) evalRuleContextCode(
        createRuleContext("//test/getrule:genrule_attr"), "ruleContext.attr.s");
    assertThat(result).containsExactly(
        "name",
        "visibility",
        "transitive_configs",
        "tags",
        "generator_name",
        "generator_function",
        "generator_location",
        "features",
        "compatible_with",
        "restricted_to",
        "srcs",
        "tools",
        "toolchains",
        "outs",
        "cmd",
        "output_to_bindir",
        "local",
        "message",
        "executable",
        "stamp",
        "heuristic_label_expansion",
        "kind",
        "exec_compatible_with");
  }

  @Test
  public void testGetRuleAttributeListValue() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertThat(((SkylarkList) result)).hasSize(1);
  }

  @Test
  public void testGetRuleAttributeListValueNoGet() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertThat(((SkylarkList) result)).hasSize(1);
  }

  @Test
  public void testGetRuleAttributeStringTypeValue() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.cmd");
    assertThat((String) result).isEqualTo("dummy_cmd");
  }

  @Test
  public void testGetRuleAttributeStringTypeValueNoGet() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.cmd");
    assertThat((String) result).isEqualTo("dummy_cmd");
  }

  @Test
  public void testGetRuleAttributeBadAttributeName() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"), "No attribute 'bad'", "ruleContext.attr.bad");
  }

  @Test
  public void testGetLabel() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.label");
    assertThat(((Label) result).toString()).isEqualTo("//foo:foo");
  }

  @Test
  public void testRuleError() throws Exception {
    checkErrorContains(createRuleContext("//foo:foo"), "message", "fail('message')");
  }

  @Test
  public void testAttributeError() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"),
        "attribute srcs: message",
        "fail(attr='srcs', msg='message')");
  }

  @Test
  public void testGetExecutablePrerequisite() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:androidlib");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.executable._idlclass");
    assertThat(((Artifact) result).getFilename()).matches("^IdlClass(\\.exe){0,1}$");
  }

  @Test
  public void testCreateSpawnActionArgumentsWithExecutableFilesToRunProvider() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:androidlib");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.run(\n"
            + "  inputs = ruleContext.files.srcs,\n"
            + "  outputs = ruleContext.files.srcs,\n"
            + "  arguments = ['--a','--b'],\n"
            + "  executable = ruleContext.executable._idlclass)\n");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getCommandFilename()).matches("^.*/IdlClass(\\.exe){0,1}$");
  }

  @Test
  public void testOutputs() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Iterable<?> result = (Iterable<?>) evalRuleContextCode(ruleContext, "ruleContext.outputs.outs");
    assertThat(((Artifact) Iterables.getOnlyElement(result)).getFilename()).isEqualTo("d.txt");
  }

  @Test
  public void testSkylarkRuleContextGetDefaultShellEnv() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.configuration.default_shell_env");
    assertThat(result).isInstanceOf(SkylarkDict.class);
  }

  @Test
  public void testCheckPlaceholders() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.check_placeholders('%{name}', ['name'])");
    assertThat(result).isEqualTo(true);
  }

  @Test
  public void testCheckPlaceholdersBadPlaceholder() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.check_placeholders('%{name}', ['abc'])");
    assertThat(result).isEqualTo(false);
  }

  @Test
  public void testExpandMakeVariables() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext, "ruleContext.expand_make_variables('cmd', '$(ABC)', {'ABC': 'DEF'})");
    assertThat(result).isEqualTo("DEF");
  }

  @Test
  public void testExpandMakeVariablesShell() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.expand_make_variables('cmd', '$$ABC', {})");
    assertThat(result).isEqualTo("$ABC");
  }

  private void setUpMakeVarToolchain() throws Exception {
    scratch.file(
        "vars/vars.bzl",
        "def _make_var_supplier_impl(ctx):",
        "  val = ctx.attr.value",
        "  return [platform_common.TemplateVariableInfo({'MAKE_VAR_VALUE': val})]",
        "make_var_supplier = rule(",
        "    implementation = _make_var_supplier_impl,",
        "    attrs = {",
        "        'value': attr.string(mandatory = True),",
        "    })",
        "def _make_var_user_impl(ctx):",
        "  return []",
        "make_var_user = rule(",
        "    implementation = _make_var_user_impl,",
        ")");
    scratch.file(
        "vars/BUILD",
        "load(':vars.bzl', 'make_var_supplier', 'make_var_user')",
        "make_var_supplier(name = 'supplier', value = 'foo')",
        "cc_toolchain_alias(name = 'current_cc_toolchain')",
        "make_var_user(",
        "    name = 'vars',",
        "    toolchains = [':supplier', ':current_cc_toolchain'],",
        ")");
  }

  @Test
  public void testExpandMakeVariables_cc() throws Exception {
    setUpMakeVarToolchain();
    SkylarkRuleContext ruleContext = createRuleContext("//vars:vars");
    String result =
        (String)
            evalRuleContextCode(
                ruleContext, "ruleContext.expand_make_variables('cmd', '$(CC)', {})");
    assertThat(result).isNotEmpty();
  }

  @Test
  public void testExpandMakeVariables_toolchain() throws Exception {
    setUpMakeVarToolchain();
    SkylarkRuleContext ruleContext = createRuleContext("//vars:vars");
    Object result =
        evalRuleContextCode(
            ruleContext, "ruleContext.expand_make_variables('cmd', '$(MAKE_VAR_VALUE)', {})");
    assertThat(result).isEqualTo("foo");
  }

  @Test
  public void testVar_toolchain() throws Exception {
    setUpMakeVarToolchain();
    SkylarkRuleContext ruleContext = createRuleContext("//vars:vars");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.var['MAKE_VAR_VALUE']");
    assertThat(result).isEqualTo("foo");
  }

  @Test
  public void testConfiguration() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.configuration");
    assertThat(ruleContext.getRuleContext().getConfiguration()).isSameAs(result);
  }

  @Test
  public void testFeatures() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:cc_with_features");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.features");
    assertThat((SkylarkList<?>) result).containsExactly("cc_include_scanning", "f1", "f2");
  }

  @Test
  public void testDisabledFeatures() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:cc_with_features");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.disabled_features");
    assertThat((SkylarkList<?>) result).containsExactly("f3");
  }

  @Test
  public void testHostConfiguration() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.host_configuration");
    assertThat(ruleContext.getRuleContext().getHostConfiguration()).isSameAs(result);
  }

  @Test
  public void testWorkspaceName() throws Exception {
    assertThat(ruleClassProvider.getRunfilesPrefix()).isNotNull();
    assertThat(ruleClassProvider.getRunfilesPrefix()).isNotEmpty();
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.workspace_name");
    assertThat(ruleClassProvider.getRunfilesPrefix()).isEqualTo(result);
  }

  @Test
  public void testDeriveArtifactLegacy() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ruleContext.new_file(ruleContext.genfiles_dir," + "  'a/b.txt')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertThat(fragment.getPathString()).isEqualTo("foo/a/b.txt");
  }

  @Test
  public void testDeriveArtifact() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.new_file('a/b.txt')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertThat(fragment.getPathString()).isEqualTo("foo/a/b.txt");
  }

  @Test
  public void testDeriveTreeArtifact() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.actions.declare_directory('a/b')");
    Artifact artifact = (Artifact) result;
    PathFragment fragment = artifact.getRootRelativePath();
    assertThat(fragment.getPathString()).isEqualTo("foo/a/b");
    assertThat(artifact.isTreeArtifact()).isTrue();
  }

  @Test
  public void testDeriveTreeArtifactType() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext,
            "b = ruleContext.actions.declare_directory('a/b')\n"
            + "type(b)");
    assertThat(result).isInstanceOf(String.class);
    assertThat(result).isEqualTo("File");
  }


  @Test
  public void testDeriveTreeArtifactNextToSibling() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "b = ruleContext.actions.declare_directory('a/b')\n"
                + "ruleContext.actions.declare_directory('c', sibling=b)");
    Artifact artifact = (Artifact) result;
    PathFragment fragment = artifact.getRootRelativePath();
    assertThat(fragment.getPathString()).isEqualTo("foo/a/c");
    assertThat(artifact.isTreeArtifact()).isTrue();
  }

  @Test
  public void testParamFileLegacy() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ruleContext.new_file(ruleContext.bin_dir,"
                + "ruleContext.files.tools[0], '.params')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertThat(fragment.getPathString()).isEqualTo("foo/t.exe.params");
  }

  @Test
  public void testParamFileSuffixLegacy() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ruleContext.new_file(ruleContext.files.tools[0], "
                + "ruleContext.files.tools[0].basename + '.params')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertThat(fragment.getPathString()).isEqualTo("foo/t.exe.params");
  }

  @Test
  public void testParamFileSuffix() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ruleContext.actions.declare_file(ruleContext.files.tools[0].basename + '.params', "
                + "sibling = ruleContext.files.tools[0])");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertThat(fragment.getPathString()).isEqualTo("foo/t.exe.params");
  }


  @Test
  public void testLabelKeyedStringDictConvertsToTargetToStringMap() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(),",
        "  }",
        ")");

    scratch.file(
        "BUILD",
        "filegroup(name='dep')",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={':dep': 'value'})");

    invalidatePackages();
    SkylarkRuleContext context = createRuleContext("//:r");
    Label keyLabel =
        (Label) evalRuleContextCode(context, "ruleContext.attr.label_dict.keys()[0].label");
    assertThat(keyLabel).isEqualTo(Label.parseAbsolute("//:dep", ImmutableMap.of()));
    String valueString =
        (String) evalRuleContextCode(context, "ruleContext.attr.label_dict.values()[0]");
    assertThat(valueString).isEqualTo("value");
  }

  @Test
  public void testLabelKeyedStringDictTranslatesAliases() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(),",
        "  }",
        ")");

    scratch.file(
        "BUILD",
        "filegroup(name='dep')",
        "alias(name='alias', actual='dep')",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={':alias': 'value'})");

    invalidatePackages();
    SkylarkRuleContext context = createRuleContext("//:r");
    Label keyLabel =
        (Label) evalRuleContextCode(context, "ruleContext.attr.label_dict.keys()[0].label");
    assertThat(keyLabel).isEqualTo(Label.parseAbsolute("//:dep", ImmutableMap.of()));
    String valueString =
        (String) evalRuleContextCode(context, "ruleContext.attr.label_dict.values()[0]");
    assertThat(valueString).isEqualTo("value");
  }

  @Test
  public void testLabelKeyedStringDictAcceptsDefaultValues() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(default={Label('//:default'): 'defs'}),",
        "  }",
        ")");

    scratch.file(
        "BUILD",
        "filegroup(name='default')",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r')");

    invalidatePackages();
    SkylarkRuleContext context = createRuleContext("//:r");
    Label keyLabel =
        (Label) evalRuleContextCode(context, "ruleContext.attr.label_dict.keys()[0].label");
    assertThat(keyLabel).isEqualTo(Label.parseAbsolute("//:default", ImmutableMap.of()));
    String valueString =
        (String) evalRuleContextCode(context, "ruleContext.attr.label_dict.values()[0]");
    assertThat(valueString).isEqualTo("defs");
  }

  @Test
  public void testLabelKeyedStringDictAllowsFilesWhenAllowFilesIsTrue() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(allow_files=True),",
        "  }",
        ")");

    scratch.file("myfile.cc");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={'myfile.cc': 'value'})");

    invalidatePackages();
    createRuleContext("//:r");
    assertNoEvents();
  }

  @Test
  public void testLabelKeyedStringDictAllowsFilesOfAppropriateTypes() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(allow_files=['.cc']),",
        "  }",
        ")");

    scratch.file("myfile.cc");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={'myfile.cc': 'value'})");

    invalidatePackages();
    createRuleContext("//:r");
    assertNoEvents();
  }

  @Test
  public void testLabelKeyedStringDictForbidsFilesOfIncorrectTypes() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(allow_files=['.cc']),",
        "  }",
        ")");

    scratch.file("myfile.cpp");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={'myfile.cpp': 'value'})");

    invalidatePackages();
    getConfiguredTarget("//:r");
    assertContainsEvent("file '//:myfile.cpp' is misplaced here (expected .cc)");
  }

  @Test
  public void testLabelKeyedStringDictForbidsFilesWhenAllowFilesIsFalse() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(allow_files=False),",
        "  }",
        ")");

    scratch.file("myfile.cpp");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={'myfile.cpp': 'value'})");

    invalidatePackages();
    getConfiguredTarget("//:r");
    assertContainsEvent("in label_dict attribute of my_rule rule //:r: "
        + "source file '//:myfile.cpp' is misplaced here (expected no files)");
  }

  @Test
  public void testLabelKeyedStringDictAllowsRulesWithRequiredProviders() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(providers=[['my_provider']]),",
        "  }",
        ")",
        "def _dep_impl(ctx):",
        "  return struct(my_provider=5)",
        "my_dep_rule = rule(",
        "  implementation = _dep_impl,",
        "  attrs = {}",
        ")");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule', 'my_dep_rule')",
        "my_dep_rule(name='dep')",
        "my_rule(name='r',",
        "        label_dict={':dep': 'value'})");

    invalidatePackages();
    createRuleContext("//:r");
    assertNoEvents();
  }

  @Test
  public void testLabelKeyedStringDictForbidsRulesMissingRequiredProviders() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(providers=[['my_provider']]),",
        "  }",
        ")",
        "def _dep_impl(ctx):",
        "  return",
        "my_dep_rule = rule(",
        "  implementation = _dep_impl,",
        "  attrs = {}",
        ")");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule', 'my_dep_rule')",
        "my_dep_rule(name='dep')",
        "my_rule(name='r',",
        "        label_dict={':dep': 'value'})");

    invalidatePackages();
    getConfiguredTarget("//:r");
    assertContainsEvent("in label_dict attribute of my_rule rule //:r: "
        + "'//:dep' does not have mandatory providers: 'my_provider'");
  }

  @Test
  public void testLabelKeyedStringDictForbidsEmptyDictWhenAllowEmptyIsFalse() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(allow_empty=False),",
        "  }",
        ")");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={})");

    invalidatePackages();
    getConfiguredTarget("//:r");
    assertContainsEvent("in label_dict attribute of my_rule rule //:r: "
        + "attribute must be non empty");
  }

  @Test
  public void testLabelKeyedStringDictAllowsEmptyDictWhenAllowEmptyIsTrue() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(allow_empty=True),",
        "  }",
        ")");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r',",
        "        label_dict={})");

    invalidatePackages();
    createRuleContext("//:r");
    assertNoEvents();
  }

  @Test
  public void testLabelKeyedStringDictForbidsMissingAttributeWhenMandatoryIsTrue()
      throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(mandatory=True),",
        "  }",
        ")");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r')");

    invalidatePackages();
    getConfiguredTarget("//:r");
    assertContainsEvent("missing value for mandatory attribute 'label_dict' in 'my_rule' rule");
  }

  @Test
  public void testLabelKeyedStringDictAllowsMissingAttributeWhenMandatoryIsFalse()
      throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'label_dict': attr.label_keyed_string_dict(mandatory=False),",
        "  }",
        ")");

    scratch.file(
        "BUILD",
        "load('//:my_rule.bzl', 'my_rule')",
        "my_rule(name='r')");

    invalidatePackages();
    createRuleContext("//:r");
    assertNoEvents();
  }

  @Test
  public void testLabelAttributeDefault() throws Exception {
    scratch.file(
        "my_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'explicit_dep': attr.label(default = Label('//:dep')),",
        "    '_implicit_dep': attr.label(default = Label('//:dep')),",
        "    'explicit_dep_list': attr.label_list(default = [Label('//:dep')]),",
        "    '_implicit_dep_list': attr.label_list(default = [Label('//:dep')]),",
        "  }",
        ")");

    scratch.file(
        "BUILD", "filegroup(name='dep')", "load('//:my_rule.bzl', 'my_rule')", "my_rule(name='r')");

    invalidatePackages();
    SkylarkRuleContext context = createRuleContext("//:r");
    Label explicitDepLabel =
        (Label) evalRuleContextCode(context, "ruleContext.attr.explicit_dep.label");
    assertThat(explicitDepLabel).isEqualTo(Label.parseAbsolute("//:dep", ImmutableMap.of()));
    Label implicitDepLabel =
        (Label) evalRuleContextCode(context, "ruleContext.attr._implicit_dep.label");
    assertThat(implicitDepLabel).isEqualTo(Label.parseAbsolute("//:dep", ImmutableMap.of()));
    Label explicitDepListLabel =
        (Label) evalRuleContextCode(context, "ruleContext.attr.explicit_dep_list[0].label");
    assertThat(explicitDepListLabel).isEqualTo(Label.parseAbsolute("//:dep", ImmutableMap.of()));
    Label implicitDepListLabel =
        (Label) evalRuleContextCode(context, "ruleContext.attr._implicit_dep_list[0].label");
    assertThat(implicitDepListLabel).isEqualTo(Label.parseAbsolute("//:dep", ImmutableMap.of()));
  }

  @Test
  public void testRelativeLabelInExternalRepository() throws Exception {
    scratch.file("external_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "external_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'internal_dep': attr.label(default = Label('//:dep'))",
        "  }",
        ")");

    scratch.file("BUILD",
        "filegroup(name='dep')");

    scratch.file("/r/WORKSPACE");
    scratch.file("/r/a/BUILD",
        "load('@//:external_rule.bzl', 'external_rule')",
        "external_rule(name='r')");

    scratch.overwriteFile("WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .add("local_repository(name='r', path='/r')")
            .build());

    invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchain labels.
    SkylarkRuleContext context = createRuleContext("@r//a:r");
    Label depLabel = (Label) evalRuleContextCode(context, "ruleContext.attr.internal_dep.label");
    assertThat(depLabel).isEqualTo(Label.parseAbsolute("//:dep", ImmutableMap.of()));
  }

  @Test
  public void testCallerRelativeLabelInExternalRepository() throws Exception {
    scratch.file("BUILD");
    scratch.file("external_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "external_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'internal_dep': attr.label(",
        "        default = Label('//:dep', relative_to_caller_repository = True)",
        "    )",
        "  }",
        ")");

    scratch.file("/r/WORKSPACE");
    scratch.file("/r/BUILD",
        "filegroup(name='dep')");

    scratch.file("/r/a/BUILD",
        "load('@//:external_rule.bzl', 'external_rule')",
        "external_rule(name='r')");

    scratch.overwriteFile("WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .add("local_repository(name='r', path='/r')")
            .build());

    invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchain labels.
    SkylarkRuleContext context = createRuleContext("@r//a:r");
    Label depLabel = (Label) evalRuleContextCode(context, "ruleContext.attr.internal_dep.label");
    assertThat(depLabel).isEqualTo(Label.parseAbsolute("@r//:dep", ImmutableMap.of()));
  }

  @Test
  public void testExternalWorkspaceLoad() throws Exception {
    scratch.file(
        "/r1/BUILD",
        "filegroup(name = 'test',",
        " srcs = ['test.txt'],",
        " visibility = ['//visibility:public'],",
        ")");
    scratch.file("/r1/WORKSPACE");
    scratch.file("/r2/BUILD", "exports_files(['test.bzl'])");
    scratch.file(
        "/r2/test.bzl",
        "def macro(name, path):",
        "  native.local_repository(name = name, path = path)"
    );
    scratch.file("/r2/WORKSPACE");
    scratch.file(
        "/r2/other_test.bzl",
        "def other_macro(name, path):",
        "  print(name + ': ' + path)"
    );
    scratch.file("BUILD");

    scratch.overwriteFile("WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .add("local_repository(name='r2', path='/r2')")
            .add("load('@r2//:test.bzl', 'macro')")
            .add("macro('r1', '/r1')")
            .add("NEXT_NAME = 'r3'")
            // We can still refer to r2 in other chunks:
            .add("load('@r2//:other_test.bzl', 'other_macro')")
            .add("macro(NEXT_NAME, '/r2')") // and we can still use macro outside of its chunk.
            .build());

    invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchain labels.
    assertThat(getConfiguredTarget("@r1//:test")).isNotNull();
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testLoadBlockRepositoryRedefinition() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("/bar/WORKSPACE");
    scratch.file("/bar/bar.txt");
    scratch.file("/bar/BUILD", "filegroup(name = 'baz', srcs = ['bar.txt'])");
    scratch.file("/baz/WORKSPACE");
    scratch.file("/baz/baz.txt");
    scratch.file("/baz/BUILD", "filegroup(name = 'baz', srcs = ['baz.txt'])");
    scratch.overwriteFile("WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .add("local_repository(name = 'foo', path = '/bar')")
            .add("local_repository(name = 'foo', path = '/baz')")
            .build());

    invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchain labels.
    assertThat(
            (List<Label>)
                getConfiguredTargetAndData("@foo//:baz")
                    .getTarget()
                    .getAssociatedRule()
                    .getAttributeContainer()
                    .getAttr("srcs"))
        .contains(Label.parseAbsolute("@foo//:baz.txt", ImmutableMap.of()));

    scratch.overwriteFile("BUILD");
    scratch.overwriteFile("bar.bzl", "dummy = 1");

    scratch.overwriteFile("WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .add("local_repository(name = 'foo', path = '/bar')")
            .add("load('//:bar.bzl', 'dummy')")
            .add("local_repository(name = 'foo', path = '/baz')")
            .build());

    try {
      invalidatePackages(/*alsoConfigs=*/false); // Repository shuffling messes with toolchains.
      createRuleContext("@foo//:baz");
      fail("Should have failed because repository 'foo' is overloading after a load!");
    } catch (Exception ex) {
      assertContainsEvent(
          "Cannot redefine repository after any load statement in the WORKSPACE file "
              + "(for repository 'foo')");
    }
  }

  @Test
  public void testAccessingRunfiles() throws Exception {
    scratch.file("test/a.py");
    scratch.file("test/b.py");
    scratch.file("test/__init__.py");
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  return",
        "skylark_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep': attr.label(),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:rule.bzl', 'skylark_rule')",
        "py_binary(name = 'lib', srcs = ['lib.py', 'lib2.py'])",
        "skylark_rule(name = 'foo', dep = ':lib')",
        "py_binary(name = 'lib_with_init', srcs = ['lib_with_init.py', 'lib2.py', '__init__.py'])",
        "skylark_rule(name = 'foo_with_init', dep = ':lib_with_init')");

    SkylarkRuleContext ruleContext = createRuleContext("//test:foo");
    Object filenames =
        evalRuleContextCode(
            ruleContext, "[f.short_path for f in ruleContext.attr.dep.default_runfiles.files]");
    assertThat(filenames).isInstanceOf(SkylarkList.class);
    SkylarkList filenamesList = (SkylarkList) filenames;
    assertThat(filenamesList).containsAllOf("test/lib.py", "test/lib2.py");
    Object emptyFilenames =
        evalRuleContextCode(
            ruleContext, "list(ruleContext.attr.dep.default_runfiles.empty_filenames)");
    assertThat(emptyFilenames).isInstanceOf(SkylarkList.class);
    SkylarkList emptyFilenamesList = (SkylarkList) emptyFilenames;
    assertThat(emptyFilenamesList).containsExactly("test/__init__.py");

    SkylarkRuleContext ruleWithInitContext = createRuleContext("//test:foo_with_init");
    Object noEmptyFilenames =
        evalRuleContextCode(
            ruleWithInitContext, "list(ruleContext.attr.dep.default_runfiles.empty_filenames)");
    assertThat(noEmptyFilenames).isInstanceOf(SkylarkList.class);
    SkylarkList noEmptyFilenamesList = (SkylarkList) noEmptyFilenames;
    assertThat(noEmptyFilenamesList).isEmpty();
  }

  @Test
  public void testAccessingRunfilesSymlinks() throws Exception {
    scratch.file("test/a.py");
    scratch.file("test/b.py");
    scratch.file(
        "test/rule.bzl",
        "def symlink_impl(ctx):",
        "  symlinks = {",
        "    'symlink_' + f.short_path: f",
        "    for f in ctx.files.symlink",
        "  }",
        "  return struct(",
        "    runfiles = ctx.runfiles(",
        "      symlinks=symlinks,",
        "    )",
        "  )",
        "symlink_rule = rule(",
        "  implementation = symlink_impl,",
        "  attrs = {",
        "    'symlink': attr.label(allow_files=True),",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:rule.bzl', 'symlink_rule')",
        "symlink_rule(name = 'lib_with_symlink', symlink = ':a.py')",
        "sh_binary(",
        "  name = 'test_with_symlink',",
        "  srcs = ['test/b.py'],",
        "  data = [':lib_with_symlink'],",
        ")");
    SkylarkRuleContext ruleWithSymlinkContext = createRuleContext("//test:test_with_symlink");
    Object symlinkPaths =
        evalRuleContextCode(
            ruleWithSymlinkContext,
            "[s.path for s in",
            "ruleContext.attr.data[0].data_runfiles.symlinks]");
    assertThat(symlinkPaths).isInstanceOf(SkylarkList.class);
    SkylarkList<String> symlinkPathsList = (SkylarkList<String>) symlinkPaths;
    assertThat(symlinkPathsList).containsExactly("symlink_test/a.py").inOrder();
    Object symlinkFilenames =
        evalRuleContextCode(
            ruleWithSymlinkContext,
            "[s.target_file.short_path for s in",
            "ruleContext.attr.data[0].data_runfiles.symlinks]");
    assertThat(symlinkFilenames).isInstanceOf(SkylarkList.class);
    SkylarkList<String> symlinkFilenamesList = (SkylarkList<String>) symlinkFilenames;
    assertThat(symlinkFilenamesList).containsExactly("test/a.py").inOrder();
  }

  @Test
  public void testAccessingRunfilesRootSymlinks() throws Exception {
    scratch.file("test/a.py");
    scratch.file("test/b.py");
    scratch.file(
        "test/rule.bzl",
        "def root_symlink_impl(ctx):",
        "  root_symlinks = {",
        "    'root_symlink_' + f.short_path: f",
        "    for f in ctx.files.root_symlink",
        "  }",
        "  return struct(",
        "    runfiles = ctx.runfiles(",
        "      root_symlinks=root_symlinks,",
        "    )",
        "  )",
        "root_symlink_rule = rule(",
        "  implementation = root_symlink_impl,",
        "  attrs = {",
        "    'root_symlink': attr.label(allow_files=True)",
        "  },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:rule.bzl', 'root_symlink_rule')",
        "root_symlink_rule(name = 'lib_with_root_symlink', root_symlink = ':a.py')",
        "sh_binary(",
        "  name = 'test_with_root_symlink',",
        "  srcs = ['test/b.py'],",
        "  data = [':lib_with_root_symlink'],",
        ")");
    SkylarkRuleContext ruleWithRootSymlinkContext =
        createRuleContext("//test:test_with_root_symlink");
    Object rootSymlinkPaths =
        evalRuleContextCode(
            ruleWithRootSymlinkContext,
            "[s.path for s in",
            "ruleContext.attr.data[0].data_runfiles.root_symlinks]");
    assertThat(rootSymlinkPaths).isInstanceOf(SkylarkList.class);
    SkylarkList<String> rootSymlinkPathsList = (SkylarkList<String>) rootSymlinkPaths;
    assertThat(rootSymlinkPathsList).containsExactly("root_symlink_test/a.py").inOrder();
    Object rootSymlinkFilenames =
        evalRuleContextCode(
            ruleWithRootSymlinkContext,
            "[s.target_file.short_path for s in",
            "ruleContext.attr.data[0].data_runfiles.root_symlinks]");
    assertThat(rootSymlinkFilenames).isInstanceOf(SkylarkList.class);
    SkylarkList<String> rootSymlinkFilenamesList = (SkylarkList<String>) rootSymlinkFilenames;
    assertThat(rootSymlinkFilenamesList).containsExactly("test/a.py").inOrder();
  }

  @Test
  public void testExternalShortPath() throws Exception {
    scratch.file("/bar/WORKSPACE");
    scratch.file("/bar/bar.txt");
    scratch.file("/bar/BUILD", "exports_files(['bar.txt'])");
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(name = 'foo', path = '/bar')");
    scratch.file(
        "test/BUILD",
        "genrule(",
        "    name = 'lib',",
        "    srcs = ['@foo//:bar.txt'],",
        "    cmd = 'echo $(SRCS) $@',",
        "    outs = ['lib.out'],",
        "    executable = 1,",
        ")");
    invalidatePackages();
    SkylarkRuleContext ruleContext = createRuleContext("//test:lib");
    String filename = evalRuleContextCode(ruleContext, "ruleContext.files.srcs[0].short_path")
        .toString();
    assertThat(filename).isEqualTo("../foo/bar.txt");
  }

  // Borrowed from Scratch.java.
  private static String linesAsString(String... lines) {
    StringBuilder builder = new StringBuilder();
    for (String line : lines) {
      builder.append(line);
      builder.append('\n');
    }
    return builder.toString();
  }

  // The common structure of the following actions tests is a rule under test depended upon by
  // a testing rule, where the rule under test has one output and one caller-supplied action.

  private String getSimpleUnderTestDefinition(String actionLine, boolean withSkylarkTestable) {
    return linesAsString(
      "def _undertest_impl(ctx):",
      "  out = ctx.outputs.out",
      "  " + actionLine,
      "undertest_rule = rule(",
      "  implementation = _undertest_impl,",
      "  outputs = {'out': '%{name}.txt'},",
      withSkylarkTestable ? "  _skylark_testable = True," : "",
      ")");
  }

  private String getSimpleUnderTestDefinition(String actionLine) {
    return getSimpleUnderTestDefinition(actionLine, true);
  }

  private String getSimpleNontestableUnderTestDefinition(String actionLine) {
    return getSimpleUnderTestDefinition(actionLine, false);
  }

  private final String testingRuleDefinition =
    linesAsString(
      "def _testing_impl(ctx):",
      "  pass",
      "testing_rule = rule(",
      "  implementation = _testing_impl,",
      "  attrs = {'dep': attr.label()},",
      ")");

  private final String simpleBuildDefinition =
    linesAsString(
      "load(':rules.bzl', 'undertest_rule', 'testing_rule')",
      "undertest_rule(",
      "    name = 'undertest',",
      ")",
      "testing_rule(",
      "    name = 'testing',",
      "    dep = ':undertest',",
      ")");

  @Test
  public void testDependencyActionsProvider() throws Exception {
    scratch.file("test/rules.bzl",
        getSimpleUnderTestDefinition(
            "ctx.actions.run_shell(outputs=[out], command='echo foo123 > ' + out.path)"),
        testingRuleDefinition);
    scratch.file("test/BUILD",
        simpleBuildDefinition);
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");

    Object provider = evalRuleContextCode(ruleContext, "ruleContext.attr.dep[Actions]");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider()).isEqualTo(ActionsProvider.INSTANCE);
    update("actions", provider);

    Object mapping = eval("actions.by_file");
    assertThat(mapping).isInstanceOf(SkylarkDict.class);
    assertThat((SkylarkDict<?, ?>) mapping).hasSize(1);
    update("file", eval("list(ruleContext.attr.dep.files)[0]"));
    Object actionUnchecked = eval("actions.by_file[file]");
    assertThat(actionUnchecked).isInstanceOf(ActionAnalysisMetadata.class);
  }

  @Test
  public void testNoAccessToDependencyActionsWithoutSkylarkTest() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("test/rules.bzl",
        getSimpleNontestableUnderTestDefinition(
            "ctx.actions.run_shell(outputs=[out], command='echo foo123 > ' + out.path)"),
        testingRuleDefinition);
    scratch.file("test/BUILD",
        simpleBuildDefinition);
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");

    try {
      evalRuleContextCode(ruleContext, "ruleContext.attr.dep[Actions]");
      fail("Should have failed due to trying to access actions of a rule not marked "
          + "_skylark_testable");
    } catch (Exception e) {
      assertThat(e).hasMessageThat().contains(
          "<target //test:undertest> (rule 'undertest_rule') doesn't contain "
              + "declared provider 'Actions'");
    }
  }

  @Test
  public void testAbstractActionInterface() throws Exception {
    scratch.file("test/rules.bzl",
        "def _undertest_impl(ctx):",
        "  out1 = ctx.outputs.out1",
        "  out2 = ctx.outputs.out2",
        "  ctx.actions.write(output=out1, content='foo123')",
        "  ctx.actions.run_shell(outputs=[out2], inputs=[out1],",
        "                        command='cp ' + out1.path + ' ' + out2.path)",
        "  return struct(out1=out1, out2=out2)",
        "undertest_rule = rule(",
        "  implementation = _undertest_impl,",
        "  outputs = {'out1': '%{name}1.txt',",
        "             'out2': '%{name}2.txt'},",
        "  _skylark_testable = True,",
        ")",
        testingRuleDefinition);
    scratch.file("test/BUILD",
        simpleBuildDefinition);
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");
    update("ruleContext", ruleContext);
    update("file1", eval("ruleContext.attr.dep.out1"));
    update("file2", eval("ruleContext.attr.dep.out2"));
    update("action1", eval("ruleContext.attr.dep[Actions].by_file[file1]"));
    update("action2", eval("ruleContext.attr.dep[Actions].by_file[file2]"));

    assertThat(eval("action1.inputs")).isInstanceOf(SkylarkNestedSet.class);
    assertThat(eval("action1.outputs")).isInstanceOf(SkylarkNestedSet.class);

    assertThat(eval("action1.argv")).isEqualTo(Runtime.NONE);
    assertThat(eval("action2.content")).isEqualTo(Runtime.NONE);
    assertThat(eval("action1.substitutions")).isEqualTo(Runtime.NONE);

    assertThat(eval("list(action1.inputs)")).isEqualTo(eval("[]"));
    assertThat(eval("list(action1.outputs)")).isEqualTo(eval("[file1]"));
    assertThat(eval("list(action2.inputs)")).isEqualTo(eval("[file1]"));
    assertThat(eval("list(action2.outputs)")).isEqualTo(eval("[file2]"));
  }

  // For created_actions() tests, the "undertest" rule represents both the code under test and the
  // Skylark user test code itself.

  @Test
  public void testCreatedActions() throws Exception {
    // createRuleContext() gives us the context for a rule upon entry into its analysis function.
    // But we need to inspect the result of calling created_actions() after the rule context has
    // been modified by creating actions. So we'll call created_actions() from within the analysis
    // function and pass it along as a provider.
    scratch.file("test/rules.bzl",
        "def _undertest_impl(ctx):",
        "  out1 = ctx.outputs.out1",
        "  out2 = ctx.outputs.out2",
        "  ctx.actions.run_shell(outputs=[out1], command='echo foo123 > ' + out1.path,",
        "                        mnemonic='foo')",
        "  v = ctx.created_actions().by_file",
        "  ctx.actions.run_shell(outputs=[out2], command='echo bar123 > ' + out2.path)",
        "  return struct(v=v, out1=out1, out2=out2)",
        "undertest_rule = rule(",
        "  implementation = _undertest_impl,",
        "  outputs = {'out1': '%{name}1.txt',",
        "             'out2': '%{name}2.txt'},",
        "  _skylark_testable = True,",
        ")",
        testingRuleDefinition
        );
    scratch.file("test/BUILD",
        simpleBuildDefinition);
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");

    Object mapUnchecked = evalRuleContextCode(ruleContext, "ruleContext.attr.dep.v");
    assertThat(mapUnchecked).isInstanceOf(SkylarkDict.class);
    SkylarkDict<?, ?> map = (SkylarkDict<?, ?>) mapUnchecked;
    // Should only have the first action because created_actions() was called
    // before the second action was created.
    Object file = eval("ruleContext.attr.dep.out1");
    assertThat(map).hasSize(1);
    assertThat(map).containsKey(file);
    Object actionUnchecked = map.get(file);
    assertThat(actionUnchecked).isInstanceOf(ActionAnalysisMetadata.class);
    assertThat(((ActionAnalysisMetadata) actionUnchecked).getMnemonic()).isEqualTo("foo");
  }

  @Test
  public void testNoAccessToCreatedActionsWithoutSkylarkTest() throws Exception {
    scratch.file("test/rules.bzl",
        getSimpleNontestableUnderTestDefinition(
            "ctx.actions.run_shell(outputs=[out], command='echo foo123 > ' + out.path)")
        );
    scratch.file("test/BUILD",
        "load(':rules.bzl', 'undertest_rule')",
        "undertest_rule(",
        "    name = 'undertest',",
        ")");
    SkylarkRuleContext ruleContext = createRuleContext("//test:undertest");

    Object result = evalRuleContextCode(ruleContext, "ruleContext.created_actions()");
    assertThat(result).isEqualTo(Runtime.NONE);
  }

  @Test
  public void testSpawnActionInterface() throws Exception {
    scratch.file("test/rules.bzl",
        getSimpleUnderTestDefinition(
            "ctx.actions.run_shell(outputs=[out], command='echo foo123 > ' + out.path)"),
        testingRuleDefinition);
    scratch.file("test/BUILD",
        simpleBuildDefinition);
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");
    update("ruleContext", ruleContext);
    update("file", eval("list(ruleContext.attr.dep.files)[0]"));
    update("action", eval("ruleContext.attr.dep[Actions].by_file[file]"));

    assertThat(eval("type(action)")).isEqualTo("Action");

    Object argvUnchecked = eval("action.argv");
    assertThat(argvUnchecked).isInstanceOf(SkylarkList.MutableList.class);
    SkylarkList.MutableList<?> argv = (SkylarkList.MutableList<?>) argvUnchecked;
    assertThat(argv).hasSize(3);
    assertThat(argv.isImmutable()).isTrue();
    Object result = eval("action.argv[2].startswith('echo foo123')");
    assertThat((Boolean) result).isTrue();
  }

  @Test
  public void testRunShellUsesHelperScriptForLongCommand() throws Exception {
    // createRuleContext() gives us the context for a rule upon entry into its analysis function.
    // But we need to inspect the result of calling created_actions() after the rule context has
    // been modified by creating actions. So we'll call created_actions() from within the analysis
    // function and pass it along as a provider.
    scratch.file(
        "test/rules.bzl",
        "def _undertest_impl(ctx):",
        "  out1 = ctx.outputs.out1",
        "  out2 = ctx.outputs.out2",
        "  out3 = ctx.outputs.out3",
        "  ctx.actions.run_shell(outputs=[out1],",
        "                        command='( %s ; ) > $1' % (",
        "                            ' ; '.join(['echo xxx%d' % i for i in range(0, 7000)])),",
        "                        mnemonic='mnemonic1',",
        "                        arguments=[out1.path])",
        "  ctx.actions.run_shell(outputs=[out2],",
        "                        command='echo foo > ' + out2.path,",
        "                        mnemonic='mnemonic2')",
        "  ctx.actions.run_shell(outputs=[out3],",
        "                        command='( %s ; ) > $1' % (",
        "                            ' ; '.join(['echo yyy%d' % i for i in range(0, 7000)])),",
        "                        mnemonic='mnemonic3',",
        "                        arguments=[out3.path])",
        "  v = ctx.created_actions().by_file",
        "  return struct(v=v, out1=out1, out2=out2, out3=out3)",
        "",
        "undertest_rule = rule(",
        "    implementation=_undertest_impl,",
        "    outputs={'out1': '%{name}1.txt',",
        "             'out2': '%{name}2.txt',",
        "             'out3': '%{name}3.txt'},",
        "    _skylark_testable = True,",
        ")",
        testingRuleDefinition);
    scratch.file("test/BUILD", simpleBuildDefinition);
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");

    Object mapUnchecked = evalRuleContextCode(ruleContext, "ruleContext.attr.dep.v");
    assertThat(mapUnchecked).isInstanceOf(SkylarkDict.class);
    SkylarkDict<?, ?> map = (SkylarkDict<?, ?>) mapUnchecked;
    Object out1 = eval("ruleContext.attr.dep.out1");
    Object out2 = eval("ruleContext.attr.dep.out2");
    Object out3 = eval("ruleContext.attr.dep.out3");
    // 5 actions in total: 3 SpawnActions and 2 FileWriteActions for the two long commands.
    assertThat(map).hasSize(5);
    assertThat(map).containsKey(out1);
    assertThat(map).containsKey(out2);
    assertThat(map).containsKey(out3);
    Object action1Unchecked = map.get(out1);
    Object action2Unchecked = map.get(out2);
    Object action3Unchecked = map.get(out3);
    assertThat(action1Unchecked).isInstanceOf(ActionAnalysisMetadata.class);
    assertThat(action2Unchecked).isInstanceOf(ActionAnalysisMetadata.class);
    assertThat(action3Unchecked).isInstanceOf(ActionAnalysisMetadata.class);
    ActionAnalysisMetadata spawnAction1 = (ActionAnalysisMetadata) action1Unchecked;
    ActionAnalysisMetadata spawnAction2 = (ActionAnalysisMetadata) action2Unchecked;
    ActionAnalysisMetadata spawnAction3 = (ActionAnalysisMetadata) action3Unchecked;
    assertThat(spawnAction1.getMnemonic()).isEqualTo("mnemonic1");
    assertThat(spawnAction2.getMnemonic()).isEqualTo("mnemonic2");
    assertThat(spawnAction3.getMnemonic()).isEqualTo("mnemonic3");
    Artifact helper1 =
        Iterables.getOnlyElement(
            Iterables.filter(
                spawnAction1.getInputs(), a -> a.getFilename().equals("undertest.run_shell_0.sh")));
    assertThat(
            Iterables.filter(spawnAction2.getInputs(), a -> a.getFilename().contains("run_shell_")))
        .isEmpty();
    Artifact helper3 =
        Iterables.getOnlyElement(
            Iterables.filter(
                spawnAction3.getInputs(), a -> a.getFilename().equals("undertest.run_shell_2.sh")));
    assertThat(map).containsKey(helper1);
    assertThat(map).containsKey(helper3);
    Object action4Unchecked = map.get(helper1);
    Object action5Unchecked = map.get(helper3);
    assertThat(action4Unchecked).isInstanceOf(FileWriteAction.class);
    assertThat(action5Unchecked).isInstanceOf(FileWriteAction.class);
    FileWriteAction fileWriteAction1 = (FileWriteAction) action4Unchecked;
    FileWriteAction fileWriteAction2 = (FileWriteAction) action5Unchecked;
    assertThat(fileWriteAction1.getFileContents()).contains("echo xxx6999 ;");
    assertThat(fileWriteAction2.getFileContents()).contains("echo yyy6999 ;");
  }

  @Test
  public void testFileWriteActionInterface() throws Exception {
    scratch.file("test/rules.bzl",
        getSimpleUnderTestDefinition(
            "ctx.actions.write(output=out, content='foo123')"),
        testingRuleDefinition);
    scratch.file("test/BUILD",
        simpleBuildDefinition);
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");
    update("ruleContext", ruleContext);
    update("file", eval("list(ruleContext.attr.dep.files)[0]"));
    update("action", eval("ruleContext.attr.dep[Actions].by_file[file]"));

    assertThat(eval("type(action)")).isEqualTo("Action");

    Object contentUnchecked = eval("action.content");
    assertThat(contentUnchecked).isInstanceOf(String.class);
    assertThat(contentUnchecked).isEqualTo("foo123");
  }

  @Test
  public void testTemplateExpansionActionInterface() throws Exception {
    scratch.file("test/rules.bzl",
        "def _undertest_impl(ctx):",
        "  out = ctx.outputs.out",
        "  ctx.actions.expand_template(output=out,",
        "                              template=ctx.file.template, substitutions={'a': 'b'})",
        "undertest_rule = rule(",
        "  implementation = _undertest_impl,",
        "  outputs = {'out': '%{name}.txt'},",
        "  attrs = {'template': attr.label(allow_single_file=True)},",
        "  _skylark_testable = True,",
        ")",
        testingRuleDefinition);
    scratch.file("test/template.txt",
        "aaaaa",
        "bcdef");
    scratch.file("test/BUILD",
        "load(':rules.bzl', 'undertest_rule', 'testing_rule')",
        "undertest_rule(",
        "    name = 'undertest',",
        "    template = ':template.txt',",
        ")",
        "testing_rule(",
        "    name = 'testing',",
        "    dep = ':undertest',",
        ")");
    SkylarkRuleContext ruleContext = createRuleContext("//test:testing");
    update("ruleContext", ruleContext);
    update("file", eval("list(ruleContext.attr.dep.files)[0]"));
    update("action", eval("ruleContext.attr.dep[Actions].by_file[file]"));

    assertThat(eval("type(action)")).isEqualTo("Action");

    Object contentUnchecked = eval("action.content");
    assertThat(contentUnchecked).isInstanceOf(String.class);
    assertThat(contentUnchecked).isEqualTo("bbbbb\nbcdef\n");

    Object substitutionsUnchecked = eval("action.substitutions");
    assertThat(substitutionsUnchecked).isInstanceOf(SkylarkDict.class);
    assertThat(substitutionsUnchecked).isEqualTo(SkylarkDict.of(null, "a", "b"));
  }

  private void setUpCoverageInstrumentedTest() throws Exception {
    scratch.file("test/BUILD",
        "cc_library(",
        "  name = 'foo',",
        "  srcs = ['foo.cc'],",
        "  deps = [':bar'],",
        ")",
        "cc_library(",
        "  name = 'bar',",
        "  srcs = ['bar.cc'],",
        ")");
  }

  @Test
  public void testCoverageInstrumentedCoverageDisabled() throws Exception {
    setUpCoverageInstrumentedTest();
    useConfiguration("--nocollect_code_coverage", "--instrumentation_filter=.");
    SkylarkRuleContext ruleContext = createRuleContext("//test:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.coverage_instrumented()");
    assertThat((Boolean) result).isFalse();
  }

  @Test
  public void testCoverageInstrumentedFalseForSourceFileLabel() throws Exception {
    setUpCoverageInstrumentedTest();
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=.");
    SkylarkRuleContext ruleContext = createRuleContext("//test:foo");
    Object result = evalRuleContextCode(ruleContext,
        "ruleContext.coverage_instrumented(ruleContext.attr.srcs[0])");
    assertThat((Boolean) result).isFalse();
  }

  @Test
  public void testCoverageInstrumentedDoesNotMatchFilter() throws Exception {
    setUpCoverageInstrumentedTest();
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=:foo");
    SkylarkRuleContext ruleContext = createRuleContext("//test:bar");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.coverage_instrumented()");
    assertThat((Boolean) result).isFalse();
  }

  @Test
  public void testCoverageInstrumentedMatchesFilter() throws Exception {
    setUpCoverageInstrumentedTest();
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=:foo");
    SkylarkRuleContext ruleContext = createRuleContext("//test:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.coverage_instrumented()");
    assertThat((Boolean) result).isTrue();
  }

  @Test
  public void testCoverageInstrumentedDoesNotMatchFilterNonDefaultLabel() throws Exception {
    setUpCoverageInstrumentedTest();
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=:foo");
    SkylarkRuleContext ruleContext = createRuleContext("//test:foo");
    // //test:bar does not match :foo, though //test:foo would.
    Object result = evalRuleContextCode(ruleContext,
        "ruleContext.coverage_instrumented(ruleContext.attr.deps[0])");
    assertThat((Boolean) result).isFalse();
  }

  @Test
  public void testCoverageInstrumentedMatchesFilterNonDefaultLabel() throws Exception {
    setUpCoverageInstrumentedTest();
    useConfiguration("--collect_code_coverage", "--instrumentation_filter=:bar");
    SkylarkRuleContext ruleContext = createRuleContext("//test:foo");
    // //test:bar does match :bar, though //test:foo would not.
    Object result = evalRuleContextCode(ruleContext,
        "ruleContext.coverage_instrumented(ruleContext.attr.deps[0])");
    assertThat((Boolean) result).isTrue();
  }

  // A list of attributes and methods ctx objects have
  private final List<String> ctxAttributes =
      ImmutableList.of(
          "attr",
          "split_attr",
          "executable",
          "file",
          "files",
          "workspace_name",
          "label",
          "fragments",
          "host_fragments",
          "configuration",
          "host_configuration",
          "coverage_instrumented(dep)",
          "features",
          "bin_dir",
          "genfiles_dir",
          "outputs",
          "rule",
          "aspect_ids",
          "var",
          "tokenize('foo')",
          "expand('foo', [], Label('//test:main'))",
          "new_file('foo.txt')",
          "new_file(file, 'foo.txt')",
          "actions.declare_file('foo.txt')",
          "actions.declare_file('foo.txt', sibling = file)",
          "actions.declare_directory('foo.txt')",
          "actions.declare_directory('foo.txt', sibling = file)",
          "actions.do_nothing(mnemonic = 'foo', inputs = [file])",
          "actions.expand_template(template = file, output = file, substitutions = {})",
          "actions.run(executable = file, outputs = [file])",
          "actions.run_shell(command = 'foo', outputs = [file])",
          "actions.write(file, 'foo')",
          "check_placeholders('foo', [])",
          "action(command = 'foo', outputs = [file])",
          "file_action(file, 'foo')",
          "empty_action(mnemonic = 'foo', inputs = [file])",
          "template_action(template = file, output = file, substitutions = {})",
          "runfiles()",
          "resolve_command(command = 'foo')",
          "resolve_tools()");

  @Test
  public void testFrozenRuleContextHasInaccessibleAttributes() throws Exception {
    scratch.file("test/BUILD",
        "load('//test:rules.bzl', 'main_rule', 'dep_rule')",
        "dep_rule(name = 'dep')",
        "main_rule(name = 'main', deps = [':dep'])");
    scratch.file("test/rules.bzl");

    for (String attribute : ctxAttributes) {
      scratch.overwriteFile("test/rules.bzl",
          "def _main_impl(ctx):",
          "  dep = ctx.attr.deps[0]",
          "  file = ctx.outputs.file",
          "  foo = dep.dep_ctx." + attribute,
          "main_rule = rule(",
          "  implementation = _main_impl,",
          "  attrs = {",
          "    'deps': attr.label_list()",
          "  },",
          "  outputs = {'file': 'output.txt'},",
          ")",
          "def _dep_impl(ctx):",
          "  return struct(dep_ctx = ctx)",
          "dep_rule = rule(implementation = _dep_impl)");
      invalidatePackages();
      try {
        getConfiguredTarget("//test:main");
        fail("Should have been unable to access dep_ctx." + attribute);
      } catch (AssertionError e) {
        assertThat(e)
            .hasMessageThat()
            .contains(
                "cannot access field or method '"
                    + Iterables.get(Splitter.on('(').split(attribute), 0)
                    + "' of rule context for '//test:dep' outside of its own rule implementation "
                    + "function");
      }
    }
  }

  @Test
  public void testFrozenRuleContextForAspectsHasInaccessibleAttributes() throws Exception {
    List<String> attributes = new ArrayList<>();
    attributes.addAll(ctxAttributes);
    attributes.addAll(ImmutableList.of(
        "rule.attr",
        "rule.executable",
        "rule.file",
        "rule.files",
        "rule.kind"));
    scratch.file("test/BUILD",
        "load('//test:rules.bzl', 'my_rule')",
        "my_rule(name = 'dep')",
        "my_rule(name = 'mid', deps = [':dep'])",
        "my_rule(name = 'main', deps = [':mid'])");
    scratch.file("test/rules.bzl");
    for (String attribute : attributes) {
       scratch.overwriteFile("test/rules.bzl",
          "def _rule_impl(ctx):",
          "  pass",
          "def _aspect_impl(target, ctx):",
          "  if ctx.rule.attr.deps:",
          "    dep = ctx.rule.attr.deps[0]",
          "    file = ctx.actions.declare_file('file.txt')",
          "    foo = dep." + (attribute.startsWith("rule.") ? "" : "ctx.") + attribute,
          "  return struct(ctx = ctx, rule=ctx.rule)",
          "MyAspect = aspect(implementation=_aspect_impl)",
          "my_rule = rule(",
          "  implementation = _rule_impl,",
          "  attrs = {",
          "    'deps': attr.label_list(aspects = [MyAspect])",
          "  },",
          ")");
      invalidatePackages();
      try {
        getConfiguredTarget("//test:main");
        fail("Should have been unable to access dep." + attribute);
      } catch (AssertionError e) {
        assertThat(e)
            .hasMessageThat()
            .contains(
                "cannot access field or method '"
                    + Iterables.get(Splitter.on('(').split(attribute), 0)
                    + "' of rule context for '//test:dep' outside of its own rule implementation "
                    + "function");
      }
    }
  }


  private static final List<String> deprecatedActionsApi =
      ImmutableList.of(
          "new_file('foo.txt')",
          "experimental_new_directory('foo.txt')",
          "new_file(file, 'foo.txt')",
          "action(command = 'foo', outputs = [file])",
          "file_action(file, 'foo')",
          "empty_action(mnemonic = 'foo', inputs = [file])",
          "template_action(template = file, output = file, substitutions = {})"
      );

  @Test
  public void testIncompatibleNewActionsApi() throws Exception {
    scratch.file("test/BUILD",
        "load('//test:rules.bzl', 'main_rule')",
        "main_rule(name = 'main')");
    scratch.file("test/rules.bzl");

    for (String actionApi : deprecatedActionsApi) {
      scratch.overwriteFile("test/rules.bzl",
          "def _main_impl(ctx):",
          "  file = ctx.outputs.file",
          "  foo = ctx." + actionApi,
          "main_rule = rule(",
          "  implementation = _main_impl,",
          "  attrs = {",
          "    'deps': attr.label_list()",
          "  },",
          "  outputs = {'file': 'output.txt'},",
          ")"
      );
      setSkylarkSemanticsOptions("--incompatible_new_actions_api=true");
      invalidatePackages();
      try {
        getConfiguredTarget("//test:main");
        fail("Should have reported deprecation error for: " + actionApi);
      } catch (AssertionError e) {
        assertWithMessage(actionApi + " reported wrong error").that(e)
            .hasMessageThat()
            .contains("Use --incompatible_new_actions_api=false");
      }
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testMapAttributeOrdering() throws Exception {
    scratch.file("a/a.bzl",
        "key_provider = provider(fields=['keys'])",
        "def _impl(ctx):",
        "  return [key_provider(keys=ctx.attr.value.keys())]",
        "a = rule(implementation=_impl, attrs={'value': attr.string_dict()})");
    scratch.file("a/BUILD",
        "load(':a.bzl', 'a')",
        "a(name='a', value={'c': 'c', 'b': 'b', 'a': 'a', 'f': 'f', 'e': 'e', 'd': 'd'})");

    ConfiguredTarget a = getConfiguredTarget("//a");
    SkylarkKey key =
        new SkylarkKey(Label.parseAbsolute("//a:a.bzl", ImmutableMap.of()), "key_provider");

    SkylarkInfo keyInfo = (SkylarkInfo) a.get(key);
    SkylarkList<String> keys = (SkylarkList<String>) keyInfo.getValue("keys");
    assertThat(keys).containsExactly("c", "b", "a", "f", "e", "d").inOrder();
  }

  private void writeIntFlagBuildSettingFiles() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=True");
    scratch.file(
        "test/build_setting.bzl",
        "BuildSettingInfo = provider(fields = ['name', 'value'])",
        "def _impl(ctx):",
        "  return [BuildSettingInfo(name = ctx.attr.name, value = ctx.build_setting_value)]",
        "",
        "int_flag = rule(",
        "  implementation = _impl,",
        "  build_setting = config.int(flag = True),",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:build_setting.bzl', 'int_flag')",
        "int_flag(name = 'int_flag', build_setting_default = 42)");
  }

  @Test
  public void testBuildSettingValue_explicitlySet() throws Exception {
    writeIntFlagBuildSettingFiles();
    useConfiguration(ImmutableMap.of("//test:int_flag", 24));

    ConfiguredTarget buildSetting = getConfiguredTarget("//test:int_flag");
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.create(buildSetting.getLabel().getPackageIdentifier(), "build_setting.bzl"),
            "BuildSettingInfo");
    StructImpl buildSettingInfo = (StructImpl) buildSetting.get(key);

    assertThat(buildSettingInfo.getValue("value")).isEqualTo(24);
  }

  @Test
  public void testBuildSettingValue_defaultFallback() throws Exception {
    writeIntFlagBuildSettingFiles();

    ConfiguredTarget buildSetting = getConfiguredTarget("//test:int_flag");
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.create(buildSetting.getLabel().getPackageIdentifier(), "build_setting.bzl"),
            "BuildSettingInfo");
    StructImpl buildSettingInfo = (StructImpl) buildSetting.get(key);

    assertThat(buildSettingInfo.getValue("value")).isEqualTo(42);
  }

  @Test
  public void testBuildSettingValue_nonBuildSettingRule() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=True");
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  foo = ctx.build_setting_value",
        "  return []",
        "non_build_setting = rule(implementation = _impl)");
    scratch.file(
        "test/BUILD",
        "load('//test:rule.bzl', 'non_build_setting')",
        "non_build_setting(name = 'my_non_build_setting')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_non_build_setting");
    assertContainsEvent("attempting to access 'build_setting_value' of non-build setting "
        + "//test:my_non_build_setting");
  }
}
