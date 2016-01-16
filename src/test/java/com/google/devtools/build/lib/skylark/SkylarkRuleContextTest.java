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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.python.PythonSourcesProvider;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 * Tests for SkylarkRuleContext.
 */
@RunWith(JUnit4.class)
public class SkylarkRuleContextTest extends SkylarkTestCase {

  @Before
  public final void generateBuildFile() throws Exception {
    scratch.file(
        "foo/BUILD",
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
        "java_import(name = 'asr',",
        "  jars = [ 'asr.jar' ],",
        "  srcjar = 'asr-src.jar',",
        ")",
        "genrule(name = 'gl',",
        "  cmd = 'touch $(OUTS)',",
        "  srcs = ['a.go'],",
        "  outs = [ 'gl.a', 'gl.gcgox', ],",
        "  output_to_bindir = 1,",
        ")");
  }

  private void setUpAttributeErrorTest() throws Exception   {
    scratch.file("test/BUILD",
        "load('/test/macros', 'macro_native_rule', 'macro_skylark_rule', 'skylark_rule')",
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
      assertContainsEvent(
          "ERROR /workspace/test/BUILD:2:1: in deps attribute of cc_library rule //test:m_native: "
          + "java_library rule '//test:jlib' is misplaced here (expected ");
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
          + "//test:m_skylark: '//test:jlib' does not have mandatory provider 'some_provider'. "
          + "Since this rule was created by the macro 'macro_skylark_rule', the error might have "
          + "been caused by the macro implementation in /workspace/test/macros.bzl:12:36");
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
      assertContainsEvent("ERROR /workspace/test/BUILD:9:10: in deps attribute of "
          + "cc_library rule //test:cclib: java_library rule '//test:jlib' is misplaced here "
          + "(expected ");
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
          + "skylark_rule rule //test:skyrule: '//test:jlib' does not have mandatory provider "
          + "'some_provider'");
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
        "load('/test/macros', 'skylark_rule')",
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
        "load('/test/macros', 'macro_skylark_rule')",
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
        "load('/test/macros', 'macro_skylark_rule')",
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
        "load('/test/macros', 'macro_native_rule')",
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
    assertEquals(artifacts.size(), resultList.size());
    int i = 0;
    for (String artifact : artifacts) {
      assertEquals(artifact, ((Artifact) resultList.get(i++)).getFilename());
    }
  }

  @Test
  public void shouldGetPrerequisites() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.srcs");
    // Check for a known provider
    TransitiveInfoCollection tic1 = (TransitiveInfoCollection) ((SkylarkList) result).get(0);
    assertNotNull(tic1.getProvider(JavaSourceJarsProvider.class));
    // Check an unimplemented provider too
    assertNull(tic1.getProvider(PythonSourcesProvider.class));
  }

  @Test
  public void shouldGetPrerequisite() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:asr");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.srcjar");
    TransitiveInfoCollection tic = (TransitiveInfoCollection) result;
    assertThat(tic).isInstanceOf(FileConfiguredTarget.class);
    assertEquals("asr-src.jar", tic.getLabel().getName());
  }

  @Test
  public void testMiddleMan() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:jl");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.middle_man(':host_jdk')");
    assertThat(
            Iterables.getOnlyElement(((SkylarkNestedSet) result).getSet(Artifact.class))
                .getExecPathString())
        .contains("middlemen");
  }

  @Test
  public void testGetRuleAttributeListType() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertThat(result).isInstanceOf(SkylarkList.class);
  }

  @Test
  public void testGetRule() throws Exception {
    scratch.file("test/skylark/BUILD");
    scratch.file(
        "test/skylark/rulestr.bzl",
        "def rule_dict(name):",
        "  return native.rule(name)",
        "def rules_dict():",
        "  return native.rules()",
        "def nop(ctx):",
        "  pass",
        "nop_rule = rule(attrs = {'x': attr.label()}, implementation = nop)",
        "consume_rule = rule(attrs = {'s': attr.string_list()}, implementation = nop)");

    scratch.file(
        "test/getrule/BUILD",
        "load('/test/skylark/rulestr', 'rules_dict', 'rule_dict', 'nop_rule', 'consume_rule')",
        "genrule(name = 'a', outs = ['a.txt'], tools = [ '//test:bla' ], cmd = 'touch $@')",
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
    Object result = evalRuleContextCode(allContext, "ruleContext.attr.s");
    assertEquals(
        new SkylarkList.MutableList(ImmutableList.<String>of("genrule", "a", "nop_rule", "c")),
        result);

    result = evalRuleContextCode(createRuleContext("//test/getrule:a_str"), "ruleContext.attr.s");
    assertEquals(
        new SkylarkList.MutableList(
            ImmutableList.<String>of("genrule", "a", ":a.txt", "//test:bla")),
        result);

    result = evalRuleContextCode(createRuleContext("//test/getrule:c_str"), "ruleContext.attr.s");
    assertEquals(
        new SkylarkList.MutableList(ImmutableList.<String>of("nop_rule", "c", ":a")), result);

    result =
        evalRuleContextCode(createRuleContext("//test/getrule:genrule_attr"), "ruleContext.attr.s");
    assertEquals(
        new SkylarkList.MutableList(
            ImmutableList.<String>of(
                "cmd",
                "compatible_with",
                "features",
                "generator_function",
                "generator_location",
                "generator_name",
                "kind",
                "message",
                "name",
                "outs",
                "restricted_to",
                "srcs",
                "tags",
                "tools",
                "visibility")),
        result);
  }

  @Test
  public void testGetRuleAttributeListValue() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertEquals(1, ((SkylarkList) result).size());
  }

  @Test
  public void testGetRuleAttributeListValueNoGet() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertEquals(1, ((SkylarkList) result).size());
  }

  @Test
  public void testGetRuleAttributeStringTypeValue() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.cmd");
    assertEquals("dummy_cmd", (String) result);
  }

  @Test
  public void testGetRuleAttributeStringTypeValueNoGet() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.cmd");
    assertEquals("dummy_cmd", (String) result);
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
    assertEquals("//foo:foo", ((Label) result).toString());
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
    SkylarkRuleContext ruleContext = createRuleContext("//foo:jl");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.executable._ijar");
    assertEquals("ijar", ((Artifact) result).getFilename());
  }

  @Test
  public void testCreateSpawnActionArgumentsWithExecutableFilesToRunProvider() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:jl");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.action(\n"
            + "  inputs = ruleContext.files.srcs,\n"
            + "  outputs = ruleContext.files.srcs,\n"
            + "  arguments = ['--a','--b'],\n"
            + "  executable = ruleContext.executable._ijar)\n");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getCommandFilename()).endsWith("/ijar");
  }

  @Test
  public void testOutputs() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Iterable<?> result = (Iterable<?>) evalRuleContextCode(ruleContext, "ruleContext.outputs.outs");
    assertEquals("d.txt", ((Artifact) Iterables.getOnlyElement(result)).getFilename());
  }

  @Test
  public void testSkylarkRuleContextStr() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "'%s' % ruleContext");
    assertEquals("//foo:foo", result);
  }

  @Test
  public void testSkylarkRuleContextGetDefaultShellEnv() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.configuration.default_shell_env");
    assertThat(result).isInstanceOf(ImmutableMap.class);
  }

  @Test
  public void testCheckPlaceholders() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.check_placeholders('%{name}', ['name'])");
    assertEquals(true, result);
  }

  @Test
  public void testCheckPlaceholdersBadPlaceholder() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.check_placeholders('%{name}', ['abc'])");
    assertEquals(false, result);
  }

  @Test
  public void testExpandMakeVariables() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext, "ruleContext.expand_make_variables('cmd', '$(ABC)', {'ABC': 'DEF'})");
    assertEquals("DEF", result);
  }

  @Test
  public void testExpandMakeVariablesShell() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.expand_make_variables('cmd', '$$ABC', {})");
    assertEquals("$ABC", result);
  }

  @Test
  public void testConfiguration() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.configuration");
    assertSame(result, ruleContext.getRuleContext().getConfiguration());
  }

  @Test
  public void testHostConfiguration() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.host_configuration");
    assertSame(result, ruleContext.getRuleContext().getHostConfiguration());
  }

  @Test
  public void testWorkspaceName() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.workspace_name");
    assertSame(result, TestConstants.WORKSPACE_NAME);
  }

  @Test
  public void testDeriveArtifactLegacy() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ruleContext.new_file(ruleContext.configuration.genfiles_dir," + "  'a/b.txt')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertEquals("foo/a/b.txt", fragment.getPathString());
  }

  @Test
  public void testDeriveArtifact() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.new_file('a/b.txt')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertEquals("foo/a/b.txt", fragment.getPathString());
  }

  @Test
  public void testParamFileLegacy() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ruleContext.new_file(ruleContext.configuration.bin_dir,"
                + "ruleContext.files.tools[0], '.params')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertEquals("foo/t.exe.params", fragment.getPathString());
  }

  @Test
  public void testParamFileSuffix() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(
        ruleContext,
        "ruleContext.new_file(ruleContext.files.tools[0], "
        + "ruleContext.files.tools[0].basename + '.params')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertEquals("foo/t.exe.params", fragment.getPathString());
  }

  @Test
  public void testRelativeLabelInExternalRepository() throws Exception {
    scratch.file("BUILD");
    scratch.file("external_rule.bzl",
        "def _impl(ctx):",
        "  return",
        "external_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'internal_dep': attr.label(default = Label('//:dep'))",
        "  }",
        ")");

    scratch.file("/r/BUILD",
        "filegroup(name='dep')");

    scratch.file("/r/a/BUILD",
        "load('/external_rule', 'external_rule')",
        "external_rule(name='r')");

    scratch.overwriteFile("WORKSPACE",
        "local_repository(name='r', path='/r')");

    invalidatePackages();
    SkylarkRuleContext context = createRuleContext("@r//a:r");
    Label depLabel = (Label) evalRuleContextCode(context, "ruleContext.attr.internal_dep.label");
    assertThat(depLabel).isEqualTo(Label.parseAbsolute("@r//:dep"));
  }
}
