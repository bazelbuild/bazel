// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.python.PythonSourcesProvider;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/**
 * Tests for SkylarkRuleContext.
 */
public class SkylarkRuleContextTest extends SkylarkTestCase {

  @Override
  public void setUp() throws Exception {
    super.setUp();
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

  public void testNativeRuleAttributeErrorWithMacro() throws Exception {
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
  public void testSkylarkRuleAttributeErrorWithMacro() throws Exception {
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
  public void testNativeRuleAttributeErrorWithoutMacro() throws Exception {
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

  public void testSkylarkRuleAttributeErrorWithoutMacro() throws Exception {
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

  public void testGetPrerequisiteArtifacts() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.files.srcs");
    assertArtifactList(result, ImmutableList.of("a.txt", "b.img"));
  }

  public void disabledTestGetPrerequisiteArtifact() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.file.tools");
    assertEquals("t.exe", ((Artifact) result).getFilename());
  }

  public void disabledTestGetPrerequisiteArtifactNoFile() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo2");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.file.srcs");
    assertSame(Runtime.NONE, result);
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

  public void testGetPrerequisites() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.srcs");
    // Check for a known provider
    TransitiveInfoCollection tic1 = (TransitiveInfoCollection) ((SkylarkList) result).get(0);
    assertNotNull(tic1.getProvider(JavaSourceJarsProvider.class));
    // Check an unimplemented provider too
    assertNull(tic1.getProvider(PythonSourcesProvider.class));
  }

  public void testGetPrerequisite() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:asr");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.srcjar");
    TransitiveInfoCollection tic = (TransitiveInfoCollection) result;
    assertThat(tic).isInstanceOf(FileConfiguredTarget.class);
    assertEquals("asr-src.jar", tic.getLabel().getName());
  }

  public void testMiddleMan() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:jl");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.middle_man(':host_jdk')");
    assertThat(
            Iterables.getOnlyElement(((SkylarkNestedSet) result).getSet(Artifact.class))
                .getExecPathString())
        .contains("middlemen");
  }

  public void testGetRuleAttributeListType() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertThat(result).isInstanceOf(SkylarkList.class);
  }

  public void testGetRuleAttributeListValue() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertEquals(1, ((SkylarkList) result).size());
  }

  public void testGetRuleAttributeListValueNoGet() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.outs");
    assertEquals(1, ((SkylarkList) result).size());
  }

  public void testGetRuleAttributeStringTypeValue() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.cmd");
    assertEquals("dummy_cmd", (String) result);
  }

  public void testGetRuleAttributeStringTypeValueNoGet() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.attr.cmd");
    assertEquals("dummy_cmd", (String) result);
  }

  public void testGetRuleAttributeBadAttributeName() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"), "No attribute 'bad'", "ruleContext.attr.bad");
  }

  public void testGetLabel() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.label");
    assertEquals("//foo:foo", ((Label) result).toString());
  }

  public void testRuleError() throws Exception {
    checkErrorContains(createRuleContext("//foo:foo"), "message", "fail('message')");
  }

  public void testAttributeError() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"),
        "attribute srcs: message",
        "fail(attr='srcs', msg='message')");
  }

  public void testGetExecutablePrerequisite() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:jl");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.executable._ijar");
    assertEquals("ijar", ((Artifact) result).getFilename());
  }

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

  public void testOutputs() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Iterable<?> result = (Iterable<?>) evalRuleContextCode(ruleContext, "ruleContext.outputs.outs");
    assertEquals("d.txt", ((Artifact) Iterables.getOnlyElement(result)).getFilename());
  }

  public void testSkylarkRuleContextStr() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "'%s' % ruleContext");
    assertEquals("//foo:foo", result);
  }

  public void testSkylarkRuleContextGetDefaultShellEnv() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.configuration.default_shell_env");
    assertThat(result).isInstanceOf(ImmutableMap.class);
  }

  public void testCheckPlaceholders() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.check_placeholders('%{name}', ['name'])");
    assertEquals(true, result);
  }

  public void testCheckPlaceholdersBadPlaceholder() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.check_placeholders('%{name}', ['abc'])");
    assertEquals(false, result);
  }

  public void testExpandMakeVariables() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext, "ruleContext.expand_make_variables('cmd', '$(ABC)', {'ABC': 'DEF'})");
    assertEquals("DEF", result);
  }

  public void testExpandMakeVariablesShell() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.expand_make_variables('cmd', '$$ABC', {})");
    assertEquals("$ABC", result);
  }

  public void testConfiguration() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.configuration");
    assertSame(result, ruleContext.getRuleContext().getConfiguration());
  }

  public void testHostConfiguration() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.host_configuration");
    assertSame(result, ruleContext.getRuleContext().getHostConfiguration());
  }

  public void testWorkspaceName() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.workspace_name");
    assertSame(result, TestConstants.WORKSPACE_NAME);
  }

  public void testDeriveArtifactLegacy() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ruleContext.new_file(ruleContext.configuration.genfiles_dir," + "  'a/b.txt')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertEquals("foo/a/b.txt", fragment.getPathString());
  }

  public void testDeriveArtifact() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(ruleContext, "ruleContext.new_file('a/b.txt')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertEquals("foo/a/b.txt", fragment.getPathString());
  }

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

  public void testParamFileSuffix() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result = evalRuleContextCode(
        ruleContext,
        "ruleContext.new_file(ruleContext.files.tools[0], "
        + "ruleContext.files.tools[0].basename + '.params')");
    PathFragment fragment = ((Artifact) result).getRootRelativePath();
    assertEquals("foo/t.exe.params", fragment.getPathString());
  }
}
