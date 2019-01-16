// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpanderImpl;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.skylark.SkylarkActionFactory;
import com.google.devtools.build.lib.analysis.skylark.SkylarkActionFactory.Args;
import com.google.devtools.build.lib.analysis.skylark.SkylarkCustomCommandLine;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OsUtils;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for skylark functions relating to rule implemenetation. */
@RunWith(JUnit4.class)
public class SkylarkRuleImplementationFunctionsTest extends SkylarkTestCase {
  @Rule public ExpectedException thrown = ExpectedException.none();

  @SkylarkSignature(
    name = "mock",
    documented = false,
    parameters = {
      @Param(name = "mandatory", doc = ""),
      @Param(name = "optional", doc = "", defaultValue = "None"),
      @Param(name = "mandatory_key", doc = "", positional = false, named = true),
      @Param(
        name = "optional_key",
        doc = "",
        defaultValue = "'x'",
        positional = false,
        named = true
      )
    },
    useEnvironment = true
  )
  private BuiltinFunction mockFunc;

  /**
   * Used for {@link #testStackTraceWithoutOriginalMessage()} and {@link
   * #testNoStackTraceOnInterrupt}.
   */
  @SkylarkSignature(name = "throw", documented = false)
  BuiltinFunction throwFunction;

  @Before
  public final void createBuildFile() throws Exception {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',",
        "  cmd = 'dummy_cmd',",
        "  srcs = ['a.txt', 'b.img'],",
        "  tools = ['t.exe'],",
        "  outs = ['c.txt'])",
        "genrule(name = 'bar',",
        "  cmd = 'dummy_cmd',",
        "  srcs = [':jl', ':gl'],",
        "  outs = ['d.txt'])",
        "genrule(name = 'baz',",
        "  cmd = 'dummy_cmd',",
        "  outs = ['e.txt'])",
        "java_library(name = 'jl',",
        "  srcs = ['a.java'])",
        "genrule(name = 'gl',",
        "  cmd = 'touch $(OUTS)',",
        "  srcs = ['a.go'],",
        "  outs = [ 'gl.a', 'gl.gcgox', ],",
        "  output_to_bindir = 1,",
        ")",
        // The taraget below is used by testResolveCommand and testResolveTools
        "sh_binary(name = 'mytool',",
        "  srcs = ['mytool.sh'],",
        "  data = ['file1.dat', 'file2.dat'],",
        ")",
        // The taraget below is used by testResolveCommand and testResolveTools
        "genrule(name = 'resolve_me',",
        "  cmd = 'aa',",
        "  tools = [':mytool', 't.exe'],",
        "  srcs = ['file3.dat', 'file4.dat'],",
        "  outs = ['r1.txt', 'r2.txt'],",
        ")");
  }

  private void setupSkylarkFunction(String line) throws Exception {
    mockFunc =
        new BuiltinFunction("mock") {
          @SuppressWarnings("unused")
          public Object invoke(
              Object mandatory,
              Object optional,
              Object mandatoryKey,
              Object optionalKey,
              Environment env) {
            return EvalUtils.optionMap(
                env,
                "mandatory",
                mandatory,
                "optional",
                optional,
                "mandatory_key",
                mandatoryKey,
                "optional_key",
                optionalKey);
          }
        };
    assertThat(mockFunc.isConfigured()).isFalse();
    mockFunc.configure(
        SkylarkRuleImplementationFunctionsTest.class
            .getDeclaredField("mockFunc")
            .getAnnotation(SkylarkSignature.class));
    update("mock", mockFunc);
    eval(line);
  }

  private void checkSkylarkFunctionError(String errorMsg, String line) throws Exception {
    try {
      setupSkylarkFunction(line);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().isEqualTo(errorMsg);
    }
  }

  @Test
  public void testSkylarkFunctionPosArgs() throws Exception {
    setupSkylarkFunction("a = mock('a', 'b', mandatory_key='c')");
    Map<?, ?> params = (Map<?, ?>) lookup("a");
    assertThat(params.get("mandatory")).isEqualTo("a");
    assertThat(params.get("optional")).isEqualTo("b");
    assertThat(params.get("mandatory_key")).isEqualTo("c");
    assertThat(params.get("optional_key")).isEqualTo("x");
  }

  @Test
  public void testSkylarkFunctionKwArgs() throws Exception {
    setupSkylarkFunction("a = mock(optional='b', mandatory='a', mandatory_key='c')");
    Map<?, ?> params = (Map<?, ?>) lookup("a");
    assertThat(params.get("mandatory")).isEqualTo("a");
    assertThat(params.get("optional")).isEqualTo("b");
    assertThat(params.get("mandatory_key")).isEqualTo("c");
    assertThat(params.get("optional_key")).isEqualTo("x");
  }

  @Test
  public void testSkylarkFunctionTooFewArguments() throws Exception {
    checkSkylarkFunctionError(
        "insufficient arguments received by mock("
            + "mandatory, optional = None, *, mandatory_key, optional_key = \"x\") "
            + "(got 0, expected at least 1)",
        "mock()");
  }

  @Test
  public void testSkylarkFunctionTooManyArguments() throws Exception {
    checkSkylarkFunctionError(
        "too many (3) positional arguments in call to "
            + "mock(mandatory, optional = None, *, mandatory_key, optional_key = \"x\")",
        "mock('a', 'b', 'c')");
  }

  @Test
  public void testSkylarkFunctionAmbiguousArguments() throws Exception {
    checkSkylarkFunctionError(
        "argument 'mandatory' passed both by position and by name "
            + "in call to mock(mandatory, optional = None, *, mandatory_key, optional_key = \"x\")",
        "mock('by position', mandatory='by_key', mandatory_key='c')");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensionsWithNestedSet() throws Exception {
    Object result = eval("[x + x for x in depset([1, 2, 3])]");
    assertThat((Iterable<Object>) result).containsExactly(2, 4, 6).inOrder();
  }

  @Test
  public void testCreateSpawnActionCreatesSpawnAction() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    createTestSpawnAction(ruleContext);
    ActionAnalysisMetadata action =
        Iterables.getOnlyElement(
            ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action).isInstanceOf(SpawnAction.class);
  }

  @Test
  public void testCreateSpawnActionArgumentsWithCommand() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    createTestSpawnAction(ruleContext);
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertArtifactFilenames(action.getInputs(), "a.txt", "b.img");
    assertArtifactFilenames(action.getOutputs(), "a.txt", "b.img");
    MoreAsserts.assertContainsSublist(
        action.getArguments(), "-c", "dummy_command", "", "--a", "--b");
    assertThat(action.getMnemonic()).isEqualTo("DummyMnemonic");
    assertThat(action.getProgressMessage()).isEqualTo("dummy_message");
    assertThat(action.getIncompleteEnvironmentForTesting()).isEqualTo(targetConfig.getLocalShellEnvironment());
  }

  @Test
  public void testCreateSpawnActionArgumentsWithExecutable() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.run(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  executable = ruleContext.files.tools[0])");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertArtifactFilenames(action.getInputs(), "a.txt", "b.img", "t.exe");
    assertArtifactFilenames(action.getOutputs(), "a.txt", "b.img");
    MoreAsserts.assertContainsSublist(action.getArguments(), "foo/t.exe", "--a", "--b");
  }

  @Test
  public void testCreateActionWithDepsetInput() throws Exception {
    // Same test as above, with depset as inputs.
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  executable = ruleContext.files.tools[0])");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertArtifactFilenames(action.getInputs(), "a.txt", "b.img", "t.exe");
    assertArtifactFilenames(action.getOutputs(), "a.txt", "b.img");
    MoreAsserts.assertContainsSublist(action.getArguments(), "foo/t.exe", "--a", "--b");
  }

  @Test
  public void testCreateSpawnActionArgumentsBadExecutable() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"),
        "expected value of type 'File or string' for parameter 'executable', "
            + "for call to method run(",
        "ruleContext.actions.run(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  executable = 123)");
  }

  @Test
  public void testCreateSpawnActionShellCommandList() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.run_shell(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  mnemonic = 'DummyMnemonic',",
        "  command = ['dummy_command', '--arg1', '--arg2'],",
        "  progress_message = 'dummy_message')");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly("dummy_command", "--arg1", "--arg2")
        .inOrder();
  }

  @Test
  public void testCreateSpawnActionEnvAndExecInfo() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.run_shell(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  env = {'a' : 'b'},",
        "  execution_requirements = {'timeout' : '10', 'block-network' : 'foo'},",
        "  mnemonic = 'DummyMnemonic',",
        "  command = 'dummy_command',",
        "  progress_message = 'dummy_message')");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getIncompleteEnvironmentForTesting()).containsExactly("a", "b");
    // We expect "timeout" to be filtered by TargetUtils.
    assertThat(action.getExecutionInfo()).containsExactly("block-network", "foo");
  }

  @Test
  public void testCreateSpawnActionUnknownParam() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "unexpected keyword 'bad_param', for call to method run(",
        "f = ruleContext.actions.declare_file('foo.sh')",
        "ruleContext.actions.run(outputs=[], bad_param = 'some text', executable = f)");
  }

  @Test
  public void testCreateSpawnActionNoExecutable() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "You must specify either 'command' or 'executable' argument",
        "ruleContext.action(outputs=[])");
  }

  private Object createTestSpawnAction(SkylarkRuleContext ruleContext) throws Exception {
    return evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.run_shell(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  mnemonic = 'DummyMnemonic',",
        "  command = 'dummy_command',",
        "  progress_message = 'dummy_message',",
        "  use_default_shell_env = True)");
  }

  @Test
  public void testCreateSpawnActionBadGenericArg() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"),
        "expected type 'File' for 'outputs' element but got type 'string' instead",
        "l = ['a', 'b']",
        "ruleContext.actions.run_shell(",
        "  outputs = l,",
        "  command = 'dummy_command')");
  }

  @Test
  public void testRunShellArgumentsWithCommandSequence() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"),
        "'arguments' must be empty if 'command' is a sequence of strings",
        "ruleContext.actions.run_shell(outputs = ruleContext.files.srcs,",
        "  command = [\"echo\", \"'hello world'\", \"&&\", \"touch\"],",
        "  arguments = [ruleContext.files.srcs[0].path])");
  }

  private void setupToolInInputsTest(String... ruleImpl) throws Exception {
    ImmutableList.Builder<String> lines = ImmutableList.builder();
    lines.add("def _main_rule_impl(ctx):");
    for (String line : ruleImpl) {
      lines.add("  " + line);
    }
    lines.add(
        "my_rule = rule(",
        "  _main_rule_impl,",
        "  attrs = { ",
        "    'exe' : attr.label(executable = True, allow_files = True, cfg='host'),",
        "  },",
        ")");
    scratch.file("bar/bar.bzl", lines.build().toArray(new String[] {}));
    scratch.file(
        "bar/BUILD",
        "load('//bar:bar.bzl', 'my_rule')",
        "sh_binary(",
        "  name = 'mytool',",
        "  srcs = ['mytool.sh'],",
        "  data = ['file1.dat', 'file2.dat'],",
        ")",
        "my_rule(",
        "  name = 'my_rule',",
        "  exe = ':mytool',",
        ")");
  }

  @Test
  public void testCreateSpawnActionWithToolInInputsLegacy() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=false");
    setupToolInInputsTest(
        "output = ctx.actions.declare_file('bar.out')",
        "ctx.actions.run_shell(",
        "  inputs = ctx.attr.exe.files,",
        "  outputs = [output],",
        "  command = 'boo bar baz',",
        ")");
    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//bar:my_rule");
    SpawnAction action = (SpawnAction) Iterables.getOnlyElement(target.getActions());
    assertThat(action.getTools()).isNotEmpty();
  }

  @Test
  public void testCreateSpawnActionWithToolAttribute() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=true");
    setupToolInInputsTest(
        "output = ctx.actions.declare_file('bar.out')",
        "ctx.actions.run_shell(",
        "  inputs = [],",
        "  tools = ctx.attr.exe.files,",
        "  outputs = [output],",
        "  command = 'boo bar baz',",
        ")");
    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//bar:my_rule");
    SpawnAction action = (SpawnAction) Iterables.getOnlyElement(target.getActions());
    assertThat(action.getTools()).isNotEmpty();
  }

  @Test
  public void testCreateSpawnActionWithToolAttributeIgnoresToolsInInputs() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=true");
    setupToolInInputsTest(
        "output = ctx.actions.declare_file('bar.out')",
        "ctx.actions.run_shell(",
        "  inputs = ctx.attr.exe.files,",
        "  tools = ctx.attr.exe.files,",
        "  outputs = [output],",
        "  command = 'boo bar baz',",
        ")");
    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//bar:my_rule");
    SpawnAction action = (SpawnAction) Iterables.getOnlyElement(target.getActions());
    assertThat(action.getTools()).isNotEmpty();
  }

  @Test
  public void testCreateSpawnActionWithToolInInputsFailAtAnalysisTime() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=true");
    setupToolInInputsTest(
        "output = ctx.actions.declare_file('bar.out')",
        "ctx.actions.run_shell(",
        "  inputs = ctx.attr.exe.files,",
        "  outputs = [output],",
        "  command = 'boo bar baz',",
        ")");
    try {
      getConfiguredTarget("//bar:my_rule");
    } catch (Throwable t) {
      // Expected
    }
    assertThat(eventCollector).hasSize(1);
    assertThat(eventCollector.iterator().next().getMessage())
        .containsMatch("Found tool\\(s\\) '.*' in inputs");
  }

  @Test
  public void testCreateFileAction() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.write(",
        "  output = ruleContext.files.srcs[0],",
        "  content = 'hello world',",
        "  is_executable = False)");
    FileWriteAction action =
        (FileWriteAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(Iterables.getOnlyElement(action.getOutputs()).getExecPathString())
        .isEqualTo("foo/a.txt");
    assertThat(action.getFileContents()).isEqualTo("hello world");
    assertThat(action.makeExecutable()).isFalse();
  }

  @Test
  public void testEmptyAction() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");

    checkEmptyAction(ruleContext, "mnemonic = 'test'");
    checkEmptyAction(ruleContext, "mnemonic = 'test', inputs = ruleContext.files.srcs");
    checkEmptyAction(ruleContext, "mnemonic = 'test', inputs = depset(ruleContext.files.srcs)");

    checkErrorContains(
        ruleContext,
        "parameter 'mnemonic' has no default value, for call to method "
            + "do_nothing(mnemonic, inputs = []) of 'actions'",
        "ruleContext.actions.do_nothing(inputs = ruleContext.files.srcs)");
  }

  private void checkEmptyAction(SkylarkRuleContext ruleContext, String namedArgs) throws Exception {
    assertThat(
            evalRuleContextCode(
                ruleContext, String.format("ruleContext.actions.do_nothing(%s)", namedArgs)))
        .isEqualTo(Runtime.NONE);
  }

  @Test
  public void testEmptyActionWithExtraAction() throws Exception {
    scratch.file(
        "test/empty.bzl",
        "def _impl(ctx):",
        "  ctx.actions.do_nothing(",
        "      inputs = ctx.files.srcs,",
        "      mnemonic = 'EA',",
        "  )",
        "empty_action_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       \"srcs\": attr.label_list(allow_files=True),",
        "    }",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:empty.bzl', 'empty_action_rule')",
        "empty_action_rule(name = 'my_empty_action',",
        "                srcs = ['foo.in', 'other_foo.in'])",
        "action_listener(name = 'listener',",
        "                mnemonics = ['EA'],",
        "                extra_actions = [':extra'])",
        "extra_action(name = 'extra',",
        "             cmd='')");

    getPseudoActionViaExtraAction("//test:my_empty_action", "//test:listener");
  }

  @Test
  public void testExpandLocation() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");

    // If there is only a single target, both "location" and "locations" should work
    runExpansion(ruleContext, "location :jl", "[blaze]*-out/.*/bin/foo/libjl.jar");
    runExpansion(ruleContext, "locations :jl", "[blaze]*-out/.*/bin/foo/libjl.jar");

    runExpansion(ruleContext, "location //foo:jl", "[blaze]*-out/.*/bin/foo/libjl.jar");

    // Multiple targets and "location" should result in an error
    checkReportedErrorStartsWith(
        ruleContext,
        "in genrule rule //foo:bar: label '//foo:gl' "
            + "in $(location) expression expands to more than one file, please use $(locations "
            + "//foo:gl) instead.",
        "ruleContext.expand_location('$(location :gl)')");

    // We have to use "locations" for multiple targets
    runExpansion(
        ruleContext,
        "locations :gl",
        "[blaze]*-out/.*/bin/foo/gl.a [blaze]*-out/.*/bin/foo/gl.gcgox");

    // LocationExpander just returns the input string if there is no label
    runExpansion(ruleContext, "location", "\\$\\(location\\)");

    checkReportedErrorStartsWith(
        ruleContext,
        "in genrule rule //foo:bar: label '//foo:abc' in $(locations) expression "
            + "is not a declared prerequisite of this rule",
        "ruleContext.expand_location('$(locations :abc)')");
  }

  /** Regression test to check that expand_location allows ${var} and $$. */
  @Test
  public void testExpandLocationWithDollarSignsAndCurlys() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    assertThat((String)
        evalRuleContextCode(
            ruleContext, "ruleContext.expand_location('${abc} $(echo) $$ $')"))
        .isEqualTo("${abc} $(echo) $$ $");
  }

  /**
   * Invokes ctx.expand_location() with the given parameters and checks whether this led to the
   * expected result
   * @param ruleContext The rule context
   * @param command Either "location" or "locations". This only matters when the label has multiple
   * targets
   * @param expectedPattern Regex pattern that matches the expected result
   */
  private void runExpansion(SkylarkRuleContext ruleContext, String command, String expectedPattern)
      throws Exception {
    assertMatches(
        "Expanded string",
        expectedPattern,
        (String)
            evalRuleContextCode(
                ruleContext, String.format("ruleContext.expand_location('$(%s)')", command)));
  }

  private void assertMatches(String description, String expectedPattern, String computedValue)
      throws Exception {
    assertWithMessage(
            Printer.format(
                "%s %r did not match pattern '%s'", description, computedValue, expectedPattern))
        .that(Pattern.matches(expectedPattern, computedValue))
        .isTrue();
  }

  @Test
  public void testResolveCommandMakeVariables() throws Exception {
    evalRuleContextCode(
        createRuleContext("//foo:resolve_me"),
        "inputs, argv, manifests = ruleContext.resolve_command(",
        "  command='I got the $(HELLO) on a $(DAVE)', ",
        "  make_variables={'HELLO': 'World', 'DAVE': type('')})");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (MutableList) lookup("argv");
    assertThat(argv).hasSize(3);
    assertMatches("argv[0]", "^.*/bash" + OsUtils.executableExtension() + "$", argv.get(0));
    assertThat(argv.get(1)).isEqualTo("-c");
    assertThat(argv.get(2)).isEqualTo("I got the World on a string");
  }

  @Test
  public void testResolveCommandInputs() throws Exception {
    evalRuleContextCode(
        createRuleContext("//foo:resolve_me"),
        "inputs, argv, input_manifests = ruleContext.resolve_command(",
        "   tools=ruleContext.attr.tools)");
    @SuppressWarnings("unchecked")
    List<Artifact> inputs = (List<Artifact>) (List<?>) (MutableList) lookup("inputs");
    assertArtifactFilenames(
        inputs,
        "mytool.sh",
        "mytool",
        "foo_Smytool" + OsUtils.executableExtension() + "-runfiles",
        "t.exe");
    @SuppressWarnings("unchecked")
    CompositeRunfilesSupplier runfilesSupplier =
        new CompositeRunfilesSupplier((List<RunfilesSupplier>) lookup("input_manifests"));
    assertThat(runfilesSupplier.getMappings(ArtifactPathResolver.IDENTITY)).hasSize(1);
  }

  @Test
  public void testResolveCommandExpandLocations() throws Exception {
    evalRuleContextCode(
        createRuleContext("//foo:resolve_me"),
        "def foo():", // no for loops at top-level
        "  label_dict = {}",
        "  all = []",
        "  for dep in ruleContext.attr.srcs + ruleContext.attr.tools:",
        "    all.extend(list(dep.files))",
        "    label_dict[dep.label] = list(dep.files)",
        "  return ruleContext.resolve_command(",
        "    command='A$(locations //foo:mytool) B$(location //foo:file3.dat)',",
        "    attribute='cmd', expand_locations=True, label_dict=label_dict)",
        "inputs, argv, manifests = foo()");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (MutableList) lookup("argv");
    assertThat(argv).hasSize(3);
    assertMatches("argv[0]", "^.*/bash" + OsUtils.executableExtension() + "$", argv.get(0));
    assertThat(argv.get(1)).isEqualTo("-c");
    assertMatches("argv[2]", "A.*/mytool .*/mytool.sh B.*file3.dat", argv.get(2));
  }

  @Test
  public void testResolveCommandExecutionRequirements() throws Exception {
    // Tests that requires-darwin execution requirements result in the usage of /bin/bash.
    evalRuleContextCode(
        createRuleContext("//foo:resolve_me"),
        "inputs, argv, manifests = ruleContext.resolve_command(",
        "  execution_requirements={'requires-darwin': ''})");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (MutableList) lookup("argv");
    assertMatches("argv[0]", "^/bin/bash$", argv.get(0));
  }

  @Test
  public void testResolveCommandScript() throws Exception {
    evalRuleContextCode(
        createRuleContext("//foo:resolve_me"),
        "def foo():", // no for loops at top-level
        "  s = 'a'",
        "  for i in range(1,17): s = s + s", // 2**17 > CommandHelper.maxCommandLength (=64000)
        "  return ruleContext.resolve_command(",
        "    command=s)",
        "argv = foo()[1]");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (MutableList) lookup("argv");
    assertThat(argv).hasSize(2);
    assertMatches("argv[0]", "^.*/bash" + OsUtils.executableExtension() + "$", argv.get(0));
    assertMatches("argv[1]", "^.*/resolve_me[.][a-z0-9]+[.]script[.]sh$", argv.get(1));
  }

  @Test
  public void testResolveTools() throws Exception {
    evalRuleContextCode(
        createRuleContext("//foo:resolve_me"),
        "inputs, input_manifests = ruleContext.resolve_tools(",
        "   tools=ruleContext.attr.tools)");
    assertArtifactFilenames(
        ((SkylarkNestedSet) lookup("inputs")).getSet(Artifact.class),
        "mytool.sh",
        "mytool",
        "foo_Smytool" + OsUtils.executableExtension() + "-runfiles",
        "t.exe");
    @SuppressWarnings("unchecked")
    CompositeRunfilesSupplier runfilesSupplier =
        new CompositeRunfilesSupplier((List<RunfilesSupplier>) lookup("input_manifests"));
    assertThat(runfilesSupplier.getMappings(ArtifactPathResolver.IDENTITY)).hasSize(1);
  }

  @Test
  public void testBadParamTypeErrorMessage() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "expected value of type 'string or Args' for parameter 'content'",
        "ruleContext.actions.write(",
        "  output = ruleContext.files.srcs[0],",
        "  content = 1,",
        "  is_executable = False)");
  }

  @Test
  public void testCreateTemplateAction() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.expand_template(",
        "  template = ruleContext.files.srcs[0],",
        "  output = ruleContext.files.srcs[1],",
        "  substitutions = {'a': 'b'},",
        "  is_executable = False)");

    TemplateExpansionAction action = (TemplateExpansionAction) Iterables.getOnlyElement(
        ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(Iterables.getOnlyElement(action.getInputs()).getExecPathString())
        .isEqualTo("foo/a.txt");
    assertThat(Iterables.getOnlyElement(action.getOutputs()).getExecPathString())
        .isEqualTo("foo/b.img");
    assertThat(Iterables.getOnlyElement(action.getSubstitutions()).getKey()).isEqualTo("a");
    assertThat(Iterables.getOnlyElement(action.getSubstitutions()).getValue()).isEqualTo("b");
    assertThat(action.makeExecutable()).isFalse();
  }

  /**
   * Simulates the fact that the Parser currently uses Latin1 to read BUILD files, while users
   * usually write those files using UTF-8 encoding. Currently, the string-valued 'substitutions'
   * parameter of the template_action function contains a hack that assumes its input is a UTF-8
   * encoded string which has been ingested as Latin 1. The hack converts the string to its
   * "correct" UTF-8 value. Once {@link
   * com.google.devtools.build.lib.syntax.ParserInputSource#create(byte[],
   * com.google.devtools.build.lib.vfs.PathFragment)} parses files using UTF-8 and the hack for the
   * substituations parameter is removed, this test will fail.
   */
  @Test
  public void testCreateTemplateActionWithWrongEncoding() throws Exception {
    // The following array contains bytes that represent a string of length two when treated as
    // UTF-8 and a string of length four when treated as ISO-8859-1 (a.k.a. Latin 1).
    byte[] bytesToDecode = {(byte) 0xC2, (byte) 0xA2, (byte) 0xC2, (byte) 0xA2};
    Charset latin1 = StandardCharsets.ISO_8859_1;
    Charset utf8 = StandardCharsets.UTF_8;
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "ruleContext.actions.expand_template(",
        "  template = ruleContext.files.srcs[0],",
        "  output = ruleContext.files.srcs[1],",
        "  substitutions = {'a': '" + new String(bytesToDecode, latin1) + "'},",
        "  is_executable = False)");
    TemplateExpansionAction action = (TemplateExpansionAction) Iterables.getOnlyElement(
        ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    List<Substitution> substitutions = action.getSubstitutions();
    assertThat(substitutions).hasSize(1);
    assertThat(substitutions.get(0).getValue()).isEqualTo(new String(bytesToDecode, utf8));
  }

  @Test
  public void testRunfilesAddFromDependencies() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Object result =
        evalRuleContextCode(ruleContext, "ruleContext.runfiles(collect_default = True)");
    assertThat(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)))
        .contains("libjl.jar");
  }

  @Test
  public void testRunfilesBadListGenericType() throws Exception {
    checkErrorContains(
        "expected type 'File' for 'files' element but got type 'string' instead",
        "ruleContext.runfiles(files = ['some string'])");
  }

  @Test
  public void testRunfilesBadSetGenericType() throws Exception {
    checkErrorContains(
        "expected value of type 'depset of Files or NoneType' for parameter 'transitive_files', "
            + "for call to method runfiles(",
        "ruleContext.runfiles(transitive_files=depset([1, 2, 3]))");
  }

  @Test
  public void testRunfilesBadMapGenericType() throws Exception {
    checkErrorContains(
        "expected type 'string' for 'symlinks' key but got type 'int' instead",
        "ruleContext.runfiles(symlinks = {123: ruleContext.files.srcs[0]})");
    checkErrorContains(
        "expected type 'File' for 'symlinks' value but got type 'int' instead",
        "ruleContext.runfiles(symlinks = {'some string': 123})");
    checkErrorContains(
        "expected type 'string' for 'root_symlinks' key but got type 'int' instead",
        "ruleContext.runfiles(root_symlinks = {123: ruleContext.files.srcs[0]})");
    checkErrorContains(
        "expected type 'File' for 'root_symlinks' value but got type 'int' instead",
        "ruleContext.runfiles(root_symlinks = {'some string': 123})");
  }

  @Test
  public void testRunfilesArtifactsFromArtifact() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "artifacts = ruleContext.files.tools",
            "ruleContext.runfiles(files = artifacts)");
    assertThat(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result))).contains("t.exe");
  }

  @Test
  public void testRunfilesArtifactsFromIterableArtifacts() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "artifacts = ruleContext.files.srcs",
            "ruleContext.runfiles(files = artifacts)");
    assertThat(ImmutableList.of("a.txt", "b.img"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromNestedSetArtifacts() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "ftb = depset() + ruleContext.files.srcs",
            "ruleContext.runfiles(transitive_files = ftb)");
    assertThat(ImmutableList.of("a.txt", "b.img"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromDefaultAndFiles() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:bar");
    Object result =
        evalRuleContextCode(
            ruleContext,
            "artifacts = ruleContext.files.srcs",
            // It would be nice to write [DEFAULT] + artifacts, but artifacts
            // is an ImmutableList and Skylark interprets it as a tuple.
            "ruleContext.runfiles(collect_default = True, files = artifacts)");
    // From DEFAULT only libjl.jar comes, see testRunfilesAddFromDependencies().
    assertThat(ImmutableList.of("libjl.jar", "gl.a", "gl.gcgox"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromSymlink() throws Exception {
    Object result =
        evalRuleContextCode(
            "artifacts = ruleContext.files.srcs",
            "ruleContext.runfiles(symlinks = {'sym1': artifacts[0]})");
    assertThat(ImmutableList.of("a.txt"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromRootSymlink() throws Exception {
    Object result =
        evalRuleContextCode(
            "artifacts = ruleContext.files.srcs",
            "ruleContext.runfiles(root_symlinks = {'sym1': artifacts[0]})");
    assertThat(ImmutableList.of("a.txt"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesSymlinkConflict() throws Exception {
    // Two different artifacts mapped to same path in runfiles
    Object result =
        evalRuleContextCode(
            "artifacts = ruleContext.files.srcs",
            "prefix = ruleContext.workspace_name + '/' if ruleContext.workspace_name else ''",
            "ruleContext.runfiles(",
            "root_symlinks = {prefix + 'sym1': artifacts[0]},",
            "symlinks = {'sym1': artifacts[1]})");
    Runfiles runfiles = (Runfiles) result;
    reporter.removeHandler(failFastHandler); // So it doesn't throw an exception.
    runfiles.getRunfilesInputs(reporter, null, ArtifactPathResolver.IDENTITY);
    assertContainsEvent("ERROR <no location>: overwrote runfile");
  }

  private Iterable<Artifact> getRunfileArtifacts(Object runfiles) {
    return ((Runfiles) runfiles).getAllArtifacts();
  }

  @Test
  public void testRunfilesBadKeywordArguments() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "unexpected keyword 'bad_keyword', for call to method runfiles(",
        "ruleContext.runfiles(bad_keyword = '')");
  }

  @Test
  public void testNsetContainsList() throws Exception {
    checkErrorContains(
        "depsets cannot contain items of type 'list'", "depset() + [ruleContext.files.srcs]");
  }

  @Test
  public void testCmdJoinPaths() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(
            ruleContext, "f = depset(ruleContext.files.srcs)", "cmd_helper.join_paths(':', f)");
    assertThat(result).isEqualTo("foo/a.txt:foo/b.img");
  }

  @Test
  public void testStructPlusArtifactErrorMessage() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "unsupported operand type(s) for +: 'File' and 'struct'",
        "ruleContext.files.tools[0] + struct(a = 1)");
  }

  @Test
  public void testNoSuchProviderErrorMessage() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:bar"),
        "<target //foo:jl> (rule 'java_library') doesn't have provider 'my_provider'",
        "ruleContext.attr.srcs[0].my_provider");
  }

  @Test
  public void testFilesForRuleConfiguredTarget() throws Exception {
    Object result =
        evalRuleContextCode(createRuleContext("//foo:foo"), "ruleContext.attr.srcs[0].files");
    assertThat(ActionsTestUtil.baseNamesOf(((SkylarkNestedSet) result).getSet(Artifact.class)))
        .isEqualTo("a.txt");
  }

  @Test
  public void testDefaultProvider() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "def _impl(ctx):",
        "    default = DefaultInfo(",
        "        runfiles=ctx.runfiles(ctx.files.runs),",
        "    )",
        "    foo = foo_provider()",
        "    return [foo, default]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'runs': attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'foo_provider')",
        "def _impl(ctx):",
        "    provider = ctx.attr.deps[0][DefaultInfo]",
        "    return struct(",
        "        is_provided = DefaultInfo in ctx.attr.deps[0],",
        "        provider = provider,",
        "        dir = str(sorted(dir(provider))),",
        "        rule_data_runfiles = provider.data_runfiles,",
        "        rule_default_runfiles = provider.default_runfiles,",
        "        rule_files = provider.files,",
        "        rule_files_to_run = provider.files_to_run,",
        "        rule_file_executable = provider.files_to_run.executable",
        "    )",
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule', runs = ['run.file', 'run2.file'])",
        "bar_rule(name = 'my_rule', deps = [':dep_rule', 'file.txt'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");

    assertThat((Boolean) configuredTarget.get("is_provided")).isTrue();

    Object provider = configuredTarget.get("provider");
    assertThat(provider).isInstanceOf(DefaultInfo.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(DefaultInfo.PROVIDER.getKey());

    assertThat(configuredTarget.get("dir"))
        .isEqualTo(
            "[\"data_runfiles\", \"default_runfiles\", \"files\", \"files_to_run\", \"to_json\", "
                + "\"to_proto\"]");

    assertThat(configuredTarget.get("rule_data_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) configuredTarget.get("rule_data_runfiles")).getAllArtifacts(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(configuredTarget.get("rule_default_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) configuredTarget.get("rule_default_runfiles")).getAllArtifacts(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(configuredTarget.get("rule_files")).isInstanceOf(SkylarkNestedSet.class);
    assertThat(configuredTarget.get("rule_files_to_run")).isInstanceOf(FilesToRunProvider.class);
    assertThat(configuredTarget.get("rule_file_executable")).isEqualTo(Runtime.NONE);
  }

  @Test
  public void testDefaultProviderInStruct() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "def _impl(ctx):",
        "    default = DefaultInfo(",
        "        runfiles=ctx.runfiles(ctx.files.runs),",
        "    )",
        "    foo = foo_provider()",
        "    return struct(providers=[foo, default])",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'runs': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'foo_provider')",
        "def _impl(ctx):",
        "    provider = ctx.attr.deps[0][DefaultInfo]",
        "    return struct(",
        "        is_provided = DefaultInfo in ctx.attr.deps[0],",
        "        provider = provider,",
        "        dir = str(sorted(dir(provider))),",
        "        rule_data_runfiles = provider.data_runfiles,",
        "        rule_default_runfiles = provider.default_runfiles,",
        "        rule_files = provider.files,",
        "        rule_files_to_run = provider.files_to_run,",
        "    )",
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule', runs = ['run.file', 'run2.file'])",
        "bar_rule(name = 'my_rule', deps = [':dep_rule', 'file.txt'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");

    assertThat((Boolean) configuredTarget.get("is_provided")).isTrue();

    Object provider = configuredTarget.get("provider");
    assertThat(provider).isInstanceOf(DefaultInfo.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(DefaultInfo.PROVIDER.getKey());

    assertThat(configuredTarget.get("dir"))
        .isEqualTo(
            "[\"data_runfiles\", \"default_runfiles\", \"files\", \"files_to_run\", \"to_json\", "
                + "\"to_proto\"]");

    assertThat(configuredTarget.get("rule_data_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) configuredTarget.get("rule_data_runfiles")).getAllArtifacts(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(configuredTarget.get("rule_default_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) configuredTarget.get("rule_default_runfiles")).getAllArtifacts(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(configuredTarget.get("rule_files")).isInstanceOf(SkylarkNestedSet.class);
    assertThat(configuredTarget.get("rule_files_to_run")).isInstanceOf(FilesToRunProvider.class);
  }

  @Test
  public void testDefaultProviderInvalidConfiguration() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "def _impl(ctx):",
        "    default = DefaultInfo(",
        "        runfiles=ctx.runfiles(ctx.files.runs),",
        "    )",
        "    foo = foo_provider()",
        "    return struct(providers=[foo, default], files=depset([]))",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'runs': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "foo_rule(name = 'my_rule', runs = ['run.file', 'run2.file'])");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "Provider 'files' should be specified in DefaultInfo "
                + "if it's provided explicitly.");
  }

  @Test
  public void testDefaultProviderOnFileTarget() throws Exception {
    scratch.file(
        "test/bar.bzl",
        "def _impl(ctx):",
        "    provider = ctx.attr.deps[0][DefaultInfo]",
        "    return struct(",
        "        is_provided = DefaultInfo in ctx.attr.deps[0],",
        "        provider = provider,",
        "        dir = str(sorted(dir(provider))),",
        "        file_data_runfiles = provider.data_runfiles,",
        "        file_default_runfiles = provider.default_runfiles,",
        "        file_files = provider.files,",
        "        file_files_to_run = provider.files_to_run,",
        "    )",
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':bar.bzl', 'bar_rule')",
        "bar_rule(name = 'my_rule', deps = ['file.txt'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");

    assertThat((Boolean) configuredTarget.get("is_provided")).isTrue();

    Object provider = configuredTarget.get("provider");
    assertThat(provider).isInstanceOf(DefaultInfo.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(DefaultInfo.PROVIDER.getKey());

    assertThat(configuredTarget.get("dir"))
        .isEqualTo(
            "[\"data_runfiles\", \"default_runfiles\", \"files\", \"files_to_run\", \"to_json\", "
                + "\"to_proto\"]");

    assertThat(configuredTarget.get("file_data_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) configuredTarget.get("file_data_runfiles")).getAllArtifacts(),
                String::valueOf))
        .isEmpty();

    assertThat(configuredTarget.get("file_default_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) configuredTarget.get("file_default_runfiles")).getAllArtifacts(),
                String::valueOf))
        .isEmpty();

    assertThat(configuredTarget.get("file_files")).isInstanceOf(SkylarkNestedSet.class);
    assertThat(configuredTarget.get("file_files_to_run")).isInstanceOf(FilesToRunProvider.class);
  }

  @Test
  public void testDefaultProviderProvidedImplicitly() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "def _impl(ctx):",
        "    foo = foo_provider()",
        "    return [foo]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        ")"
    );
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'foo_provider')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[DefaultInfo]",  // The goal is to test this object
        "    return struct(",               // so we return it here
        "        default = provider,",
        "    )",
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    Object provider = configuredTarget.get("default");
    assertThat(provider).isInstanceOf(DefaultInfo.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(DefaultInfo.PROVIDER.getKey());
  }

  @Test
  public void testDefaultProviderUnknownFields() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "def _impl(ctx):",
        "    default = DefaultInfo(",
        "        foo=ctx.runfiles(),",
        "    )",
        "    return [default]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "foo_rule(name = 'my_rule')"
    );
    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains("unexpected keyword 'foo', for call to function DefaultInfo(");
  }

  @Test
  public void testDeclaredProviders() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "foobar_provider = provider()",
        "def _impl(ctx):",
        "    foo = foo_provider()",
        "    foobar = foobar_provider()",
        "    return [foo, foobar]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       \"srcs\": attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'foo_provider')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[foo_provider]",     // The goal is to test this object
        "    return struct(proxy = provider)",  // so we return it here
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'srcs': attr.label_list(allow_files=True),",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    Object provider = configuredTarget.get("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new SkylarkKey(
                Label.parseAbsolute("//test:foo.bzl", ImmutableMap.of()), "foo_provider"));
  }

  @Test
  public void testAdvertisedProviders() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "FooInfo = provider()",
        "BarInfo = provider()",
        "def _impl(ctx):",
        "    foo = FooInfo()",
        "    bar = BarInfo()",
        "    return [foo, bar]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    provides = [FooInfo, BarInfo]",
        ")");
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'FooInfo')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    proxy = dep[FooInfo]", // The goal is to test this object
        "    return struct(proxy = proxy)", // so we return it here
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    Object provider = configuredTarget.get("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new SkylarkKey(Label.parseAbsolute("//test:foo.bzl", ImmutableMap.of()), "FooInfo"));
  }

  @Test
  public void testLacksAdvertisedDeclaredProvider() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "FooInfo = provider()",
        "def _impl(ctx):",
        "    default = DefaultInfo(",
        "        runfiles=ctx.runfiles(ctx.files.runs),",
        "    )",
        "    return struct(providers=[default])",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'runs': attr.label_list(allow_files=True),",
        "    },",
        "    provides = [FooInfo, DefaultInfo]",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "foo_rule(name = 'my_rule', runs = ['run.file', 'run2.file'])");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains("rule advertised the 'FooInfo' provider, "
            + "but this provider was not among those returned");
  }

  @Test
  public void testLacksAdvertisedNativeProvider() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "FooInfo = provider()",
        "def _impl(ctx):",
        "    MyFooInfo = FooInfo()",
        "    return struct(providers=[MyFooInfo])",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    provides = [FooInfo, JavaInfo]",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "foo_rule(name = 'my_rule')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains("rule advertised the 'JavaInfo' provider, "
            + "but this provider was not among those returned");
  }

  @Test
  public void testBadlySpecifiedProvides() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "def _impl(ctx):",
        "    return struct()",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    provides = [1]",
        ")");
    scratch.file("test/BUILD", "load(':foo.bzl', 'foo_rule')", "foo_rule(name = 'my_rule')");


    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "element in 'provides' is of unexpected type. "
                + "Should be list of providers, but got item of type int");
  }

  @Test
  public void testSingleDeclaredProvider() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "def _impl(ctx):",
        "    return foo_provider(a=123)",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       \"srcs\": attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'foo_provider')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[foo_provider]", // The goal is to test this object
        "    return struct(proxy = provider)", // so we return it here
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'srcs': attr.label_list(allow_files=True),",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    Object provider = configuredTarget.get("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new SkylarkKey(
                Label.parseAbsolute("//test:foo.bzl", ImmutableMap.of()), "foo_provider"));
    assertThat(((StructImpl) provider).getValue("a")).isEqualTo(123);
  }

  @Test
  public void testDeclaredProvidersAliasTarget() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "foobar_provider = provider()",
        "def _impl(ctx):",
        "    foo = foo_provider()",
        "    foobar = foobar_provider()",
        "    return [foo, foobar]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       \"srcs\": attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'foo_provider')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[foo_provider]",     // The goal is to test this object
        "    return struct(proxy = provider)",  // so we return it here
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'srcs': attr.label_list(allow_files=True),",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'foo_rule')",
        "alias(name = 'dep_rule', actual=':foo_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    Object provider = configuredTarget.get("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new SkylarkKey(
                Label.parseAbsolute("//test:foo.bzl", ImmutableMap.of()), "foo_provider"));
  }

  @Test
  public void testDeclaredProvidersWrongKey() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "unused_provider = provider()",
        "def _impl(ctx):",
        "    foo = foo_provider()",
        "    return [foo]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       \"srcs\": attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'unused_provider')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[unused_provider]",  // Should throw an error here
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'srcs': attr.label_list(allow_files=True),",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "<target //test:dep_rule> (rule 'foo_rule') doesn't contain "
                + "declared provider 'unused_provider'");
  }

  @Test
  public void testDeclaredProvidersInvalidKey() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "def _impl(ctx):",
        "    foo = foo_provider()",
        "    return [foo]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       \"srcs\": attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/bar.bzl",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep['foo_provider']",  // Should throw an error here
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'srcs': attr.label_list(allow_files=True),",
        "       'deps': attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'foo_rule')",
        "load(':bar.bzl', 'bar_rule')",
        "foo_rule(name = 'dep_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains("Type Target only supports indexing by object constructors, got string instead");
  }

  @Test
  public void testDeclaredProvidersFileTarget() throws Exception {
    scratch.file(
        "test/bar.bzl",
        "unused_provider = provider()",
        "def _impl(ctx):",
        "    src = ctx.attr.srcs[0]",
        "    provider = src[unused_provider]",  // Should throw an error here
        "bar_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'srcs': attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':bar.bzl', 'bar_rule')",
        "bar_rule(name = 'my_rule', srcs = ['input.txt'])");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "<input file target //test:input.txt> doesn't contain "
                + "declared provider 'unused_provider'");
  }

  @Test
  public void testDeclaredProvidersInOperator() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "bar_provider = provider()",
        "",
        "def _inner_impl(ctx):",
        "    foo = foo_provider()",
        "    return [foo]",
        "inner_rule = rule(",
        "    implementation = _inner_impl,",
        ")",
        "",
        "def _outer_impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    return struct(",
        "        foo = (foo_provider in dep),",  // Should be true
        "        bar = (bar_provider in dep),",  // Should be false
        "    )",
        "outer_rule = rule(",
        "    implementation = _outer_impl,",
        "    attrs = {",
        "       'deps': attr.label_list(),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'inner_rule', 'outer_rule')",
        "inner_rule(name = 'dep_rule')",
        "outer_rule(name = 'my_rule', deps = [':dep_rule'])");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    Object foo = configuredTarget.get("foo");
    assertThat(foo).isInstanceOf(Boolean.class);
    assertThat((Boolean) foo).isTrue();
    Object bar = configuredTarget.get("bar");
    assertThat(bar).isInstanceOf(Boolean.class);
    assertThat((Boolean) bar).isFalse();
  }

  @Test
  public void testDeclaredProvidersInOperatorInvalidKey() throws Exception {
    scratch.file(
        "test/foo.bzl",
        "foo_provider = provider()",
        "bar_provider = provider()",
        "",
        "def _inner_impl(ctx):",
        "    foo = foo_provider()",
        "    return [foo]",
        "inner_rule = rule(",
        "    implementation = _inner_impl,",
        ")",
        "",
        "def _outer_impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    'foo_provider' in dep",  // Should throw an error here
        "outer_rule = rule(",
        "    implementation = _outer_impl,",
        "    attrs = {",
        "       'deps': attr.label_list(),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'inner_rule', 'outer_rule')",
        "inner_rule(name = 'dep_rule')",
        "outer_rule(name = 'my_rule', deps = [':dep_rule'])");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains("Type Target only supports querying by object constructors, got string instead");
  }

  @Test
  public void testFilesForFileConfiguredTarget() throws Exception {
    Object result =
        evalRuleContextCode(createRuleContext("//foo:bar"), "ruleContext.attr.srcs[0].files");
    assertThat(ActionsTestUtil.baseNamesOf(((SkylarkNestedSet) result).getSet(Artifact.class)))
        .isEqualTo("libjl.jar");
  }

  @Test
  public void testCtxStructFieldsCustomErrorMessages() throws Exception {
    checkErrorContains("No attribute 'foo' in attr.", "ruleContext.attr.foo");
    checkErrorContains("No attribute 'foo' in outputs.", "ruleContext.outputs.foo");
    checkErrorContains("No attribute 'foo' in files.", "ruleContext.files.foo");
    checkErrorContains("No attribute 'foo' in file.", "ruleContext.file.foo");
    checkErrorContains("No attribute 'foo' in executable.", "ruleContext.executable.foo");
  }

  @Test
  public void testBinDirPath() throws Exception {
    SkylarkRuleContext ctx = createRuleContext("//foo:bar");
    Object result = evalRuleContextCode(ctx, "ruleContext.bin_dir.path");
    assertThat(result).isEqualTo(ctx.getConfiguration().getBinFragment().getPathString());
  }

  @Test
  public void testEmptyLabelListTypeAttrInCtx() throws Exception {
    SkylarkRuleContext ctx = createRuleContext("//foo:baz");
    Object result = evalRuleContextCode(ctx, "ruleContext.attr.srcs");
    assertThat(result).isEqualTo(MutableList.empty());
  }

  @Test
  public void testDefinedMakeVariable() throws Exception {
    useConfiguration("--define=FOO=bar");
    SkylarkRuleContext ctx = createRuleContext("//foo:baz");
    String foo = (String) evalRuleContextCode(ctx, "ruleContext.var['FOO']");
    assertThat(foo).isEqualTo("bar");
  }

  @Test
  public void testCodeCoverageConfigurationAccess() throws Exception {
    SkylarkRuleContext ctx = createRuleContext("//foo:baz");
    boolean coverage =
        (Boolean) evalRuleContextCode(ctx, "ruleContext.configuration.coverage_enabled");
    assertThat(ctx.getRuleContext().getConfiguration().isCodeCoverageEnabled()).isEqualTo(coverage);
  }

  @Override
  protected void checkErrorContains(String errorMsg, String... lines) throws Exception {
    super.checkErrorContains(createRuleContext("//foo:foo"), errorMsg, lines);
  }

  /**
   * Checks whether the given (invalid) statement leads to the expected error
   */
  private void checkReportedErrorStartsWith(
      SkylarkRuleContext ruleContext, String errorMsg, String... statements) throws Exception {
    // If the component under test relies on Reporter and EventCollector for error handling, any
    // error would lead to an asynchronous AssertionFailedError thanks to failFastHandler in
    // FoundationTestCase.
    //
    // Consequently, we disable failFastHandler and check all events for the expected error message
    reporter.removeHandler(failFastHandler);

    Object result = evalRuleContextCode(ruleContext, statements);

    String first = null;
    int count = 0;

    try {
      for (Event evt : eventCollector) {
        if (evt.getMessage().startsWith(errorMsg)) {
          return;
        }

        ++count;
        first = evt.getMessage();
      }

      if (count == 0) {
        fail(
            String.format(
                "checkReportedErrorStartsWith(): There was no error; the result is '%s'", result));
      } else {
        fail(
            String.format(
                "Found %d error(s), but none with the expected message '%s'. First error: '%s'",
                count, errorMsg, first));
      }
    } finally {
      eventCollector.clear();
    }
  }

  @Test
  public void testStackTraceWithoutOriginalMessage() throws Exception {
    setupThrowFunction(
        new BuiltinFunction("throw") {
          @SuppressWarnings("unused")
          public Object invoke() throws Exception {
            throw new ThereIsNoMessageException();
          }
        });

    checkEvalErrorContains(
        "There Is No Message: SkylarkRuleImplementationFunctionsTest",
        "throw()");
  }

  @Test
  public void testNoStackTraceOnInterrupt() throws Exception {
    setupThrowFunction(
        new BuiltinFunction("throw") {
          @SuppressWarnings("unused")
          public Object invoke() throws Exception {
            throw new InterruptedException();
          }
        });
    try {
      eval("throw()");
      fail("Expected an InterruptedException");
    } catch (InterruptedException ex) {
      // Expected.
    }
  }

  @Test
  public void testGlobInImplicitOutputs() throws Exception {
    scratch.file(
        "test/glob.bzl",
        "def _impl(ctx):",
        "  ctx.empty_action(",
        "    inputs = [],",
        "  )",
        "def _foo():",
        "  return native.glob(['*'])",
        "glob_rule = rule(",
        "  implementation = _impl,",
        "  outputs = _foo,",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:glob.bzl', 'glob_rule')",
        "glob_rule(name = 'my_glob',",
        "  srcs = ['foo.bar', 'other_foo.bar'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:my_glob");
    assertContainsEvent("native.glob() can only be called during the loading phase");
  }

  @Test
  public void testRuleFromBzlFile() throws Exception {
    scratch.file("test/rule.bzl", "def _impl(ctx): return", "foo = rule(implementation = _impl)");
    scratch.file("test/ext.bzl", "load('//test:rule.bzl', 'foo')", "a = 1", "foo(name = 'x')");
    scratch.file("test/BUILD", "load('//test:ext.bzl', 'a')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:x");
    assertContainsEvent("Cannot instantiate a rule when loading a .bzl file");
  }

  @Test
  public void testImplicitOutputsFromGlob() throws Exception {
    scratch.file(
        "test/glob.bzl",
        "def _impl(ctx):",
        "  outs = ctx.outputs",
        "  for i in ctx.attr.srcs:",
        "    o = getattr(outs, 'foo_' + i.label.name)",
        "    ctx.actions.write(",
        "      output = o,",
        "      content = 'hoho')",
        "",
        "def _foo(srcs):",
        "  outs = {}",
        "  for i in srcs:",
        "    outs['foo_' + i.name] = i.name + '.out'",
        "  return outs",
        "",
        "glob_rule = rule(",
        "    attrs = {",
        "        'srcs': attr.label_list(allow_files = True),",
        "    },",
        "    outputs = _foo,",
        "    implementation = _impl,",
        ")");
    scratch.file("test/a.bar", "a");
    scratch.file("test/b.bar", "b");
    scratch.file(
        "test/BUILD",
        "load('//test:glob.bzl', 'glob_rule')",
        "glob_rule(name = 'my_glob', srcs = glob(['*.bar']))");
    ConfiguredTarget ct = getConfiguredTarget("//test:my_glob");
    assertThat(ct).isNotNull();
    assertThat(getGeneratingAction(getBinArtifact("a.bar.out", ct))).isNotNull();
    assertThat(getGeneratingAction(getBinArtifact("b.bar.out", ct))).isNotNull();
  }

  @Test
  public void testBuiltInFunctionAsRuleImplementation() throws Exception {
    // Using built-in functions as rule implementations shouldn't cause runtime errors
    scratch.file(
        "test/rule.bzl",
        "silly_rule = rule(",
        "    implementation = int,",
        "    attrs = {",
        "       \"srcs\": attr.label_list(allow_files=True),",
        "    }",
        ")"
    );
    scratch.file(
        "test/BUILD",
        "load('//test:rule.bzl', 'silly_rule')",
        "silly_rule(name = 'silly')");
    thrown.handleAssertionErrors(); // Compatibility with JUnit 4.11
    thrown.expect(AssertionError.class);
    thrown.expectMessage("<rule context for //test:silly> is not of type string or int or bool");
    getConfiguredTarget("//test:silly");
  }

  @Test
  public void testArgsScalarAdd() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "args.add('--foo')",
        "args.add('-')",
        "args.add('foo', format='format%s')",
        "args.add('-')",
        "args.add('--foo', 'val')",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly("foo/t.exe", "--foo", "-", "formatfoo", "-", "--foo", "val")
        .inOrder();
  }

  @Test
  public void testArgsScalarAddThrowsWithVectorArg() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_old_style_args_add=true");
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "Args#add no longer accepts vectorized",
        "args = ruleContext.actions.args()",
        "args.add([1, 2])",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
  }

  @Test
  public void testArgsAddAll() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "args.add_all([1, 2])",
        "args.add('-')",
        "args.add_all('--foo', [1, 2])",
        "args.add('-')",
        "args.add_all([1, 2], before_each='-before')",
        "args.add('-')",
        "args.add_all([1, 2], format_each='format/%s')",
        "args.add('-')",
        "args.add_all(ruleContext.files.srcs)",
        "args.add('-')",
        "args.add_all(ruleContext.files.srcs, format_each='format/%s')",
        "args.add('-')",
        "args.add_all([1, 2], terminate_with='--terminator')",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly(
            "foo/t.exe",
            "1",
            "2",
            "-",
            "--foo",
            "1",
            "2",
            "-",
            "-before",
            "1",
            "-before",
            "2",
            "-",
            "format/1",
            "format/2",
            "-",
            "foo/a.txt",
            "foo/b.img",
            "-",
            "format/foo/a.txt",
            "format/foo/b.img",
            "-",
            "1",
            "2",
            "--terminator")
        .inOrder();
  }

  @Test
  public void testArgsAddAllWithMapEach() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "def add_one(val): return str(val + 1)",
        "def expand_to_many(val): return ['hey', 'hey']",
        "args = ruleContext.actions.args()",
        "args.add_all([1, 2], map_each=add_one)",
        "args.add('-')",
        "args.add_all([1, 2], map_each=expand_to_many)",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly("foo/t.exe", "2", "3", "-", "hey", "hey", "hey", "hey")
        .inOrder();
  }

  @Test
  public void testOmitIfEmpty() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "def add_one(val): return str(val + 1)",
        "def filter(val): return None",
        "args = ruleContext.actions.args()",
        "args.add_joined([], join_with=',')",
        "args.add('-')",
        "args.add_joined([], join_with=',', omit_if_empty=False)",
        "args.add('-')",
        "args.add_all('--foo', [])",
        "args.add('-')",
        "args.add_all('--foo', [], omit_if_empty=False)",
        "args.add('-')",
        "args.add_all('--foo', [1], map_each=filter, terminate_with='hello')",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly(
            "foo/t.exe",
            // Nothing
            "-",
            "", // Empty string was joined and added
            "-",
            // Nothing
            "-",
            "--foo", // Arg added regardless
            "-"
            // Nothing, all values were filtered
            )
        .inOrder();
  }

  @Test
  public void testUniquify() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "def add_one(val): return str(val + 1)",
        "args = ruleContext.actions.args()",
        "args.add_all(['a', 'b', 'a'])",
        "args.add('-')",
        "args.add_all(['a', 'b', 'a', 'c', 'b'], uniquify=True)",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly("foo/t.exe", "a", "b", "a", "-", "a", "b", "c")
        .inOrder();
  }

  @Test
  public void testArgsAddJoined() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "def add_one(val): return str(val + 1)",
        "args = ruleContext.actions.args()",
        "args.add_joined([1, 2], join_with=':')",
        "args.add('-')",
        "args.add_joined([1, 2], join_with=':', format_each='format/%s')",
        "args.add('-')",
        "args.add_joined([1, 2], join_with=':', format_each='format/%s', format_joined='--foo=%s')",
        "args.add('-')",
        "args.add_joined([1, 2], join_with=':', map_each=add_one)",
        "args.add('-')",
        "args.add_joined(ruleContext.files.srcs, join_with=':')",
        "args.add('-')",
        "args.add_joined(ruleContext.files.srcs, join_with=':', format_each='format/%s')",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly(
            "foo/t.exe",
            "1:2",
            "-",
            "format/1:format/2",
            "-",
            "--foo=format/1:format/2",
            "-",
            "2:3",
            "-",
            "foo/a.txt:foo/b.img",
            "-",
            "format/foo/a.txt:format/foo/b.img")
        .inOrder();
  }

  @Test
  public void testLazyArgsLegacy() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_old_style_args_add=false");
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "def map_scalar(val): return 'mapped' + val",
        "def map_vector(vals): return [x + 1 for x in vals]",
        "args = ruleContext.actions.args()",
        "args.add('--foo')",
        "args.add('foo', format='format%s')",
        "args.add('foo', map_fn=map_scalar)",
        "args.add([1, 2])",
        "args.add([1, 2], join_with=':')",
        "args.add([1, 2], before_each='-before')",
        "args.add([1, 2], format='format/%s')",
        "args.add([1, 2], map_fn=map_vector)",
        "args.add([1, 2], format='format/%s', join_with=':')",
        "args.add(ruleContext.files.srcs)",
        "args.add(ruleContext.files.srcs, format='format/%s')",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly(
            "foo/t.exe",
            "--foo",
            "formatfoo",
            "mappedfoo",
            "1",
            "2",
            "1:2",
            "-before",
            "1",
            "-before",
            "2",
            "format/1",
            "format/2",
            "2",
            "3",
            "format/1:format/2",
            "foo/a.txt",
            "foo/b.img",
            "format/foo/a.txt",
            "format/foo/b.img")
        .inOrder();
  }

  @Test
  public void testLegacyLazyArgMapFnReturnsWrongArgumentCount() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_old_style_args_add=false");
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "def bad_fn(args): return [0]",
        "args.add([1, 2], map_fn=bad_fn)",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    try {
      action.getArguments();
      fail();
    } catch (CommandLineExpansionException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("map_fn must return a list of the same length as the input");
    }
  }

  @Test
  public void testMultipleLazyArgsMixedWithStrings() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "foo_args = ruleContext.actions.args()",
        "foo_args.add('--foo')",
        "bar_args = ruleContext.actions.args()",
        "bar_args.add('--bar')",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['hello', foo_args, 'world', bar_args, 'works'],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments())
        .containsExactly("foo/t.exe", "hello", "--foo", "world", "--bar", "works")
        .inOrder();
  }

  @Test
  public void testLazyArgsWithParamFile() throws Exception {
    scratch.file(
        "test/main_rule.bzl",
        "def _impl(ctx):",
        "  args = ctx.actions.args()",
        "  args.add('--foo')",
        "  args.use_param_file('--file=%s', use_always=True)",
        "  output=ctx.actions.declare_file('out')",
        "  ctx.actions.run_shell(",
        "    inputs = [output],",
        "    outputs = [output],",
        "    arguments = [args],",
        "    command = 'touch out',",
        "  )",
        "main_rule = rule(implementation = _impl)");
    scratch.file(
        "test/BUILD", "load('//test:main_rule.bzl', 'main_rule')", "main_rule(name='main')");
    ConfiguredTarget ct = getConfiguredTarget("//test:main");
    Artifact output = getBinArtifact("out", ct);
    SpawnAction action = (SpawnAction) getGeneratingAction(output);
    assertThat(paramFileArgsForAction(action)).containsExactly("--foo");
  }

  @Test
  public void testWriteArgsToParamFile() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "args.add('--foo')",
        "output=ruleContext.actions.declare_file('out')",
        "ruleContext.actions.write(",
        "  output=output,",
        "  content=args,",
        ")");
    List<ActionAnalysisMetadata> actions =
        ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions();
    Optional<ActionAnalysisMetadata> action =
        actions.stream().filter(a -> a instanceof ParameterFileWriteAction).findFirst();
    assertThat(action.isPresent()).isTrue();
    ParameterFileWriteAction paramAction = (ParameterFileWriteAction) action.get();
    assertThat(paramAction.getArguments()).containsExactly("--foo");
  }

  @Test
  public void testLazyArgsWithParamFileInvalidFormatString() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkError(
        ruleContext,
        "Invalid value for parameter \"param_file_arg\": Expected string with a single \"%s\"",
        "args = ruleContext.actions.args()\n" + "args.use_param_file('--file=')");
    checkError(
        ruleContext,
        "Invalid value for parameter \"param_file_arg\": Expected string with a single \"%s\"",
        "args = ruleContext.actions.args()\n" + "args.use_param_file('--file=%s%s')");
  }

  @Test
  public void testLazyArgsWithParamFileInvalidFormat() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkError(
        ruleContext,
        "Invalid value for parameter \"format\": Expected one of \"shell\", \"multiline\"",
        "args = ruleContext.actions.args()\n" + "args.set_param_file_format('illegal')");
  }

  @Test
  public void testScalarJoinWithErrorMessage() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkError(
        ruleContext,
        "'join_with' is not supported for scalar arguments",
        "args = ruleContext.actions.args()\n" + "args.add(1, join_with=':')");
  }

  @Test
  public void testScalarBeforeEachErrorMessage() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkError(
        ruleContext,
        "'before_each' is not supported for scalar arguments",
        "args = ruleContext.actions.args()\n" + "args.add(1, before_each='illegal')");
  }

  @Test
  public void testArgsAddInvalidTypesForArgAndValues() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_old_style_args_add=true");
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkError(
        ruleContext,
        "expected value of type 'string' for arg name, got 'Integer'",
        "args = ruleContext.actions.args()",
        "args.add(1, 'value')");
    checkError(
        ruleContext,
        "expected value of type 'string' for arg name, got 'Integer'",
        "args = ruleContext.actions.args()",
        "args.add_all(1, [1, 2])");
    checkError(
        ruleContext,
        "expected value of type 'sequence or depset' for values, got 'Integer'",
        "args = ruleContext.actions.args()",
        "args.add_all(1)");
    checkErrorContains(
        ruleContext,
        "expected value of type 'sequence or depset' for parameter 'values'",
        "args = ruleContext.actions.args()",
        "args.add_all('--foo', 1)");
  }

  @Test
  public void testLazyArgIllegalLegacyFormatString() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_old_style_args_add=false");
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "args.add('foo', format='format/%s%s')", // Expects two args, will only be given one
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    try {
      action.getArguments();
      fail();
    } catch (CommandLineExpansionException e) {
      assertThat(e.getMessage()).contains("not enough arguments");
    }
  }

  @Test
  public void testLazyArgIllegalFormatString() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_disallow_old_style_args_add=true");
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkError(
        ruleContext,
        "Invalid value for parameter \"format\": Expected string with a single \"%s\"",
        "args = ruleContext.actions.args()",
        "args.add('foo', format='illegal_format')", // Expects two args, will only be given one
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
  }

  @Test
  public void testLazyArgMapEachWrongArgCount() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "map_each must be a function that accepts a single",
        "args = ruleContext.actions.args()",
        "def bad_fn(val, val2): return str(val)",
        "args.add_all([1, 2], map_each=bad_fn)",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
  }

  @Test
  public void testLazyArgMapEachThrowsError() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "def bad_fn(val): 'hello'.nosuchmethod()",
        "args.add_all([1, 2], map_each=bad_fn)",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    try {
      action.getArguments();
      fail();
    } catch (CommandLineExpansionException e) {
      assertThat(e.getMessage()).contains("type 'string' has no method nosuchmethod()");
    }
  }

  @Test
  public void testLazyArgMapEachReturnsNone() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "def none_fn(val): return None if val == 'nokeep' else val",
        "args.add_all(['keep', 'nokeep'], map_each=none_fn)",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getArguments()).containsExactly("foo/t.exe", "keep").inOrder();
  }

  @Test
  public void testLazyArgMapEachReturnsWrongType() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "def bad_fn(val): return 1",
        "args.add_all([1, 2], map_each=bad_fn)",
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  executable = ruleContext.files.tools[0],",
        ")");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    try {
      action.getArguments();
      fail();
    } catch (CommandLineExpansionException e) {
      assertThat(e.getMessage())
          .contains("Expected map_each to return string, None, or list of strings, found Integer");
    }
  }

  @Test
  public void createShellWithLazyArgs() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    evalRuleContextCode(
        ruleContext,
        "args = ruleContext.actions.args()",
        "args.add('--foo')",
        "ruleContext.actions.run_shell(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = [args],",
        "  mnemonic = 'DummyMnemonic',",
        "  command = 'dummy_command',",
        "  progress_message = 'dummy_message',",
        "  use_default_shell_env = True)");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    List<String> args = action.getArguments();
    // We don't need to assert the entire arg list, just check that
    // the dummy empty string is inserted followed by '--foo'
    assertThat(args.get(args.size() - 2)).isEmpty();
    assertThat(Iterables.getLast(args)).isEqualTo("--foo");
  }

  @Test
  public void testLazyArgsObjectImmutability() throws Exception {
    scratch.file(
        "test/BUILD",
        "load('//test:rules.bzl', 'main_rule', 'dep_rule')",
        "dep_rule(name = 'dep')",
        "main_rule(name = 'main', deps = [':dep'])");
    scratch.file(
        "test/rules.bzl",
        "def _main_impl(ctx):",
        "  dep = ctx.attr.deps[0]",
        "  args = dep.dep_arg",
        "  args.add('hello')",
        "main_rule = rule(",
        "  implementation = _main_impl,",
        "  attrs = {",
        "    'deps': attr.label_list()",
        "  },",
        "  outputs = {'file': 'output.txt'},",
        ")",
        "def _dep_impl(ctx):",
        "  args = ctx.actions.args()",
        "  return struct(dep_arg = args)",
        "dep_rule = rule(implementation = _dep_impl)");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:main"));
    assertThat(e).hasMessageThat().contains("cannot modify frozen value");
  }

  @Test
  public void testConfigurationField_SkylarkSplitTransitionProhibited() throws Exception {
    scratch.file(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
        "    packages = [",
        "        '//...',",
        "    ],",
        ")");

    scratch.file(
        "test/rule.bzl",
        "def _foo_impl(ctx):",
        "  return struct()",
        "",
        "def _foo_transition_impl(settings):",
        "  return {'t1': {}, 't2': {}}",
        "foo_transition = transition(implementation=_foo_transition_impl, inputs=[], outputs=[])",
        "",
        "foo = rule(",
        "  implementation = _foo_impl,",
        "  attrs = {",
        "    '_whitelist_function_transition': attr.label(",
        "        default = '//tools/whitelists/function_transition_whitelist'),",
        "    '_attr': attr.label(",
        "        cfg = foo_transition,",
        "        default = configuration_field(fragment='cpp', name = 'cc_toolchain'))})");

    scratch.file("test/BUILD", "load('//test:rule.bzl', 'foo')", "foo(name='foo')");

    setSkylarkSemanticsOptions("--experimental_starlark_config_transitions=true");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:foo");
    assertContainsEvent("late-bound attributes must not have a split configuration transition");
  }

  @Test
  public void testConfigurationField_NativeSplitTransitionProviderProhibited() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _foo_impl(ctx):",
        "  return struct()",
        "",
        "foo = rule(",
        "  implementation = _foo_impl,",
        "  attrs = {",
        "    '_attr': attr.label(",
        "        cfg = apple_common.multi_arch_split,",
        "        default = configuration_field(fragment='cpp', name = 'cc_toolchain'))})");

    scratch.file("test/BUILD", "load('//test:rule.bzl', 'foo')", "foo(name='foo')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:foo");
    assertContainsEvent("late-bound attributes must not have a split configuration transition");
  }

  @Test
  public void testConfigurationField_NativeSplitTransitionProhibited() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _foo_impl(ctx):",
        "  return struct()",
        "",
        "foo = rule(",
        "  implementation = _foo_impl,",
        "  attrs = {",
        "    '_attr': attr.label(",
        "        cfg = android_common.multi_cpu_configuration,",
        "        default = configuration_field(fragment='cpp', name = 'cc_toolchain'))})");

    scratch.file("test/BUILD", "load('//test:rule.bzl', 'foo')", "foo(name='foo')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:foo");
    assertContainsEvent("late-bound attributes must not have a split configuration transition");
  }

  @Test
  public void testConfigurationField_invalidFragment() throws Exception {
    scratch.file("test/main_rule.bzl",
        "def _impl(ctx):",
        "  return struct()",

        "main_rule = rule(implementation = _impl,",
        "    attrs = { '_myattr': attr.label(",
        "        default = configuration_field(",
        "        fragment = 'notarealfragment', name = 'method_name')),",
        "    },",
        ")");

    scratch.file("test/BUILD",
        "load('//test:main_rule.bzl', 'main_rule')",
        "main_rule(name='main')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:main"));
    assertThat(expected).hasMessageThat()
        .contains("invalid configuration fragment name 'notarealfragment'");
  }

  @Test
  public void testConfigurationField_doesNotChangeFragmentAccess() throws Exception {
    scratch.file("test/main_rule.bzl",
        "def _impl(ctx):",
        "  return struct(platform = ctx.fragments.apple.single_arch_platform)",

        "main_rule = rule(implementation = _impl,",
        "    attrs = { '_myattr': attr.label(",
        "        default = configuration_field(",
        "        fragment = 'apple', name = 'xcode_config_label')),",
        "    },",
        "    fragments = [],",
        ")");

    scratch.file("test/BUILD",
        "load('//test:main_rule.bzl', 'main_rule')",
        "main_rule(name='main')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:main"));

    assertThat(expected).hasMessageThat()
        .contains("has to declare 'apple' as a required fragment in target configuration");
  }

  @Test
  public void testConfigurationField_invalidFieldName() throws Exception {
    scratch.file("test/main_rule.bzl",
        "def _impl(ctx):",
        "  return struct()",

        "main_rule = rule(implementation = _impl,",
        "    attrs = { '_myattr': attr.label(",
        "        default = configuration_field(",
        "        fragment = 'apple', name = 'notarealfield')),",
        "    },",
        "    fragments = ['apple'],",
        ")");

    scratch.file("test/BUILD",
        "load('//test:main_rule.bzl', 'main_rule')",
        "main_rule(name='main')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:main"));

    assertThat(expected).hasMessageThat()
        .contains("invalid configuration field name 'notarealfield' on fragment 'apple'");
  }

  // Verifies that configuration_field can only be used on 'private' attributes.
  @Test
  public void testConfigurationField_invalidVisibility() throws Exception {
    scratch.file("test/main_rule.bzl",
        "def _impl(ctx):",
        "  return struct()",

        "main_rule = rule(implementation = _impl,",
        "    attrs = { 'myattr': attr.label(",
        "        default = configuration_field(",
        "        fragment = 'apple', name = 'xcode_config_label')),",
        "    },",
        "    fragments = ['apple'],",
        ")");

    scratch.file("test/BUILD",
        "load('//test:main_rule.bzl', 'main_rule')",
        "main_rule(name='main')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:main"));

    assertThat(expected).hasMessageThat()
        .contains("When an attribute value is a function, "
            + "the attribute must be private (i.e. start with '_')");
  }

  // Verifies that configuration_field can only be used on 'label' attributes.
  @Test
  public void testConfigurationField_invalidAttributeType() throws Exception {
    scratch.file("test/main_rule.bzl",
        "def _impl(ctx):",
        "  return struct()",

        "main_rule = rule(implementation = _impl,",
        "    attrs = { '_myattr': attr.int(",
        "        default = configuration_field(",
        "        fragment = 'apple', name = 'xcode_config_label')),",
        "    },",
        "    fragments = ['apple'],",
        ")");

    scratch.file("test/BUILD",
        "load('//test:main_rule.bzl', 'main_rule')",
        "main_rule(name='main')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:main"));

    assertThat(expected)
        .hasMessageThat()
        .contains(
            "expected value of type 'int or function' for parameter 'default', "
                + "for call to method int(");
  }

  @Test
  public void testSkylarkCustomCommandLineKeyComputation() throws Exception {
    ImmutableList.Builder<SkylarkCustomCommandLine> commandLines = ImmutableList.builder();
    commandLines.add(getCommandLine("ruleContext.actions.args()"));
    commandLines.add(
        getCommandLine("args = ruleContext.actions.args()", "args.add('foo')", "args"));
    commandLines.add(
        getCommandLine("args = ruleContext.actions.args()", "args.add('--foo', 'foo')", "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()", "args.add('foo', format='--foo=%s')", "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()", "args.add_all(['foo', 'bar'])", "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()", "args.add_all('-foo', ['foo', 'bar'])", "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['foo', 'bar'], format_each='format%s')",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['foo', 'bar'], before_each='-I')",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['boing', 'boing', 'boing'])",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['boing', 'boing', 'boing'], uniquify=True)",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['foo', 'bar'], terminate_with='baz')",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_joined(['foo', 'bar'], join_with=',')",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_joined(['foo', 'bar'], join_with=',', format_joined='--foo=%s')",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "def _map_each(s): return s + '_mapped'",
            "args.add_all(['foo', 'bar'], map_each=_map_each)",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "values = depset(['a', 'b'])",
            "args.add_all(values)",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "def _map_each(s): return s + '_mapped'",
            "values = depset(['a', 'b'])",
            "args.add_all(values, map_each=_map_each)",
            "args"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "def _map_each(s): return s + '_mapped_again'",
            "values = depset(['a', 'b'])",
            "args.add_all(values, map_each=_map_each)",
            "args"));

    // Ensure all these command lines have distinct keys
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Map<String, SkylarkCustomCommandLine> digests = new HashMap<>();
    for (SkylarkCustomCommandLine commandLine : commandLines.build()) {
      Fingerprint fingerprint = new Fingerprint();
      commandLine.addToFingerprint(actionKeyContext, fingerprint);
      String digest = fingerprint.hexDigestAndReset();
      SkylarkCustomCommandLine previous = digests.putIfAbsent(digest, commandLine);
      if (previous != null) {
        fail(
            String.format(
                "Found two command lines with identical digest %s: '%s' and '%s'",
                digest,
                Joiner.on(' ').join(previous.arguments()),
                Joiner.on(' ').join(commandLine.arguments())));
      }
    }

    // Ensure errors are handled
    MoreAsserts.assertThrows(
        CommandLineExpansionException.class,
        () -> {
          SkylarkCustomCommandLine commandLine =
              getCommandLine(
                  "args = ruleContext.actions.args()",
                  "def _bad_fn(s): return s.doesnotexist()",
                  "values = depset(['a', 'b'])",
                  "args.add_all(values, map_each=_bad_fn)",
                  "args");
          commandLine.addToFingerprint(actionKeyContext, new Fingerprint());
        });
  }

  private SkylarkCustomCommandLine getCommandLine(String... lines) throws Exception {
    return ((SkylarkActionFactory.Args) evalRuleContextCode(lines)).build();
  }

  @Test
  public void testPrintArgs() throws Exception {
    Args args =
        (Args)
            evalRuleContextCode(
                "args = ruleContext.actions.args()", "args.add_all(['--foo', '--bar'])", "args");
    assertThat(Printer.debugPrint(args)).isEqualTo("--foo --bar");
  }

  @Test
  public void testDirectoryInArgs() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_expand_directories");
    SkylarkList<?> result =
        (SkylarkList<?>)
            evalRuleContextCode(
                "args = ruleContext.actions.args()",
                "directory = ruleContext.actions.declare_directory('dir')",
                "def _short_path(f): return f.short_path", // For easier assertions
                "args.add_all([directory], map_each=_short_path)",
                "args, directory");
    Args args = (Args) result.get(0);
    Artifact directory = (Artifact) result.get(1);
    CommandLine commandLine = args.build();

    // When asking for arguments without an artifact expander we just return the directory
    assertThat(commandLine.arguments()).containsExactly("foo/dir");

    // Now ask for one with an expanded directory
    Artifact file1 = getBinArtifactWithNoOwner("foo/dir/file1");
    Artifact file2 = getBinArtifactWithNoOwner("foo/dir/file2");
    ArtifactExpanderImpl artifactExpander =
        new ArtifactExpanderImpl(
            ImmutableMap.of(directory, ImmutableList.of(file1, file2)), ImmutableMap.of());
    assertThat(commandLine.arguments(artifactExpander))
        .containsExactly("foo/dir/file1", "foo/dir/file2");
  }

  @Test
  public void testDirectoryInArgsIncompatibleFlagOff() throws Exception {
    setSkylarkSemanticsOptions("--noincompatible_expand_directories");
    SkylarkList<?> result =
        (SkylarkList<?>)
            evalRuleContextCode(
                "args = ruleContext.actions.args()",
                "directory = ruleContext.actions.declare_directory('dir')",
                "def _short_path(f): return f.short_path", // For easier assertions
                "args.add_all([directory], map_each=_short_path)",
                "args, directory");
    Args args = (Args) result.get(0);
    Artifact directory = (Artifact) result.get(1);
    CommandLine commandLine = args.build();

    Artifact file1 = getBinArtifactWithNoOwner("foo/dir/file1");
    Artifact file2 = getBinArtifactWithNoOwner("foo/dir/file2");
    ArtifactExpanderImpl artifactExpander =
        new ArtifactExpanderImpl(
            ImmutableMap.of(directory, ImmutableList.of(file1, file2)), ImmutableMap.of());
    // Do not expand the directory, incompatible flag is off
    assertThat(commandLine.arguments(artifactExpander)).containsExactly("foo/dir");
  }

  @Test
  public void testDirectoryInArgsExpandDirectories() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_expand_directories");
    SkylarkList<?> result =
        (SkylarkList<?>)
            evalRuleContextCode(
                "args = ruleContext.actions.args()",
                "directory = ruleContext.actions.declare_directory('dir')",
                "def _short_path(f): return f.short_path", // For easier assertions
                "args.add_all([directory], map_each=_short_path, expand_directories=True)",
                "args.add_all([directory], map_each=_short_path, expand_directories=False)",
                "args, directory");
    Args args = (Args) result.get(0);
    Artifact directory = (Artifact) result.get(1);
    CommandLine commandLine = args.build();

    Artifact file1 = getBinArtifactWithNoOwner("foo/dir/file1");
    Artifact file2 = getBinArtifactWithNoOwner("foo/dir/file2");
    ArtifactExpanderImpl artifactExpander =
        new ArtifactExpanderImpl(
            ImmutableMap.of(directory, ImmutableList.of(file1, file2)), ImmutableMap.of());
    // First expanded, then not expanded (two separate calls)
    assertThat(commandLine.arguments(artifactExpander))
        .containsExactly("foo/dir/file1", "foo/dir/file2", "foo/dir");
  }

  @Test
  public void testDirectoryInScalarArgsFails() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_expand_directories");
    checkErrorContains(
        "Cannot add directories to Args#add",
        "args = ruleContext.actions.args()",
        "directory = ruleContext.actions.declare_directory('dir')",
        "args.add(directory)");
  }

  @Test
  public void testDirectoryInScalarArgsIsOkWithoutIncompatibleFlag() throws Exception {
    setSkylarkSemanticsOptions("--noincompatible_expand_directories");
    Args args =
        (Args)
            evalRuleContextCode(
                "args = ruleContext.actions.args()",
                "directory = ruleContext.actions.declare_directory('dir')",
                "args.add(directory)",
                "args");
    assertThat(Iterables.getOnlyElement(args.build().arguments())).endsWith("foo/dir");
  }

  @Test
  public void testParamFileHasDirectoryAsInput() throws Exception {
    setSkylarkSemanticsOptions("--incompatible_expand_directories");
    SkylarkRuleContext ctx = dummyRuleContext();
    SkylarkList<?> result =
        (SkylarkList<?>)
            evalRuleContextCode(
                ctx,
                "args = ruleContext.actions.args()",
                "directory = ruleContext.actions.declare_directory('dir')",
                "args.add_all([directory])",
                "params = ruleContext.actions.declare_file('params')",
                "ruleContext.actions.write(params, args)",
                "params, directory");
    Artifact params = (Artifact) result.get(0);
    Artifact directory = (Artifact) result.get(1);
    ActionAnalysisMetadata action =
        ctx.getRuleContext().getAnalysisEnvironment().getLocalGeneratingAction(params);
    assertThat(action.getInputs()).contains(directory);
  }

  private void setupThrowFunction(BuiltinFunction func) throws Exception {
    throwFunction = func;
    throwFunction.configure(
        getClass().getDeclaredField("throwFunction").getAnnotation(SkylarkSignature.class));
    update("throw", throwFunction);
  }

  private static class ThereIsNoMessageException extends EvalException {
    public ThereIsNoMessageException() {
      super(null, "This is not the message you are looking for."); // Unused dummy message
    }

    @Override
    public String getMessage() {
      return "";
    }
  }
}
