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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DefaultProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.OsUtils;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkRuleImplementationFunctions.
 */
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
        // The two below are used by testResolveCommand
        "sh_binary(name = 'mytool',",
        "  srcs = ['mytool.sh'],",
        "  data = ['file1.dat', 'file2.dat'],",
        ")",
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
      assertThat(e).hasMessage(errorMsg);
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
    assertThat(action.getEnvironment()).isEqualTo(targetConfig.getLocalShellEnvironment());
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
        "expected file or PathFragment for executable but got string instead",
        "ruleContext.actions.run(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  executable = 'xyz.exe')");
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
        "env = {'a' : 'b'}",
        "ruleContext.actions.run_shell(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  env = env,",
        "  execution_requirements = env,",
        "  mnemonic = 'DummyMnemonic',",
        "  command = 'dummy_command',",
        "  progress_message = 'dummy_message')");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getEnvironment()).containsExactly("a", "b");
    assertThat(action.getExecutionInfo()).containsExactly("a", "b");
  }

  @Test
  public void testCreateSpawnActionUnknownParam() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "unexpected keyword 'bad_param', in method run("
            + "list outputs, string bad_param, File executable) of 'actions'",
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
  public void testCreateSpawnActionCommandsListTooShort() throws Exception {
    checkErrorContains(
        createRuleContext("//foo:foo"),
        "'command' list has to be of size at least 3",
        "ruleContext.actions.run_shell(",
        "  outputs = ruleContext.files.srcs,",
        "  command = ['dummy_command', '--arg'])");
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
        "parameter 'mnemonic' has no default value, in method do_nothing(list inputs) of 'actions'",
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
        "load('/test/empty', 'empty_action_rule')",
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
    assertArtifactFilenames(inputs, "mytool.sh", "mytool", "foo_Smytool-runfiles", "t.exe");
    @SuppressWarnings("unchecked")
    CompositeRunfilesSupplier runfilesSupplier =
        new CompositeRunfilesSupplier((List<RunfilesSupplier>) lookup("input_manifests"));
    assertThat(runfilesSupplier.getMappings()).hasSize(1);
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
    assertMatches("argv[1]", "^.*/resolve_me[.]script[.]sh$", argv.get(1));
  }

  @Test
  public void testBadParamTypeErrorMessage() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    checkErrorContains(
        ruleContext,
        "Cannot convert parameter 'content' to type string, in method "
            + "write(File output, int content, bool is_executable) of 'actions'",
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
        "  executable = False)");

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
   * "correct" UTF-8 value. Once {@link com.google.devtools.build.lib.syntax.ParserInputSource#create(com.google.devtools.build.lib.vfs.Path)}
   * parses files using UTF-8 and the hack for the substituations parameter is removed, this test
   * will fail.
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
        "  executable = False)");
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
        "expected depset of Files or NoneType for 'transitive_files' while calling runfiles "
            + "but got depset of ints instead: depset([1, 2, 3])",
        "ruleContext.runfiles(transitive_files=depset([1, 2, 3]))");
  }

  @Test
  public void testRunfilesBadMapGenericType() throws Exception {
    checkErrorContains(
        "expected type 'string' for 'symlinks' key " + "but got type 'int' instead",
        "ruleContext.runfiles(symlinks = {123: ruleContext.files.srcs[0]})");
    checkErrorContains(
        "expected type 'File' for 'symlinks' value " + "but got type 'int' instead",
        "ruleContext.runfiles(symlinks = {'some string': 123})");
    checkErrorContains(
        "expected type 'string' for 'root_symlinks' key " + "but got type 'int' instead",
        "ruleContext.runfiles(root_symlinks = {123: ruleContext.files.srcs[0]})");
    checkErrorContains(
        "expected type 'File' for 'root_symlinks' value " + "but got type 'int' instead",
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
    reporter.removeHandler(failFastHandler); // So it doesn't throw exception
    runfiles.getRunfilesInputs(reporter, null);
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
        "unexpected keyword 'bad_keyword' in call to runfiles(self: ctx, ",
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
        "target (rule class of 'java_library') " + "doesn't have provider 'my_provider'.",
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
    assertThat(provider).isInstanceOf(DefaultProvider.class);
    assertThat(((DefaultProvider) provider).getConstructor().getPrintableName())
        .isEqualTo("DefaultInfo");

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
    assertThat(provider).isInstanceOf(DefaultProvider.class);
    assertThat(((DefaultProvider) provider).getConstructor().getPrintableName())
        .isEqualTo("DefaultInfo");

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
    assertThat(provider).isInstanceOf(DefaultProvider.class);
    SkylarkClassObject defaultProvider = (DefaultProvider) provider;
    assertThat((defaultProvider).getConstructor().getPrintableName())
        .isEqualTo("DefaultInfo");
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
    try {
      getConfiguredTarget("//test:my_rule");
      fail();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("Invalid key for default provider: foo");
    }
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
    assertThat(provider).isInstanceOf(SkylarkClassObject.class);
    assertThat(((SkylarkClassObject) provider).getConstructor().getPrintableName())
        .isEqualTo("foo_provider");
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
    assertThat(provider).isInstanceOf(SkylarkClassObject.class);
    assertThat(((SkylarkClassObject) provider).getConstructor().getPrintableName())
        .isEqualTo("foo_provider");
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

    try {
      getConfiguredTarget("//test:my_rule");
      fail();
    } catch (AssertionError expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains("Object of type Target doesn't " + "contain declared provider unused_provider");
    }
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

    try {
      getConfiguredTarget("//test:my_rule");
      fail();
    } catch (AssertionError expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains(
              "Type Target only supports indexing " + "by object constructors, got string instead");
    }
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

    try {
      getConfiguredTarget("//test:my_rule");
      fail();
    } catch (AssertionError expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains("Object of type Target doesn't " + "contain declared provider unused_provider");
    }
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

    try {
      getConfiguredTarget("//test:my_rule");
      fail();
    } catch (AssertionError expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains(
              "Type Target only supports querying by object " + "constructors, got string instead");
    }
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
    assertThat(result).isEqualTo(MutableList.EMPTY);
  }

  @Test
  public void testDefinedMakeVariable() throws Exception {
    SkylarkRuleContext ctx = createRuleContext("//foo:baz");
    String java = (String) evalRuleContextCode(ctx, "ruleContext.var['JAVA']");
    // Get the last path segment
    java = java.substring(java.lastIndexOf('/'));
    assertThat(java).isEqualTo("/java" + OsUtils.executableExtension());
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
        "load('/test/glob', 'glob_rule')",
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
        "load('/test/glob', 'glob_rule')",
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
        "load('/test/rule', 'silly_rule')",
        "silly_rule(name = 'silly')");
    thrown.handleAssertionErrors(); // Compatibility with JUnit 4.11
    thrown.expect(AssertionError.class);
    thrown.expectMessage("<rule context for //test:silly> is not of type string or int or bool");
    getConfiguredTarget("//test:silly");
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
