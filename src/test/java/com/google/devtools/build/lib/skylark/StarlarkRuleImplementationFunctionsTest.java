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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Action;
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
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.skylark.Args;
import com.google.devtools.build.lib.analysis.skylark.StarlarkRuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skylark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
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

/** Tests for Starlark functions relating to rule implementation. */
@RunWith(JUnit4.class)
@SkylarkGlobalLibrary // needed for CallUtils.getBuiltinCallable, sadly
public class StarlarkRuleImplementationFunctionsTest extends BuildViewTestCase {

  private final EvaluationTestCase ev = new BazelEvaluationTestCase();

  private StarlarkRuleContext createRuleContext(String label) throws Exception {
    return new StarlarkRuleContext(
        getRuleContextForStarlark(getConfiguredTarget(label)), null, getStarlarkSemantics());
  }

  @Rule public ExpectedException thrown = ExpectedException.none();

  // def mock(mandatory, optional=None, *, mandatory_key, optional_key='x')
  @SkylarkCallable(
      name = "mock",
      documented = false,
      parameters = {
        @Param(name = "mandatory", doc = "", named = true),
        @Param(name = "optional", doc = "", defaultValue = "None", noneable = true, named = true),
        @Param(name = "mandatory_key", doc = "", positional = false, named = true),
        @Param(
            name = "optional_key",
            doc = "",
            defaultValue = "'x'",
            positional = false,
            named = true)
      },
      useStarlarkThread = true)
  public Object mock(
      Object mandatory,
      Object optional,
      Object mandatoryKey,
      Object optionalKey,
      StarlarkThread thread) {
    Map<String, Object> m = new HashMap<>();
    m.put("mandatory", mandatory);
    m.put("optional", optional);
    m.put("mandatory_key", mandatoryKey);
    m.put("optional_key", optionalKey);
    return m;
  }

  @Before
  public final void createBuildFile() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");

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
        // The target below is used by testResolveCommand and testResolveTools
        "sh_binary(name = 'mytool',",
        "  srcs = ['mytool.sh'],",
        "  data = ['file1.dat', 'file2.dat'],",
        ")",
        // The target below is used by testResolveCommand and testResolveTools
        "genrule(name = 'resolve_me',",
        "  cmd = 'aa',",
        "  tools = [':mytool', 't.exe'],",
        "  srcs = ['file3.dat', 'file4.dat'],",
        "  outs = ['r1.txt', 'r2.txt'],",
        ")");
  }

  private void setRuleContext(StarlarkRuleContext ctx) throws Exception {
    ev.update("ruleContext", ctx);
  }

  private static void assertArtifactFilenames(Iterable<Artifact> artifacts, String... expected) {
    ImmutableList.Builder<String> filenames = ImmutableList.builder();
    for (Artifact a : artifacts) {
      filenames.add(a.getFilename());
    }
    assertThat(filenames.build()).containsAtLeastElementsIn(Lists.newArrayList(expected));
  }

  private StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//myinfo:myinfo.bzl", ImmutableMap.of()), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  // Defines all @StarlarkCallable-annotated methods (mock, throw, ...) in the environment.
  private void defineTestMethods() throws Exception {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, this);
    for (Map.Entry<String, Object> entry : env.build().entrySet()) {
      ev.update(entry.getKey(), entry.getValue());
    }
  }

  private void checkStarlarkFunctionError(String errorSubstring, String line) throws Exception {
    defineTestMethods();
    EvalException e = assertThrows(EvalException.class, () -> ev.exec(line));
    assertThat(e).hasMessageThat().contains(errorSubstring);
  }

  // TODO(adonovan): move these tests of the interpreter core into lib.syntax.

  @Test
  public void testStarlarkFunctionPosArgs() throws Exception {
    defineTestMethods();
    ev.exec("a = mock('a', 'b', mandatory_key='c')");
    Map<?, ?> params = (Map<?, ?>) ev.lookup("a");
    assertThat(params.get("mandatory")).isEqualTo("a");
    assertThat(params.get("optional")).isEqualTo("b");
    assertThat(params.get("mandatory_key")).isEqualTo("c");
    assertThat(params.get("optional_key")).isEqualTo("x");
  }

  @Test
  public void testStarlarkFunctionKwArgs() throws Exception {
    defineTestMethods();
    ev.exec("a = mock(optional='b', mandatory='a', mandatory_key='c')");
    Map<?, ?> params = (Map<?, ?>) ev.lookup("a");
    assertThat(params.get("mandatory")).isEqualTo("a");
    assertThat(params.get("optional")).isEqualTo("b");
    assertThat(params.get("mandatory_key")).isEqualTo("c");
    assertThat(params.get("optional_key")).isEqualTo("x");
  }

  @Test
  public void testStarlarkFunctionTooFewArguments() throws Exception {
    checkStarlarkFunctionError(
        "missing 1 required positional argument: mandatory", "mock(mandatory_key='y')");
  }

  @Test
  public void testStarlarkFunctionTooManyArguments() throws Exception {
    checkStarlarkFunctionError(
        "mock() accepts no more than 2 positional arguments but got 3",
        "mock('a', 'b', 'c', mandatory_key='y')");
  }

  @Test
  public void testStarlarkFunctionAmbiguousArguments() throws Exception {
    checkStarlarkFunctionError(
        "mock() got multiple values for argument 'mandatory'",
        "mock('by position', mandatory='by_key', mandatory_key='c')");
  }

  @Test
  public void testCreateSpawnActionCreatesSpawnAction() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    createTestSpawnAction(ruleContext);
    ActionAnalysisMetadata action =
        Iterables.getOnlyElement(
            ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action).isInstanceOf(SpawnAction.class);
  }

  @Test
  public void testArtifactPath() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    String result = (String) ev.eval("ruleContext.files.tools[0].path");
    assertThat(result).isEqualTo("foo/t.exe");
  }

  @Test
  public void testArtifactShortPath() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    String result = (String) ev.eval("ruleContext.files.tools[0].short_path");
    assertThat(result).isEqualTo("foo/t.exe");
  }

  @Test
  public void testCreateSpawnActionArgumentsWithCommand() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    createTestSpawnAction(ruleContext);
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertArtifactFilenames(action.getInputs().toList(), "a.txt", "b.img");
    assertArtifactFilenames(action.getOutputs(), "a.txt", "b.img");
    MoreAsserts.assertContainsSublist(
        action.getArguments(), "-c", "dummy_command", "", "--a", "--b");
    assertThat(action.getMnemonic()).isEqualTo("DummyMnemonic");
    assertThat(action.getProgressMessage()).isEqualTo("dummy_message");
    assertThat(action.getIncompleteEnvironmentForTesting())
        .isEqualTo(targetConfig.getLocalShellEnvironment());
  }

  @Test
  public void testCreateSpawnActionArgumentsWithExecutable() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
        "ruleContext.actions.run(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  executable = ruleContext.files.tools[0])");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertArtifactFilenames(action.getInputs().toList(), "a.txt", "b.img", "t.exe");
    assertArtifactFilenames(action.getOutputs(), "a.txt", "b.img");
    MoreAsserts.assertContainsSublist(action.getArguments(), "foo/t.exe", "--a", "--b");
  }

  @Test
  public void testCreateActionWithDepsetInput() throws Exception {
    // Same test as above, with depset as inputs.
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
        "ruleContext.actions.run(",
        "  inputs = depset(ruleContext.files.srcs),",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  executable = ruleContext.files.tools[0])");
    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertArtifactFilenames(action.getInputs().toList(), "a.txt", "b.img", "t.exe");
    assertArtifactFilenames(action.getOutputs(), "a.txt", "b.img");
    MoreAsserts.assertContainsSublist(action.getArguments(), "foo/t.exe", "--a", "--b");
  }

  @Test
  public void testCreateSpawnActionArgumentsBadExecutable() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "got value of type 'int', want 'File or string or FilesToRunProvider'",
        "ruleContext.actions.run(",
        "  inputs = ruleContext.files.srcs,",
        "  outputs = ruleContext.files.srcs,",
        "  arguments = ['--a','--b'],",
        "  executable = 123)");
  }

  @Test
  public void testCreateSpawnActionShellCommandList() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "run() got unexpected keyword argument 'bad_param'",
        "f = ruleContext.actions.declare_file('foo.sh')",
        "ruleContext.actions.run(outputs=[], bad_param = 'some text', executable = f)");
  }

  private Object createTestSpawnAction(StarlarkRuleContext ruleContext) throws Exception {
    setRuleContext(ruleContext);
    return ev.eval(
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
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "at index 0 of outputs, got element of type string, want File",
        "l = ['a', 'b']",
        "ruleContext.actions.run_shell(",
        "  outputs = l,",
        "  command = 'dummy_command')");
  }

  @Test
  public void testRunShellArgumentsWithCommandSequence() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
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
    setStarlarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=false");
    setupToolInInputsTest(
        "output = ctx.actions.declare_file('bar.out')",
        "ctx.actions.run_shell(",
        "  inputs = ctx.attr.exe.files,",
        "  outputs = [output],",
        "  command = 'boo bar baz',",
        ")");
    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//bar:my_rule");
    SpawnAction action = (SpawnAction) Iterables.getOnlyElement(target.getActions());
    assertThat(action.getTools().toList()).isNotEmpty();
  }

  @Test
  public void testCreateSpawnActionWithToolAttribute() throws Exception {
    setStarlarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=true");
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
    assertThat(action.getTools().toList()).isNotEmpty();
  }

  @Test
  public void testCreateSpawnActionWithToolAttributeIgnoresToolsInInputs() throws Exception {
    setStarlarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=true");
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
    assertThat(action.getTools().toList()).isNotEmpty();
  }

  @Test
  public void testCreateSpawnActionWithToolInInputsFailAtAnalysisTime() throws Exception {
    setStarlarkSemanticsOptions("--incompatible_no_support_tools_in_action_inputs=true");
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    setRuleContext(createRuleContext("//foo:foo"));
    checkEmptyAction("mnemonic = 'test'");
    checkEmptyAction("mnemonic = 'test', inputs = ruleContext.files.srcs");
    checkEmptyAction("mnemonic = 'test', inputs = depset(ruleContext.files.srcs)");

    ev.checkEvalErrorContains(
        "do_nothing() missing 1 required named argument: mnemonic",
        "ruleContext.actions.do_nothing(inputs = ruleContext.files.srcs)");
  }

  private void checkEmptyAction(String namedArgs) throws Exception {
    assertThat(ev.eval(String.format("ruleContext.actions.do_nothing(%s)", namedArgs)))
        .isEqualTo(Starlark.NONE);
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:bar");
    setRuleContext(ruleContext);

    // If there is only a single target, both "location" and "locations" should work
    runExpansion("location :jl", "[blaze]*-out/.*/bin/foo/libjl.jar");
    runExpansion("locations :jl", "[blaze]*-out/.*/bin/foo/libjl.jar");

    runExpansion("location //foo:jl", "[blaze]*-out/.*/bin/foo/libjl.jar");

    // Multiple targets and "location" should result in an error
    checkReportedErrorStartsWith(
        "in genrule rule //foo:bar: label '//foo:gl' "
            + "in $(location) expression expands to more than one file, please use $(locations "
            + "//foo:gl) instead.",
        "ruleContext.expand_location('$(location :gl)')");

    // We have to use "locations" for multiple targets
    runExpansion(
        "locations :gl",
        "[blaze]*-out/.*/bin/foo/gl.a [blaze]*-out/.*/bin/foo/gl.gcgox");

    // LocationExpander just returns the input string if there is no label
    runExpansion("location", "\\$\\(location\\)");

    checkReportedErrorStartsWith(
        "in genrule rule //foo:bar: label '//foo:abc' in $(locations) expression "
            + "is not a declared prerequisite of this rule",
        "ruleContext.expand_location('$(locations :abc)')");
  }

  /** Regression test to check that expand_location allows ${var} and $$. */
  @Test
  public void testExpandLocationWithDollarSignsAndCurlys() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:bar");
    setRuleContext(ruleContext);
    assertThat((String) ev.eval("ruleContext.expand_location('${abc} $(echo) $$ $')"))
        .isEqualTo("${abc} $(echo) $$ $");
  }

  /**
   * Invokes ctx.expand_location() with the given parameters and checks whether this led to the
   * expected result
   *
   * @param command Either "location" or "locations". This only matters when the label has multiple
   *     targets
   * @param expectedPattern Regex pattern that matches the expected result
   */
  private void runExpansion(String command, String expectedPattern) throws Exception {
    assertMatches(
        "Expanded string",
        expectedPattern,
        (String) ev.eval(String.format("ruleContext.expand_location('$(%s)')", command)));
  }

  private void assertMatches(String description, String expectedPattern, String computedValue)
      throws Exception {
    assertWithMessage(
            Starlark.format(
                "%s %r did not match pattern '%s'", description, computedValue, expectedPattern))
        .that(Pattern.matches(expectedPattern, computedValue))
        .isTrue();
  }

  @Test
  public void testResolveCommandMakeVariables() throws Exception {
    setRuleContext(createRuleContext("//foo:resolve_me"));
    ev.exec(
        "inputs, argv, manifests = ruleContext.resolve_command(",
        "  command='I got the $(HELLO) on a $(DAVE)', ",
        "  make_variables={'HELLO': 'World', 'DAVE': type('')})");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (StarlarkList) ev.lookup("argv");
    assertThat(argv).hasSize(3);
    assertMatches("argv[0]", "^.*/bash" + OsUtils.executableExtension() + "$", argv.get(0));
    assertThat(argv.get(1)).isEqualTo("-c");
    assertThat(argv.get(2)).isEqualTo("I got the World on a string");
  }

  @Test
  public void testResolveCommandInputs() throws Exception {
    setRuleContext(createRuleContext("//foo:resolve_me"));
    ev.exec(
        "inputs, argv, input_manifests = ruleContext.resolve_command(",
        "   tools=ruleContext.attr.tools)");
    @SuppressWarnings("unchecked")
    List<Artifact> inputs = (List<Artifact>) (List<?>) (StarlarkList) ev.lookup("inputs");
    assertArtifactFilenames(
        inputs,
        "mytool.sh",
        "mytool",
        "foo_Smytool" + OsUtils.executableExtension() + "-runfiles",
        "t.exe");
    @SuppressWarnings("unchecked")
    RunfilesSupplier runfilesSupplier =
        CompositeRunfilesSupplier.fromSuppliers(
            (List<RunfilesSupplier>) ev.lookup("input_manifests"));
    assertThat(runfilesSupplier.getMappings(ArtifactPathResolver.IDENTITY)).hasSize(1);
  }

  @Test
  public void testResolveCommandExpandLocations() throws Exception {
    setRuleContext(createRuleContext("//foo:resolve_me"));
    ev.exec(
        "def foo():", // no for loops at top-level
        "  label_dict = {}",
        "  all = []",
        "  for dep in ruleContext.attr.srcs + ruleContext.attr.tools:",
        "    all.extend(dep.files.to_list())",
        "    label_dict[dep.label] = dep.files.to_list()",
        "  return ruleContext.resolve_command(",
        "    command='A$(locations //foo:mytool) B$(location //foo:file3.dat)',",
        "    attribute='cmd', expand_locations=True, label_dict=label_dict)",
        "inputs, argv, manifests = foo()");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (StarlarkList) ev.lookup("argv");
    assertThat(argv).hasSize(3);
    assertMatches("argv[0]", "^.*/bash" + OsUtils.executableExtension() + "$", argv.get(0));
    assertThat(argv.get(1)).isEqualTo("-c");
    assertMatches("argv[2]", "A.*/mytool .*/mytool.sh B.*file3.dat", argv.get(2));
  }

  @Test
  public void testResolveCommandExecutionRequirements() throws Exception {
    // Tests that requires-darwin execution requirements result in the usage of /bin/bash.
    setRuleContext(createRuleContext("//foo:resolve_me"));
    ev.exec(
        "inputs, argv, manifests = ruleContext.resolve_command(",
        "  execution_requirements={'requires-darwin': ''})");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (StarlarkList) ev.lookup("argv");
    assertMatches("argv[0]", "^/bin/bash$", argv.get(0));
  }

  @Test
  public void testResolveCommandScript() throws Exception {
    setRuleContext(createRuleContext("//foo:resolve_me"));
    ev.exec(
        "def foo():", // no for loops at top-level
        "  s = 'a'",
        "  for i in range(1,17): s = s + s", // 2**17 > CommandHelper.maxCommandLength (=64000)
        "  return ruleContext.resolve_command(",
        "    command=s)",
        "argv = foo()[1]");
    @SuppressWarnings("unchecked")
    List<String> argv = (List<String>) (List<?>) (StarlarkList) ev.lookup("argv");
    assertThat(argv).hasSize(2);
    assertMatches("argv[0]", "^.*/bash" + OsUtils.executableExtension() + "$", argv.get(0));
    assertMatches("argv[1]", "^.*/resolve_me[.][a-z0-9]+[.]script[.]sh$", argv.get(1));
  }

  @Test
  public void testResolveTools() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:resolve_me");
    setRuleContext(ruleContext);
    ev.exec(
        "inputs, input_manifests = ruleContext.resolve_tools(tools=ruleContext.attr.tools)",
        "ruleContext.actions.run(",
        "    outputs = [ruleContext.actions.declare_file('x.out')],",
        "    inputs = inputs,",
        "    input_manifests = input_manifests,",
        "    executable = 'dummy',",
        ")");
    assertArtifactFilenames(
        ((Depset) ev.lookup("inputs")).getSet(Artifact.class).toList(),
        "mytool.sh",
        "mytool",
        "foo_Smytool" + OsUtils.executableExtension() + "-runfiles",
        "t.exe");
    @SuppressWarnings("unchecked")
    RunfilesSupplier runfilesSupplier =
        CompositeRunfilesSupplier.fromSuppliers(
            (List<RunfilesSupplier>) ev.lookup("input_manifests"));
    assertThat(runfilesSupplier.getMappings(ArtifactPathResolver.IDENTITY)).hasSize(1);

    SpawnAction action =
        (SpawnAction)
            Iterables.getOnlyElement(
                ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsAtLeast(
            "mytool.sh",
            "mytool",
            "foo_Smytool" + OsUtils.executableExtension() + "-runfiles",
            "t.exe");
  }

  @Test
  public void testBadParamTypeErrorMessage() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "got value of type 'int', want 'string or Args'",
        "ruleContext.actions.write(",
        "  output = ruleContext.files.srcs[0],",
        "  content = 1,",
        "  is_executable = False)");
  }

  @Test
  public void testCreateTemplateAction() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
        "ruleContext.actions.expand_template(",
        "  template = ruleContext.files.srcs[0],",
        "  output = ruleContext.files.srcs[1],",
        "  substitutions = {'a': 'b'},",
        "  is_executable = False)");

    TemplateExpansionAction action = (TemplateExpansionAction) Iterables.getOnlyElement(
        ruleContext.getRuleContext().getAnalysisEnvironment().getRegisteredActions());
    assertThat(action.getInputs().getSingleton().getExecPathString()).isEqualTo("foo/a.txt");
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
   * com.google.devtools.build.lib.syntax.ParserInput#create(byte[],
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    setRuleContext(createRuleContext("//foo:bar"));
    Object result = ev.eval("ruleContext.runfiles(collect_default = True)");
    assertThat(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)))
        .contains("libjl.jar");
  }

  @Test
  public void testRunfilesBadListGenericType() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "at index 0 of files, got element of type string, want File",
        "ruleContext.runfiles(files = ['some string'])");
  }

  @Test
  public void testRunfilesBadSetGenericType() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "got a depset of 'int', expected a depset of 'File'",
        "ruleContext.runfiles(transitive_files=depset([1, 2, 3]))");
  }

  @Test
  public void testRunfilesBadMapGenericType() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "got dict<int, File> for 'symlinks', want dict<string, File>",
        "ruleContext.runfiles(symlinks = {123: ruleContext.files.srcs[0]})");
    ev.checkEvalErrorContains(
        "got dict<string, int> for 'symlinks', want dict<string, File>",
        "ruleContext.runfiles(symlinks = {'some string': 123})");
    ev.checkEvalErrorContains(
        "got dict<int, File> for 'root_symlinks', want dict<string, File>",
        "ruleContext.runfiles(root_symlinks = {123: ruleContext.files.srcs[0]})");
    ev.checkEvalErrorContains(
        "got dict<string, int> for 'root_symlinks', want dict<string, File>",
        "ruleContext.runfiles(root_symlinks = {'some string': 123})");
  }

  @Test
  public void testRunfilesArtifactsFromArtifact() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    Object result = ev.eval("ruleContext.runfiles(files = ruleContext.files.tools)");
    assertThat(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result))).contains("t.exe");
  }

  @Test
  public void testRunfilesArtifactsFromIterableArtifacts() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    Object result = ev.eval("ruleContext.runfiles(files = ruleContext.files.srcs)");
    assertThat(ImmutableList.of("a.txt", "b.img"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromNestedSetArtifacts() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    Object result =
        ev.eval("ruleContext.runfiles(transitive_files = depset(ruleContext.files.srcs))");
    assertThat(ImmutableList.of("a.txt", "b.img"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromDefaultAndFiles() throws Exception {
    setRuleContext(createRuleContext("//foo:bar"));
    // It would be nice to write [DEFAULT] + ruleContext.files.srcs, but artifacts
    // is an ImmutableList and Starlark interprets it as a tuple.
    Object result =
        ev.eval("ruleContext.runfiles(collect_default = True, files = ruleContext.files.srcs)");
    // From DEFAULT only libjl.jar comes, see testRunfilesAddFromDependencies().
    assertThat(ImmutableList.of("libjl.jar", "gl.a", "gl.gcgox"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromSymlink() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    Object result = ev.eval("ruleContext.runfiles(symlinks = {'sym1': ruleContext.files.srcs[0]})");
    assertThat(ImmutableList.of("a.txt"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesArtifactsFromRootSymlink() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    Object result =
        ev.eval("ruleContext.runfiles(root_symlinks = {'sym1': ruleContext.files.srcs[0]})");
    assertThat(ImmutableList.of("a.txt"))
        .isEqualTo(ActionsTestUtil.baseArtifactNames(getRunfileArtifacts(result)));
  }

  @Test
  public void testRunfilesSymlinkConflict() throws Exception {
    // Two different artifacts mapped to same path in runfiles
    setRuleContext(createRuleContext("//foo:foo"));
    ev.exec("prefix = ruleContext.workspace_name + '/' if ruleContext.workspace_name else ''");
    Object result =
        ev.eval(
            "ruleContext.runfiles(",
            "  root_symlinks = {prefix + 'sym1': ruleContext.files.srcs[0]},",
            "  symlinks = {'sym1': ruleContext.files.srcs[1]})");
    Runfiles runfiles = (Runfiles) result;
    reporter.removeHandler(failFastHandler); // So it doesn't throw an exception.
    runfiles.getRunfilesInputs(reporter, null, ArtifactPathResolver.IDENTITY);
    assertContainsEvent("ERROR <no location>: overwrote runfile");
  }

  private static Iterable<Artifact> getRunfileArtifacts(Object runfiles) {
    return ((Runfiles) runfiles).getAllArtifacts().toList();
  }

  @Test
  public void testRunfilesBadKeywordArguments() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "runfiles() got unexpected keyword argument 'bad_keyword'",
        "ruleContext.runfiles(bad_keyword = '')");
  }

  @Test
  public void testNsetContainsList() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "depset elements must not be mutable values", "depset([[ruleContext.files.srcs]])");
  }

  @Test
  public void testCmdJoinPaths() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    Object result = ev.eval("cmd_helper.join_paths(':', depset(ruleContext.files.srcs))");
    assertThat(result).isEqualTo("foo/a.txt:foo/b.img");
  }

  @Test
  public void testStructPlusArtifactErrorMessage() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "unsupported binary operation: File + struct",
        "ruleContext.files.tools[0] + struct(a = 1)");
  }

  @Test
  public void testNoSuchProviderErrorMessage() throws Exception {
    setRuleContext(createRuleContext("//foo:bar"));
    ev.checkEvalErrorContains(
        "<target //foo:jl> (rule 'java_library') doesn't have provider 'my_provider'",
        "ruleContext.attr.srcs[0].my_provider");
  }

  @Test
  public void testFilesForRuleConfiguredTarget() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    Object result = ev.eval("ruleContext.attr.srcs[0].files");
    assertThat(ActionsTestUtil.baseNamesOf(((Depset) result).getSet(Artifact.class)))
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    provider = ctx.attr.deps[0][DefaultInfo]",
        "    return [MyInfo(",
        "        is_provided = DefaultInfo in ctx.attr.deps[0],",
        "        provider = provider,",
        "        dir = str(sorted(dir(provider))),",
        "        rule_data_runfiles = provider.data_runfiles,",
        "        rule_default_runfiles = provider.default_runfiles,",
        "        rule_files = provider.files,",
        "        rule_files_to_run = provider.files_to_run,",
        "        rule_file_executable = provider.files_to_run.executable",
        "    )]",
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
    StructImpl myInfo = getMyInfoFromTarget(configuredTarget);
    assertThat((Boolean) myInfo.getValue("is_provided")).isTrue();

    Object provider = myInfo.getValue("provider");
    assertThat(provider).isInstanceOf(DefaultInfo.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(DefaultInfo.PROVIDER.getKey());

    assertThat(myInfo.getValue("dir"))
        .isEqualTo(
            "[\"data_runfiles\", \"default_runfiles\", \"files\", \"files_to_run\", \"to_json\", "
                + "\"to_proto\"]");

    assertThat(myInfo.getValue("rule_data_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) myInfo.getValue("rule_data_runfiles")).getAllArtifacts().toList(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(myInfo.getValue("rule_default_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) myInfo.getValue("rule_default_runfiles")).getAllArtifacts().toList(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(myInfo.getValue("rule_files")).isInstanceOf(Depset.class);
    assertThat(myInfo.getValue("rule_files_to_run")).isInstanceOf(FilesToRunProvider.class);
    assertThat(myInfo.getValue("rule_file_executable")).isEqualTo(Starlark.NONE);
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
        "    return [foo, default]",
        "foo_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'runs': attr.label_list(allow_files=True),",
        "    }",
        ")");
    scratch.file(
        "test/bar.bzl",
        "load(':foo.bzl', 'foo_provider')",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    provider = ctx.attr.deps[0][DefaultInfo]",
        "    return [MyInfo(",
        "        is_provided = DefaultInfo in ctx.attr.deps[0],",
        "        provider = provider,",
        "        dir = str(sorted(dir(provider))),",
        "        rule_data_runfiles = provider.data_runfiles,",
        "        rule_default_runfiles = provider.default_runfiles,",
        "        rule_files = provider.files,",
        "        rule_files_to_run = provider.files_to_run,",
        "    )]",
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
    StructImpl myInfo = getMyInfoFromTarget(configuredTarget);

    assertThat((Boolean) myInfo.getValue("is_provided")).isTrue();

    Object provider = myInfo.getValue("provider");
    assertThat(provider).isInstanceOf(DefaultInfo.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(DefaultInfo.PROVIDER.getKey());

    assertThat(myInfo.getValue("dir"))
        .isEqualTo(
            "[\"data_runfiles\", \"default_runfiles\", \"files\", \"files_to_run\", \"to_json\", "
                + "\"to_proto\"]");

    assertThat(myInfo.getValue("rule_data_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) myInfo.getValue("rule_data_runfiles")).getAllArtifacts().toList(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(myInfo.getValue("rule_default_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) myInfo.getValue("rule_default_runfiles")).getAllArtifacts().toList(),
                String::valueOf))
        .containsExactly(
            "File:[/workspace[source]]test/run.file", "File:[/workspace[source]]test/run2.file");

    assertThat(myInfo.getValue("rule_files")).isInstanceOf(Depset.class);
    assertThat(myInfo.getValue("rule_files_to_run")).isInstanceOf(FilesToRunProvider.class);
  }

  @Test
  public void testDefaultProviderInvalidConfiguration() throws Exception {
    setStarlarkSemanticsOptions("--incompatible_disallow_struct_provider_syntax=false");
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    provider = ctx.attr.deps[0][DefaultInfo]",
        "    return [MyInfo(",
        "        is_provided = DefaultInfo in ctx.attr.deps[0],",
        "        provider = provider,",
        "        dir = str(sorted(dir(provider))),",
        "        file_data_runfiles = provider.data_runfiles,",
        "        file_default_runfiles = provider.default_runfiles,",
        "        file_files = provider.files,",
        "        file_files_to_run = provider.files_to_run,",
        "    )]",
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
    StructImpl myInfo = getMyInfoFromTarget(configuredTarget);

    assertThat((Boolean) myInfo.getValue("is_provided")).isTrue();

    Object provider = myInfo.getValue("provider");
    assertThat(provider).isInstanceOf(DefaultInfo.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(DefaultInfo.PROVIDER.getKey());

    assertThat(myInfo.getValue("dir"))
        .isEqualTo(
            "[\"data_runfiles\", \"default_runfiles\", \"files\", \"files_to_run\", \"to_json\", "
                + "\"to_proto\"]");

    assertThat(myInfo.getValue("file_data_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) myInfo.getValue("file_data_runfiles")).getAllArtifacts().toList(),
                String::valueOf))
        .isEmpty();

    assertThat(myInfo.getValue("file_default_runfiles")).isInstanceOf(Runfiles.class);
    assertThat(
            Iterables.transform(
                ((Runfiles) myInfo.getValue("file_default_runfiles")).getAllArtifacts().toList(),
                String::valueOf))
        .isEmpty();

    assertThat(myInfo.getValue("file_files")).isInstanceOf(Depset.class);
    assertThat(myInfo.getValue("file_files_to_run")).isInstanceOf(FilesToRunProvider.class);
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[DefaultInfo]", // The goal is to test this object
        "    return [MyInfo(", // so we return it here
        "        default = provider,",
        "    )]",
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
    Object provider = getMyInfoFromTarget(configuredTarget).getValue("default");
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
        .contains("DefaultInfo() got unexpected keyword argument 'foo'");
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[foo_provider]", // The goal is to test this object
        "    return [MyInfo(proxy = provider)]", // so we return it here
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
    Object provider = getMyInfoFromTarget(configuredTarget).getValue("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new StarlarkProvider.Key(
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    proxy = dep[FooInfo]", // The goal is to test this object
        "    return [MyInfo(proxy = proxy)]", // so we return it here
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
    Object provider = getMyInfoFromTarget(configuredTarget).getValue("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new StarlarkProvider.Key(
                Label.parseAbsolute("//test:foo.bzl", ImmutableMap.of()), "FooInfo"));
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
        "    return [default]",
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
        "    return [MyFooInfo]",
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
        "    return []",
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[foo_provider]", // The goal is to test this object
        "    return [MyInfo(proxy = provider)]", // so we return it here
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
    Object provider = getMyInfoFromTarget(configuredTarget).getValue("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new StarlarkProvider.Key(
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "    dep = ctx.attr.deps[0]",
        "    provider = dep[foo_provider]", // The goal is to test this object
        "    return [MyInfo(proxy = provider)]", // so we return it here
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
        "foo_rule(name = 'foo_rule')",
        "alias(name = 'dep_rule', actual=':foo_rule')",
        "bar_rule(name = 'my_rule', deps = [':dep_rule'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    Object provider = getMyInfoFromTarget(configuredTarget).getValue("proxy");
    assertThat(provider).isInstanceOf(StructImpl.class);
    assertThat(((StructImpl) provider).getProvider().getKey())
        .isEqualTo(
            new StarlarkProvider.Key(
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
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
        "    return [MyInfo(",
        "        foo = (foo_provider in dep),", // Should be true
        "        bar = (bar_provider in dep),", // Should be false
        "    )]",
        "outer_rule = rule(",
        "    implementation = _outer_impl,",
        "    attrs = {",
        "       'deps': attr.label_list(),",
        "    }",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':foo.bzl', 'inner_rule', 'outer_rule')",
        "inner_rule(name = 'dep_rule')",
        "outer_rule(name = 'my_rule', deps = [':dep_rule'])");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//test:my_rule");
    StructImpl myInfo = getMyInfoFromTarget(configuredTarget);

    Object foo = myInfo.getValue("foo");
    assertThat(foo).isInstanceOf(Boolean.class);
    assertThat((Boolean) foo).isTrue();
    Object bar = myInfo.getValue("bar");
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
  public void testReturnNonExportedProvider() throws Exception {
    scratch.file(
        "test/my_rule.bzl",
        "def _rule_impl(ctx):",
        "    foo_provider = provider()",
        "    foo = foo_provider()",
        "    return [foo]",
        "",
        "my_rule = rule(",
        "    implementation = _rule_impl,",
        ")");
    scratch.file("test/BUILD", "load(':my_rule.bzl', 'my_rule')", "my_rule(name = 'my_rule')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:my_rule"));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "cannot return a non-exported provider instance from a rule implementation function.");
  }

  @Test
  public void testFilesForFileConfiguredTarget() throws Exception {
    setRuleContext(createRuleContext("//foo:bar"));
    Object result = ev.eval("ruleContext.attr.srcs[0].files");
    assertThat(ActionsTestUtil.baseNamesOf(((Depset) result).getSet(Artifact.class)))
        .isEqualTo("libjl.jar");
  }

  @Test
  public void testCtxStructFieldsCustomErrorMessages() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains("No attribute 'foo' in attr.", "ruleContext.attr.foo");
    ev.checkEvalErrorContains("No attribute 'foo' in outputs.", "ruleContext.outputs.foo");
    ev.checkEvalErrorContains("No attribute 'foo' in files.", "ruleContext.files.foo");
    ev.checkEvalErrorContains("No attribute 'foo' in file.", "ruleContext.file.foo");
    ev.checkEvalErrorContains("No attribute 'foo' in executable.", "ruleContext.executable.foo");
  }

  @Test
  public void testBinDirPath() throws Exception {
    StarlarkRuleContext ctx = createRuleContext("//foo:bar");
    setRuleContext(ctx);
    Object result = ev.eval("ruleContext.bin_dir.path");
    assertThat(result).isEqualTo(ctx.getConfiguration().getBinFragment().getPathString());
  }

  @Test
  public void testEmptyLabelListTypeAttrInCtx() throws Exception {
    setRuleContext(createRuleContext("//foo:baz"));
    Object result = ev.eval("ruleContext.attr.srcs");
    assertThat(result).isEqualTo(StarlarkList.empty());
  }

  @Test
  public void testDefinedMakeVariable() throws Exception {
    useConfiguration("--define=FOO=bar");
    setRuleContext(createRuleContext("//foo:baz"));
    String foo = (String) ev.eval("ruleContext.var['FOO']");
    assertThat(foo).isEqualTo("bar");
  }

  @Test
  public void testCodeCoverageConfigurationAccess() throws Exception {
    StarlarkRuleContext ctx = createRuleContext("//foo:baz");
    setRuleContext(ctx);
    boolean coverage = (Boolean) ev.eval("ruleContext.configuration.coverage_enabled");
    assertThat(ctx.getRuleContext().getConfiguration().isCodeCoverageEnabled()).isEqualTo(coverage);
  }

  /** Checks whether the given (invalid) statement leads to the expected error */
  private void checkReportedErrorStartsWith(String errorMsg, String... statements)
      throws Exception {
    // If the component under test relies on Reporter and EventCollector for error handling, any
    // error would lead to an asynchronous AssertionFailedError thanks to failFastHandler in
    // FoundationTestCase.
    //
    // Consequently, we disable failFastHandler and check all events for the expected error message
    reporter.removeHandler(failFastHandler);

    Object result = ev.eval(statements);

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

  @SkylarkCallable(name = "throw1", documented = false)
  public Object throw1() throws Exception {
    class ThereIsNoMessageException extends EvalException {
      ThereIsNoMessageException() {
        super(null, "This is not the message you are looking for."); // Unused dummy message
      }

      @Override
      public String getMessage() {
        return "";
      }
    }
    throw new ThereIsNoMessageException();
  }

  @Test
  public void testStackTraceWithoutOriginalMessage() throws Exception {
    defineTestMethods();
    ev.checkEvalErrorContains(
        "There Is No Message: StarlarkRuleImplementationFunctionsTest", "throw1()");
  }

  @SkylarkCallable(name = "throw2", documented = false)
  public Object throw2() throws Exception {
    throw new InterruptedException();
  }

  @Test
  public void testNoStackTraceOnInterrupt() throws Exception {
    defineTestMethods();
    assertThrows(InterruptedException.class, () -> ev.eval("throw2()"));
  }

  @Test
  public void testGlobInImplicitOutputs() throws Exception {
    scratch.file(
        "test/glob.bzl",
        "def _impl(ctx):",
        "  ctx.actions.do_nothing(",
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
    assertContainsEvent("The native module can be accessed only from a BUILD thread.");
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
    // This confusing message shows why we should distinguish
    // built-ins and Starlark functions in their repr strings.
    thrown.expectMessage(
        "in call to rule(), parameter 'implementation' got value of type 'function', want"
            + " 'function'");
    getConfiguredTarget("//test:silly");
  }

  @Test
  public void testArgsScalarAdd() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "Args.add() doesn't accept vectorized arguments",
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
  public void testMultipleLazyArgsMixedWithStrings() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "Invalid value for parameter \"param_file_arg\": Expected string with a single \"--file=\"",
        "args = ruleContext.actions.args()\n" + "args.use_param_file('--file=')");
    ev.checkEvalErrorContains(
        "Invalid value for parameter \"param_file_arg\": "
            + "Expected string with a single \"--file=%s%s\"",
        "args = ruleContext.actions.args()\n" + "args.use_param_file('--file=%s%s')");
  }

  @Test
  public void testLazyArgsWithParamFileInvalidFormat() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "Invalid value for parameter \"format\": Expected one of \"shell\", \"multiline\"",
        "args = ruleContext.actions.args()\n" + "args.set_param_file_format('illegal')");
  }

  @Test
  public void testArgsAddInvalidTypesForArgAndValues() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "expected value of type 'string' for arg name, got 'Integer'",
        "args = ruleContext.actions.args()",
        "args.add(1, 'value')");
    ev.checkEvalErrorContains(
        "expected value of type 'string' for arg name, got 'Integer'",
        "args = ruleContext.actions.args()",
        "args.add_all(1, [1, 2])");
    ev.checkEvalErrorContains(
        "expected value of type 'sequence or depset' for values, got 'Integer'",
        "args = ruleContext.actions.args()",
        "args.add_all(1)");
    ev.checkEvalErrorContains(
        "in call to add_all(), parameter 'values' got value of type 'int', want 'sequence or"
            + " depset'",
        "args = ruleContext.actions.args()",
        "args.add_all('--foo', 1)");
  }

  @Test
  public void testLazyArgIllegalFormatString() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
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
  public void testMapEachAcceptsBuiltinFunction() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    // map_each accepts a non-Starlark built-in function such as str.
    ev.exec("ruleContext.actions.args().add_all(['foo'], map_each = str)");
  }

  @Test
  public void testLazyArgMapEachThrowsError() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    CommandLineExpansionException e =
        assertThrows(CommandLineExpansionException.class, () -> action.getArguments());
    assertThat(e).hasMessageThat().contains("'string' value has no field or method 'nosuchmethod'");
  }

  @Test
  public void testLazyArgMapEachReturnsNone() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
    CommandLineExpansionException e =
        assertThrows(CommandLineExpansionException.class, () -> action.getArguments());
    assertThat(e.getMessage())
        .contains("Expected map_each to return string, None, or list of strings, found Integer");
  }

  @Test
  public void createShellWithLazyArgs() throws Exception {
    StarlarkRuleContext ruleContext = createRuleContext("//foo:foo");
    setRuleContext(ruleContext);
    ev.exec(
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
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _main_impl(ctx):",
        "  dep = ctx.attr.deps[0]",
        "  args = dep[MyInfo].dep_arg",
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
        "  return [MyInfo(dep_arg = args)]",
        "dep_rule = rule(implementation = _dep_impl)");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:main"));
    assertThat(e).hasMessageThat().contains("trying to mutate a frozen Args value");
  }

  @Test
  public void testConfigurationField_StarlarkSplitTransitionProhibited() throws Exception {
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
        "  return []",
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

    setStarlarkSemanticsOptions("--experimental_starlark_config_transitions=true");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:foo");
    assertContainsEvent("late-bound attributes must not have a split configuration transition");
  }

  @Test
  public void testConfigurationField_NativeSplitTransitionProviderProhibited() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def _foo_impl(ctx):",
        "  return []",
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
        "  return []",
        "",
        "foo = rule(",
        "  implementation = _foo_impl,",
        "  attrs = {",
        "    '_attr': attr.label(",
        "        cfg = android_common.multi_cpu_configuration,",
        "        default = configuration_field(fragment='cpp', name = 'cc_toolchain'))})");
    setStarlarkSemanticsOptions("--experimental_google_legacy_api");

    scratch.file("test/BUILD", "load('//test:rule.bzl', 'foo')", "foo(name='foo')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:foo");
    assertContainsEvent("late-bound attributes must not have a split configuration transition");
  }

  @Test
  public void testConfigurationField_invalidFragment() throws Exception {
    scratch.file(
        "test/main_rule.bzl",
        "def _impl(ctx):",
        "  return []",
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
    scratch.file(
        "test/main_rule.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _impl(ctx):",
        "  return [MyInfo(platform = ctx.fragments.apple.single_arch_platform)]",
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
    scratch.file(
        "test/main_rule.bzl",
        "def _impl(ctx):",
        "  return []",
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
    scratch.file(
        "test/main_rule.bzl",
        "def _impl(ctx):",
        "  return []",
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

  @Test
  public void testFilesToRunInActionsRun() throws Exception {
    scratch.file(
        "a/a.bzl",
        "def _impl(ctx):",
        "    f = ctx.actions.declare_file('output')",
        "    ctx.actions.run(",
        "        inputs = [],",
        "        outputs = [f],",
        "        executable = ctx.attr._tool[DefaultInfo].files_to_run)",
        "    return [DefaultInfo(files=depset([f]))]",
        "r = rule(implementation=_impl, attrs = {'_tool': attr.label(default='//a:tool')})");

    scratch.file(
        "a/BUILD",
        "load(':a.bzl', 'r')",
        "r(name='r')",
        "sh_binary(name='tool', srcs=['tool.sh'], data=['data'])");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    Action action =
        getGeneratingAction(r.getProvider(FileProvider.class).getFilesToBuild().getSingleton());
    assertThat(ActionsTestUtil.baseArtifactNames(action.getRunfilesSupplier().getArtifacts()))
        .containsAtLeast("tool", "tool.sh", "data");
  }

  @Test
  public void testFilesToRunInActionsTools() throws Exception {
    scratch.file(
        "a/a.bzl",
        "def _impl(ctx):",
        "    f = ctx.actions.declare_file('output')",
        "    ctx.actions.run(",
        "        inputs = [],",
        "        outputs = [f],",
        "        tools = [ctx.attr._tool[DefaultInfo].files_to_run],",
        "        executable = 'a/tool')",
        "    return [DefaultInfo(files=depset([f]))]",
        "r = rule(implementation=_impl, attrs = {'_tool': attr.label(default='//a:tool')})");

    scratch.file(
        "a/BUILD",
        "load(':a.bzl', 'r')",
        "r(name='r')",
        "sh_binary(name='tool', srcs=['tool.sh'], data=['data'])");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    Action action =
        getGeneratingAction(r.getProvider(FileProvider.class).getFilesToBuild().getSingleton());
    assertThat(ActionsTestUtil.baseArtifactNames(action.getRunfilesSupplier().getArtifacts()))
        .containsAtLeast("tool", "tool.sh", "data");
  }

  // Verifies that configuration_field can only be used on 'label' attributes.
  @Test
  public void testConfigurationField_invalidAttributeType() throws Exception {
    scratch.file(
        "test/main_rule.bzl",
        "def _impl(ctx):",
        "  return []",
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
            "in call to int(), parameter 'default' got value of type 'LateBoundDefault', want"
                + " 'int'");
  }

  @Test
  public void testStarlarkCustomCommandLineKeyComputation() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));

    ImmutableList.Builder<CommandLine> commandLines = ImmutableList.builder();

    commandLines.add(getCommandLine("args = ruleContext.actions.args()"));
    commandLines.add(getCommandLine("args = ruleContext.actions.args()", "args.add('foo')"));
    commandLines.add(
        getCommandLine("args = ruleContext.actions.args()", "args.add('--foo', 'foo')"));
    commandLines.add(
        getCommandLine("args = ruleContext.actions.args()", "args.add('foo', format='--foo=%s')"));
    commandLines.add(
        getCommandLine("args = ruleContext.actions.args()", "args.add_all(['foo', 'bar'])"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()", "args.add_all('-foo', ['foo', 'bar'])"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['foo', 'bar'], format_each='format%s')"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()", "args.add_all(['foo', 'bar'], before_each='-I')"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()", "args.add_all(['boing', 'boing', 'boing'])"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['boing', 'boing', 'boing'], uniquify=True)"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_all(['foo', 'bar'], terminate_with='baz')"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()", "args.add_joined(['foo', 'bar'], join_with=',')"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "args.add_joined(['foo', 'bar'], join_with=',', format_joined='--foo=%s')"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "def _map_each(s): return s + '_mapped'",
            "args.add_all(['foo', 'bar'], map_each=_map_each)"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "values = depset(['a', 'b'])",
            "args.add_all(values)"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "def _map_each(s): return s + '_mapped'",
            "values = depset(['a', 'b'])",
            "args.add_all(values, map_each=_map_each)"));
    commandLines.add(
        getCommandLine(
            "args = ruleContext.actions.args()",
            "def _map_each(s): return s + '_mapped_again'",
            "values = depset(['a', 'b'])",
            "args.add_all(values, map_each=_map_each)"));

    // Ensure all these command lines have distinct keys
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Map<String, CommandLine> digests = new HashMap<>();
    for (CommandLine commandLine : commandLines.build()) {
      Fingerprint fingerprint = new Fingerprint();
      commandLine.addToFingerprint(actionKeyContext, fingerprint);
      String digest = fingerprint.hexDigestAndReset();
      CommandLine previous = digests.putIfAbsent(digest, commandLine);
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
    CommandLine commandLine =
        getCommandLine(
            "args = ruleContext.actions.args()",
            "def _bad_fn(s): return s.doesnotexist()",
            "values = depset(['a', 'b'])",
            "args.add_all(values, map_each=_bad_fn)");
    assertThrows(
        CommandLineExpansionException.class,
        () -> commandLine.addToFingerprint(actionKeyContext, new Fingerprint()));
  }

  private CommandLine getCommandLine(String... lines) throws Exception {
    ev.exec(lines);
    return ((Args) ev.eval("args")).build();
  }

  @Test
  public void testPrintArgs() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.exec("args = ruleContext.actions.args()", "args.add_all(['--foo', '--bar'])");
    Args args = (Args) ev.eval("args");
    assertThat(Printer.getPrinter().debugPrint(args).toString()).isEqualTo("--foo --bar");
  }

  @Test
  public void testDirectoryInArgs() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.exec(
        "args = ruleContext.actions.args()",
        "directory = ruleContext.actions.declare_directory('dir')",
        "def _short_path(f): return f.short_path", // For easier assertions
        "args.add_all([directory], map_each=_short_path)");
    Sequence<?> result = (Sequence<?>) ev.eval("args, directory");
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
  public void testDirectoryInArgsExpandDirectories() throws Exception {
    setRuleContext(createRuleContext("//foo:foo"));
    ev.exec(
        "args = ruleContext.actions.args()",
        "directory = ruleContext.actions.declare_directory('dir')",
        "def _short_path(f): return f.short_path", // For easier assertions
        "args.add_all([directory], map_each=_short_path, expand_directories=True)",
        "args.add_all([directory], map_each=_short_path, expand_directories=False)");
    Sequence<?> result = (Sequence<?>) ev.eval("args, directory");
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
    setRuleContext(createRuleContext("//foo:foo"));
    ev.checkEvalErrorContains(
        "Cannot add directories to Args#add",
        "args = ruleContext.actions.args()",
        "directory = ruleContext.actions.declare_directory('dir')",
        "args.add(directory)");
  }

  @Test
  public void testParamFileHasDirectoryAsInput() throws Exception {
    StarlarkRuleContext ctx = createRuleContext("//foo:foo");
    setRuleContext(ctx);
    ev.exec(
        "args = ruleContext.actions.args()",
        "directory = ruleContext.actions.declare_directory('dir')",
        "args.add_all([directory])",
        "params = ruleContext.actions.declare_file('params')",
        "ruleContext.actions.write(params, args)");
    Sequence<?> result = (Sequence<?>) ev.eval("params, directory");
    Artifact params = (Artifact) result.get(0);
    Artifact directory = (Artifact) result.get(1);
    ActionAnalysisMetadata action =
        ctx.getRuleContext().getAnalysisEnvironment().getLocalGeneratingAction(params);
    assertThat(action.getInputs().toList()).contains(directory);
  }
}
