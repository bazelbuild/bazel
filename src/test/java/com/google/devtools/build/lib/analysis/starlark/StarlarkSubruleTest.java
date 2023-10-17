// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.analysis.starlark.StarlarkSubrule.getRuleAttrName;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkSubruleApi;
import com.google.devtools.build.lib.testutil.TestConstants;
import net.starlark.java.eval.BuiltinFunction;
import net.starlark.java.eval.Sequence;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class StarlarkSubruleTest extends BuildViewTestCase {

  private final BazelEvaluationTestCase ev = new BazelEvaluationTestCase("//subrule_testing:label");
  private final BazelEvaluationTestCase evOutsideAllowlist =
      new BazelEvaluationTestCase("//foo:bar");

  @Test
  public void testSubruleFunctionSymbol_notVisibleInBUILD() throws Exception {
    scratch.file("foo/BUILD", "subrule");

    checkLoadingPhaseError("//foo", "'subrule' is not defined");
  }

  @Test
  // checks that 'subrule' symbol visibility in bzl files, not whether it's callable
  public void testSubruleFunctionSymbol_isVisibleInBzl() throws Exception {
    Object subruleFunction = ev.eval("subrule");

    assertNoEvents();
    assertThat(subruleFunction).isNotNull();
    assertThat(subruleFunction).isInstanceOf(BuiltinFunction.class);
  }

  @Test
  public void testSubruleInstantiation_inAllowlistedPackage_succeeds() throws Exception {
    Object subrule = ev.eval("subrule(implementation = lambda : 0 )");

    assertThat(subrule).isNotNull();
    assertThat(subrule).isInstanceOf(StarlarkSubruleApi.class);
  }

  @Test
  public void testSubrule_isCallableOnlyFromRuleOrAspectImplementation() throws Exception {
    ev.execAndExport("x = subrule(implementation = lambda : 'dummy result')");

    ev.checkEvalErrorContains("x can only be called from a rule or aspect implementation", "x()");
  }

  @Test
  public void testSubrule_isCallableOnlyAfterExport() throws Exception {
    ev.checkEvalErrorContains(
        "Invalid subrule hasn't been exported by a bzl file",
        "unexported = [subrule(implementation = lambda: None)]",
        "unexported[0]()");
  }

  @Test
  public void testSubrule_ruleMustDeclareSubrule() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "_my_subrule = subrule(implementation = lambda : '')",
        "",
        "def _rule_impl(ctx):",
        "  _my_subrule()",
        "",
        "my_rule = rule(implementation = _rule_impl)");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error)
        .hasMessageThat()
        .contains(
            "Error in _my_subrule: rule 'my_rule' must declare '_my_subrule' in" + " 'subrules'");
  }

  @Test
  public void testSubrule_aspectMustDeclareSubrule() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "_my_subrule = subrule(implementation = lambda ctx: 'dummy aspect result')",
        "",
        "def _aspect_impl(ctx,target):",
        "  res = _my_subrule()",
        "",
        "_my_aspect = aspect(implementation = _aspect_impl)",
        "",
        "my_rule = rule(",
        "  implementation = lambda ctx: [],",
        "  attrs = {'dep' : attr.label(mandatory = True, aspects = [_my_aspect])},",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "java_library(name = 'bar')",
        "my_rule(name = 'foo', dep = 'bar')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error)
        .hasMessageThat()
        .contains(
            "Error in _my_subrule: aspect '//subrule_testing:myrule.bzl%_my_aspect' must"
                + " declare '_my_subrule' in 'subrules'");
  }

  @Test
  public void testSubrule_implementationMustAcceptSubruleContext() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "_my_subrule = subrule(implementation = lambda : '')",
        "",
        "def _rule_impl(ctx):",
        "  _my_subrule()",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error)
        .hasMessageThat()
        .contains("Error: lambda() does not accept positional arguments, but got 1");
  }

  @Test
  public void testSubrule_isCallableFromRule() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "_my_subrule = subrule(implementation = lambda ctx: 'dummy rule result')",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    StructImpl provider =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo");

    assertThat(provider).isNotNull();
    assertThat(provider.getValue("result")).isEqualTo("dummy rule result");
  }

  @Test
  public void testSubrule_isCallableFromAspect() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "_my_subrule = subrule(implementation = lambda ctx: 'dummy aspect result')",
        "",
        "MyInfo = provider()",
        "def _aspect_impl(ctx,target):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "_my_aspect = aspect(implementation = _aspect_impl, subrules = [_my_subrule])",
        "",
        "my_rule = rule(",
        "  implementation = lambda ctx: [ctx.attr.dep[MyInfo]],",
        "  attrs = {'dep' : attr.label(mandatory = True, aspects = [_my_aspect])},",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "java_library(name = 'bar')",
        "my_rule(name = 'foo', dep = 'bar')");

    StructImpl provider =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo");

    assertThat(provider).isNotNull();
    assertThat(provider.getValue("result")).isEqualTo("dummy aspect result");
  }

  @Test
  public void testSubrule_subruleContextExposesRuleLabel() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  return 'called in: ' + str(ctx.label)",
        "_my_subrule = subrule(implementation = _subrule_impl)",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    StructImpl provider =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo");

    assertThat(provider).isNotNull();
    assertThat(provider.getValue("result")).isEqualTo("called in: @@//subrule_testing:foo");
  }

  @Test
  public void testSubrule_subruleContextExposesActionsApi() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.write(out, 'subrule file content')",
        "  return out",
        "_my_subrule = subrule(implementation = _subrule_impl)",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    Artifact artifact =
        (Artifact)
            getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo")
                .getValue("result");

    assertThat(artifact).isNotNull();
    assertThat(artifact.getFilename()).isEqualTo("foo.out");
    assertThat(((FileWriteAction) getGeneratingAction(artifact)).getFileContents())
        .isEqualTo("subrule file content");
  }

  @Test
  public void testSubruleActions_run_doesNotAllowSettingToolchain() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run(toolchain = 'foo', executable = '/path/to/tool', outputs = [out])",
        "",
        "_my_subrule = subrule(implementation = _subrule_impl)",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  _my_subrule()",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error).hasMessageThat().contains("'toolchain' may not be specified in subrules");
  }

  @Test
  public void testSubruleActions_run_doesNotAllowSettingExecGroup() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run(exec_group = 'foo', executable = '/path/to/tool', outputs = [out])",
        "",
        "_my_subrule = subrule(implementation = _subrule_impl)",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  _my_subrule()",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error).hasMessageThat().contains("'exec_group' may not be specified in subrules");
  }

  @Test
  public void testSubruleContext_cannotBeUsedOutsideImplementationFunction() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  return ctx",
        "",
        "_my_subrule = subrule(implementation = _subrule_impl)",
        "",
        "def _rule_impl(ctx):",
        "  subrule_ctx = _my_subrule()",
        "  subrule_ctx.label",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error)
        .hasMessageThat()
        .contains(
            "Error: cannot access field or method 'label' of subrule context outside of its own"
                + " implementation function");
  }

  @Test
  public void testRuleContext_cannotBeUsedInSubruleImplementation() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, rule_ctx):",
        "  rule_ctx.label",
        "",
        "_my_subrule = subrule(implementation = _subrule_impl)",
        "",
        "def _rule_impl(ctx):",
        "  subrule_ctx = _my_subrule(ctx)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error)
        .hasMessageThat()
        .contains(
            "Error: cannot access field or method 'label' of rule context for"
                + " '//subrule_testing:foo' outside of its own rule implementation function");
  }

  @Test
  public void testSubruleAttrs_publicAttributesAreNotPermitted() throws Exception {
    ev.checkEvalErrorContains(
        "illegal attribute name 'foo': subrules may only define private attributes",
        "subrule(implementation = lambda: None, attrs = {'foo': attr.string()})");
  }

  @Test
  public void testSubruleAttrs_computedDefaultsAreNotPermitted() throws Exception {
    ev.checkEvalErrorContains(
        "for attribute '_foo': subrules cannot define computed defaults.",
        "subrule(",
        "  implementation = lambda: None,",
        "  attrs = {'_foo': attr.label(default = lambda: '')}",
        ")");
  }

  @Test
  public void testSubruleAttrs_onlyLabelsOrLabelListsPermitted() throws Exception {
    ev.checkEvalErrorContains(
        "bad type for attribute '_foo': subrule attributes may only be label or lists of labels.",
        "subrule(",
        "  implementation = lambda: None,",
        "  attrs = {'_foo': attr.int()}",
        ")");
  }

  @Test
  public void testSubruleAttrs_attributeMustHaveDefaultValue() throws Exception {
    ev.checkEvalErrorContains(
        "for attribute '_foo': no default value specified",
        "subrule(",
        "  implementation = lambda: None,",
        "  attrs = {'_foo': attr.label()}",
        ")");
  }

  @Test
  public void testSubruleAttrs_cannotHaveStarlarkTransitions() throws Exception {
    ev.checkEvalErrorContains(
        "bad cfg for attribute '_foo': subrules may only have target/exec attributes.",
        "my_transition = transition(implementation = lambda: None, inputs = [], outputs = [])",
        "_my_subrule = subrule(",
        "  implementation = lambda: None,",
        "  attrs = {'_foo': attr.label(cfg = my_transition)}",
        ")");
  }

  @Test
  public void testSubruleAttrs_cannotHaveNativeTransitions() throws Exception {
    ev.update("native_transition", (StarlarkExposedRuleTransitionFactory) data -> null);
    ev.checkEvalErrorContains(
        "bad cfg for attribute '_foo': subrules may only have target/exec attributes.",
        "_my_subrule = subrule(",
        "  implementation = lambda: None,",
        "  attrs = {'_foo': attr.label(cfg = native_transition)}",
        ")");
  }

  @Test
  public void testSubruleAttrs_notVisibleInRuleCtx() throws Exception {
    scratch.file("default/BUILD", "genrule(name = 'default', outs = ['a'], cmd = '')");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  return ",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_foo' : attr.label(default = '//default')},",
        ")",
        "MyInfo=provider()",
        "def _rule_impl(ctx):",
        "  res = dir(ctx.attr)",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    ImmutableList<String> ruleClassAttributes =
        getRuleContext(getConfiguredTarget("//subrule_testing:foo"))
            .getRule()
            .getRuleClassObject()
            .getAttributes()
            .stream()
            .map(Attribute::getName)
            .collect(toImmutableList());
    ImmutableList<String> attributesVisibleToStarlark =
        Sequence.cast(
                getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo")
                    .getValue("result"),
                String.class,
                "")
            .getImmutableList();
    String ruleAttrName =
        getRuleAttrName(
            Label.parseCanonical("//subrule_testing:myrule.bzl"),
            "_my_subrule",
            "_foo",
            AttributeValueSource.DIRECT);

    assertThat(ruleClassAttributes).contains(ruleAttrName);
    assertThat(attributesVisibleToStarlark).doesNotContain(ruleAttrName);
  }

  @Test
  public void testSubruleAttrs_notVisibleInAspectCtx() throws Exception {
    scratch.file("default/BUILD", "genrule(name = 'default', outs = ['a'], cmd = '')");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "_my_subrule = subrule(",
        "  implementation = lambda: None,",
        "  attrs = {'_foo' : attr.label(default = '//default')},",
        ")",
        "MyInfo=provider()",
        "def _aspect_impl(target, ctx):",
        "  res = dir(ctx.attr)",
        "  return MyInfo(result = res)",
        "my_aspect = aspect(implementation = _aspect_impl, subrules = [_my_subrule])",
        "def _rule_impl(ctx):",
        "  return ctx.attr.dep[MyInfo]",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {'dep' : attr.label(aspects = [my_aspect])}",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo', dep = '//default')");

    ImmutableList<String> attributesVisibleToStarlark =
        Sequence.cast(
                getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo")
                    .getValue("result"),
                String.class,
                "")
            .getImmutableList();
    String ruleAttrName =
        getRuleAttrName(
            Label.parseCanonical("//subrule_testing:myrule.bzl"),
            "_my_subrule",
            "_foo",
            AttributeValueSource.DIRECT);

    assertThat(attributesVisibleToStarlark).doesNotContain(ruleAttrName);
  }

  @Test
  public void testSubruleAttrs_overridingImplicitAttributeValueFails() throws Exception {
    scratch.file("default/BUILD", "genrule(name = 'default', outs = ['a'], cmd = '')");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, _foo):",
        "  return ",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_foo' : attr.label(default = '//default')},",
        ")",
        "",
        "def _rule_impl(ctx):",
        "  res = _my_subrule(_foo = '//override')",
        "  return []",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error)
        .hasMessageThat()
        .contains(
            "Error in _my_subrule: got invalid named argument: '_foo' is an implicit dependency and"
                + " cannot be overridden");
  }

  @Test
  public void testSubruleAttrs_implicitLabelDepsAreResolvedToTargets() throws Exception {
    scratch.file(
        "some/pkg/BUILD",
        //
        "genrule(name = 'tool', cmd = '', outs = ['tool.exe'])");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, _tool):",
        "  return _tool",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_tool' : attr.label(default = '//some/pkg:tool')},",
        ")",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    StructImpl provider =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo");

    assertThat(provider).isNotNull();
    Object value = provider.getValue("result");
    assertThat(value).isInstanceOf(ConfiguredTarget.class);
    assertThat(((ConfiguredTarget) value).getLabel().toString()).isEqualTo("//some/pkg:tool");
  }

  @Test
  public void testSubruleAttr_executableAttrIsPassedAsFilesToRun() throws Exception {
    scratch.file(
        "my/BUILD",
        //
        "cc_binary(name = 'tool')");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, _tool):",
        "  return _tool",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_tool' : attr.label(default = '//my:tool', executable = True, cfg = 'exec')},",
        ")",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    Object result =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo")
            .getValue("result");

    assertThat(result).isInstanceOf(FilesToRunProvider.class);
    assertThat(((FilesToRunProvider) result).getExecutable().getRootRelativePathString())
        .isEqualTo("my/tool");
  }

  @Test
  public void testSubruleAction_executableMustBeFilesToRunProvider() throws Exception {
    scratch.file(
        "my/BUILD",
        //
        "cc_binary(name = 'tool')");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, _tool):",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run(executable = _tool.executable, outputs = [out])",
        "  return out",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_tool' : attr.label(default = '//my:tool', executable = True, cfg = 'exec')},",
        ")",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    AssertionError error =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//subrule_testing:foo"));

    assertThat(error)
        .hasMessageThat()
        .contains("Error in run: for 'executable', expected FilesToRunProvider, got File");
  }

  @Test
  public void testSubruleAttrs_lateBoundDefaultsAreResolved() throws Exception {
    scratch.file(
        "my/BUILD",
        //
        "cc_binary(name = 'tool')");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, _tool):",
        "  return _tool",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_tool' : attr.label(",
        "         default = configuration_field(fragment = 'coverage', name = 'output_generator')",
        "  )},",
        ")",
        "",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  res = _my_subrule()",
        "  return MyInfo(result = res)",
        "",
        "my_rule = rule(implementation = _rule_impl, subrules = [_my_subrule])");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");
    // TODO: b/293304174 - use a custom fragment instead of coverage
    useConfiguration("--collect_code_coverage", "--coverage_output_generator=//my:tool");

    StructImpl provider =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo");

    assertThat(provider).isNotNull();
    Object value = provider.getValue("result");
    assertThat(value).isInstanceOf(ConfiguredTarget.class);
    assertThat(((ConfiguredTarget) value).getLabel().toString()).isEqualTo("//my:tool");
  }

  @Test
  public void testSubruleToolchains_cannotRequireMoreThanOne() throws Exception {
    ev.checkEvalErrorContains(
        "subrules may require at most 1 toolchain",
        "_my_subrule = subrule(",
        "  implementation = lambda: None,",
        "  toolchains = ['//t1', '//t2'],",
        ")");
  }

  @Test
  public void testSubruleToolchains_cannotAccessUnrequestedToolchain() throws Exception {
    useConfiguration("--incompatible_auto_exec_groups");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  ctx.toolchains['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "']",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        ")",
        "",
        "def _rule_impl(ctx):",
        "  _my_subrule()",
        "",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  subrules = [_my_subrule],",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    assertThrows(
        TestConstants.JAVA_TOOLCHAIN_TYPE + " was requested but only types [] are configured",
        AssertionError.class,
        () -> getConfiguredTarget("//subrule_testing:foo"));
  }

  @Test
  public void testSubruleToolchains_cannotAccessToolchainFromRule() throws Exception {
    useConfiguration("--incompatible_auto_exec_groups");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  ctx.toolchains['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "']",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        ")",
        "",
        "def _rule_impl(ctx):",
        "  _my_subrule()",
        "",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  subrules = [_my_subrule],",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    assertThrows(
        TestConstants.JAVA_TOOLCHAIN_TYPE + " was requested but only types [] are configured",
        AssertionError.class,
        () -> getConfiguredTarget("//subrule_testing:foo"));
  }

  @Test
  public void testSubruleToolchains_requestedToolchainIsResolved_inRule() throws Exception {
    useConfiguration("--incompatible_auto_exec_groups");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  return ctx.toolchains['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "']",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        ")",
        "MyInfo = provider()",
        "def _rule_impl(ctx):",
        "  return [MyInfo(result = _my_subrule())]",
        "",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  subrules = [_my_subrule],",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    ToolchainInfo toolchainInfo =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo")
            .getValue("result", ToolchainInfo.class);

    assertThat(toolchainInfo).isNotNull();
    assertThat(toolchainInfo.getValue("java", StarlarkInfo.class).getProvider().getKey())
        .isEqualTo(JavaToolchainProvider.PROVIDER.getKey());
  }

  @Test
  public void testSubruleToolchains_requstedToolchainIsResolved_inAspect() throws Exception {
    useConfiguration("--incompatible_auto_exec_groups");
    scratch.file("default/BUILD", "genrule(name = 'default', outs = ['a'], cmd = '')");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  return ctx.toolchains['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "']",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        ")",
        "MyInfo=provider()",
        "def _aspect_impl(target, ctx):",
        "  return MyInfo(result = _my_subrule())",
        "my_aspect = aspect(implementation = _aspect_impl, subrules = [_my_subrule])",
        "def _rule_impl(ctx):",
        "  return ctx.attr.dep[MyInfo]",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  attrs = {'dep' : attr.label(aspects = [my_aspect])}",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo', dep = '//default')");

    ToolchainInfo toolchainInfo =
        getProvider("//subrule_testing:foo", "//subrule_testing:myrule.bzl", "MyInfo")
            .getValue("result", ToolchainInfo.class);

    assertThat(toolchainInfo).isNotNull();
    assertThat(toolchainInfo.getValue("java", StarlarkInfo.class).getProvider().getKey())
        .isEqualTo(JavaToolchainProvider.PROVIDER.getKey());
  }

  /**
   * This test case exists purely as a companion to the test case {@link
   * #testSubruleToolchains_requestedToolchainIsSuppliedToAction} below. To ensure the latter does
   * not spuriously pass, we test the inverse case here. If this test begins to fail due to changes
   * to the AEG checking code, it very likely means the other test below needs to be updated
   * appropriately.
   */
  @Test
  public void testSubruleToolchains_aegValidationFailsWithoutToolchain() throws Exception {
    useConfiguration("--incompatible_auto_exec_groups");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run(outputs = [out], executable = '/bin/ls', tools = [depset()])",
        "  return out",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        ")",
        "def _rule_impl(ctx):",
        "  return [DefaultInfo(files = depset([_my_subrule()]))]",
        "",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  subrules = [_my_subrule],",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    assertThrows(
        "Couldn't identify if tools are from implicit dependencies or a toolchain. Please set the"
            + " toolchain parameter.",
        AssertionError.class,
        () -> getConfiguredTarget("//subrule_testing:foo"));
  }

  @Test
  public void testSubruleToolchains_requestedToolchainIsSuppliedToAction() throws Exception {
    useConfiguration("--incompatible_auto_exec_groups");
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx):",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run(outputs = [out], executable = '/bin/ls', tools = [depset()])",
        "  return out",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  toolchains = ['" + TestConstants.JAVA_TOOLCHAIN_TYPE + "'],",
        ")",
        "def _rule_impl(ctx):",
        "  return [DefaultInfo(files = depset([_my_subrule()]))]",
        "",
        "my_rule = rule(",
        "  implementation = _rule_impl,",
        "  subrules = [_my_subrule],",
        ")");
    scratch.file(
        "subrule_testing/BUILD",
        //
        "load('myrule.bzl', 'my_rule')",
        "my_rule(name = 'foo')");

    ConfiguredTarget target = getConfiguredTarget("//subrule_testing:foo");
    Action action = getGeneratingAction(target, "subrule_testing/foo.out");

    assertThat(action).isNotNull();
    assertThat(action.getOwner())
        .isEqualTo(getRuleContext(target).getActionOwner(TestConstants.JAVA_TOOLCHAIN_TYPE));
  }

  @Test
  public void testSubruleInstantiation_outsideAllowlist_failsWithPrivateAPIError()
      throws Exception {
    evOutsideAllowlist.checkEvalErrorContains(
        "'//foo:bar' cannot use private API", "subrule(implementation = lambda: 0 )");
  }

  @Test
  public void testSubrulesParamForRule_isPrivateAPI() throws Exception {
    evOutsideAllowlist.checkEvalErrorContains(
        "'//foo:bar' cannot use private API", "rule(implementation = lambda: 0, subrules = [1])");
  }

  @Test
  public void testSubrulesParamForAspect_isPrivateAPI() throws Exception {
    evOutsideAllowlist.checkEvalErrorContains(
        "'//foo:bar' cannot use private API", "aspect(implementation = lambda: 0, subrules = [1])");
  }

  private StructImpl getProvider(String targetLabel, String providerLabel, String providerName)
      throws LabelSyntaxException {
    ConfiguredTarget target = getConfiguredTarget(targetLabel);
    Provider.Key key = new StarlarkProvider.Key(Label.parseCanonical(providerLabel), providerName);
    return (StructImpl) target.get(key);
  }
}
