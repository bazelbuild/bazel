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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkSubruleApi;
import net.starlark.java.eval.BuiltinFunction;
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
    ev.exec("x = subrule(implementation = lambda : 'dummy result' )");

    ev.checkEvalErrorContains(
        "subrule(lambda) can only be called from a rule or aspect implementation", "x()");
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
            "Error in subrule(lambda): rule 'my_rule' must declare 'subrule(lambda)' in"
                + " 'subrules'");
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
            "Error in subrule(lambda): aspect '//subrule_testing:myrule.bzl%_my_aspect' must"
                + " declare 'subrule(lambda)' in 'subrules'");
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
    assertThat(provider.getValue("result")).isEqualTo("called in: @//subrule_testing:foo");
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
  public void testSubruleAttrs_attributeMustHaveDefaultValue() throws Exception {
    ev.checkEvalErrorContains(
        "for attribute '_foo': no default value specified",
        "subrule(",
        "  implementation = lambda: None,",
        "  attrs = {'_foo': attr.label()}",
        ")");
  }

  @Test
  public void testSubruleAttrs_overridingImplicitAttributeValueFails() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, _foo):",
        "  return ",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_foo' : attr.string(default = 'default value')},",
        ")",
        "",
        "def _rule_impl(ctx):",
        "  res = _my_subrule(_foo = 'override')",
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
        .contains("Error in subrule(_subrule_impl): got invalid named argument: '_foo'");
  }

  @Test
  public void testSubruleAttrs_implicitDepsArePassedToImplementation() throws Exception {
    scratch.file(
        "subrule_testing/myrule.bzl",
        "def _subrule_impl(ctx, _foo):",
        "  return _foo",
        "_my_subrule = subrule(",
        "  implementation = _subrule_impl,",
        "  attrs = {'_foo' : attr.string(default = 'my attribute value')},",
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
    assertThat(provider.getValue("result")).isEqualTo("my attribute value");
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
