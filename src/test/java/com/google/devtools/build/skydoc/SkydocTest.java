// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.runfiles.Runfiles;
import com.google.devtools.build.skydoc.SkydocMain.StarlarkEvaluationException;
import com.google.devtools.build.skydoc.rendering.DocstringParseException;
import com.google.devtools.build.skydoc.rendering.LabelRenderer;
import com.google.devtools.build.skydoc.rendering.ProtoRenderer;
import com.google.devtools.build.skydoc.rendering.StarlarkFunctionInfoExtractor;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Java tests for Skydoc. */
@RunWith(JUnit4.class)
public final class SkydocTest {

  @Rule public TemporaryFolder runfilesDir = new TemporaryFolder(TestUtils.tmpDirFile());

  private SkydocMain skydocMain;

  @Before
  public void setUp() throws IOException {
    skydocMain =
        new SkydocMain(
            "io_bazel",
            Runfiles.preload(
                ImmutableMap.of("RUNFILES_DIR", runfilesDir.getRoot().getAbsolutePath())));
  }

  @Test
  public void testStarlarkEvaluationError() throws Exception {
    scratchRunfile(
        "io_bazel/test/a.bzl", //
        "def f(): 1//0",
        "f()");
    StarlarkEvaluationException ex =
        assertThrows(
            StarlarkEvaluationException.class,
            () ->
                skydocMain.eval(
                    StarlarkSemantics.DEFAULT,
                    Label.parseCanonicalUnchecked("//test:a.bzl"),
                    ImmutableMap.builder(),
                    ImmutableMap.builder(),
                    ImmutableMap.builder(),
                    ImmutableMap.builder()));
    String msg = ex.getMessage();
    assertThat(msg).contains("Traceback");
    assertThat(msg).contains("line 2, column 2, in <toplevel>");
    assertThat(msg).contains("line 1, column 11, in f");
    assertThat(msg).contains("Error: integer division by zero");
  }

  @Test
  public void testRuleInfoAttrs() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "def rule_impl(ctx):",
        "  return []",
        "",
        "my_rule = rule(",
        "    doc = 'This is my rule. It does stuff.',",
        "    implementation = rule_impl,",
        "    attrs = {",
        "        'a': attr.label(mandatory=True, allow_single_file=True),",
        "        'b': attr.string_dict(mandatory=True),",
        "        'c': attr.output(mandatory=True),",
        "        'd': attr.bool(default=False, mandatory=False),",
        "    },",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ruleInfoMap,
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());
    Map<String, RuleInfo> ruleInfos = ruleInfoMap.buildOrThrow();
    assertThat(ruleInfos).hasSize(1);

    RuleInfo ruleInfo = Iterables.getOnlyElement(ruleInfos.values());
    assertThat(ruleInfo.getRuleName()).isEqualTo("my_rule");
    assertThat(ruleInfo.getDocString()).isEqualTo("This is my rule. It does stuff.");
    assertThat(getAttrNames(ruleInfo)).containsExactly("name", "a", "b", "c", "d").inOrder();
    assertThat(getAttrTypes(ruleInfo))
        .containsExactly(
            AttributeType.NAME,
            AttributeType.LABEL,
            AttributeType.STRING_DICT,
            AttributeType.OUTPUT,
            AttributeType.BOOLEAN)
        .inOrder();
  }

  private static Iterable<String> getAttrNames(RuleInfo ruleInfo) {
    return ruleInfo.getAttributeList().stream()
        .map(attr -> attr.getName())
        .collect(Collectors.toList());
  }

  private static Iterable<String> getAttrNames(AspectInfo aspectInfo) {
    return aspectInfo.getAttributeList().stream()
        .map(attr -> attr.getName())
        .collect(Collectors.toList());
  }

  private static Iterable<AttributeType> getAttrTypes(RuleInfo ruleInfo) {
    return ruleInfo.getAttributeList().stream()
        .map(attr -> attr.getType())
        .collect(Collectors.toList());
  }

  private static Iterable<AttributeType> getAttrTypes(AspectInfo aspectInfo) {
    return aspectInfo.getAttributeList().stream()
        .map(attr -> attr.getType())
        .collect(Collectors.toList());
  }

  @Test
  public void testMultipleRuleNames() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "def rule_impl(ctx):",
        "  return []",
        "",
        "rule_one = rule(",
        "    doc = 'Rule one',",
        "    implementation = rule_impl,",
        ")",
        "",
        "rule(",
        "    doc = 'This rule is not named',",
        "    implementation = rule_impl,",
        ")",
        "",
        "rule(",
        "    doc = 'This rule also is not named',",
        "    implementation = rule_impl,",
        ")",
        "",
        "rule_two = rule(",
        "    doc = 'Rule two',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ruleInfoMap,
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());

    assertThat(ruleInfoMap.buildOrThrow().keySet()).containsExactly("rule_one", "rule_two");
  }

  @Test
  public void testRuleWithMultipleExports() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "def rule_impl(ctx):",
        "  return []",
        "",
        "rule_one = rule(",
        "    doc = 'Rule one',",
        "    implementation = rule_impl,",
        ")",
        "",
        "rule_two = rule_one");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ruleInfoMap,
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());

    assertThat(ruleInfoMap.buildOrThrow().keySet()).containsExactly("rule_one", "rule_two");
  }

  @Test
  public void testRulesAcrossMultipleFiles() throws Exception {
    scratchRunfile("io_bazel/lib/rule_impl.bzl", "def rule_impl(ctx):", "  return []");

    scratchRunfile("io_bazel/deps/foo/other_root.bzl", "doc_string = 'Dep rule'");

    scratchRunfile(
        "io_bazel/deps/foo/dep_rule.bzl",
        "load('//lib:rule_impl.bzl', 'rule_impl')",
        "load(':other_root.bzl', 'doc_string')",
        "",
        "_hidden_rule = rule(",
        "    doc = doc_string,",
        "    implementation = rule_impl,",
        ")",
        "",
        "dep_rule = rule(",
        "    doc = doc_string,",
        "    implementation = rule_impl,",
        ")");

    scratchRunfile(
        "io_bazel/test/main.bzl",
        "load('//lib:rule_impl.bzl', 'rule_impl')",
        "load('//deps/foo:dep_rule.bzl', 'dep_rule')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:main.bzl"),
            ruleInfoMapBuilder,
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.buildOrThrow();

    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
  }

  @Test
  public void testRulesAcrossRepository() throws Exception {
    scratchRunfile("dep_repo/lib/rule_impl.bzl", "def rule_impl(ctx):", "  return []");

    scratchRunfile("io_bazel/deps/foo/docstring.bzl", "doc_string = 'Dep rule'");

    scratchRunfile(
        "io_bazel/deps/foo/dep_rule.bzl",
        "load('@dep_repo//lib:rule_impl.bzl', 'rule_impl')",
        "load(':docstring.bzl', 'doc_string')",
        "",
        "_hidden_rule = rule(",
        "    doc = doc_string,",
        "    implementation = rule_impl,",
        ")",
        "",
        "dep_rule = rule(",
        "    doc = doc_string,",
        "    implementation = rule_impl,",
        ")");

    scratchRunfile(
        "io_bazel/test/main.bzl",
        "load('@dep_repo//lib:rule_impl.bzl', 'rule_impl')",
        "load('//deps/foo:dep_rule.bzl', 'dep_rule')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:main.bzl"),
            ruleInfoMapBuilder,
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.buildOrThrow();

    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
  }

  @Test
  public void testRulesAcrossRepositorySiblingRepositoryLayout() throws Exception {
    scratchRunfile("dep_repo/lib/rule_impl.bzl", "def rule_impl(ctx):", "  return []");

    scratchRunfile("io_bazel/deps/foo/docstring.bzl", "doc_string = 'Dep rule'");

    scratchRunfile(
        "io_bazel/deps/foo/dep_rule.bzl",
        "load('@dep_repo//lib:rule_impl.bzl', 'rule_impl')",
        "load(':docstring.bzl', 'doc_string')",
        "",
        "_hidden_rule = rule(",
        "    doc = doc_string,",
        "    implementation = rule_impl,",
        ")",
        "",
        "dep_rule = rule(",
        "    doc = doc_string,",
        "    implementation = rule_impl,",
        ")");

    scratchRunfile(
        "io_bazel/test/main.bzl",
        "load('@dep_repo//lib:rule_impl.bzl', 'rule_impl')",
        "load('//deps/foo:dep_rule.bzl', 'dep_rule')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.builder()
                .setBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT, true)
                .build(),
            Label.parseCanonicalUnchecked("//test:main.bzl"),
            ruleInfoMapBuilder,
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.buildOrThrow();

    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
  }

  @Test
  public void testLoadOwnRepository() throws Exception {
    scratchRunfile("io_bazel/deps/foo/dep_rule.bzl", "def rule_impl(ctx):", "  return []");

    scratchRunfile(
        "io_bazel/test/main.bzl",
        "load('@io_bazel//deps/foo:dep_rule.bzl', 'rule_impl')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:main.bzl"),
            ruleInfoMapBuilder,
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.buildOrThrow();

    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
  }

  @Test
  public void testSkydocCrashesOnCycle() throws Exception {
    scratchRunfile(
        "io_bazel/dep/dep.bzl",
        "load('//test:main.bzl', 'some_var')",
        "def rule_impl(ctx):",
        "  return []");

    scratchRunfile(
        "io_bazel/test/main.bzl",
        "load('//dep:dep.bzl', 'rule_impl')",
        "",
        "some_var = 1",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    StarlarkEvaluationException expected =
        assertThrows(
            StarlarkEvaluationException.class,
            () ->
                skydocMain.eval(
                    StarlarkSemantics.DEFAULT,
                    Label.parseCanonicalUnchecked("//test:main.bzl"),
                    ImmutableMap.builder(),
                    ImmutableMap.builder(),
                    ImmutableMap.builder(),
                    ImmutableMap.builder()));

    assertThat(expected)
        .hasMessageThat()
        .contains("cycle with " + runfilesDir.getRoot() + "/io_bazel/test/main.bzl");
  }

  @Test
  public void testMalformedFunctionDocstring() throws Exception {
    scratchRunfile(
        "io_bazel/test/main.bzl",
        "def check_sources(name,",
        "                  required_param,",
        "                  bool_param = True,",
        "                  srcs = []):",
        "    \"\"\"Runs some checks on the given source files.",
        "",
        "    This rule runs checks on a given set of source files.",
        "    Use `bazel build` to run the check.",
        "",
        "    Args:",
        "        name: A unique name for this rule.",
        "        required_param:",
        "        bool_param: ..oh hey I forgot to document required_param!",
        "    \"\"\"",
        "    pass");

    ImmutableMap.Builder<String, StarlarkFunction> functionInfoBuilder = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:main.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            functionInfoBuilder,
            ImmutableMap.builder());

    StarlarkFunction checkSourcesFn = functionInfoBuilder.buildOrThrow().get("check_sources");
    DocstringParseException expected =
        assertThrows(
            DocstringParseException.class,
            () ->
                StarlarkFunctionInfoExtractor.fromNameAndFunction(
                    "check_sources",
                    checkSourcesFn,
                    /* withOriginKey= */ false,
                    LabelRenderer.DEFAULT));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "Unable to generate documentation for function check_sources "
                + "(defined at "
                + runfilesDir.getRoot()
                + "/io_bazel/test/main.bzl:1:5) "
                + "due to malformed docstring. Parse errors:");
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "/test/main.bzl:1:5 line 8: invalid parameter documentation "
                + "(expected format: \"parameter_name: documentation\").");
  }

  @Test
  public void testFuncInfoParams() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "def check_function(foo, bar, baz):",
        "  \"\"\"Runs some checks on the given function parameter.",
        "  ",
        "  This rule runs checks on a given function parameter.",
        "  ",
        "  Args:",
        "    foo: A unique parameter for this rule.",
        "    bar: A unique parameter for this rule.",
        "    baz: A unique parameter for this rule.",
        "  ",
        "  \"\"\"",
        "  pass");

    ImmutableMap.Builder<String, StarlarkFunction> funcInfoMap = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            funcInfoMap,
            ImmutableMap.builder());

    Map<String, StarlarkFunction> functions = funcInfoMap.buildOrThrow();
    assertThat(functions).hasSize(1);

    ModuleInfo moduleInfo =
        new ProtoRenderer().appendStarlarkFunctionInfos(functions).getModuleInfo().build();
    StarlarkFunctionInfo funcInfo = moduleInfo.getFuncInfo(0);
    assertThat(funcInfo.getFunctionName()).isEqualTo("check_function");
    assertThat(getParamNames(funcInfo)).containsExactly("foo", "bar", "baz").inOrder();
  }

  private static Iterable<String> getParamNames(StarlarkFunctionInfo funcInfo) {
    return funcInfo.getParameterList().stream()
        .map(param -> param.getName())
        .collect(Collectors.toList());
  }

  @Test
  public void testProviderInfo() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "MyExampleInfo = provider(",
        "  doc = 'Stores information about example.',",
        "  fields = {",
        "    'name' : 'A string representing a random name.',",
        "    'city' : 'A string representing a city.',",
        "  },",
        ")",
        "pass");

    ImmutableMap.Builder<String, ProviderInfo> providerInfoMap = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            providerInfoMap,
            ImmutableMap.builder(),
            ImmutableMap.builder());

    Map<String, ProviderInfo> providers = providerInfoMap.buildOrThrow();
    assertThat(providers).hasSize(1);

    ModuleInfo moduleInfo =
        new ProtoRenderer().appendProviderInfos(providers.values()).getModuleInfo().build();
    ProviderInfo providerInfo = moduleInfo.getProviderInfo(0);
    assertThat(providerInfo.getProviderName()).isEqualTo("MyExampleInfo");
    assertThat(providerInfo.getDocString()).isEqualTo("Stores information about example.");
    assertThat(getFieldNames(providerInfo)).containsExactly("name", "city").inOrder();
    assertThat(getFieldDocString(providerInfo))
        .containsExactly("A string representing a random name.", "A string representing a city.")
        .inOrder();
  }

  private static Iterable<String> getFieldNames(ProviderInfo providerInfo) {
    return providerInfo.getFieldInfoList().stream()
        .map(field -> field.getName())
        .collect(Collectors.toList());
  }

  private static Iterable<String> getFieldDocString(ProviderInfo providerInfo) {
    return providerInfo.getFieldInfoList().stream()
        .map(field -> field.getDocString())
        .collect(Collectors.toList());
  }

  @Test
  public void testAspectInfo() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "def my_aspect_impl(ctx):\n"
            + "    return []\n"
            + "\n"
            + "my_aspect = aspect(\n"
            + "    implementation = my_aspect_impl,\n"
            + "    doc = \"This is my aspect. It does stuff.\",\n"
            + "    attr_aspects = [\"deps\"],\n"
            + "    attrs = {\n"
            + "        \"first\": attr.label(mandatory = True, allow_single_file = True),\n"
            + "        \"second\": attr.string_dict(mandatory = True),\n"
            + "        \"_third\": attr.label(mandatory = True, allow_single_file = True),\n"
            + "    },\n"
            + ")");

    ImmutableMap.Builder<String, AspectInfo> aspectInfoMap = ImmutableMap.builder();

    Module unused =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            aspectInfoMap);
    Map<String, AspectInfo> aspectInfos = aspectInfoMap.buildOrThrow();
    assertThat(aspectInfos).hasSize(1);

    ModuleInfo moduleInfo =
        new ProtoRenderer().appendAspectInfos(aspectInfos.values()).getModuleInfo().build();
    AspectInfo aspectInfo = moduleInfo.getAspectInfo(0);
    assertThat(aspectInfo.getAspectName()).isEqualTo("my_aspect");
    assertThat(aspectInfo.getDocString()).isEqualTo("This is my aspect. It does stuff.");
    assertThat(getAttrNames(aspectInfo)).containsExactly("name", "first", "second").inOrder();
    assertThat(getAttrTypes(aspectInfo))
        .containsExactly(AttributeType.NAME, AttributeType.LABEL, AttributeType.STRING_DICT)
        .inOrder();
    assertThat(aspectInfo.getAspectAttributeList()).containsExactly("deps");
  }

  @Test
  public void testModuleDocstring() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "\"\"\"Input file to test module docstring\"\"\"",
        "def check_function(foo):",
        "  \"\"\"Runs some checks on the given function parameter.",
        " ",
        "Args:",
        "foo: A unique parameter for this rule.",
        "\"\"\"",
        "pass");

    Module module =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());

    assertThat(module.getDocumentation()).isEqualTo("Input file to test module docstring");
  }

  @Test
  public void testnoModuleDoc() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "def check_function(foo):",
        "  \"\"\"Runs some checks input file with no module docstring.",
        " ",
        "  Args:",
        "  foo: A parameter.",
        "  \"\"\"",
        "pass");
    Module module =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());
    ProtoRenderer renderer =
        SkydocMain.render(
            module,
            ImmutableSet.of(),
            ImmutableMap.of(),
            ImmutableMap.of(),
            ImmutableMap.of(),
            ImmutableMap.of());

    ModuleInfo moduleInfo = renderer.getModuleInfo().build();
    String moduleDoc = moduleInfo.getModuleDocstring();
    assertThat(moduleDoc).isEmpty();
  }

  @Test
  public void testMultipleLineModuleDoc() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl",
        "\"\"\"Input file to test",
        "multiple lines module docstring\"\"\"",
        "def check_function(foo):",
        "  \"\"\"Runs some checks on the given function parameter.",
        "  ",
        "  Args:",
        "  foo: A unique parameter for this rule.",
        "  \"\"\"",
        "pass");
    Module module =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());
    ProtoRenderer renderer =
        SkydocMain.render(
            module,
            ImmutableSet.of(),
            ImmutableMap.of(),
            ImmutableMap.of(),
            ImmutableMap.of(),
            ImmutableMap.of());

    ModuleInfo moduleInfo = renderer.getModuleInfo().build();
    String moduleDoc = moduleInfo.getModuleDocstring();
    assertThat(moduleDoc).isEqualTo("Input file to test\nmultiple lines module docstring");
  }

  @Test
  public void testModuleDocAcrossFiles() throws Exception {
    scratchRunfile(
        "io_bazel/test/othertest.bzl", //
        "\"\"\"Should be displayed.\"\"\"",
        "load(':test.bzl', 'check_function')",
        "pass");
    scratchRunfile(
        "io_bazel/test/test.bzl", //
        "\"\"\"Should not be displayed.\"\"\"",
        "def check_function():",
        "  pass");
    Module module =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:othertest.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            ImmutableMap.builder());
    ProtoRenderer renderer =
        SkydocMain.render(
            module,
            ImmutableSet.of(),
            ImmutableMap.of(),
            ImmutableMap.of(),
            ImmutableMap.of(),
            ImmutableMap.of());

    ModuleInfo moduleInfo = renderer.getModuleInfo().build();
    String moduleDoc = moduleInfo.getModuleDocstring();
    assertThat(moduleDoc).isEqualTo("Should be displayed.");
  }

  @Test
  public void testDefaultSymbolFilter() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl", //
        "def foo():",
        "  pass",
        "def bar():",
        "  pass",
        "def _baz():",
        "  pass");
    ImmutableMap.Builder<String, StarlarkFunction> functionInfoBuilder = ImmutableMap.builder();
    Module module =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            functionInfoBuilder,
            ImmutableMap.builder());
    ProtoRenderer renderer =
        SkydocMain.render(
            module,
            ImmutableSet.of(),
            ImmutableMap.of(),
            ImmutableMap.of(),
            functionInfoBuilder.buildOrThrow(),
            ImmutableMap.of());

    ModuleInfo moduleInfo = renderer.getModuleInfo().build();
    ImmutableList<String> documentedFunctions =
        moduleInfo.getFuncInfoList().stream()
            .map(StarlarkFunctionInfo::getFunctionName)
            .collect(toImmutableList());
    assertThat(documentedFunctions).containsExactly("foo", "bar");
  }

  @Test
  public void testCustomSymbolFilter() throws Exception {
    scratchRunfile(
        "io_bazel/test/test.bzl", //
        "def foo():",
        "  pass",
        "def bar():",
        "  pass",
        "def _baz():",
        "  pass");
    ImmutableMap.Builder<String, StarlarkFunction> functionInfoBuilder = ImmutableMap.builder();
    Module module =
        skydocMain.eval(
            StarlarkSemantics.DEFAULT,
            Label.parseCanonicalUnchecked("//test:test.bzl"),
            ImmutableMap.builder(),
            ImmutableMap.builder(),
            functionInfoBuilder,
            ImmutableMap.builder());
    ProtoRenderer renderer =
        SkydocMain.render(
            module,
            ImmutableSet.of("bar", "_baz"),
            ImmutableMap.of(),
            ImmutableMap.of(),
            functionInfoBuilder.buildOrThrow(),
            ImmutableMap.of());

    ModuleInfo moduleInfo = renderer.getModuleInfo().build();
    ImmutableList<String> documentedFunctions =
        moduleInfo.getFuncInfoList().stream()
            .map(StarlarkFunctionInfo::getFunctionName)
            .collect(toImmutableList());
    assertThat(documentedFunctions).containsExactly("bar", "_baz");
  }

  private void scratchRunfile(String path, String... lines) throws Exception {
    Path file = runfilesDir.getRoot().toPath().resolve(path.replace('/', File.separatorChar));
    Files.createDirectories(file.getParent());
    Files.write(file, Arrays.asList(lines), StandardCharsets.UTF_8);
  }
}
