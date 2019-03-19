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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skydoc.rendering.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import com.google.devtools.build.skydoc.rendering.UserDefinedFunctionInfo;
import com.google.devtools.build.skydoc.rendering.UserDefinedFunctionInfo.DocstringParseException;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Java tests for Skydoc.
 */
@RunWith(JUnit4.class)
public final class SkydocTest extends SkylarkTestCase {

  private SkydocMain skydocMain;

  @Before
  public void setUp() {
    skydocMain =
        new SkydocMain(
            new SkylarkFileAccessor() {

              @Override
              public ParserInputSource inputSource(String pathString) throws IOException {
                Path path = fileSystem.getPath("/" + pathString);
                byte[] bytes = FileSystemUtils.asByteSource(path).read();
                return ParserInputSource.create(bytes, path.asFragment());
              }

              @Override
              public boolean fileExists(String pathString) {
                return fileSystem.exists(fileSystem.getPath("/" + pathString));
              }
            },
            "io_bazel",
            ImmutableList.of("/other_root", "."));
  }

  @Test
  public void testRuleInfoAttrs() throws Exception {
    scratch.file(
        "/test/test.bzl",
        "def rule_impl(ctx):",
        "  return struct()",
        "",
        "my_rule = rule(",
        "    doc = 'This is my rule. It does stuff.',",
        "    implementation = rule_impl,",
        "    attrs = {",
        "        'a': attr.label(mandatory=True, allow_files=True, single_file=True),",
        "        'b': attr.string_dict(mandatory=True),",
        "        'c': attr.output(mandatory=True),",
        "        'd': attr.bool(default=False, mandatory=False),",
        "    },",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();

    skydocMain.eval(
        StarlarkSemantics.DEFAULT_SEMANTICS,
        Label.parseAbsoluteUnchecked("//test:test.bzl"),
        ruleInfoMap,
        ImmutableMap.builder(),
        ImmutableMap.builder());
    Map<String, RuleInfo> ruleInfos = ruleInfoMap.build();
    assertThat(ruleInfos).hasSize(1);

    Entry<String, RuleInfo> ruleInfo = Iterables.getOnlyElement(ruleInfos.entrySet());
    assertThat(ruleInfo.getKey()).isEqualTo("my_rule");
    assertThat(ruleInfo.getValue().getDocString()).isEqualTo("This is my rule. It does stuff.");
    assertThat(getAttrNames(ruleInfo.getValue())).containsExactly(
        "name", "a", "b", "c", "d").inOrder();
    assertThat(getAttrTypes(ruleInfo.getValue())).containsExactly(
        AttributeInfo.Type.NAME,
        AttributeInfo.Type.LABEL,
        AttributeInfo.Type.STRING_DICT,
        AttributeInfo.Type.OUTPUT,
        AttributeInfo.Type.BOOLEAN).inOrder();
  }

  private static Iterable<String> getAttrNames(RuleInfo ruleInfo) {
    return ruleInfo.getAttributes().stream().map(attr -> attr.getName())
        .collect(Collectors.toList());
  }

  private static Iterable<AttributeInfo.Type> getAttrTypes(RuleInfo ruleInfo) {
    return ruleInfo.getAttributes().stream().map(attr -> attr.getType())
        .collect(Collectors.toList());
  }

  @Test
  public void testMultipleRuleNames() throws Exception {
    scratch.file(
        "/test/test.bzl",
        "def rule_impl(ctx):",
        "  return struct()",
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

    skydocMain.eval(
        StarlarkSemantics.DEFAULT_SEMANTICS,
        Label.parseAbsoluteUnchecked("//test:test.bzl"),
        ruleInfoMap,
        ImmutableMap.builder(),
        ImmutableMap.builder());

    assertThat(ruleInfoMap.build().keySet()).containsExactly("rule_one", "rule_two");
  }

  @Test
  public void testRulesAcrossMultipleFiles() throws Exception {
    scratch.file(
        "/lib/rule_impl.bzl",
        "def rule_impl(ctx):",
        "  return struct()");

    scratch.file("/other_root/deps/foo/other_root.bzl", "doc_string = 'Dep rule'");

    scratch.file(
        "/deps/foo/dep_rule.bzl",
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

    scratch.file(
        "/test/main.bzl",
        "load('//lib:rule_impl.bzl', 'rule_impl')",
        "load('//deps/foo:dep_rule.bzl', 'dep_rule')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    skydocMain.eval(
        StarlarkSemantics.DEFAULT_SEMANTICS,
        Label.parseAbsoluteUnchecked("//test:main.bzl"),
        ruleInfoMapBuilder,
        ImmutableMap.builder(),
        ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.build();

    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
  }

  @Test
  public void testRulesAcrossRepository() throws Exception {
    scratch.file(
        "/external/dep_repo/lib/rule_impl.bzl",
        "def rule_impl(ctx):",
        "  return struct()");

    scratch.file(
        "/deps/foo/docstring.bzl",
        "doc_string = 'Dep rule'");

    scratch.file(
        "/deps/foo/dep_rule.bzl",
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

    scratch.file(
        "/test/main.bzl",
        "load('@dep_repo//lib:rule_impl.bzl', 'rule_impl')",
        "load('//deps/foo:dep_rule.bzl', 'dep_rule')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    skydocMain.eval(
        StarlarkSemantics.DEFAULT_SEMANTICS,
        Label.parseAbsoluteUnchecked("//test:main.bzl"),
        ruleInfoMapBuilder,
        ImmutableMap.builder(),
        ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.build();

    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
  }

  @Test
  public void testLoadOwnRepository() throws Exception {
    scratch.file("/deps/foo/dep_rule.bzl", "def rule_impl(ctx):", "  return []");

    scratch.file(
        "/test/main.bzl",
        "load('@io_bazel//deps/foo:dep_rule.bzl', 'rule_impl')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    skydocMain.eval(
        StarlarkSemantics.DEFAULT_SEMANTICS,
        Label.parseAbsoluteUnchecked("//test:main.bzl"),
        ruleInfoMapBuilder,
        ImmutableMap.builder(),
        ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.build();

    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
  }

  @Test
  public void testSkydocCrashesOnCycle() throws Exception {
    scratch.file(
        "/dep/dep.bzl",
        "load('//test:main.bzl', 'some_var')",
        "def rule_impl(ctx):",
        "  return struct()");

    scratch.file(
        "/test/main.bzl",
        "load('//dep:dep.bzl', 'rule_impl')",
        "",
        "some_var = 1",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    IllegalStateException expected =
        assertThrows(
            IllegalStateException.class,
            () ->
                skydocMain.eval(
                    StarlarkSemantics.DEFAULT_SEMANTICS,
                    Label.parseAbsoluteUnchecked("//test:main.bzl"),
                    ruleInfoMapBuilder,
                    ImmutableMap.builder(),
                    ImmutableMap.builder()));

    assertThat(expected).hasMessageThat().contains("cycle with test/main.bzl");
  }

  @Test
  public void testMalformedFunctionDocstring() throws Exception {
    scratch.file(
        "/test/main.bzl",
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

    ImmutableMap.Builder<String, UserDefinedFunction> functionInfoBuilder = ImmutableMap.builder();

    skydocMain.eval(
        StarlarkSemantics.DEFAULT_SEMANTICS,
        Label.parseAbsoluteUnchecked("//test:main.bzl"),
        ImmutableMap.builder(),
        ImmutableMap.builder(),
        functionInfoBuilder);

    UserDefinedFunction checkSourcesFn = functionInfoBuilder.build().get("check_sources");
    DocstringParseException expected =
        assertThrows(
            DocstringParseException.class,
            () -> UserDefinedFunctionInfo.fromNameAndFunction("check_sources", checkSourcesFn));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "Unable to generate documentation for function check_sources "
                + "(defined at /test/main.bzl:1:5) due to malformed docstring. Parse errors:");
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "/test/main.bzl:1:5 line 8: invalid parameter documentation "
                + "(expected format: \"parameter_name: documentation\").");
  }
}
