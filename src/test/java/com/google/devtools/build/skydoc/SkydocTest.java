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
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDescriptor.Type;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
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
    skydocMain = new SkydocMain(new SkylarkFileAccessor() {

      @Override
      public ParserInputSource inputSource(String pathString) throws IOException {
        Path path = fileSystem.getPath("/" + pathString);
        byte[] bytes = FileSystemUtils.asByteSource(path).read();
        return ParserInputSource.create(bytes, path.asFragment());
      }
    });
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
    ImmutableList.Builder<RuleInfo> unexportedRuleInfos = ImmutableList.builder();

    skydocMain.eval(
        Label.parseAbsoluteUnchecked("//test:test.bzl"),
        ruleInfoMap,
        unexportedRuleInfos,
        ImmutableMap.builder());
    Map<String, RuleInfo> ruleInfos = ruleInfoMap.build();
    assertThat(ruleInfos).hasSize(1);

    Entry<String, RuleInfo> ruleInfo = Iterables.getOnlyElement(ruleInfos.entrySet());
    assertThat(ruleInfo.getKey()).isEqualTo("my_rule");
    assertThat(ruleInfo.getValue().getDocString()).isEqualTo("This is my rule. It does stuff.");
    assertThat(getAttrNames(ruleInfo.getValue())).containsExactly(
        "name", "a", "b", "c", "d").inOrder();
    assertThat(getAttrTypes(ruleInfo.getValue())).containsExactly(
        Type.STRING.getDescription(),
        Type.LABEL.getDescription(),
        Type.STRING_DICT.getDescription(),
        Type.LABEL.getDescription(),
        Type.BOOLEAN.getDescription()).inOrder();
    assertThat(unexportedRuleInfos.build()).isEmpty();
  }

  private static Iterable<String> getAttrNames(RuleInfo ruleInfo) {
    return ruleInfo.getAttributes().stream().map(attr -> attr.getName())
        .collect(Collectors.toList());
  }

  private static Iterable<String> getAttrTypes(RuleInfo ruleInfo) {
    return ruleInfo.getAttributes().stream().map(attr -> attr.getTypeString())
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
    ImmutableList.Builder<RuleInfo> unexportedRuleInfos = ImmutableList.builder();

    skydocMain.eval(
        Label.parseAbsoluteUnchecked("//test:test.bzl"),
        ruleInfoMap,
        unexportedRuleInfos,
        ImmutableMap.builder());

    assertThat(ruleInfoMap.build().keySet()).containsExactly("rule_one", "rule_two");

    assertThat(unexportedRuleInfos.build().stream()
            .map(ruleInfo -> ruleInfo.getDocString())
            .collect(Collectors.toList()))
        .containsExactly("This rule is not named", "This rule also is not named");
  }

  @Test
  public void testRulesAcrossMultipleFiles() throws Exception {
    scratch.file(
        "/lib/rule_impl.bzl",
        "def rule_impl(ctx):",
        "  return struct()");

    scratch.file(
        "/deps/foo/docstring.bzl",
        "doc_string = 'Dep rule'");

    scratch.file(
        "/deps/foo/dep_rule.bzl",
        "load('//lib:rule_impl.bzl', 'rule_impl')",
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
        "load('//lib:rule_impl.bzl', 'rule_impl')",
        "load('//deps/foo:dep_rule.bzl', 'dep_rule')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    skydocMain.eval(
        Label.parseAbsoluteUnchecked("//test:main.bzl"),
        ruleInfoMapBuilder,
        ImmutableList.builder(),
        ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.build();

    // dep_rule is available here, even though it was not defined in main.bzl, because it is
    // imported in main.bzl. Thus, it's a top-level symbol in main.bzl.
    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule", "dep_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
    assertThat(ruleInfoMap.get("dep_rule").getDocString()).isEqualTo("Dep rule");
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
        Label.parseAbsoluteUnchecked("//test:main.bzl"),
        ruleInfoMapBuilder,
        ImmutableList.builder(),
        ImmutableMap.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.build();

    // dep_rule is available here, even though it was not defined in main.bzl, because it is
    // imported in main.bzl. Thus, it's a top-level symbol in main.bzl.
    assertThat(ruleInfoMap.keySet()).containsExactly("main_rule", "dep_rule");
    assertThat(ruleInfoMap.get("main_rule").getDocString()).isEqualTo("Main rule");
    assertThat(ruleInfoMap.get("dep_rule").getDocString()).isEqualTo("Dep rule");
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
        assertThrows(IllegalStateException.class,
            () -> skydocMain.eval(
                Label.parseAbsoluteUnchecked("//test:main.bzl"),
                ruleInfoMapBuilder,
                ImmutableList.builder(),
                ImmutableMap.builder()));

    assertThat(expected).hasMessageThat().contains("cycle with test/main.bzl");
  }
}
