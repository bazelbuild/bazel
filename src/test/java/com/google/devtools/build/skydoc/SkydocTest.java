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
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
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
        Path path = fileSystem.getPath(pathString);
        byte[] bytes = null;
        try (InputStream in = path.getInputStream()) {
          bytes = new ByteSource() {
            @Override
            public InputStream openStream() throws IOException {
              return in;
            }
          }.read();
        }

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
        "        'first': attr.label(mandatory=True, allow_files=True, single_file=True),",
        "        'second': attr.string_dict(mandatory=True),",
        "        'third': attr.output(mandatory=True),",
        "        'fourth': attr.bool(default=False, mandatory=False),",
        "    },",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
    ImmutableList.Builder<RuleInfo> unexportedRuleInfos = ImmutableList.builder();

    skydocMain.eval(
        Paths.get("/test/test.bzl"),
        ruleInfoMap,
        unexportedRuleInfos);
    Map<String, RuleInfo> ruleInfos = ruleInfoMap.build();
    assertThat(ruleInfos).hasSize(1);

    Entry<String, RuleInfo> ruleInfo = Iterables.getOnlyElement(ruleInfos.entrySet());
    assertThat(ruleInfo.getKey()).isEqualTo("my_rule");
    assertThat(ruleInfo.getValue().getDocString()).isEqualTo("This is my rule. It does stuff.");
    assertThat(ruleInfo.getValue().getAttrNames()).containsExactly(
        "first", "second", "third", "fourth");
    assertThat(unexportedRuleInfos.build()).isEmpty();
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
        Paths.get("/test/test.bzl"),
        ruleInfoMap,
        unexportedRuleInfos);

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
        "some_var = 1",
        "",
        "dep_rule = rule(",
        "    doc = doc_string,",
        "    implementation = rule_impl,",
        ")");

    scratch.file(
        "/test/main.bzl",
        "load('//lib:rule_impl.bzl', 'rule_impl')",
        "load('//deps/foo:dep_rule.bzl', 'some_var')",
        "",
        "main_rule = rule(",
        "    doc = 'Main rule',",
        "    implementation = rule_impl,",
        ")");

    ImmutableMap.Builder<String, RuleInfo> ruleInfoMapBuilder = ImmutableMap.builder();

    skydocMain.eval(
        Paths.get("/test/main.bzl"),
        ruleInfoMapBuilder,
        ImmutableList.builder());

    Map<String, RuleInfo> ruleInfoMap = ruleInfoMapBuilder.build();

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
                Paths.get("/test/main.bzl"),
                ruleInfoMapBuilder,
                ImmutableList.builder()));

    assertThat(expected).hasMessageThat().contains("cycle with /test/main.bzl");
  }
}
