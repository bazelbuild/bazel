// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaFileParseResult;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaParserStep;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRule;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRuleVariable;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaVariableValue;
import com.google.devtools.build.lib.util.Pair;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NinjaScope} */
@RunWith(JUnit4.class)
public class NinjaScopeTest {
  @Test
  public void testSortVariables() {
    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("cba"));
    parseResult.addVariable("abc", 1, NinjaVariableValue.createPlainText("cba1"));
    parseResult.addVariable("abc", 14, NinjaVariableValue.createPlainText("cba2"));

    parseResult.sortResults();

    List<Long> offsets =
        parseResult.getVariables().get("abc").stream()
            .map(Pair::getFirst)
            .collect(Collectors.toList());
    assertThat(offsets).isInOrder();
  }

  @Test
  public void testSortRules() {
    // We can just use the same rule value here.
    NinjaRule rule = rule("rule1");

    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addRule(10, rule);
    parseResult.addRule(1115, rule);
    parseResult.addRule(5, rule);

    parseResult.sortResults();

    List<Long> offsets =
        parseResult.getRules().get(rule.getName()).stream()
            .map(Pair::getFirst)
            .collect(Collectors.toList());
    assertThat(offsets).isInOrder();
  }

  @Test
  public void testMerge() {
    NinjaRule rule1 = rule("rule1");
    NinjaRule rule2 = rule("rule2");

    NinjaFileParseResult parseResult1 = new NinjaFileParseResult();
    parseResult1.addRule(10, rule1);
    parseResult1.addVariable("from1", 7, NinjaVariableValue.createPlainText("111"));
    parseResult1.addVariable("abc", 5, NinjaVariableValue.createPlainText("5"));
    parseResult1.addVariable("abc", 115, NinjaVariableValue.createPlainText("7"));

    NinjaFileParseResult parseResult2 = new NinjaFileParseResult();
    parseResult2.addRule(10, rule2);
    parseResult2.addVariable("from2", 20017, NinjaVariableValue.createPlainText("222"));
    parseResult2.addVariable("abc", 2005, NinjaVariableValue.createPlainText("15"));
    parseResult2.addVariable("abc", 20015, NinjaVariableValue.createPlainText("17"));

    NinjaFileParseResult result =
        NinjaFileParseResult.merge(ImmutableList.of(parseResult1, parseResult2));
    assertThat(result.getRules()).hasSize(2);
    assertThat(result.getRules()).containsKey("rule1");
    assertThat(result.getRules()).containsKey("rule2");

    assertThat(result.getVariables()).hasSize(3);
    assertThat(result.getVariables()).containsKey("from1");
    assertThat(result.getVariables()).containsKey("from2");
    assertThat(result.getVariables()).containsKey("abc");

    List<Pair<Long, NinjaVariableValue>> abc = result.getVariables().get("abc");
    assertThat(abc).hasSize(4);
    assertThat(abc.stream().map(Pair::getFirst).collect(Collectors.toList())).isInOrder();
  }

  @Test
  public void testFindVariable() throws Exception {
    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("cba"));
    parseResult.addVariable("abc", 5, NinjaVariableValue.createPlainText("cba1"));
    parseResult.addVariable("abc", 14, NinjaVariableValue.createPlainText("cba2"));

    parseResult.sortResults();
    NinjaScope scope = new NinjaScope();
    parseResult.expandIntoScope(scope, Maps.newHashMap());

    assertThat(scope.findExpandedVariable(1, "not_there")).isNull();
    assertThat(scope.findExpandedVariable(1, "abc")).isNull();
    assertThat(scope.findExpandedVariable(6, "abc")).isEqualTo("cba1");

    assertThat(scope.findExpandedVariable(13, "abc")).isEqualTo("cba");
    assertThat(scope.findExpandedVariable(130, "abc")).isEqualTo("cba2");
  }

  @Test
  public void testFindVariableErrors() throws Exception {
    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("cba"));
    parseResult.addVariable("abc", 5, NinjaVariableValue.createPlainText("cba1"));
    parseResult.addVariable("abc", 14, NinjaVariableValue.createPlainText("cba2"));

    parseResult.sortResults();
    NinjaScope scope = new NinjaScope();
    parseResult.expandIntoScope(scope, Maps.newHashMap());

    IllegalStateException exception =
        assertThrows(IllegalStateException.class, () -> scope.findExpandedVariable(5, "abc"));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo("Trying to interpret declaration as reference.");
  }

  @Test
  public void testFindRule() throws Exception {
    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addRule(10, rule("rule1", "10"));
    parseResult.addRule(1115, rule("rule1", "1115"));
    parseResult.addRule(5, rule("rule1", "5"));

    parseResult.sortResults();
    NinjaScope scope = new NinjaScope();
    parseResult.expandIntoScope(scope, Maps.newHashMap());

    assertThat(scope.findRule(1, "non-existent")).isNull();
    assertThat(scope.findRule(1, "rule1")).isNull();

    NinjaRule rule1 = scope.findRule(6, "rule1");
    assertThat(rule1).isNotNull();
    assertThat(rule1.getVariables().get(NinjaRuleVariable.COMMAND).getRawText()).isEqualTo("5");

    rule1 = scope.findRule(15, "rule1");
    assertThat(rule1).isNotNull();
    assertThat(rule1.getVariables().get(NinjaRuleVariable.COMMAND).getRawText()).isEqualTo("10");
  }

  @Test
  public void testFindVariableInParentScope() throws Exception {
    NinjaFileParseResult parentParseResult = new NinjaFileParseResult();
    parentParseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parentParseResult.addVariable("edf", 120, NinjaVariableValue.createPlainText("edf"));
    parentParseResult.addVariable("xyz", 1000, NinjaVariableValue.createPlainText("xyz"));

    // This is subninja scope, not include scope.
    NinjaFileParseResult childParseResult = new NinjaFileParseResult();
    parentParseResult.addSubNinjaScope(140, scope -> childParseResult);
    // Shadows this variable from parent.
    childParseResult.addVariable("edf", 1, NinjaVariableValue.createPlainText("11111"));

    parentParseResult.sortResults();
    NinjaScope scope = new NinjaScope();
    parentParseResult.expandIntoScope(scope, Maps.newHashMap());

    assertThat(scope.getSubNinjaScopes()).hasSize(1);
    NinjaScope child = scope.getSubNinjaScopes().iterator().next();

    assertThat(child.findExpandedVariable(2, "abc")).isEqualTo("abc");
    assertThat(child.findExpandedVariable(2, "edf")).isEqualTo("11111");
    assertThat(child.findExpandedVariable(2, "xyz")).isNull();
  }

  @Test
  public void testfindExpandedVariableInIncludedScope() throws Exception {
    NinjaFileParseResult parentParseResult = new NinjaFileParseResult();
    parentParseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parentParseResult.addVariable("edf", 120, NinjaVariableValue.createPlainText("edf"));
    parentParseResult.addVariable("xyz", 1000, NinjaVariableValue.createPlainText("xyz"));

    NinjaFileParseResult childParseResult = new NinjaFileParseResult();
    parentParseResult.addIncludeScope(140, scope -> childParseResult);
    // Shadows this variable from parent.
    childParseResult.addVariable("edf", 1, NinjaVariableValue.createPlainText("11111"));
    childParseResult.addVariable("child", 2, NinjaVariableValue.createPlainText("child"));

    NinjaFileParseResult childParseResult2 = new NinjaFileParseResult();
    parentParseResult.addIncludeScope(200, scope -> childParseResult2);
    childParseResult2.addVariable("edf", 1, NinjaVariableValue.createPlainText("22222"));

    parentParseResult.sortResults();
    NinjaScope scope = new NinjaScope();
    parentParseResult.expandIntoScope(scope, Maps.newHashMap());

    assertThat(scope.findExpandedVariable(160, "edf")).isEqualTo("11111");
    assertThat(scope.findExpandedVariable(220, "edf")).isEqualTo("22222");
    assertThat(scope.findExpandedVariable(125, "edf")).isEqualTo("edf");
    assertThat(scope.findExpandedVariable(145, "child")).isEqualTo("child");
  }

  @Test
  public void testFindInRecursivelyIncluded() throws Exception {
    NinjaFileParseResult parentParseResult = new NinjaFileParseResult();
    parentParseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parentParseResult.addVariable("edf", 120, NinjaVariableValue.createPlainText("edf"));
    parentParseResult.addVariable("xyz", 1000, NinjaVariableValue.createPlainText("xyz"));

    NinjaFileParseResult childParseResult1 = new NinjaFileParseResult();
    parentParseResult.addIncludeScope(140, scope -> childParseResult1);
    // Shadows this variable from parent.
    childParseResult1.addVariable("edf", 1, NinjaVariableValue.createPlainText("11111"));
    childParseResult1.addVariable("child", 2, NinjaVariableValue.createPlainText("child"));

    NinjaFileParseResult childParseResult2 = new NinjaFileParseResult();
    childParseResult1.addIncludeScope(3, scope -> childParseResult2);
    childParseResult2.addVariable("edf", 1, NinjaVariableValue.createPlainText("22222"));

    parentParseResult.sortResults();
    NinjaScope scope = new NinjaScope();
    parentParseResult.expandIntoScope(scope, Maps.newHashMap());

    assertThat(scope.findExpandedVariable(220, "edf")).isEqualTo("22222");
  }

  @Test
  public void testVariableExpand() throws Exception {
    NinjaFileParseResult parseResult = new NinjaFileParseResult();
    parseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parseResult.addVariable("edf", 120, parseValue("=> $abc = ?"));
    parseResult.addVariable("abc", 130, NinjaVariableValue.createPlainText("redefined"));
    parseResult.addVariable("edf", 180, parseValue("now$: $abc!"));

    parseResult.sortResults();
    NinjaScope scope = new NinjaScope();
    parseResult.expandIntoScope(scope, Maps.newHashMap());

    assertThat(scope.findExpandedVariable(15, "abc")).isEqualTo("abc");
    assertThat(scope.findExpandedVariable(150, "edf")).isEqualTo("=> abc = ?");
    assertThat(scope.findExpandedVariable(140, "abc")).isEqualTo("redefined");
    assertThat(scope.findExpandedVariable(181, "edf")).isEqualTo("now: redefined!");
  }

  @Test
  public void testExpandWithParentChild() throws Exception {
    NinjaFileParseResult parentParseResult = new NinjaFileParseResult();
    parentParseResult.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parentParseResult.addVariable("edf", 120, parseValue("$abc === ${ abc }"));

    NinjaFileParseResult includeParseResult = new NinjaFileParseResult();
    parentParseResult.addIncludeScope(140, scope -> includeParseResult);
    includeParseResult.addVariable("included", 1, parseValue("<$abc and ${ edf }>"));

    NinjaFileParseResult childParseResult = new NinjaFileParseResult();
    parentParseResult.addSubNinjaScope(150, scope -> childParseResult);
    childParseResult.addVariable("subninja", 2, parseValue("$edf = ${ included }*"));

    parentParseResult.sortResults();
    NinjaScope parentScope = new NinjaScope();
    parentParseResult.expandIntoScope(parentScope, Maps.newHashMap());

    assertThat(parentScope.getIncludedScopes()).hasSize(1);
    NinjaScope includeScope = parentScope.getIncludedScopes().iterator().next();
    assertThat(parentScope.getSubNinjaScopes()).hasSize(1);
    NinjaScope childScope = parentScope.getSubNinjaScopes().iterator().next();

    assertThat(includeScope.findExpandedVariable(2, "included")).isEqualTo("<abc and abc === abc>");
    assertThat(childScope.findExpandedVariable(3, "subninja"))
        .isEqualTo("abc === abc = <abc and abc === abc>*");
    assertThat(parentScope.findExpandedVariable(150, "included"))
        .isEqualTo("<abc and abc === abc>");
  }

  private static NinjaRule rule(String name) {
    return rule(name, "command");
  }

  private static NinjaRule rule(String name, String command) {
    return new NinjaRule(
        ImmutableSortedMap.of(
            NinjaRuleVariable.NAME, NinjaVariableValue.createPlainText(name),
            NinjaRuleVariable.COMMAND, NinjaVariableValue.createPlainText(command)));
  }

  private static NinjaVariableValue parseValue(String text) throws Exception {
    ByteBuffer bb = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    NinjaLexer lexer = new NinjaLexer(new ByteBufferFragment(bb, 0, bb.limit()));
    return new NinjaParserStep(lexer).parseVariableValue();
  }
}
