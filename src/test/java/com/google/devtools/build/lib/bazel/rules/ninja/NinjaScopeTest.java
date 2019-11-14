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
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.lexer.NinjaLexer;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaParser;
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

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope} */
@RunWith(JUnit4.class)
public class NinjaScopeTest {
  @Test
  public void testSortVariables() {
    NinjaScope scope = new NinjaScope();
    scope.addVariable("abc", 12, NinjaVariableValue.createPlainText("cba"));
    scope.addVariable("abc", 1, NinjaVariableValue.createPlainText("cba1"));
    scope.addVariable("abc", 14, NinjaVariableValue.createPlainText("cba2"));

    scope.sortResults();

    List<Integer> offsets =
        scope.getVariables().get("abc").stream().map(Pair::getFirst).collect(Collectors.toList());
    assertThat(offsets).isInOrder();
  }

  @Test
  public void testSortRules() {
    // We can just use the same rule value here.
    NinjaRule rule = rule("rule1");

    NinjaScope scope = new NinjaScope();
    scope.addRule(10, rule);
    scope.addRule(1115, rule);
    scope.addRule(5, rule);

    scope.sortResults();

    List<Integer> offsets =
        scope.getRules().get(rule.getName()).stream()
            .map(Pair::getFirst)
            .collect(Collectors.toList());
    assertThat(offsets).isInOrder();
  }

  @Test
  public void testMerge() {
    NinjaRule rule1 = rule("rule1");
    NinjaRule rule2 = rule("rule2");

    NinjaScope scope1 = new NinjaScope();
    scope1.addRule(10, rule1);
    scope1.addVariable("from1", 7, NinjaVariableValue.createPlainText("111"));
    scope1.addVariable("abc", 5, NinjaVariableValue.createPlainText("5"));
    scope1.addVariable("abc", 115, NinjaVariableValue.createPlainText("7"));

    NinjaScope scope2 = new NinjaScope();
    scope2.addRule(10, rule2);
    scope2.addVariable("from2", 20017, NinjaVariableValue.createPlainText("222"));
    scope2.addVariable("abc", 2005, NinjaVariableValue.createPlainText("15"));
    scope2.addVariable("abc", 20015, NinjaVariableValue.createPlainText("17"));

    NinjaScope result = NinjaScope.mergeScopeParts(ImmutableList.of(scope1, scope2));
    assertThat(result.getRules()).hasSize(2);
    assertThat(result.getRules()).containsKey("rule1");
    assertThat(result.getRules()).containsKey("rule2");

    assertThat(result.getVariables()).hasSize(3);
    assertThat(result.getVariables()).containsKey("from1");
    assertThat(result.getVariables()).containsKey("from2");
    assertThat(result.getVariables()).containsKey("abc");

    List<Pair<Integer, NinjaVariableValue>> abc = result.getVariables().get("abc");
    assertThat(abc).hasSize(4);
    assertThat(abc.stream().map(Pair::getFirst).collect(Collectors.toList())).isInOrder();
  }

  @Test
  public void testFindVariable() {
    NinjaScope scope = new NinjaScope();
    scope.addVariable("abc", 12, NinjaVariableValue.createPlainText("cba"));
    scope.addVariable("abc", 5, NinjaVariableValue.createPlainText("cba1"));
    scope.addVariable("abc", 14, NinjaVariableValue.createPlainText("cba2"));

    scope.sortResults();

    assertThat(scope.findVariable(1, "not_there")).isNull();
    assertThat(scope.findVariable(1, "abc")).isNull();
    NinjaVariableValue abc = scope.findVariable(6, "abc");
    assertThat(abc).isNotNull();
    assertThat(abc.getRawText()).isEqualTo("cba1");

    abc = scope.findVariable(13, "abc");
    assertThat(abc).isNotNull();
    assertThat(abc.getRawText()).isEqualTo("cba");

    abc = scope.findVariable(130, "abc");
    assertThat(abc).isNotNull();
    assertThat(abc.getRawText()).isEqualTo("cba2");
  }

  @Test
  public void testFindVariableErrors() {
    NinjaScope scope = new NinjaScope();
    scope.addVariable("abc", 12, NinjaVariableValue.createPlainText("cba"));
    scope.addVariable("abc", 5, NinjaVariableValue.createPlainText("cba1"));
    scope.addVariable("abc", 14, NinjaVariableValue.createPlainText("cba2"));

    scope.sortResults();

    IllegalStateException exception =
        assertThrows(IllegalStateException.class, () -> scope.findVariable(5, "abc"));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo("Trying to interpret declaration as reference.");
  }

  @Test
  public void testFindRule() {
    NinjaScope scope = new NinjaScope();
    scope.addRule(10, rule("rule1", "10"));
    scope.addRule(1115, rule("rule1", "1115"));
    scope.addRule(5, rule("rule1", "5"));

    scope.sortResults();

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
  public void testFindVariableInParentScope() {
    NinjaScope parent = new NinjaScope();
    parent.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parent.addVariable("edf", 120, NinjaVariableValue.createPlainText("edf"));
    parent.addVariable("xyz", 1000, NinjaVariableValue.createPlainText("xyz"));

    // This is subninja scope, not include scope.
    NinjaScope child = new NinjaScope(parent, 140);
    // Shadows this variable from parent.
    child.addVariable("edf", 1, NinjaVariableValue.createPlainText("11111"));

    NinjaVariableValue abcVar = child.findVariable(2, "abc");
    assertThat(abcVar).isNotNull();
    assertThat(abcVar.getRawText()).isEqualTo("abc");

    NinjaVariableValue edfVar = child.findVariable(2, "edf");
    assertThat(edfVar).isNotNull();
    assertThat(edfVar.getRawText()).isEqualTo("11111");

    assertThat(child.findVariable(2, "xyz")).isNull();
  }

  @Test
  public void testFindVariableInIncludedScope() {
    NinjaScope parent = new NinjaScope();
    parent.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parent.addVariable("edf", 120, NinjaVariableValue.createPlainText("edf"));
    parent.addVariable("xyz", 1000, NinjaVariableValue.createPlainText("xyz"));

    NinjaScope child = parent.createIncludeScope(140);
    // Shadows this variable from parent.
    child.addVariable("edf", 1, NinjaVariableValue.createPlainText("11111"));
    child.addVariable("child", 2, NinjaVariableValue.createPlainText("child"));

    NinjaScope child2 = parent.createIncludeScope(200);
    child2.addVariable("edf", 1, NinjaVariableValue.createPlainText("22222"));

    NinjaVariableValue edfVar = parent.findVariable(160, "edf");
    assertThat(edfVar).isNotNull();
    assertThat(edfVar.getRawText()).isEqualTo("11111");

    NinjaVariableValue edfVarFromChild2 = parent.findVariable(220, "edf");
    assertThat(edfVarFromChild2).isNotNull();
    assertThat(edfVarFromChild2.getRawText()).isEqualTo("22222");

    NinjaVariableValue edfVarBefore = parent.findVariable(125, "edf");
    assertThat(edfVarBefore).isNotNull();
    assertThat(edfVarBefore.getRawText()).isEqualTo("edf");

    NinjaVariableValue childVar = parent.findVariable(145, "child");
    assertThat(childVar).isNotNull();
    assertThat(childVar.getRawText()).isEqualTo("child");
  }

  @Test
  public void testFindInRecursivelyIncluded() {
    NinjaScope parent = new NinjaScope();
    parent.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parent.addVariable("edf", 120, NinjaVariableValue.createPlainText("edf"));
    parent.addVariable("xyz", 1000, NinjaVariableValue.createPlainText("xyz"));

    NinjaScope child = parent.createIncludeScope(140);
    // Shadows this variable from parent.
    child.addVariable("edf", 1, NinjaVariableValue.createPlainText("11111"));
    child.addVariable("child", 2, NinjaVariableValue.createPlainText("child"));

    NinjaScope child2 = child.createIncludeScope(3);
    child2.addVariable("edf", 1, NinjaVariableValue.createPlainText("22222"));

    NinjaVariableValue edfVarFromChild2 = parent.findVariable(220, "edf");
    assertThat(edfVarFromChild2).isNotNull();
    assertThat(edfVarFromChild2.getRawText()).isEqualTo("22222");
  }

  @Test
  public void testVariableExpand() throws GenericParsingException {
    NinjaScope scope = new NinjaScope();
    scope.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    scope.addVariable("edf", 120, parseValue("=> $abc = ?"));
    scope.addVariable("abc", 130, NinjaVariableValue.createPlainText("redefined"));
    scope.addVariable("edf", 180, parseValue("now$: $abc!"));

    scope.expandVariables();

    assertThat(scope.findExpandedVariable(15, "abc")).isEqualTo("abc");
    assertThat(scope.findExpandedVariable(150, "edf")).isEqualTo("=> abc = ?");
    assertThat(scope.findExpandedVariable(140, "abc")).isEqualTo("redefined");
    assertThat(scope.findExpandedVariable(181, "edf")).isEqualTo("now: redefined!");
  }

  @Test
  public void testExpandWithParentChild() throws GenericParsingException {
    NinjaScope parent = new NinjaScope();
    parent.addVariable("abc", 12, NinjaVariableValue.createPlainText("abc"));
    parent.addVariable("edf", 120, parseValue("$abc === ${ abc }"));

    NinjaScope includeScope = parent.createIncludeScope(140);
    includeScope.addVariable("included", 1, parseValue("<$abc and ${ edf }>"));

    NinjaScope child = new NinjaScope(parent, 150);
    child.addVariable("subninja", 2, parseValue("$edf = ${ included }*"));

    parent.expandVariables();
    child.expandVariables();

    assertThat(includeScope.findExpandedVariable(2, "included")).isEqualTo("<abc and abc === abc>");
    assertThat(child.findExpandedVariable(3, "subninja"))
        .isEqualTo("abc === abc = <abc and abc === abc>*");
    assertThat(parent.findExpandedVariable(150, "included")).isEqualTo("<abc and abc === abc>");
  }

  private static NinjaRule rule(String name) {
    return rule(name, "command");
  }

  private static NinjaRule rule(String name, String command) {
    return new NinjaRule(ImmutableSortedMap.of(
        NinjaRuleVariable.NAME, NinjaVariableValue.createPlainText(name),
        NinjaRuleVariable.COMMAND, NinjaVariableValue.createPlainText(command)));
  }

  private static NinjaVariableValue parseValue(String text) throws GenericParsingException {
    ByteBuffer bb = ByteBuffer.wrap(text.getBytes(StandardCharsets.ISO_8859_1));
    NinjaLexer lexer = new NinjaLexer(new ByteBufferFragment(bb, 0, bb.limit()));
    return new NinjaParser(lexer).parseVariableValue("test");
  }
}
