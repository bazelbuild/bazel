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
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRule;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRuleVariable;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaVariableValue;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.util.Pair;
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
    scope.addVariable("abc", 12, value("cba"));
    scope.addVariable("abc", 1, value("cba1"));
    scope.addVariable("abc", 14, value("cba2"));

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
    scope1.addVariable("from1", 7, value("111"));
    scope1.addVariable("abc", 5, value("5"));
    scope1.addVariable("abc", 115, value("7"));

    NinjaScope scope2 = new NinjaScope();
    scope2.addRule(10, rule2);
    scope2.addVariable("from2", 20017, value("222"));
    scope2.addVariable("abc", 2005, value("15"));
    scope2.addVariable("abc", 20015, value("17"));

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
  public void testFindVariable() throws Exception {
    NinjaScope scope = new NinjaScope();
    scope.addVariable("abc", 12, value("cba"));
    scope.addVariable("abc", 5, value("cba1"));
    scope.addVariable("abc", 14, value("cba2"));

    scope.sortResults();

    assertThat(scope.findVariable(1, "not_there")).isNull();
    assertThat(scope.findVariable(1, "abc")).isNull();
    NinjaVariableValue abc = scope.findVariable(6, "abc");
    assertThat(abc).isNotNull();
    assertThat(abc.getText()).isEqualTo("cba1");

    abc = scope.findVariable(13, "abc");
    assertThat(abc).isNotNull();
    assertThat(abc.getText()).isEqualTo("cba");

    abc = scope.findVariable(130, "abc");
    assertThat(abc).isNotNull();
    assertThat(abc.getText()).isEqualTo("cba2");
  }

  @Test
  public void testFindVariableErrors() {
    NinjaScope scope = new NinjaScope();
    scope.addVariable("abc", 12, value("cba"));
    scope.addVariable("abc", 5, value("cba1"));
    scope.addVariable("abc", 14, value("cba2"));

    scope.sortResults();

    IllegalStateException exception =
        assertThrows(IllegalStateException.class, () -> scope.findVariable(5, "abc"));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo("Trying to interpret declaration as reference.");
  }

  @Test
  public void testFindRule() throws Exception {
    NinjaScope scope = new NinjaScope();
    scope.addRule(10, rule("rule1", "10"));
    scope.addRule(1115, rule("rule1", "1115"));
    scope.addRule(5, rule("rule1", "5"));

    scope.sortResults();

    assertThat(scope.findRule(1, "non-existent")).isNull();
    assertThat(scope.findRule(1, "rule1")).isNull();

    NinjaRule rule1 = scope.findRule(6, "rule1");
    assertThat(rule1).isNotNull();
    assertThat(rule1.getVariables().get(NinjaRuleVariable.COMMAND).getText()).isEqualTo("5");

    rule1 = scope.findRule(15, "rule1");
    assertThat(rule1).isNotNull();
    assertThat(rule1.getVariables().get(NinjaRuleVariable.COMMAND).getText()).isEqualTo("10");
  }

  private static NinjaRule rule(String name) {
    return rule(name, "command");
  }

  private static NinjaRule rule(String name, String command) {
    return new NinjaRule(
        ImmutableSortedMap.of(
            NinjaRuleVariable.NAME, value(name),
            NinjaRuleVariable.COMMAND, value(command)));
  }

  private static NinjaVariableValue value(String text) {
    return new NinjaVariableValue(text, ImmutableSortedKeyListMultimap.of());
  }
}
