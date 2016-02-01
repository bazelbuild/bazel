// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the validation process of Skylark files.
 */
@RunWith(JUnit4.class)
public class ValidationTest extends EvaluationTestCase {

  @Test
  public void testAssignmentNotValidLValue() {
    checkError("can only assign to variables and tuples, not to ''a''", "'a' = 1");
  }

  @Test
  public void testTopLevelForStatement() throws Exception {
    checkError("'For' is not allowed as a top level statement", "for i in [1,2,3]: a = i\n");
  }

  @Test
  public void testReturnOutsideFunction() throws Exception {
    checkError("Return statements must be inside a function", "return 2\n");
  }

  @Test
  public void testTwoFunctionsWithTheSameName() throws Exception {
    checkError(
        "Variable foo is read only", "def foo():", "  return 1", "def foo(x, y):", "  return 1");
  }

  @Test
  public void testFunctionLocalVariable() throws Exception {
    checkError(
        "name 'a' is not defined",
        "def func2(b):",
        "  c = b",
        "  c = a",
        "def func1():",
        "  a = 1",
        "  func2(2)");
  }

  @Test
  public void testFunctionLocalVariableDoesNotEffectGlobalValidationEnv() throws Exception {
    checkError("name 'a' is not defined", "def func1():", "  a = 1", "def func2(b):", "  b = a");
  }

  @Test
  public void testFunctionParameterDoesNotEffectGlobalValidationEnv() throws Exception {
    checkError("name 'a' is not defined", "def func1(a):", "  return a", "def func2():", "  b = a");
  }

  @Test
  public void testLocalValidationEnvironmentsAreSeparated() throws Exception {
    parse("def func1():", "  a = 1", "def func2():", "  a = 'abc'\n");
  }

  @Test
  public void testBuiltinSymbolsAreReadOnly() throws Exception {
    checkError("Variable repr is read only", "repr = 1");
  }

  @Test
  public void testSkylarkGlobalVariablesAreReadonly() throws Exception {
    checkError("Variable a is read only", "a = 1", "a = 2");
  }

  @Test
  public void testFunctionDefRecursion() throws Exception {
    parse("def func():", "  func()\n");
  }

  @Test
  public void testMutualRecursion() throws Exception {
    parse("def foo(i):", "  bar(i)", "def bar(i):", "  foo(i)", "foo(4)");
  }

  @Test
  public void testFunctionDefinedBelow() {
    parse("def bar(): a = foo() + 'a'", "def foo(): return 1\n");
  }

  @Test
  public void testFunctionDoesNotExist() {
    checkError("function 'foo' does not exist", "def bar(): a = foo() + 'a'");
  }

  @Test
  public void testStructMembersAreImmutable() {
    checkError(
        "can only assign to variables and tuples, not to 's.x'",
        "s = struct(x = 'a')",
        "s.x = 'b'\n");
  }

  @Test
  public void testStructDictMembersAreImmutable() {
    checkError(
        "can only assign to variables and tuples, not to 's.x['b']'",
        "s = struct(x = {'a' : 1})",
        "s.x['b'] = 2\n");
  }

  @Test
  public void testTupleLiteralWorksForDifferentTypes() throws Exception {
    parse("('a', 1)");
  }

  @Test
  public void testDictLiteralDifferentValueTypeWorks() throws Exception {
    parse("{'a': 1, 'b': 'c'}");
  }

  @Test
  public void testNoneAssignment() throws Exception {
    parse("def func():", "  a = None", "  a = 2", "  a = None\n");
  }

  @Test
  public void testNoneIsAnyType() throws Exception {
    parse("None + None");
    parse("2 == None");
    parse("None > 'a'");
    parse("[] in None");
    parse("5 * None");
  }

  // Skylark built-in functions specific tests

  @Test
  public void testFuncReturningDictAssignmentAsLValue() throws Exception {
    checkError(
        "can only assign to variables and tuples, not to 'my_dict()['b']'",
        "def my_dict():",
        "  return {'a': 1}",
        "def func():",
        "  my_dict()['b'] = 2",
        "  return d\n");
  }

  @Test
  public void testEmptyLiteralGenericIsSetInLaterConcatWorks() {
    parse("def func():", "  s = {}", "  s['a'] = 'b'\n");
  }

  @Test
  public void testReadOnlyWorksForSimpleBranching() {
    parse("if 1:", "  v = 'a'", "else:", "  v = 'b'");
  }

  @Test
  public void testReadOnlyWorksForNestedBranching() {
    parse(
        "if 1:",
        "  if 0:",
        "    v = 'a'",
        "  else:",
        "    v = 'b'",
        "else:",
        "  if 0:",
        "    v = 'c'",
        "  else:",
        "    v = 'd'\n");
  }

  @Test
  public void testReadOnlyWorksForDifferentLevelBranches() {
    checkError("Variable v is read only", "if 1:", "  if 1:", "    v = 'a'", "  v = 'b'\n");
  }

  @Test
  public void testReadOnlyWorksWithinSimpleBranch() {
    checkError(
        "Variable v is read only", "if 1:", "  v = 'a'", "else:", "  v = 'b'", "  v = 'c'\n");
  }

  @Test
  public void testReadOnlyWorksWithinNestedBranch() {
    checkError(
        "Variable v is read only",
        "if 1:",
        "  v = 'a'",
        "else:",
        "  if 1:",
        "    v = 'b'",
        "  else:",
        "    v = 'c'",
        "    v = 'd'\n");
  }

  @Test
  public void testReadOnlyWorksAfterSimpleBranch() {
    checkError("Variable v is read only", "if 1:", "  v = 'a'", "else:", "  w = 'a'", "v = 'b'");
  }

  @Test
  public void testReadOnlyWorksAfterNestedBranch() {
    checkError("Variable v is read only", "if 1:", "  if 1:", "    v = 'a'", "v = 'b'");
  }

  @Test
  public void testReadOnlyWorksAfterNestedBranch2() {
    checkError(
        "Variable v is read only",
        "if 1:",
        "  v = 'a'",
        "else:",
        "  if 0:",
        "    w = 1",
        "v = 'b'\n");
  }

  @Test
  public void testModulesReadOnlyInFuncDefBody() {
    parse("def func():", "  cmd_helper = set()");
  }

  @Test
  public void testBuiltinGlobalFunctionsReadOnlyInFuncDefBody() {
    parse("def func():", "  rule = 'abc'");
  }

  @Test
  public void testBuiltinGlobalFunctionsReadOnlyAsFuncDefArg() {
    parse("def func(rule):", "  return rule");
  }

  @Test
  public void testFunctionReturnsFunction() {
    parse(
        "def rule(*, implementation): return None",
        "def impl(ctx): return None",
        "",
        "skylark_rule = rule(implementation = impl)",
        "",
        "def macro(name):",
        "  skylark_rule(name = name)");
  }

  @Test
  public void testTypeForBooleanLiterals() {
    parse("len([1, 2]) == 0 and True");
    parse("len([1, 2]) == 0 and False");
  }

  @Test
  public void testDollarErrorDoesNotLeak() throws Exception {
    setFailFast(false);
    parseFile(
        "def GenerateMapNames():", "  a = 2", "  b = [3, 4]", "  if a not b:", "    print(a)");
    assertContainsError("syntax error at 'b': expected in");
    // Parser uses "$error" symbol for error recovery.
    // It should not be used in error messages.
    for (Event event : getEventCollector()) {
      assertThat(event.getMessage()).doesNotContain("$error$");
    }
  }

  @Test
  public void testParentWithSkylarkModule() throws Exception {
    Class<?> emptyTupleClass = Tuple.EMPTY.getClass();
    Class<?> tupleClass = Tuple.of(1, "a", "b").getClass();
    Class<?> mutableListClass = new MutableList(Tuple.of(1, 2, 3), env).getClass();

    assertThat(mutableListClass).isEqualTo(MutableList.class);
    assertThat(MutableList.class.isAnnotationPresent(SkylarkModule.class)).isTrue();
    assertThat(EvalUtils.getParentWithSkylarkModule(MutableList.class))
        .isEqualTo(MutableList.class);
    assertThat(EvalUtils.getParentWithSkylarkModule(emptyTupleClass)).isEqualTo(Tuple.class);
    assertThat(EvalUtils.getParentWithSkylarkModule(tupleClass)).isEqualTo(Tuple.class);
    // TODO(bazel-team): make a tuple not a list anymore.
    assertThat(EvalUtils.getParentWithSkylarkModule(tupleClass)).isEqualTo(Tuple.class);

    // TODO(bazel-team): fix that?
    assertThat(ClassObject.class.isAnnotationPresent(SkylarkModule.class)).isFalse();
    assertThat(ClassObject.SkylarkClassObject.class.isAnnotationPresent(SkylarkModule.class))
        .isTrue();
    assertThat(
            EvalUtils.getParentWithSkylarkModule(ClassObject.SkylarkClassObject.class)
                == ClassObject.SkylarkClassObject.class)
        .isTrue();
    assertThat(EvalUtils.getParentWithSkylarkModule(ClassObject.class)).isNull();
  }

  @Test
  public void testSkylarkTypeEquivalence() throws Exception {
    // All subclasses of SkylarkList are made equivalent
    Class<?> emptyTupleClass = Tuple.EMPTY.getClass();
    Class<?> tupleClass = Tuple.of(1, "a", "b").getClass();
    Class<?> mutableListClass = new MutableList(Tuple.of(1, 2, 3), env).getClass();

    assertThat(SkylarkType.of(mutableListClass)).isEqualTo(SkylarkType.LIST);
    assertThat(SkylarkType.of(emptyTupleClass)).isEqualTo(SkylarkType.TUPLE);
    assertThat(SkylarkType.of(tupleClass)).isEqualTo(SkylarkType.TUPLE);
    assertThat(SkylarkType.TUPLE).isNotEqualTo(SkylarkType.LIST);

    // Also for ClassObject
    assertThat(SkylarkType.of(ClassObject.SkylarkClassObject.class)).isEqualTo(SkylarkType.STRUCT);
    // TODO(bazel-team): fix that?
    assertThat(SkylarkType.of(ClassObject.class)).isNotEqualTo(SkylarkType.STRUCT);

    // Also test for these bazel classes, to avoid some regression.
    // TODO(bazel-team): move to some other place to remove dependency of syntax tests on Artifact?
    assertThat(SkylarkType.of(Artifact.SpecialArtifact.class))
        .isEqualTo(SkylarkType.of(Artifact.class));
    assertThat(SkylarkType.of(RuleConfiguredTarget.class)).isNotEqualTo(SkylarkType.STRUCT);
  }

  @Test
  public void testSkylarkTypeInclusion() throws Exception {
    assertThat(SkylarkType.INT.includes(SkylarkType.BOTTOM)).isTrue();
    assertThat(SkylarkType.BOTTOM.includes(SkylarkType.INT)).isFalse();
    assertThat(SkylarkType.TOP.includes(SkylarkType.INT)).isTrue();

    SkylarkType combo1 = SkylarkType.Combination.of(SkylarkType.LIST, SkylarkType.INT);
    assertThat(SkylarkType.LIST.includes(combo1)).isTrue();

    SkylarkType union1 =
        SkylarkType.Union.of(SkylarkType.MAP, SkylarkType.LIST, SkylarkType.STRUCT);
    assertThat(union1.includes(SkylarkType.MAP)).isTrue();
    assertThat(union1.includes(SkylarkType.STRUCT)).isTrue();
    assertThat(union1.includes(combo1)).isTrue();
    assertThat(union1.includes(SkylarkType.STRING)).isFalse();

    SkylarkType union2 =
        SkylarkType.Union.of(
            SkylarkType.LIST, SkylarkType.MAP, SkylarkType.STRING, SkylarkType.INT);
    SkylarkType inter1 = SkylarkType.intersection(union1, union2);
    assertThat(inter1.includes(SkylarkType.MAP)).isTrue();
    assertThat(inter1.includes(SkylarkType.LIST)).isTrue();
    assertThat(inter1.includes(combo1)).isTrue();
    assertThat(inter1.includes(SkylarkType.INT)).isFalse();
  }

  private void parse(String... lines) {
    parseFile(lines);
    assertNoWarningsOrErrors();
  }

  private void checkError(String errorMsg, String... lines) {
    setFailFast(false);
    parseFile(lines);
    assertContainsError(errorMsg);
  }
}
