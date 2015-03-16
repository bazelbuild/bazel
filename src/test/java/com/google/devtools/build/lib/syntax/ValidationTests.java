// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;

/**
 * Tests for the validation process of Skylark files.
 */
@RunWith(JUnit4.class)
public class ValidationTests extends AbstractParserTestCase {

  @Test
  public void testIncompatibleLiteralTypesStringInt() {
    checkError("bad variable 'a': int is incompatible with string at /some/file.txt",
        "def foo():\n",
        "  a = '1'",
        "  a = 1");
  }

  @Test
  public void testIncompatibleLiteralTypesDictString() {
    checkError("bad variable 'a': int is incompatible with dict of ints at /some/file.txt:3:3",
        "def foo():\n",
        "  a = {1 : 'x'}",
        "  a = 1");
  }

  @Test
  public void testIncompatibleLiteralTypesInIf() {
    checkError("bad variable 'a': int is incompatible with string at /some/file.txt",
        "def foo():\n",
        "  if 1:",
        "    a = 'a'",
        "  else:",
        "    a = 1");
  }

  @Test
  public void testAssignmentNotValidLValue() {
    checkError("can only assign to variables, not to ''a''", "'a' = 1");
  }

  @Test
  public void testForNotIterable() throws Exception {
    checkError("type 'int' is not iterable",
          "def func():\n"
        + "  for i in 5: a = i\n");
  }

  @Test
  public void testForIterableWithUknownArgument() throws Exception {
    parse("def func(x=None):\n"
        + "  for i in x: a = i\n");
  }

  @Test
  public void testForNotIterableBinaryExpression() throws Exception {
    checkError("type 'int' is not iterable",
          "def func():\n"
        + "  for i in 1 + 1: a = i\n");
  }

  @Test
  public void testOptionalArgument() throws Exception {
    checkError("type 'int' is not iterable",
          "def func(x=5):\n"
        + "  for i in x: a = i\n");
  }

  @Test
  public void testOptionalArgumentHasError() throws Exception {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
          "def func(x=5+'a'):\n"
        + "  return 0\n");
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
  public void testTwoReturnTypes() throws Exception {
    checkError("bad return type of foo: string is incompatible with int at /some/file.txt:3:5",
        "def foo(x):",
        "  if x:",
        "    return 1",
        "  else:",
        "    return 'a'");
  }

  @Test
  public void testTwoFunctionsWithTheSameName() throws Exception {
    checkError("function foo already exists",
        "def foo():",
        "  return 1",
        "def foo(x, y):",
        "  return 1");
  }

  @Test
  public void testDynamicTypeCheck() throws Exception {
    checkError("bad variable 'a': string is incompatible with int at /some/file.txt:2:3",
        "def foo():",
        "  a = 1",
        "  a = '1'");
  }

  @Test
  public void testFunctionLocalVariable() throws Exception {
    checkError("name 'a' is not defined",
        "def func2(b):",
        "  c = b",
        "  c = a",
        "def func1():",
        "  a = 1",
        "  func2(2)");
  }

  @Test
  public void testFunctionLocalVariableDoesNotEffectGlobalValidationEnv() throws Exception {
    checkError("name 'a' is not defined",
        "def func1():",
        "  a = 1",
        "def func2(b):",
        "  b = a");
  }

  @Test
  public void testFunctionParameterDoesNotEffectGlobalValidationEnv() throws Exception {
    checkError("name 'a' is not defined",
        "def func1(a):",
        "  return a",
        "def func2():",
        "  b = a");
  }

  @Test
  public void testLocalValidationEnvironmentsAreSeparated() throws Exception {
    parse(
          "def func1():\n"
        + "  a = 1\n"
        + "def func2():\n"
        + "  a = 'abc'\n");
  }

  @Test
  public void testListComprehensionNotIterable() throws Exception {
    checkError("type 'int' is not iterable",
        "[i for i in 1 for j in [2]]");
  }

  @Test
  public void testListComprehensionNotIterable2() throws Exception {
    checkError("type 'int' is not iterable",
        "[i for i in [1] for j in 123]");
  }

  @Test
  public void testListIsNotComparable() {
    checkError("list of strings is not comparable", "['a'] > 1");
  }

  @Test
  public void testStringCompareToInt() {
    checkError("bad comparison: int is incompatible with string", "'a' > 1");
  }

  @Test
  public void testInOnInt() {
    checkError("operand 'in' only works on strings, dictionaries, "
        + "lists, sets or tuples, not on a(n) int", "1 in 2");
  }

  @Test
  public void testUnsupportedOperator() {
    checkError("unsupported operand type(s) for -: 'string' and 'int'", "'a' - 1");
  }

  @Test
  public void testBuiltinSymbolsAreReadOnly() throws Exception {
    checkError("Variable rule is read only", "rule = 1");
  }

  @Test
  public void testSkylarkGlobalVariablesAreReadonly() throws Exception {
    checkError("Variable a is read only",
        "a = 1\n"
        + "a = 2");
  }

  @Test
  public void testFunctionDefRecursion() throws Exception {
    checkError("function 'func' does not exist",
        "def func():\n"
      + "  func()\n");
  }

  @Test
  public void testMutualRecursion() throws Exception {
    checkError("function 'bar' does not exist",
        "def foo(i):\n"
      + "  bar(i)\n"
      + "def bar(i):\n"
      + "  foo(i)\n"
      + "foo(4)");
  }

  @Test
  public void testFunctionReturnValue() {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
          "def foo(): return 1\n"
        + "a = foo() + 'a'\n");
  }

  @Test
  public void testFunctionReturnValueInFunctionDef() {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
          "def foo(): return 1\n"
        + "def bar(): a = foo() + 'a'\n");
  }

  @Test
  public void testFunctionDoesNotExistInFunctionDef() {
    checkError("function 'foo' does not exist",
          "def bar(): a = foo() + 'a'\n"
        + "def foo(): return 1\n");
  }

  @Test
  public void testStructMembersAreImmutable() {
    checkError("can only assign to variables, not to 's.x'",
        "s = struct(x = 'a')\n"
      + "s.x = 'b'\n");
  }

  @Test
  public void testStructDictMembersAreImmutable() {
    checkError("can only assign to variables, not to 's.x['b']'",
        "s = struct(x = {'a' : 1})\n"
      + "s.x['b'] = 2\n");
  }

  @Test
  public void testTupleAssign() throws Exception {
    // TODO(bazel-team): fix our code so 'tuple' not 'list' gets printed.
    checkError("unsupported operand type(s) for +: 'list' and 'dict of ints'",
        "d = (1, 2)\n"
      + "d[0] = 2\n");
  }

  @Test
  public void testAssignOnNonCollection() throws Exception {
    checkError("unsupported operand type(s) for +: 'string' and 'dict of ints'",
        "d = 'abc'\n"
      + "d[0] = 2");
  }

  @Test
  public void testNsetBadRightOperand() throws Exception {
    checkError("can only concatenate nested sets with other nested sets or list of items, "
        + "not 'string'", "set() + 'a'");
  }

  @Test
  public void testNsetBadItemType() throws Exception {
    checkError("bad nested set: set of ints is incompatible with set of strings "
        + "at /some/file.txt:1:1",
        "(set() + ['a']) + [1]");
  }

  @Test
  public void testNsetBadNestedItemType() throws Exception {
    checkError("bad nested set: set of ints is incompatible with set of strings "
        + "at /some/file.txt:1:1",
        "(set() + ['b']) + (set() + [1])");
  }

  @Test
  public void testTypeInferenceForMethodLibraryFunction() throws Exception {
    checkError("bad variable 'l': string is incompatible with int at /some/file.txt:2:3",
          "def foo():\n"
        + "  l = len('abc')\n"
        + "  l = 'a'");
  }

  @Test
  public void testListLiteralBadTypes() throws Exception {
    checkError("bad list literal: int is incompatible with string at /some/file.txt:1:1",
        "['a', 1]");
  }

  @Test
  public void testTupleLiteralWorksForDifferentTypes() throws Exception {
    parse("('a', 1)");
  }

  @Test
  public void testDictLiteralBadKeyTypes() throws Exception {
    checkError("bad dict literal: int is incompatible with string at /some/file.txt:1:1",
        "{'a': 1, 1: 2}");
  }

  @Test
  public void testDictLiteralDifferentValueTypeWorks() throws Exception {
    parse("{'a': 1, 'b': 'c'}");
  }

  @Test
  public void testListConcatBadTypes() throws Exception {
    checkError("bad list concatenation: list of ints is incompatible with list of strings"
        + " at /some/file.txt:1:1",
        "['a'] + [1]");
  }

  @Test
  public void testDictConcatBadKeyTypes() throws Exception {
    checkError("bad dict concatenation: dict of ints is incompatible with dict of strings "
        + "at /some/file.txt:1:1",
        "{'a': 1} + {1: 2}");
  }

  @Test
  public void testDictLiteralBadKeyType() throws Exception {
    checkError("Dict cannot contain composite type 'list of strings' as key", "{['a']: 1}");
  }

  @Test
  public void testAndTypeInfer() throws Exception {
    checkError("unsupported operand type(s) for +: 'string' and 'int'", "('a' and 'b') + 1");
  }

  @Test
  public void testOrTypeInfer() throws Exception {
    checkError("unsupported operand type(s) for +: 'string' and 'int'", "('' or 'b') + 1");
  }

  @Test
  public void testAndDifferentTypes() throws Exception {
    checkError("bad and operator: int is incompatible with string at /some/file.txt:1:1",
        "'ab' and 3");
  }

  @Test
  public void testOrDifferentTypes() throws Exception {
    checkError("bad or operator: int is incompatible with string at /some/file.txt:1:1",
        "'ab' or 3");
  }

  @Test
  public void testOrNone() throws Exception {
    parse("a = None or 3");
  }

  @Test
  public void testNoneAssignment() throws Exception {
    parse("def func():\n"
        + "  a = None\n"
        + "  a = 2\n"
        + "  a = None\n");
  }

  @Test
  public void testNoneAssignmentError() throws Exception {
    checkError("bad variable 'a': string is incompatible with int at /some/file.txt",
          "def func():\n"
        + "  a = None\n"
        + "  a = 2\n"
        + "  a = None\n"
        + "  a = 'b'\n");
  }

  @Test
  public void testNoneIsAnyType() throws Exception {
    parse("None + None");
    parse("2 == None");
    parse("None > 'a'");
    parse("[] in None");
    parse("5 * None");
  }

  @Test
  public void testDictComprehensionNotOnList() throws Exception {
    checkError("Dict comprehension elements must be a list", "{k : k for k in 'abc'}");
  }

  @Test
  public void testTypeInferenceForUserDefinedFunction() throws Exception {
    checkError("bad variable 'a': string is incompatible with int at /some/file.txt",
          "def func():\n"
        + "  return 'a'\n"
        + "def foo():\n"
        + "  a = 1\n"
        + "  a = func()\n");
  }

  @Test
  public void testCallingNonFunction() {
    checkError("a is not a function",
        "a = '1':\n"
      + "a()\n");
  }

  @Test
  public void testFuncallArgument() {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
        "def foo(x): return x\n"
      + "a = foo(1 + 'a')");
  }

  // Skylark built-in functions specific tests

  @Test
  public void testTypeInferenceForSkylarkBuiltinGlobalFunction() throws Exception {
    checkError("bad variable 'a': string is incompatible with function at /some/file.txt:3:3",
          "def impl(ctx): return None\n"
        + "def foo():\n"
        + "  a = rule(impl)\n"
        + "  a = 'a'\n");
  }

  @Test
  public void testTypeInferenceForSkylarkBuiltinObjectFunction() throws Exception {
    checkError("bad variable 'a': string is incompatible with Attribute at /some/file.txt",
        "def foo():\n"
        + "  a = attr.int()\n"
        + "  a = 'a'\n");
  }

  @Test
  public void testFuncReturningDictAssignmentAsLValue() throws Exception {
    checkError("can only assign to variables, not to 'dict([])['b']'",
          "def dict():\n"
        + "  return {'a': 1}\n"
        + "def func():\n"
        + "  dict()['b'] = 2\n"
        + "  return d\n");
  }

  @Test
  public void testListIndexAsLValue() {
    checkError("unsupported operand type(s) for +: 'list of ints' and 'dict of ints'",
        "def func():\n"
      + "  l = [1]\n"
      + "  l[0] = 2\n"
      + "  return l\n");
  }

  @Test
  public void testStringIndexAsLValue() {
    checkError("unsupported operand type(s) for +: 'string' and 'dict of ints'",
        "def func():\n"
      + "  s = 'abc'\n"
      + "  s[0] = 'd'\n"
      + "  return s\n");
  }

  @Test
  public void testEmptyLiteralGenericIsSetInLaterConcatWorks() {
    parse("def func():\n"
        + "  s = {}\n"
        + "  s['a'] = 'b'\n");
  }

  @Test
  public void testTypeIsInferredForStructs() {
    checkError("unsupported operand type(s) for +: 'struct' and 'string'",
        "(struct(a = 1) + struct(b = 1)) + 'x'");
  }

  @Test
  public void testReadOnlyWorksForSimpleBranching() {
    parse("if 1:\n"
        + "  v = 'a'\n"
        + "else:\n"
        + "  v = 'b'");
  }

  @Test
  public void testReadOnlyWorksForNestedBranching() {
    parse("if 1:\n"
        + "  if 0:\n"
        + "    v = 'a'\n"
        + "  else:\n"
        + "    v = 'b'\n"
        + "else:\n"
        + "  if 0:\n"
        + "    v = 'c'\n"
        + "  else:\n"
        + "    v = 'd'\n");
  }

  @Test
  public void testTypeCheckWorksForSimpleBranching() {
    checkError("bad variable 'v': int is incompatible with string at /some/file.txt:2:3",
          "if 1:\n"
        + "  v = 'a'\n"
        + "else:\n"
        + "  v = 1");
  }

  @Test
  public void testTypeCheckWorksForNestedBranching() {
    checkError("bad variable 'v': int is incompatible with string at /some/file.txt:5:5",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  if 0:\n"
      + "    v = 'b'\n"
      + "  else:\n"
      + "    v = 1\n");
  }

  @Test
  public void testTypeCheckWorksForDifferentLevelBranches() {
    checkError("bad variable 'v': int is incompatible with string at /some/file.txt:2:3",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  if 0:\n"
      + "    v = 1\n");
  }

  @Test
  public void testReadOnlyWorksForDifferentLevelBranches() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  if 1:\n"
      + "    v = 'a'\n"
      + "  v = 'b'\n");
  }

  @Test
  public void testReadOnlyWorksWithinSimpleBranch() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  v = 'b'\n"
      + "  v = 'c'\n");
  }

  @Test
  public void testReadOnlyWorksWithinNestedBranch() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  if 1:\n"
      + "    v = 'b'\n"
      + "  else:\n"
      + "    v = 'c'\n"
      + "    v = 'd'\n");
  }

  @Test
  public void testReadOnlyWorksAfterSimpleBranch() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  w = 'a'\n"
      + "v = 'b'");
  }

  @Test
  public void testReadOnlyWorksAfterNestedBranch() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  if 1:\n"
      + "    v = 'a'\n"
      + "v = 'b'");
  }

  @Test
  public void testReadOnlyWorksAfterNestedBranch2() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  if 0:\n"
      + "    w = 1\n"
      + "v = 'b'\n");
  }

  @Test
  public void testModulesReadOnlyInFuncDefBody() {
    checkError("Variable cmd_helper is read only",
        "def func():",
        "  cmd_helper = set()");
  }

  @Test
  public void testBuiltinGlobalFunctionsReadOnlyInFuncDefBody() {
    checkError("Variable rule is read only",
        "def func():",
        "  rule = 'abc'");
  }

  @Test
  public void testBuiltinGlobalFunctionsReadOnlyAsFuncDefArg() {
    checkError("Variable rule is read only",
        "def func(rule):",
        "  return rule");
  }

  @Test
  public void testFilesModulePlusStringErrorMessage() throws Exception {
    checkError("unsupported operand type(s) for +: 'cmd_helper (a language module)' and 'string'",
        "cmd_helper += 'a'");
  }

  @Test
  public void testFunctionReturnsFunction() {
    parse(
        "def impl(ctx):",
        "  return None",
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
  public void testLoadRelativePathOneSegment() throws Exception {
    parse("load('extension', 'a')\n");
  }

  @Test
  public void testLoadAbsolutePathMultipleSegments() throws Exception {
    parse("load('/pkg/extension', 'a')\n");
  }

  @Test
  public void testLoadRelativePathMultipleSegments() throws Exception {
    checkError("Path 'pkg/extension.bzl' is not valid. It should either start with "
        + "a slash or refer to a file in the current directory.",
        "load('pkg/extension', 'a')\n");
  }

  @Test
  public void testParentWithSkylarkModule() throws Exception {
    Class<?> emptyListClass = SkylarkList.EMPTY_LIST.getClass();
    Class<?> simpleListClass = SkylarkList.list(Arrays.<Integer>asList(1, 2, 3), SkylarkType.INT)
        .getClass();
    Class<?> tupleClass = SkylarkList.tuple(Arrays.<Object>asList(1, "a", "b")).getClass();

    assertThat(SkylarkList.class.isAnnotationPresent(SkylarkModule.class)).isTrue();
    assertThat(EvalUtils.getParentWithSkylarkModule(SkylarkList.class))
        .isEqualTo(SkylarkList.class);
    assertThat(EvalUtils.getParentWithSkylarkModule(emptyListClass)).isEqualTo(SkylarkList.class);
    assertThat(EvalUtils.getParentWithSkylarkModule(simpleListClass)).isEqualTo(SkylarkList.class);
    // TODO(bazel-team): make a tuple not a list anymore.
    assertThat(EvalUtils.getParentWithSkylarkModule(tupleClass)).isEqualTo(SkylarkList.class);

    // TODO(bazel-team): fix that?
    assertThat(ClassObject.class.isAnnotationPresent(SkylarkModule.class))
        .isFalse();
    assertThat(ClassObject.SkylarkClassObject.class.isAnnotationPresent(SkylarkModule.class))
        .isTrue();
    assertThat(EvalUtils.getParentWithSkylarkModule(ClassObject.SkylarkClassObject.class)
        == ClassObject.SkylarkClassObject.class).isTrue();
    assertThat(EvalUtils.getParentWithSkylarkModule(ClassObject.class))
        .isNull();
  }

  @Test
  public void testSkylarkTypeEquivalence() throws Exception {
    // All subclasses of SkylarkList are made equivalent
    Class<?> emptyListClass = SkylarkList.EMPTY_LIST.getClass();
    Class<?> simpleListClass = SkylarkList.list(Arrays.<Integer>asList(1, 2, 3), SkylarkType.INT)
        .getClass();
    Class<?> tupleClass = SkylarkList.tuple(Arrays.<Object>asList(1, "a", "b")).getClass();

    assertThat(SkylarkType.of(SkylarkList.class)).isEqualTo(SkylarkType.LIST);
    assertThat(SkylarkType.of(emptyListClass)).isEqualTo(SkylarkType.LIST);
    assertThat(SkylarkType.of(simpleListClass)).isEqualTo(SkylarkType.LIST);
    // TODO(bazel-team): make a tuple not a list anymore.
    assertThat(SkylarkType.of(tupleClass)).isEqualTo(SkylarkType.LIST);


    // Also for ClassObject
    assertThat(SkylarkType.of(ClassObject.SkylarkClassObject.class))
        .isEqualTo(SkylarkType.STRUCT);
    // TODO(bazel-team): fix that?
    assertThat(SkylarkType.of(ClassObject.class))
        .isNotEqualTo(SkylarkType.STRUCT);

    // Also test for these bazel classes, to avoid some regression.
    // TODO(bazel-team): move to some other place to remove dependency of syntax tests on Artifact?
    assertThat(SkylarkType.of(Artifact.SpecialArtifact.class))
        .isEqualTo(SkylarkType.of(Artifact.class));
    assertThat(SkylarkType.of(RuleConfiguredTarget.class))
        .isNotEqualTo(SkylarkType.STRUCT);
  }

  @Test
  public void testSkylarkTypeInclusion() throws Exception {
    assertThat(SkylarkType.INT.includes(SkylarkType.BOTTOM)).isTrue();
    assertThat(SkylarkType.BOTTOM.includes(SkylarkType.INT)).isFalse();
    assertThat(SkylarkType.TOP.includes(SkylarkType.INT)).isTrue();

    SkylarkType combo1 = SkylarkType.Combination.of(SkylarkType.LIST, SkylarkType.INT);
    assertThat(SkylarkType.LIST.includes(combo1)).isTrue();

    SkylarkType union1 = SkylarkType.Union.of(
        SkylarkType.MAP, SkylarkType.LIST, SkylarkType.STRUCT);
    assertThat(union1.includes(SkylarkType.MAP)).isTrue();
    assertThat(union1.includes(SkylarkType.STRUCT)).isTrue();
    assertThat(union1.includes(combo1)).isTrue();
    assertThat(union1.includes(SkylarkType.STRING)).isFalse();

    SkylarkType union2 = SkylarkType.Union.of(
        SkylarkType.LIST, SkylarkType.MAP, SkylarkType.STRING, SkylarkType.INT);
    SkylarkType inter1 = SkylarkType.intersection(union1, union2);
    assertThat(inter1.includes(SkylarkType.MAP)).isTrue();
    assertThat(inter1.includes(SkylarkType.LIST)).isTrue();
    assertThat(inter1.includes(combo1)).isTrue();
    assertThat(inter1.includes(SkylarkType.INT)).isFalse();
  }

  private void parse(String... lines) {
    parseFileForSkylark(Joiner.on("\n").join(lines));
    syntaxEvents.assertNoEvents();
  }

  private void checkError(String errorMsg, String... lines) {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark(Joiner.on("\n").join(lines));
    syntaxEvents.assertContainsEvent(errorMsg);
  }
}
