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

import com.google.common.base.Joiner;

/**
 * Tests for the validation process of Skylark files.
 */
public class ValidationTests extends AbstractParserTestCase {

  public void testIncompatibleLiteralTypesStringInt() {
    checkError("bad variable 'a': int is incompatible with string at /some/file.txt",
        "def foo():\n",
        "  a = '1'",
        "  a = 1");
  }

  public void testIncompatibleLiteralTypesDictString() {
    checkError("bad variable 'a': int is incompatible with dict of ints at /some/file.txt:3:3",
        "def foo():\n",
        "  a = {1 : 'x'}",
        "  a = 1");
  }

  public void testIncompatibleLiteralTypesInIf() {
    checkError("bad variable 'a': int is incompatible with string at /some/file.txt",
        "def foo():\n",
        "  if 1:",
        "    a = 'a'",
        "  else:",
        "    a = 1");
  }

  public void testAssignmentNotValidLValue() {
    checkError("can only assign to variables, not to ''a''", "'a' = 1");
  }

  public void testForNotIterable() throws Exception {
    checkError("type 'int' is not iterable",
          "def func():\n"
        + "  for i in 5: a = i\n");
  }

  public void testForIterableWithUknownArgument() throws Exception {
    parse("def func(x=None):\n"
        + "  for i in x: a = i\n");
  }

  public void testForNotIterableBinaryExpression() throws Exception {
    checkError("type 'int' is not iterable",
          "def func():\n"
        + "  for i in 1 + 1: a = i\n");
  }

  public void testOptionalArgument() throws Exception {
    checkError("type 'int' is not iterable",
          "def func(x=5):\n"
        + "  for i in x: a = i\n");
  }

  public void testOptionalArgumentHasError() throws Exception {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
          "def func(x=5+'a'):\n"
        + "  return 0\n");
  }

  public void testTopLevelForStatement() throws Exception {
    checkError("'For' is not allowed as a top level statement", "for i in [1,2,3]: a = i\n");
  }

  public void testReturnOutsideFunction() throws Exception {
    checkError("Return statements must be inside a function", "return 2\n");
  }

  public void testTwoReturnTypes() throws Exception {
    checkError("bad return type of foo: string is incompatible with int at /some/file.txt:3:5",
        "def foo(x):",
        "  if x:",
        "    return 1",
        "  else:",
        "    return 'a'");
  }

  public void testTwoFunctionsWithTheSameName() throws Exception {
    checkError("function foo already exists",
        "def foo():",
        "  return 1",
        "def foo(x, y):",
        "  return 1");
  }

  public void testDynamicTypeCheck() throws Exception {
    checkError("bad variable 'a': string is incompatible with int at /some/file.txt:2:3",
        "def foo():",
        "  a = 1",
        "  a = '1'");
  }

  public void testFunctionLocalVariable() throws Exception {
    checkError("name 'a' is not defined",
        "def func2(b):",
        "  c = b",
        "  c = a",
        "def func1():",
        "  a = 1",
        "  func2(2)");
  }

  public void testFunctionLocalVariableDoesNotEffectGlobalValidationEnv() throws Exception {
    checkError("name 'a' is not defined",
        "def func1():",
        "  a = 1",
        "def func2(b):",
        "  b = a");
  }

  public void testFunctionParameterDoesNotEffectGlobalValidationEnv() throws Exception {
    checkError("name 'a' is not defined",
        "def func1(a):",
        "  return a",
        "def func2():",
        "  b = a");
  }

  public void testLocalValidationEnvironmentsAreSeparated() throws Exception {
    parse(
          "def func1():\n"
        + "  a = 1\n"
        + "def func2():\n"
        + "  a = 'abc'\n");
  }

  public void testListComprehensionNotIterable() throws Exception {
    checkError("type 'int' is not iterable",
        "[i for i in 1 for j in [2]]");
  }

  public void testListComprehensionNotIterable2() throws Exception {
    checkError("type 'int' is not iterable",
        "[i for i in [1] for j in 123]");
  }

  public void testListIsNotComparable() {
    checkError("list of strings is not comparable", "['a'] > 1");
  }

  public void testStringCompareToInt() {
    checkError("bad comparison: int is incompatible with string", "'a' > 1");
  }

  public void testInOnInt() {
    checkError("operand 'in' only works on strings, dictionaries, "
        + "lists, sets or tuples, not on a(n) int", "1 in 2");
  }

  public void testUnsupportedOperator() {
    checkError("unsupported operand type(s) for -: 'string' and 'int'", "'a' - 1");
  }

  public void testBuiltinSymbolsAreReadOnly() throws Exception {
    checkError("Variable rule is read only", "rule = 1");
  }

  public void testSkylarkGlobalVariablesAreReadonly() throws Exception {
    checkError("Variable a is read only",
        "a = 1\n"
        + "a = 2");
  }

  public void testFunctionDefRecursion() throws Exception {
    checkError("function 'func' does not exist",
        "def func():\n"
      + "  func()\n");
  }

  public void testMutualRecursion() throws Exception {
    checkError("function 'bar' does not exist",
        "def foo(i):\n"
      + "  bar(i)\n"
      + "def bar(i):\n"
      + "  foo(i)\n"
      + "foo(4)");
  }

  public void testFunctionReturnValue() {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
          "def foo(): return 1\n"
        + "a = foo() + 'a'\n");
  }

  public void testFunctionReturnValueInFunctionDef() {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
          "def foo(): return 1\n"
        + "def bar(): a = foo() + 'a'\n");
  }

  public void testFunctionDoesNotExistInFunctionDef() {
    checkError("function 'foo' does not exist",
          "def bar(): a = foo() + 'a'\n"
        + "def foo(): return 1\n");
  }

  public void testStructMembersAreImmutable() {
    checkError("can only assign to variables, not to 's.x'",
        "s = struct(x = 'a')\n"
      + "s.x = 'b'\n");
  }

  public void testStructDictMembersAreImmutable() {
    checkError("can only assign to variables, not to 's.x['b']'",
        "s = struct(x = {'a' : 1})\n"
      + "s.x['b'] = 2\n");
  }

  public void testTupleAssign() throws Exception {
    // TODO(bazel-team): fix our code so 'tuple' not 'list' gets printed.
    checkError("unsupported operand type(s) for +: 'list' and 'dict of ints'",
        "d = (1, 2)\n"
      + "d[0] = 2\n");
  }

  public void testAssignOnNonCollection() throws Exception {
    checkError("unsupported operand type(s) for +: 'string' and 'dict of ints'",
        "d = 'abc'\n"
      + "d[0] = 2");
  }

  public void testNsetBadRightOperand() throws Exception {
    checkError("can only concatenate nested sets with other nested sets or list of items, "
        + "not 'string'", "set() + 'a'");
  }

  public void testNsetBadItemType() throws Exception {
    checkError("bad nested set: set of ints is incompatible with set of strings "
        + "at /some/file.txt:1:1",
        "(set() + ['a']) + [1]");
  }

  public void testNsetBadNestedItemType() throws Exception {
    checkError("bad nested set: set of ints is incompatible with set of strings "
        + "at /some/file.txt:1:1",
        "(set() + ['b']) + (set() + [1])");
  }

  public void testTypeInferenceForMethodLibraryFunction() throws Exception {
    checkError("bad variable 'l': string is incompatible with int at /some/file.txt:2:3",
          "def foo():\n"
        + "  l = len('abc')\n"
        + "  l = 'a'");
  }

  public void testListLiteralBadTypes() throws Exception {
    checkError("bad list literal: int is incompatible with string at /some/file.txt:1:1",
        "['a', 1]");
  }

  public void testTupleLiteralWorksForDifferentTypes() throws Exception {
    parse("('a', 1)");
  }

  public void testDictLiteralBadKeyTypes() throws Exception {
    checkError("bad dict literal: int is incompatible with string at /some/file.txt:1:1",
        "{'a': 1, 1: 2}");
  }

  public void testDictLiteralDifferentValueTypeWorks() throws Exception {
    parse("{'a': 1, 'b': 'c'}");
  }

  public void testListConcatBadTypes() throws Exception {
    checkError("bad list concatenation: list of ints is incompatible with list of strings"
        + " at /some/file.txt:1:1",
        "['a'] + [1]");
  }

  public void testDictConcatBadKeyTypes() throws Exception {
    checkError("bad dict concatenation: dict of ints is incompatible with dict of strings "
        + "at /some/file.txt:1:1",
        "{'a': 1} + {1: 2}");
  }

  public void testDictLiteralBadKeyType() throws Exception {
    checkError("Dict cannot contain composite type 'list of strings' as key", "{['a']: 1}");
  }

  public void testAndTypeInfer() throws Exception {
    checkError("unsupported operand type(s) for +: 'string' and 'int'", "('a' and 'b') + 1");
  }

  public void testOrTypeInfer() throws Exception {
    checkError("unsupported operand type(s) for +: 'string' and 'int'", "('' or 'b') + 1");
  }

  public void testAndDifferentTypes() throws Exception {
    checkError("bad and operator: int is incompatible with string at /some/file.txt:1:1",
        "'ab' and 3");
  }

  public void testOrDifferentTypes() throws Exception {
    checkError("bad or operator: int is incompatible with string at /some/file.txt:1:1",
        "'ab' or 3");
  }

  public void testOrNone() throws Exception {
    parse("a = None or 3");
  }

  public void testNoneAssignment() throws Exception {
    parse("def func():\n"
        + "  a = None\n"
        + "  a = 2\n"
        + "  a = None\n");
  }

  public void testNoneAssignmentError() throws Exception {
    checkError("bad variable 'a': string is incompatible with int at /some/file.txt",
          "def func():\n"
        + "  a = None\n"
        + "  a = 2\n"
        + "  a = None\n"
        + "  a = 'b'\n");
  }

  public void testDictComprehensionNotOnList() throws Exception {
    checkError("Dict comprehension elements must be a list", "{k : k for k in 'abc'}");
  }

  public void testTypeInferenceForUserDefinedFunction() throws Exception {
    checkError("bad variable 'a': string is incompatible with int at /some/file.txt",
          "def func():\n"
        + "  return 'a'\n"
        + "def foo():\n"
        + "  a = 1\n"
        + "  a = func()\n");
  }

  public void testCallingNonFunction() {
    checkError("a is not a function",
        "a = '1':\n"
      + "a()\n");
  }

  public void testFuncallArgument() {
    checkError("unsupported operand type(s) for +: 'int' and 'string'",
        "def foo(x): return x\n"
      + "a = foo(1 + 'a')");
  }

  // Skylark built-in functions specific tests

  public void testTypeInferenceForSkylarkBuiltinGlobalFunction() throws Exception {
    checkError("bad variable 'a': string is incompatible with function at /some/file.txt:3:3",
          "def impl(ctx): return None\n"
        + "def foo():\n"
        + "  a = rule(impl)\n"
        + "  a = 'a'\n");
  }

  public void testTypeInferenceForSkylarkBuiltinObjectFunction() throws Exception {
    checkError("bad variable 'a': string is incompatible with Attribute at /some/file.txt",
        "def foo():\n"
        + "  a = attr.int()\n"
        + "  a = 'a'\n");
  }

  public void testFuncReturningDictAssignmentAsLValue() throws Exception {
    checkError("can only assign to variables, not to 'dict([])['b']'",
          "def dict():\n"
        + "  return {'a': 1}\n"
        + "def func():\n"
        + "  dict()['b'] = 2\n"
        + "  return d\n");
  }

  public void testListIndexAsLValue() {
    checkError("unsupported operand type(s) for +: 'list of ints' and 'dict of ints'",
        "def func():\n"
      + "  l = [1]\n"
      + "  l[0] = 2\n"
      + "  return l\n");
  }

  public void testStringIndexAsLValue() {
    checkError("unsupported operand type(s) for +: 'string' and 'dict of ints'",
        "def func():\n"
      + "  s = 'abc'\n"
      + "  s[0] = 'd'\n"
      + "  return s\n");
  }

  public void testEmptyLiteralGenericIsSetInLaterConcatWorks() {
    parse("def func():\n"
        + "  s = {}\n"
        + "  s['a'] = 'b'\n");
  }

  public void testTypeIsInferredForStructs() {
    checkError("unsupported operand type(s) for +: 'struct' and 'string'",
        "(struct(a = 1) + struct(b = 1)) + 'x'");
  }

  public void testReadOnlyWorksForSimpleBranching() {
    parse("if 1:\n"
        + "  v = 'a'\n"
        + "else:\n"
        + "  v = 'b'");
  }

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

  public void testTypeCheckWorksForSimpleBranching() {
    checkError("bad variable 'v': int is incompatible with string at /some/file.txt:2:3",
          "if 1:\n"
        + "  v = 'a'\n"
        + "else:\n"
        + "  v = 1");
  }

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

  public void testTypeCheckWorksForDifferentLevelBranches() {
    checkError("bad variable 'v': int is incompatible with string at /some/file.txt:2:3",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  if 0:\n"
      + "    v = 1\n");
  }

  public void testReadOnlyWorksForDifferentLevelBranches() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  if 1:\n"
      + "    v = 'a'\n"
      + "  v = 'b'\n");
  }

  public void testReadOnlyWorksWithinSimpleBranch() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  v = 'b'\n"
      + "  v = 'c'\n");
  }

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

  public void testReadOnlyWorksAfterSimpleBranch() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  w = 'a'\n"
      + "v = 'b'");
  }

  public void testReadOnlyWorksAfterNestedBranch() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  if 1:\n"
      + "    v = 'a'\n"
      + "v = 'b'");
  }

  public void testReadOnlyWorksAfterNestedBranch2() {
    checkError("Variable v is read only",
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  if 0:\n"
      + "    w = 1\n"
      + "v = 'b'\n");
  }

  public void testModulesReadOnlyInFuncDefBody() {
    checkError("Variable cmd_helper is read only",
        "def func():",
        "  cmd_helper = set()");
  }

  public void testBuiltinGlobalFunctionsReadOnlyInFuncDefBody() {
    checkError("Variable rule is read only",
        "def func():",
        "  rule = 'abc'");
  }

  public void testBuiltinGlobalFunctionsReadOnlyAsFuncDefArg() {
    checkError("Variable rule is read only",
        "def func(rule):",
        "  return rule");
  }

  public void testFilesModulePlusStringErrorMessage() throws Exception {
    checkError("unsupported operand type(s) for +: 'cmd_helper (a language module)' and 'string'",
        "cmd_helper += 'a'");
  }

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

  public void testTypeForBooleanLiterals() {
    parse("len([1, 2]) == 0 and True");
    parse("len([1, 2]) == 0 and False");
  }

  public void testLoadRelativePathOneSegment() throws Exception {
    parse("load('extension', 'a')\n");
  }

  public void testLoadAbsolutePathMultipleSegments() throws Exception {
    parse("load('/pkg/extension', 'a')\n");
  }

  public void testLoadRelativePathMultipleSegments() throws Exception {
    checkError("Path 'pkg/extension.bzl' is not valid. It should either start with "
        + "a slash or refer to a file in the current directory.",
        "load('pkg/extension', 'a')\n");
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
