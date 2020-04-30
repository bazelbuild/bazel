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

import static com.google.devtools.build.lib.syntax.LexerTest.assertContainsError;

import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of the Starlark resolver. */
@RunWith(JUnit4.class)
public class ResolverTest {

  private final FileOptions.Builder options = FileOptions.builder();

  // Resolves a file using the current options.
  private StarlarkFile resolveFile(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    Module module = Module.createForBuiltins(Starlark.UNIVERSE);
    return EvalUtils.parseAndValidate(input, options.build(), module);
  }

  // Assertions that parsing and resolution succeeds.
  private void assertValid(String... lines) throws SyntaxError.Exception {
    StarlarkFile file = resolveFile(lines);
    if (!file.ok()) {
      throw new SyntaxError.Exception(file.errors());
    }
  }

  // Asserts that parsing of the program succeeds but resolution fails
  // with at least the specified error.
  private void assertInvalid(String expectedError, String... lines) throws SyntaxError.Exception {
    List<SyntaxError> errors = getResolutionErrors(lines);
    assertContainsError(errors, expectedError);
  }

  // Returns the non-empty list of resolution errors of the program.
  private List<SyntaxError> getResolutionErrors(String... lines) throws SyntaxError.Exception {
    StarlarkFile file = resolveFile(lines);
    if (file.ok()) {
      throw new AssertionError("resolution succeeded unexpectedly");
    }
    return file.errors();
  }

  @Test
  public void testAssignmentNotValidLValue() throws Exception {
    assertInvalid("cannot assign to '\"a\"'", "'a' = 1");
  }

  @Test
  public void testAugmentedAssignmentWithMultipleLValues() throws Exception {
    assertInvalid(
        "cannot perform augmented assignment on a list or tuple expression", //
        "a, b += 2, 3");
  }

  @Test
  public void testReturnOutsideFunction() throws Exception {
    assertInvalid(
        "return statements must be inside a function", //
        "return 2\n");
  }

  @Test
  public void testLoadAfterStatement() throws Exception {
    options.requireLoadStatementsFirst(true);
    List<SyntaxError> errors = getResolutionErrors("a = 5", "load(':b.bzl', 'c')");
    assertContainsError(errors, ":2:1: load statements must appear before any other statement");
    assertContainsError(errors, ":1:1: \tfirst non-load statement appears here");
  }

  @Test
  public void testAllowLoadAfterStatement() throws Exception {
    options.requireLoadStatementsFirst(false);
    assertValid(
        "a = 5", //
        "load(':b.bzl', 'c')");
  }

  @Test
  public void testLoadDuplicateSymbols() throws Exception {
    assertInvalid(
        "load statement defines 'x' more than once", //
        "load('module', 'x', 'x')");
    assertInvalid(
        "load statement defines 'x' more than once", //
        "load('module', 'x', x='y')");

    // Eventually load bindings will be local,
    // at which point these errors will need adjusting.
    assertInvalid(
        "cannot reassign global 'x'", //
        "x=1; load('module', 'x')");
    assertInvalid(
        "cannot reassign global 'x'", //
        "load('module', 'x'); x=1");
  }

  @Test
  public void testForbiddenToplevelIfStatement() throws Exception {
    assertInvalid(
        "if statements are not allowed at the top level", //
        "if True: a = 2");
  }

  @Test
  public void testFunctionLocalVariable() throws Exception {
    assertInvalid(
        "name 'a' is not defined", //
        "def func2(b):",
        "  c = b",
        "  c = a",
        "def func1():",
        "  a = 1",
        "  func2(2)");
  }

  @Test
  public void testFunctionLocalVariableDoesNotEffectGlobalEnv() throws Exception {
    assertInvalid(
        "name 'a' is not defined", //
        "def func1():",
        "  a = 1",
        "def func2(b):",
        "  b = a");
  }

  @Test
  public void testFunctionParameterDoesNotEffectGlobalEnv() throws Exception {
    assertInvalid(
        "name 'a' is not defined", //
        "def func1(a):",
        "  return a",
        "def func2():",
        "  b = a");
  }

  @Test
  public void testDefinitionByItself() throws Exception {
    // Variables are assumed to be statically visible in the block (even if they might not be
    // initialized).
    assertValid("a = a");
    assertValid("a += a");
    assertValid("[[] for a in a]");
    assertValid("def f():", "  for a in a: pass");
  }

  @Test
  public void testLocalEnvironmentsAreSeparate() throws Exception {
    assertValid(
        "def func1():", //
        "  a = 1",
        "def func2():",
        "  a = 'abc'");
  }

  @Test
  public void testBuiltinsCanBeShadowed() throws Exception {
    assertValid("repr = 1");
  }

  @Test
  public void testNoGlobalReassign() throws Exception {
    List<SyntaxError> errors = getResolutionErrors("a = 1", "a = 2");
    assertContainsError(errors, ":2:1: cannot reassign global 'a'");
    assertContainsError(errors, ":1:1: 'a' previously declared here");
  }

  @Test
  public void testTwoFunctionsWithTheSameName() throws Exception {
    List<SyntaxError> errors = getResolutionErrors("def foo(): pass", "def foo(): pass");
    assertContainsError(errors, ":2:5: cannot reassign global 'foo'");
    assertContainsError(errors, ":1:5: 'foo' previously declared here");
  }

  @Test
  public void testFunctionDefRecursion() throws Exception {
    assertValid("def func():", "  func()\n");
  }

  @Test
  public void testMutualRecursion() throws Exception {
    assertValid("def foo(i):", "  bar(i)", "def bar(i):", "  foo(i)", "foo(4)");
  }

  @Test
  public void testFunctionDefinedBelow() throws Exception {
    assertValid("def bar(): a = foo() + 'a'", "def foo(): return 1\n");
  }

  @Test
  public void testGlobalDefinedBelow() throws Exception {
    assertValid("def bar(): return x", "x = 5\n");
  }

  @Test
  public void testLocalVariableDefinedBelow() throws Exception {
    assertValid(
        "def bar():",
        "    for i in range(5):",
        "        if i > 2: return x",
        "        x = i" // x is visible in the entire function block
        );
  }

  @Test
  public void testFunctionDoesNotExist() throws Exception {
    assertInvalid(
        "name 'foo' is not defined", //
        "def bar(): a = foo() + 'a'");
  }

  @Test
  public void testTupleLiteralWorksForDifferentTypes() throws Exception {
    assertValid("('a', 1)");
  }

  @Test
  public void testDictExpressionDifferentValueTypeWorks() throws Exception {
    assertValid("{'a': 1, 'b': 'c'}");
  }

  @Test
  public void testNoneAssignment() throws Exception {
    assertValid("def func():", "  a = None", "  a = 2", "  a = None\n");
  }

  @Test
  public void testNoneIsAnyType() throws Exception {
    assertValid("None + None");
    assertValid("2 == None");
    assertValid("None > 'a'");
    assertValid("[] in None");
    assertValid("5 * None");
  }

  // Starlark built-in functions specific tests

  @Test
  public void testFuncReturningDictAssignmentAsLValue() throws Exception {
    assertValid(
        "def my_dict():", //
        "  return {'a': 1}",
        "def func():",
        "  my_dict()['b'] = 2");
  }

  @Test
  public void testEmptyLiteralGenericIsSetInLaterConcatWorks() throws Exception {
    assertValid(
        "def func():", //
        "  s = {}",
        "  s['a'] = 'b'");
  }

  @Test
  public void testBuiltinGlobalFunctionsReadOnlyInFuncDefBody() throws Exception {
    assertValid("def func():", "  rule = 'abc'");
  }

  @Test
  public void testBuiltinGlobalFunctionsReadOnlyAsFuncDefArg() throws Exception {
    assertValid("def func(rule):", "  return rule");
  }

  @Test
  public void testFunctionReturnsFunction() throws Exception {
    assertValid(
        "def rule(*, implementation): return None", //
        "def impl(ctx): return None",
        "",
        "skylark_rule = rule(implementation = impl)",
        "",
        "def macro(name):",
        "  skylark_rule(name = name)");
  }

  @Test
  public void testTypeForBooleanLiterals() throws Exception {
    assertValid("len([1, 2]) == 0 and True");
    assertValid("len([1, 2]) == 0 and False");
  }

  @Test
  public void testPositionalAfterStarArg() throws Exception {
    assertInvalid(
        "positional argument is misplaced (positional arguments come first)", //
        "def fct(*args, **kwargs): pass",
        "fct(1, *[2], 3)");
  }

  @Test
  public void testTwoStarArgs() throws Exception {
    assertInvalid(
        "*arg argument is misplaced", //
        "def fct(*args, **kwargs):",
        "  pass",
        "fct(1, 2, 3, *[], *[])");
  }

  @Test
  public void testKeywordArgAfterStarArg() throws Exception {
    assertInvalid(
        "keyword argument is misplaced (keyword arguments must be before any *arg or **kwarg)", //
        "def fct(*args, **kwargs): pass",
        "fct(1, *[2], a=3)");
  }

  @Test
  public void testTopLevelForFails() throws Exception {
    assertInvalid(
        "for loops are not allowed at the top level", //
        "for i in []: 0\n");
  }

  @Test
  public void testNestedFunctionFails() throws Exception {
    assertInvalid(
        "nested functions are not allowed. Move the function to the top level", //
        "def func(a):",
        "  def bar(): return 0",
        "  return bar()",
        "");
  }

  @Test
  public void testDuplicateParameter() throws Exception {
    assertInvalid(
        "duplicate parameter: a",
        "def func(a, b, a):", //
        "  a = 1");
  }

  @Test
  public void testParameterOrdering() throws Exception {
    // ordering
    assertInvalid(
        "required parameter a may not follow **kwargs", //
        "def func(**kwargs, a): pass");
    assertInvalid(
        "required positional parameter b may not follow an optional parameter", //
        "def func(a=1, b): pass");
    assertInvalid(
        "optional parameter may not follow **kwargs", //
        "def func(**kwargs, a=1): pass");
    assertInvalid(
        "* parameter may not follow **kwargs", //
        "def func(**kwargs, *args): pass");
    assertInvalid(
        "* parameter may not follow **kwargs", //
        "def func(**kwargs, *): pass");
    assertInvalid(
        "bare * must be followed by keyword-only parameters", //
        "def func(*): pass");

    // duplicate parameters
    assertInvalid("duplicate parameter: a", "def func(a, a): pass");
    assertInvalid("duplicate parameter: a", "def func(a, a=1): pass");
    assertInvalid("duplicate parameter: a", "def func(a, *a): pass");
    assertInvalid("duplicate parameter: a", "def func(*a, a): pass");
    assertInvalid("duplicate parameter: a", "def func(*a, a=1): pass");
    assertInvalid("duplicate parameter: a", "def func(a, **a): pass");
    assertInvalid("duplicate parameter: a", "def func(*a, **a): pass");

    // multiple *
    assertInvalid("multiple * parameters not allowed", "def func(a, *, b, *): pass");
    assertInvalid("multiple * parameters not allowed", "def func(a, *args, b, *): pass");
    assertInvalid("multiple * parameters not allowed", "def func(a, *, b, *args): pass");
    assertInvalid("multiple * parameters not allowed", "def func(a, *args, b, *args): pass");

    // multiple **kwargs
    assertInvalid("multiple ** parameters not allowed", "def func(**kwargs, **kwargs): pass");

    assertValid("def f(a, b, c=1, d=2, *args, e, f=3, g, **kwargs): pass");
  }
}
