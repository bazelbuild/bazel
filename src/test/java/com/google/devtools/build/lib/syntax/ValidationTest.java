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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEvent;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions; // TODO(adonovan): break!
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of the Starlark validator. */
@RunWith(JUnit4.class)
public class ValidationTest {

  private StarlarkSemantics semantics = StarlarkSemantics.DEFAULT_SEMANTICS;

  private void setSemantics(String... options) throws OptionsParsingException {
    this.semantics =
        Options.parse(StarlarkSemanticsOptions.class, options).getOptions().toSkylarkSemantics();
  }

  // Validates a file using the current semantics.
  private StarlarkFile validateFile(String... lines) throws SyntaxError {
    ParserInput input = ParserInput.fromLines(lines);
    Module module = Module.createForBuiltins(Starlark.UNIVERSE);
    return EvalUtils.parseAndValidate(input, module, semantics);
  }

  // Assertions that parsing and validation succeeds.
  private void assertValid(String... lines) throws SyntaxError {
    StarlarkFile file = validateFile(lines);
    if (!file.ok()) {
      throw new SyntaxError(file.errors());
    }
  }

  // Asserts that parsing of the program succeeds but validation fails
  // with at least the specified error.
  private void assertInvalid(String expectedError, String... lines) throws SyntaxError {
    EventCollector errors = getValidationErrors(lines);
    assertContainsEvent(errors, expectedError);
  }

  // Returns the non-empty list of validation errors of the program.
  private EventCollector getValidationErrors(String... lines) throws SyntaxError {
    StarlarkFile file = validateFile(lines);
    if (file.ok()) {
      throw new AssertionError("validation succeeded unexpectedly");
    }
    EventCollector errors = new EventCollector();
    Event.replayEventsOn(errors, file.errors());
    return errors;
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
    assertInvalid(
        "load() statements must be called before any other statement", //
        "a = 5",
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
  public void testFunctionLocalVariableDoesNotEffectGlobalValidationEnv() throws Exception {
    assertInvalid(
        "name 'a' is not defined", //
        "def func1():",
        "  a = 1",
        "def func2(b):",
        "  b = a");
  }

  @Test
  public void testFunctionParameterDoesNotEffectGlobalValidationEnv() throws Exception {
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
  public void testLocalValidationEnvironmentsAreSeparated() throws Exception {
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
    EventCollector errors = getValidationErrors("a = 1", "a = 2");
    assertContainsEvent(errors, ":2:1: cannot reassign global 'a'");
    assertContainsEvent(errors, ":1:1: 'a' previously declared here");
  }

  @Test
  public void testTwoFunctionsWithTheSameName() throws Exception {
    EventCollector errors = getValidationErrors("def foo(): pass", "def foo(): pass");
    assertContainsEvent(errors, ":2:5: cannot reassign global 'foo'");
    assertContainsEvent(errors, ":1:5: 'foo' previously declared here");
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
  public void testModulesReadOnlyInFuncDefBody() throws Exception {
    assertValid("def func():", "  cmd_helper = depset()");
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
  public void testDollarErrorDoesNotLeak() throws Exception {
    EventCollector errors =
        getValidationErrors(
            "def GenerateMapNames():", //
            "  a = 2",
            "  b = [3, 4]",
            "  if a not b:",
            "    print(a)");
    assertContainsEvent(errors, "syntax error at 'b': expected 'in'");
    // Parser uses "$error" symbol for error recovery.
    // It should not be used in error messages.
    for (Event event : errors) {
      assertThat(event.getMessage()).doesNotContain("$error$");
    }
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
}
