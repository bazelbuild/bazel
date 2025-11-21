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
package net.starlark.java.syntax;

import static com.google.common.truth.Truth.assertThat;
import static net.starlark.java.syntax.LexerTest.assertContainsError;

import com.google.common.base.Joiner;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of the Starlark resolver. */
@RunWith(JUnit4.class)
public class ResolverTest {

  private final FileOptions.Builder options = FileOptions.builder();

  // Resolves a file using the current options,
  // in an environment with a single predeclared name, pre.
  // Errors are recorded in file.errors().
  private StarlarkFile resolveFile(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, options.build());
    Resolver.resolveFile(file, Resolver.moduleWithPredeclared("pre"));
    return file;
  }

  // Assertions that parsing and resolution succeeds.
  private void assertValid(String... lines) throws SyntaxError.Exception {
    getValidFile(lines);
  }

  // Asserts that parsing of the program succeeds but resolution fails
  // with at least the specified error.
  private void assertInvalid(String expectedError, String... lines) throws SyntaxError.Exception {
    List<SyntaxError> errors = getResolutionErrors(lines);
    assertContainsError(errors, expectedError);
  }

  private StarlarkFile getValidFile(String... lines) throws SyntaxError.Exception {
    StarlarkFile file = resolveFile(lines);
    if (!file.ok()) {
      throw new SyntaxError.Exception(file.errors());
    }
    return file;
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
  public void testDuplicateBindingWithinALoadStatement() throws Exception {
    assertInvalid(
        "load statement defines 'x' more than once", //
        "load('module', 'x', 'x')");
    assertInvalid(
        "load statement defines 'x' more than once", //
        "load('module', 'x', x='y')");
  }

  @Test
  public void testConflictsAtToplevel_default() throws Exception {
    List<SyntaxError> errors = getResolutionErrors("x=1; x=2");
    assertContainsError(errors, ":1:6: 'x' redeclared at top level");
    assertContainsError(errors, ":1:1: 'x' previously declared here");

    errors = getResolutionErrors("x=1; load('module', 'x')");
    assertContainsError(errors, ":1:22: conflicting file-local declaration of 'x'");
    assertContainsError(errors, ":1:1: 'x' previously declared as global here");
    // Also: "loads must appear first"

    errors = getResolutionErrors("load('module', 'x'); x=1");
    assertContainsError(errors, ":1:22: conflicting global declaration of 'x'");
    assertContainsError(errors, ":1:17: 'x' previously declared as file-local here");

    errors = getResolutionErrors("load('module', 'x'); load('module', 'x')");
    assertContainsError(errors, ":1:38: 'x' redeclared at top level");
    assertContainsError(errors, ":1:17: 'x' previously declared here");
  }

  @Test
  public void testConflictsAtToplevel_loadBindsGlobally() throws Exception {
    options.loadBindsGlobally(true);

    List<SyntaxError> errors = getResolutionErrors("x=1; x=2");
    assertContainsError(errors, ":1:6: 'x' redeclared at top level");
    assertContainsError(errors, ":1:1: 'x' previously declared here");

    errors = getResolutionErrors("x=1; load('module', 'x')");
    assertContainsError(errors, ":1:22: 'x' redeclared at top level");
    assertContainsError(errors, ":1:1: 'x' previously declared here");
    // Also: "loads must appear first"

    errors = getResolutionErrors("load('module', 'x'); x=1");
    assertContainsError(errors, ":1:22: 'x' redeclared at top level");
    assertContainsError(errors, ":1:17: 'x' previously declared here");

    errors = getResolutionErrors("load('module', 'x'); load('module', 'x')");
    assertContainsError(errors, ":1:38: 'x' redeclared at top level");
    assertContainsError(errors, ":1:17: 'x' previously declared here");
  }

  @Test
  public void testConflictsAtToplevel_allowToplevelRebinding() throws Exception {
    // This flag allows rebinding of globals, or of file-locals,
    // but a given name cannot be both globally and file-locally bound.
    options.allowToplevelRebinding(true);

    assertValid("x=1; x=2");

    List<SyntaxError> errors = getResolutionErrors("x=1; load('module', 'x')");
    assertContainsError(errors, ":1:22: conflicting file-local declaration of 'x'");
    assertContainsError(errors, ":1:1: 'x' previously declared as global here");
    // Also: "loads must appear first"

    errors = getResolutionErrors("load('module', 'x'); x=1");
    assertContainsError(errors, ":1:22: conflicting global declaration of 'x'");
    assertContainsError(errors, ":1:17: 'x' previously declared as file-local here");

    assertValid("load('module', 'x'); load('module', 'x')");
  }

  @Test
  public void testConflictsAtToplevel_loadBindsGlobally_allowToplevelRebinding() throws Exception {
    options.loadBindsGlobally(true);
    options.allowToplevelRebinding(true);
    options.requireLoadStatementsFirst(false);

    assertValid("x=1; x=2");
    assertValid("x=1; load('module', 'x')");
    assertValid("load('module', 'x'); x=1");
    assertValid("load('module', 'x'); load('module', 'x')");
  }

  @Test
  public void testForbiddenToplevelIfStatement() throws Exception {
    assertInvalid(
        "if statements are not allowed at the top level", //
        "if pre: a = 2");
  }

  @Test
  public void testUndefinedName() throws Exception {
    assertInvalid("name 'foo' is not defined", "[foo for x in []]");
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
    assertValid("[[] for _ in [] for a in a]");
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
    assertValid("pre = 1");
  }

  @Test
  public void testGlobalShadowsPredeclaredForEntireFile() throws Exception {
    // global 'pre' shadows predeclared of same name.
    List<SyntaxError> errors = getResolutionErrors("pre; pre = 1; pre = 2");
    assertContainsError(errors, ":1:15: 'pre' redeclared at top level");
    assertContainsError(errors, ":1:6: 'pre' previously declared here");
  }

  @Test
  public void testTwoFunctionsWithTheSameName() throws Exception {
    // Def statements act just like an assignment statement.
    List<SyntaxError> errors = getResolutionErrors("def foo(): pass", "def foo(): pass");
    assertContainsError(errors, ":2:5: 'foo' redeclared at top level");
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
        "    for i in pre(5):",
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
  public void testTopLevelForFails() throws Exception {
    assertInvalid(
        "for loops are not allowed at the top level", //
        "for i in []: 0\n");
  }

  @Test
  public void testComprehension() throws Exception {
    // The operand of the first for clause is resolved outside the comprehension block.
    assertInvalid("name 'x' is not defined", "[() for x in x]");
    assertValid("[() for x in () for x in x]"); // forward ref
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

  @Test
  public void testArgumentOrdering() throws Exception {
    // positionals go before keywords
    assertInvalid(
        "positional argument may not follow keyword", //
        "pre(a=1, 0)");

    // keywords must be unique
    assertInvalid(
        "duplicate keyword argument: a", //
        "pre(a=1, a=2)");

    // no arguments after **kwargs
    assertInvalid(
        "positional argument may not follow **kwargs", //
        "pre(**0, 0)");
    assertInvalid(
        "keyword argument a may not follow **kwargs", //
        "pre(**0, a=1)");
    assertInvalid(
        "*args may not follow **kwargs", //
        "pre(**0, *0)");
    assertInvalid(
        "multiple **kwargs not allowed", //
        "pre(**0, **0)");
    assertInvalid(
        "*args may not follow **kwargs", // also, a parse error
        "pre(**0, *)");

    // bad arguments after *args
    assertInvalid(
        "positional argument may not follow *args", //
        "pre(*0, 1)");
    assertInvalid(
        "keyword argument a may not follow *args", //
        "pre(*0, a=1)"); // Python (even v2) allows this
    assertInvalid(
        "multiple *args not allowed", //
        "pre(*0, *0)");

    assertValid("pre(0, a=0, *0, **0)");
  }

  @Test
  public void testUndefError() throws Exception {
    // Regression test for a poor error message.
    List<SyntaxError> errors = getResolutionErrors("lambda: undef");
    assertThat(errors.get(0).message()).isEqualTo("name 'undef' is not defined");
  }

  // TODO: #27728 - Add resolver behavior for type expressions, add bindingScopeAndIndex tests here.

  @Test
  public void testBindingScopeAndIndex_basic() throws Exception {
    checkBindings(
        // Assign successive indices.
        "xᴳ₀ = 0",
        // Visit LHS.
        "yᴳ₁, zᴳ₂ = 1, 2",
        // Visit function identifiers and subscripts, don't visit field names, resolve predeclareds.
        "xᴳ₀(yᴳ₁.f  , preᴾ₀[zᴳ₂])");
  }

  @Test
  public void testBindingScopeAndIndex_bindingAfterFirstUse() throws Exception {
    checkBindings(
        // Use before definition. (Dynamically invalid, but resolves just fine.)
        "xᴳ₀",
        "xᴳ₀ = 0",
        // Same in local scope, but permit reassignment.
        "def fᴳ₁():",
        "  yᴸ₀",
        "  yᴸ₀ = 0",
        "  yᴸ₀ = 0",
        "  yᴸ₀");
  }

  @Test
  public void testBindingScopeAndIndex_functionBlock() throws Exception {
    checkBindings(
        "xᴳ₀ = 0",
        "yᴳ₁ = 1",
        // Default expr resolves outside function block, for all params.
        "def fᴳ₂(xᴸ₀ = xᴳ₀, zᴸ₁ = xᴳ₀):",
        // Param available within function block, and shadows global.
        "  xᴸ₀",
        "  zᴸ₁ = 1",
        // New bindings in body are local to function block.
        "  wᴸ₂ = 2",
        // Global is referenced directly without cell/free indirection.
        "  yᴳ₁",
        // Can resolve recursive reference to current function.
        "  fᴳ₂");
  }

  @Test
  public void testBindingScopeAndIndex_nestedFunctions() throws Exception {
    checkBindings(
        "aᴳ₀ = 0", // a used in nested function but not a cell because it's global
        "bᴳ₁ = 1", // b not used in nested function
        "def fᴳ₂():",
        "  cᶜ₀ = aᴳ₀", // c used in nested function, so made a cell; still increments index
        "  dᴸ₁ = 1", // d not used in nested function, remains local
        "  def gᴸ₂():",
        "    cᶠ₀", // use of enclosing local becomes free; does not increment index
        "    eᴸ₀ = 1");
  }

  @Test
  public void testBindingScopeAndIndex_comprehensions() throws Exception {
    checkBindings(
        "xᴳ₀ = 0",
        "yᴳ₁ = 0",
        // Comprehensions have their own block.
        // First for-clause resolved outside of this block.
        // Subsequent for-clauses resolved inside this block.
        "[xᴸ₀ for xᴸ₀ in xᴳ₀ for xᴸ₀ in xᴸ₀ if yᴳ₁]");
  }

  @Test
  public void testBindingScopeAndIndex_loads() throws Exception {
    // Load statements create file-local bindings.
    // Functions that reference load bindings are closures.
    checkBindings(
        """
        load('module', aᶜ₀='a', bᴸ₁='b')
        aᶜ₀, bᴸ₁
        def fᴳ₀():
          aᶠ₀
        """);
  }

  @Test
  public void testBindingScopeAndIndex_varStatement() throws Exception {
    options.allowTypeSyntax(true);
    checkBindings(
        // Var statement creates a binding, even in the absence of assignment.
        "xᴳ₀ : T",
        // Var statement can shadow predeclared.
        "preᴳ₁ : T",
        "def fᴳ₂():",
        "  xᴳ₀",
        "  preᴳ₁");
  }

  @Test
  public void testDocComments() throws Exception {
    options.allowTypeSyntax(true);
    StarlarkFile file =
        getValidFile(
            """
            #: Doc for FOO
            #: multiline
            FOO = 1

            BAR, BAZ = (2, 3)  #: Applies to LHS list

            #: Applies to var annotation without initialier
            QUX : T
            QUUX : T #: And the trailing version...
            """);

    assertThat(file.docCommentsMap.keySet())
        .containsExactly("FOO", "BAR", "BAZ", "QUX", "QUUX")
        .inOrder();
    assertThat(file.docCommentsMap.values().stream().map(DocComments::getText))
        .containsExactly(
            "Doc for FOO\nmultiline",
            "Applies to LHS list",
            "Applies to LHS list",
            "Applies to var annotation without initialier",
            "And the trailing version...")
        .inOrder();
  }

  @Test
  public void testTypeAliasStatement_mustBeAtTopLevel() throws Exception {
    options.allowTypeSyntax(true);
    assertInvalid(
        ":2:3: type alias statement not at top level",
        """
        def f():
          type X = int
        """);
  }

  @Test
  public void testMultipleTypeAnnotationsDisallowed_topLevel() throws Exception {
    options.allowTypeSyntax(true);
    List<SyntaxError> errors =
        getResolutionErrors(
            // All four permutations of VarStatement vs annotated assignment statement.
            """
            a : int
            a : str

            b : int = 123
            b : str

            c : int
            c : str = "abc"

            d : int = 123
            d : str = "abc"
            """);
    assertContainsError(errors, ":2:1: 'a' redeclared at top level");
    assertContainsError(errors, ":5:1: 'b' redeclared at top level");
    assertContainsError(errors, ":8:1: 'c' redeclared at top level");
    assertContainsError(errors, ":11:1: 'd' redeclared at top level");
  }

  @Test
  public void testMultipleTypeAnnotationsDisallowed_localLevel() throws Exception {
    // Same as testMultipleTypeAnnotationsDisallowed_topLevel but inside a function, where
    // reassignment is always allowed.
    options.allowTypeSyntax(true);
    List<SyntaxError> errors =
        getResolutionErrors(
            // All four permutations of VarStatement vs annotated assignment statement.
            """
            def f():
                a : int
                a : str

                b : int = 123
                b : str

                c : int
                c : str = "abc"

                d : int = 123
                d : str = "abc"
            """);
    assertContainsError(errors, "type annotation on 'a' may only appear at its first declaration");
    assertContainsError(errors, "type annotation on 'b' may only appear at its first declaration");
    assertContainsError(errors, "type annotation on 'c' may only appear at its first declaration");
    assertContainsError(errors, "type annotation on 'd' may only appear at its first declaration");
  }

  @Test
  public void testMultipleTypeAnnotationsDisallowed_defStatement() throws Exception {
    options.allowTypeSyntax(true);

    assertValid(
        """
        def f():
            # Redefinition is allowed (but bad style) if second definition has no type
            # annotation.
            def a(x : int):
                pass
            def a(x):
                pass
        """);

    List<SyntaxError> errors =
        getResolutionErrors(
            """
            def f():
                # Second definition may not have a type annotation, even if first definition has
                # none.
                def b(x):
                    pass
                def b(x : int):
                    pass

                # Return type annotation counts too.
                def c(x):
                    pass
                def c(x) -> int:
                    pass

                # Even generic type vars count.
                def d(x):
                    pass
                def d[T](x):
                    pass
            """);
    // TODO: #27371 - For the case of redefining a function, the error message is a little
    // confusing. But this is also a pretty rare case.
    assertContainsError(errors, "type annotation on 'b' may only appear at its first declaration");
    assertContainsError(errors, "type annotation on 'c' may only appear at its first declaration");
    assertContainsError(errors, "type annotation on 'd' may only appear at its first declaration");
  }

  @Test
  public void testSingleAnnotationWithReassignmentIsAllowed() throws Exception {
    options.allowTypeSyntax(true);
    assertValid(
        """
        def f():
            a : int
            a = 123
        """);
  }

  @Test
  public void testAnnotationFollowedByAssignmentStillCountsAsRedeclaration() throws Exception {
    options.allowTypeSyntax(true);
    assertInvalid(
        "'a' redeclared at top level",
        """
        a : int
        a = 123
        """);
  }

  @Test
  public void testVarStatementMustPreceedAssignment() throws Exception {
    options.allowTypeSyntax(true);
    assertInvalid(
        "type annotation on 'x' may only appear at its first declaration",
        """
        def f():
            x = 123
            x : int
        """);
  }

  @Test
  public void onlyFirstAssignmentMayBeAnnotated() throws Exception {
    options.allowTypeSyntax(true);
    assertInvalid(
        "type annotation on 'x' may only appear at its first declaration",
        """
        def f():
            x = 123
            x : int = 123
        """);
  }

  @Test
  public void cannotAnnotateParamInBody() throws Exception {
    options.allowTypeSyntax(true);
    assertInvalid(
        "type annotation on 'x' may only appear at its first declaration",
        """
        def f(x):
            # Invalid even though x has no type annotation above.
            x : int
        """);
  }

  @Test
  public void testCastExpression_cannotBeLhsOfAssignment() throws Exception {
    options.allowTypeSyntax(true);
    StarlarkFile file =
        resolveFile(
            """
            cast(int, x) = 42
            cast(int, y[0]) = 42
            cast(list[int], z) += [42]
            """);
    assertThat(file.ok()).isFalse();
    assertContainsError(file.errors(), "cannot assign to 'cast(int, x)'");
    assertContainsError(file.errors(), "cannot assign to 'cast(int, y[0])'");
    assertContainsError(file.errors(), "cannot assign to 'cast(list[int], z)'");
  }

  @Test
  public void testCastExpression_value_isResolved() throws Exception {
    options.allowTypeSyntax(true);
    StarlarkFile badFile = resolveFile("cast(int, f())");
    assertThat(badFile.ok()).isFalse();
    assertContainsError(badFile.errors(), "name 'f' is not defined");

    StarlarkFile goodFile =
        resolveFile(
            """
            def f():
              return 1
            cast(int, f())
            """);
    assertThat(goodFile.ok()).isTrue();
  }

  @Test
  public void testCastExpression_type_notResolved() throws Exception {
    // TODO(brandjon): resolve the cast's type once we have type checking.
    options.allowTypeSyntax(true);
    StarlarkFile badFile = resolveFile("cast(NoSuchType[int], 42)");
    assertThat(badFile.ok()).isTrue();
  }

  // TODO(b/350661266): resolve types in isinstance().
  @Test
  public void testIsInstanceExpression_notYetSupported() throws Exception {
    options.allowTypeSyntax(true);
    StarlarkFile badFile = resolveFile("isinstance(x, list)");
    assertThat(badFile.ok()).isFalse();
    assertContainsError(badFile.errors(), "isinstance() is not yet supported");
  }

  // checkBindings verifies the binding (scope and index) of each identifier.
  // Every variable must be followed by a superscript letter (its scope)
  // and a subscript numeral (its index). They are replaced by spaces, the
  // file is resolved, and then the computed information is written over
  // the spaces. The resulting string must match the input.
  private void checkBindings(String... lines) throws Exception {
    String src = Joiner.on("\n").join(lines);
    StarlarkFile file = resolveFile(src.replaceAll("[₀₁₂₃₄₅₆₇₈₉ᴸᴳᶜᶠᴾᵁ]", " "));
    if (!file.ok()) {
      throw new AssertionError("resolution failed: " + file.errors());
    }
    String[] out = new String[] {src};
    new NodeVisitor() {
      @Override
      public void visit(Identifier id) {
        // Replace ...x__... with ...xᴸ₀...
        Resolver.Binding binding = id.getBinding();
        String suffix = "";
        if (binding != null) {
          suffix += "ᴸᴳᶜᶠᴾᵁ".charAt(binding.getScope().ordinal()); // follow order of enum
          suffix += "₀₁₂₃₄₅₆₇₈₉".charAt(binding.getIndex()); // 10 is plenty
        } else {
          suffix = "  ";
        }
        out[0] =
            out[0].substring(0, id.getEndOffset())
                + suffix
                + out[0].substring(id.getEndOffset() + 2);
      }

      @Override
      public void visit(VarStatement varStatement) {
        visit(varStatement.getIdentifier());
        // Don't visit type expression, it isn't processed.
        // TODO: #27728 - Include the type expression in these tests.
      }
    }.visit(file);
    assertThat(out[0]).isEqualTo(src);
  }
}
