// Copyright 2025 The Bazel Authors. All Rights Reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static net.starlark.java.syntax.TestUtils.assertContainsError;
import static org.junit.Assert.assertThrows;

import java.util.ArrayList;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Resolver.Module;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark type resolution. */
@RunWith(JUnit4.class)
public class TypeResolverTest {

  private final FileOptions.Builder options =
      FileOptions.builder().allowTypeSyntax(true).resolveTypeSyntax(true);

  /** Evaluates an expression string to a type in an empty environment. */
  private StarlarkType evalType(String type) throws Exception {
    Module module = TestUtils.moduleWithUniversalTypes();
    Expression expr = Expression.parseTypeExpression(ParserInput.fromLines(type), options.build());
    Resolver.resolveExpr(expr, module, options.build());
    return TypeResolver.evalTypeExpression(expr, TestUtils.moduleWithUniversalTypes());
  }

  /**
   * Parses a series of strings as a file, then resolves and type-resolves it.
   *
   * <p>Asserts that parsing and symbol resolution succeeded, but type-resolving may fail.
   */
  private StarlarkFile annotateFilePossiblyFailing(String... lines) throws Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, options.build());
    assertThat(file.ok()).isTrue();
    Module module = TestUtils.moduleWithUniversalTypes();
    Resolver.resolveFile(file, module);
    assertThat(file.ok()).isTrue();
    TypeResolver.annotateFile(file, module);
    return file;
  }

  /** As in {@link #annotateFilePossiblyFailing} but asserts that even type resolution succeeded. */
  private StarlarkFile annotateFile(String... lines) throws Exception {
    StarlarkFile file = annotateFilePossiblyFailing(lines);
    assertThat(file.ok()).isTrue();
    return file;
  }

  /** Asserts that type resolution fails with at least the specified error. */
  private void assertInvalid(String expectedError, String... lines) throws Exception {
    StarlarkFile file = annotateFilePossiblyFailing(lines);
    assertWithMessage("type resolution suceeded unexpectedly").that(file.ok()).isFalse();
    assertContainsError(file.errors(), expectedError);
  }

  /** Returns the first statement of a parsed file. */
  private <T extends Statement> T getFirstStatement(Class<T> clazz, StarlarkFile file) {
    assertThat(file.getStatements()).isNotEmpty();
    Statement stmt = file.getStatements().get(0);
    assertThat(stmt).isInstanceOf(clazz);
    return clazz.cast(stmt);
  }

  /** Returns the first statement of a function body. */
  private <T extends Statement> T getFirstStatement(Class<T> clazz, DefStatement def) {
    assertThat(def.getBody()).isNotEmpty();
    Statement stmt = def.getBody().get(0);
    assertThat(stmt).isInstanceOf(clazz);
    return clazz.cast(stmt);
  }

  /** Returns the type of an identifier. */
  @Nullable
  private StarlarkType getType(Identifier id) throws Exception {
    assertThat(id.getBinding()).isNotNull();
    return id.getBinding().getType();
  }

  /** Returns the type of a {@code def}'s resolved function. */
  @Nullable
  private Types.CallableType getType(DefStatement def) throws Exception {
    assertThat(def.getResolvedFunction()).isNotNull();
    return def.getResolvedFunction().getFunctionType();
  }

  /** Returns the type of a {@code lambda}'s resolved function. */
  @Nullable
  private Types.CallableType getType(LambdaExpression lambda) throws Exception {
    assertThat(lambda.getResolvedFunction()).isNotNull();
    return lambda.getResolvedFunction().getFunctionType();
  }

  @Test
  public void evalType_primitives() throws Exception {
    assertThat(evalType("None")).isEqualTo(Types.NONE);
    assertThat(evalType("bool")).isEqualTo(Types.BOOL);
    assertThat(evalType("int")).isEqualTo(Types.INT);
    assertThat(evalType("float")).isEqualTo(Types.FLOAT);
    assertThat(evalType("str")).isEqualTo(Types.STR);
  }

  @Test
  public void evalType_union() throws Exception {
    assertThat(evalType("int|bool")).isEqualTo(Types.union(Types.INT, Types.BOOL));
  }

  // TODO: #27370 - Rather than test applications of constructors for list and dict here, test the
  // general machinery for calling a type constructor. The actual types should be tested separately.

  @Test
  public void evalType_list() throws Exception {
    assertThat(evalType("list[int]")).isEqualTo(Types.list(Types.INT));
    assertThat(evalType("list[list[int]]")).isEqualTo(Types.list(Types.list(Types.INT)));

    var exception = assertThrows(SyntaxError.Exception.class, () -> evalType("list[int, bool]"));
    assertThat(exception).hasMessageThat().isEqualTo("list[] accepts exactly 1 argument but got 2");

    exception = assertThrows(SyntaxError.Exception.class, () -> evalType("list[[int]]"));
    assertThat(exception).hasMessageThat().isEqualTo("unexpected expression '[int]'");

    // TODO: #27370 - `list` should produce `list[Any]`.
  }

  @Test
  public void evalType_dict() throws Exception {
    assertThat(evalType("dict[int, str]")).isEqualTo(Types.dict(Types.INT, Types.STR));
    assertThat(evalType("dict[int, list[str]]"))
        .isEqualTo(Types.dict(Types.INT, Types.list(Types.STR)));

    var exception = assertThrows(SyntaxError.Exception.class, () -> evalType("dict[int]"));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo("dict[] accepts exactly 2 arguments but got 1");

    exception = assertThrows(SyntaxError.Exception.class, () -> evalType("dict[int, str, bool]"));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo("dict[] accepts exactly 2 arguments but got 3");

    exception = assertThrows(SyntaxError.Exception.class, () -> evalType("dict"));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo("expected type arguments after the type constructor 'dict'");
    // TODO: #27370 - `dict` should produce `dict[Any, Any]`.
  }

  @Test
  public void evalType_unknownIdentifier() {
    SyntaxError.Exception e = assertThrows(SyntaxError.Exception.class, () -> evalType("Foo"));

    assertThat(e).hasMessageThat().isEqualTo("name 'Foo' is not defined");
  }

  @Test
  public void evalType_badTypeApplications() {
    SyntaxError.Exception e =
        assertThrows(SyntaxError.Exception.class, () -> evalType("int[bool]"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("'int' is not a type constructor, cannot be applied to '[bool]'");

    e = assertThrows(SyntaxError.Exception.class, () -> evalType("Foo[int]"));
    assertThat(e).hasMessageThat().isEqualTo("name 'Foo' is not defined");

    e = assertThrows(SyntaxError.Exception.class, () -> evalType("list"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected type arguments after the type constructor 'list'");
  }

  @Test
  public void localCannotShadowPredeclaredType() throws Exception {
    assertInvalid(
        "local name 'int' cannot be used as a type",
        """
        def f():
            int = 123
            x : int
        """);
  }

  @Test
  public void annotationMustBeAtFirstOccurrence_assignment() throws Exception {
    assertInvalid(
        "type annotation on 'x' may only appear at its declaration",
        """
        def f():
            x = 123
            x : int = 123
        """);
  }

  @Test
  public void annotationMustBeAtFirstOccurrence_varStatementAfterAssignment() throws Exception {
    assertInvalid(
        "type annotation on 'y' may only appear at its declaration",
        """
        def f():
            x, y, z = 123
            y : int
        """);
  }

  @Test
  public void annotationMustBeAtFirstOccurrence_parameters() throws Exception {
    assertInvalid(
        "type annotation on 'x' may only appear at its declaration",
        """
        def f(x):
            # Invalid even though x has no type annotation above.
            x : int
        """);
  }

  @Test
  public void annotationMustBeAtFirstOccurence_localVar() throws Exception {
    // Also avoid assertInvalid() in this test case so we have some coverage of the declaration
    // location reporting, which is spread over two events.
    StarlarkFile file =
        annotateFilePossiblyFailing(
            """
            def f():
                x : int
                x : str
            """);
    assertThat(file.ok()).isFalse();
    assertContainsError(
        file.errors(), "3:5: type annotation on 'x' may only appear at its declaration");
    assertContainsError(file.errors(), "2:5: 'x' previously declared here");
  }

  @Test
  public void annotationMustBeAtFirstOccurrence_innerFunction() throws Exception {
    // Every function definition implicitly annotates its identifier as at least a Callable, even
    // if the definition has no parameter or return type annotations.
    assertInvalid(
        "function 'g' was previously declared",
        """
        def f():
            def g():
                pass
            def g():
                pass
        """);

    assertInvalid(
        "function 'g' was previously declared",
        """
        def f():
            g = 1
            def g():
                pass
        """);
  }

  @Test
  public void annotationMustBeAtFirstOccurrence_loadedGlobal() throws Exception {
    // These options are needed to exercise attempting to annotate a loaded symbol. Otherwise we
    // would be annotating a distinct global symbol whose name happens to clash with the loaded one.
    // That's also an error, but not the one we want to test.
    options.loadBindsGlobally(true).allowToplevelRebinding(true);

    assertInvalid(
        "type annotation on 'x' may only appear at its declaration",
        """
        load("...", "x")
        x : int = 1
        """);
  }

  @Test
  public void annotateFile_setsFunctionType_basic() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            def f(a : int, b = 1, *c : bool, d : str = "abc", e, **f : int) -> bool:
                pass
            """);
    Types.CallableType type = getType(getFirstStatement(DefStatement.class, file));

    assertThat(type).isNotNull();
    assertThat(type.getParameterNames()).containsExactly("a", "b", "d", "e").inOrder();
    assertThat(type.getParameterTypes())
        .containsExactly(Types.INT, Types.ANY, Types.STR, Types.ANY)
        .inOrder();
    assertThat(type.getNumPositionalOnlyParameters()).isEqualTo(0);
    assertThat(type.getNumPositionalParameters()).isEqualTo(2);
    assertThat(type.getMandatoryParameters()).containsExactly("a", "e").inOrder();
    assertThat(type.getVarargsType()).isEqualTo(Types.BOOL);
    assertThat(type.getKwargsType()).isEqualTo(Types.INT);
    assertThat(type.getReturnType()).isEqualTo(Types.BOOL);
  }

  @Test
  public void annotateFile_setsFunctionType_omittedDetailsHandledCorrectly() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            def f(*a, **b):
                pass
            """);
    Types.CallableType type = getType(getFirstStatement(DefStatement.class, file));

    assertThat(type).isNotNull();
    assertThat(type.getParameterNames()).isEmpty();
    assertThat(type.getParameterTypes()).isEmpty();
    assertThat(type.getVarargsType()).isEqualTo(Types.ANY);
    assertThat(type.getKwargsType()).isEqualTo(Types.ANY);
    assertThat(type.getReturnType()).isEqualTo(Types.ANY);

    file =
        annotateFile(
            """
            def f():
                pass
            """);
    type = getType(getFirstStatement(DefStatement.class, file));

    assertThat(type).isNotNull();
    assertThat(type.getVarargsType()).isNull();
    assertThat(type.getKwargsType()).isNull();
    assertThat(type.getReturnType()).isEqualTo(Types.ANY);
  }

  @Test
  public void annotateFile_reachesInnerFunctions() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            def f():
                def g(a : int):
                    pass
            """);
    var outer = getFirstStatement(DefStatement.class, file);
    var inner = getFirstStatement(DefStatement.class, outer);
    Types.CallableType type = getType(inner);

    assertThat(type).isNotNull();
    assertThat(type.getParameterNames()).containsExactly("a");
    assertThat(type.getParameterTypes()).containsExactly(Types.INT);
  }

  @Test
  public void annotateFile_setsFunctionType_onLambdas() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            lambda x: 123
            """);
    var stmt = getFirstStatement(ExpressionStatement.class, file);
    Types.CallableType type = getType((LambdaExpression) stmt.getExpression());

    assertThat(type).isNotNull();
    assertThat(type.getParameterNames()).containsExactly("x");
    assertThat(type.getParameterTypes()).containsExactly(Types.ANY);
    assertThat(type.getReturnType()).isEqualTo(Types.ANY);

    file =
        annotateFile(
            """
            lambda x: lambda y: 123
            """);
    stmt = getFirstStatement(ExpressionStatement.class, file);
    type = getType((LambdaExpression) ((LambdaExpression) stmt.getExpression()).getBody());

    assertThat(type).isNotNull();
  }

  // No type is set for the callable created for evaluating a file.
  // (There's no equivalent test for evaluating an expression, since that callable is created
  // on-the-fly by Starlark#eval.)
  @Test
  public void annotateFile_doesNotSetTypeOnStarlarkFileFunction() throws Exception {
    StarlarkFile file = annotateFile("pass");
    Types.CallableType type = file.getResolvedFunction().getFunctionType();

    assertThat(type).isNull();
  }

  @Test
  public void annotateFile_setsBindingType_nullByDefault() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            x = 1
            """);
    var stmt = getFirstStatement(AssignmentStatement.class, file);
    StarlarkType type = getType((Identifier) stmt.getLHS());

    assertThat(type).isNull();
  }

  @Test
  public void annotateFile_setsBindingType_var() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            x : int
            """);
    var stmt = getFirstStatement(VarStatement.class, file);
    StarlarkType type = getType(stmt.getIdentifier());

    assertThat(type).isEqualTo(Types.INT);
  }

  @Test
  public void annotateFile_setsBindingType_assignment() throws Exception {
    options.allowToplevelRebinding(true);

    StarlarkFile file =
        annotateFile(
            """
            x : int = 5
            x = 6  # not clobbered by annotation-less reassignment
            """);
    var stmt = getFirstStatement(AssignmentStatement.class, file);
    StarlarkType type = getType(((Identifier) stmt.getLHS()));

    assertThat(type).isEqualTo(Types.INT);
  }

  @Test
  public void annotateFile_setsBindingType_functionIdentifier() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            def f(x : int):
                pass
            """);
    var stmt = getFirstStatement(DefStatement.class, file);
    StarlarkType type = getType(stmt.getIdentifier());

    assertThat(type).isInstanceOf(Types.CallableType.class);
    assertThat(((Types.CallableType) type).getParameterTypeByPos(0)).isEqualTo(Types.INT);
  }

  @Test
  public void annotateFile_setsBindingType_functionParams() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            def f(a : int, b = 1, *c : bool, d : str = "abc", e, **f : int) -> bool:
                pass
            """);
    var stmt = getFirstStatement(DefStatement.class, file);
    ArrayList<StarlarkType> bindingTypes = new ArrayList<>();
    for (var param : stmt.getParameters()) {
      bindingTypes.add(getType(param.getIdentifier()));
    }

    assertThat(bindingTypes)
        .containsExactly(Types.INT, Types.ANY, Types.BOOL, Types.STR, Types.ANY, Types.INT)
        .inOrder();
  }

  @Test
  public void annotateFile_setsBindingType_lambdaParams() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            lambda x, y: 123
            """);
    var stmt = getFirstStatement(ExpressionStatement.class, file);
    var lambda = (LambdaExpression) stmt.getExpression();
    ArrayList<StarlarkType> bindingTypes = new ArrayList<>();
    for (var param : lambda.getParameters()) {
      bindingTypes.add(getType(param.getIdentifier()));
    }

    assertThat(bindingTypes).containsExactly(Types.ANY, Types.ANY).inOrder();
  }

  @Test
  public void annotateFile_setsBindingType_insideFunctions() throws Exception {
    StarlarkFile file =
        annotateFile(
            """
            def f():
                x : int
            """);
    var stmt = getFirstStatement(DefStatement.class, file);
    StarlarkType type = getType(getFirstStatement(VarStatement.class, stmt).getIdentifier());

    assertThat(type).isEqualTo(Types.INT);
  }

  @Test
  public void annotateFile_toleratesBareStarParam() throws Exception {
    annotateFile(
        """
        def f(*, x):
            pass
        """);
  }
}
