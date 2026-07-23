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

import com.google.common.collect.ImmutableMap;
import com.google.common.truth.BooleanSubject;
import java.util.ArrayList;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Resolver.Module;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TypeTagger}. */
@RunWith(JUnit4.class)
public class TypeTaggerTest {

  @SuppressWarnings("FieldCanBeFinal")
  private FileOptions.Builder options =
      FileOptions.builder().allowTypeSyntax(true).resolveTypeSyntax(true);

  private Module module =
      TestUtils.Module.withUniversalTypesAnd("struct", Types.STRUCT_CONSTRUCTOR);

  private TypeTagger.Loader loader = null;

  /** Extracts an expression string to a type in an empty environment. */
  private StarlarkType extractType(String type) throws Exception {
    Expression expr = Expression.parseTypeExpression(ParserInput.fromLines(type), options.build());
    Resolver.Function function = Resolver.resolveExpr(expr, module, options.build());
    return TypeTagger.extractType(expr, function, module);
  }

  /**
   * Asserts that attempting to extract an expression string to a type fails, with a syntax
   * exception whose message exactly matches the expected string.
   */
  private void assertExtractTypeFails(String type, String expectedMessage) throws Exception {
    var e = assertThrows(SyntaxError.Exception.class, () -> extractType(type));
    assertThat(e).hasMessageThat().isEqualTo(expectedMessage);
  }

  private record Result(StarlarkFile file, TypeTable typeTable) {
    /** Returns the type of an identifier. */
    @Nullable
    private StarlarkType getType(Identifier id) {
      assertThat(id.getBinding()).isNotNull();
      return typeTable().getType(id.getBinding());
    }

    @Nullable
    private Types.CallableType getType(Resolver.Function function) {
      return typeTable().getType(function);
    }

    /** Returns the type of a global. Does not support vars that are bound in a list assignment. */
    @Nullable
    private StarlarkType getType(String name) {
      for (Statement stmt : file().getStatements()) {
        switch (stmt) {
          case AssignmentStatement assign -> {
            if (assign.getLHS() instanceof Identifier id && id.getName().equals(name)) {
              return typeTable().getType(id.getBinding());
            }
          }
          case DefStatement def -> {
            if (def.getIdentifier().getName().equals(name)) {
              return typeTable().getType(def.getResolvedFunction());
            }
          }
          case TypeAliasStatement typeAlias -> {
            // TODO: #27370 - give type aliases' values a sensible type.
            if (typeAlias.getIdentifier().getName().equals(name)) {
              return typeTable().getType(typeAlias.getIdentifier().getBinding());
            }
          }
          case VarStatement var -> {
            if (var.getIdentifier().getName().equals(name)) {
              return typeTable().getType(var.getIdentifier().getBinding());
            }
          }
          default -> {}
        }
      }
      return null;
    }

    /** Returns the type of a {@code def}'s resolved function. */
    @Nullable
    private Types.CallableType getType(DefStatement def) {
      assertThat(def.getResolvedFunction()).isNotNull();
      return getType(def.getResolvedFunction());
    }

    /** Returns the type of a {@code lambda}'s resolved function. */
    @Nullable
    private Types.CallableType getType(LambdaExpression lambda) {
      assertThat(lambda.getResolvedFunction()).isNotNull();
      return getType(lambda.getResolvedFunction());
    }
  }

  /**
   * Parses a series of strings as a file, then resolves and type-tags it.
   *
   * <p>Asserts that parsing and symbol resolution succeeded, but type-tagging may fail.
   */
  private Result tagFilePossiblyFailing(String... lines) throws Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, options.build());
    assertThat(file.errors()).isEmpty();
    Resolver.resolveFile(file, module);
    assertThat(file.errors()).isEmpty();
    TypeTable typeTable = TypeTagger.tagFile(file, module, loader);
    return new Result(file, typeTable);
  }

  /** As in {@link #tagFilePossiblyFailing} but asserts that even type tagging succeeded. */
  private Result tagFile(String... lines) throws Exception {
    Result result = tagFilePossiblyFailing(lines);
    assertThat(result.typeTable().errors()).isEmpty();
    return result;
  }

  /** Asserts that type tagging fails with at least the specified error. */
  private void assertInvalid(String expectedError, String... lines) throws Exception {
    Result result = tagFilePossiblyFailing(lines);
    assertWithMessage("type tagging succeeded unexpectedly")
        .that(result.typeTable().ok())
        .isFalse();
    assertContainsError(result.typeTable().errors(), expectedError);
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

  /** Returns the resolved function of the first def statement with the given name. */
  private Resolver.Function getDefFunction(StarlarkFile file, String name) {
    ArrayList<Resolver.Function> functions = new ArrayList<>();
    new NodeVisitor() {
      @Override
      public void visit(DefStatement def) {
        if (def.getIdentifier().getName().equals(name)) {
          functions.add(def.getResolvedFunction());
        }
        super.visit(def);
      }
    }.visit(file);
    assertThat(functions).isNotEmpty();
    return functions.get(0);
  }

  private BooleanSubject assertTopLevelUsesTypeSyntax(String... lines) throws Exception {
    Result result = tagFile(lines);
    return assertThat(result.typeTable().usesTypeSyntax(result.file().getResolvedFunction()));
  }

  private BooleanSubject assertDefFunctionUsesTypeSyntax(String name, String... lines)
      throws Exception {
    Result result = tagFile(lines);
    return assertThat(result.typeTable().usesTypeSyntax(getDefFunction(result.file(), name)));
  }

  @Test
  public void staticTypeCheckingFlagRequirements() {
    options = FileOptions.builder().resolveTypeSyntax(false).tolerateInvalidTypeExpressions(false);
    assertThat(assertThrows(IllegalArgumentException.class, () -> tagFilePossiblyFailing("0")))
        .hasMessageThat()
        .contains("type tagging requires that resolveTypeSyntax is set");

    options = FileOptions.builder().resolveTypeSyntax(true).tolerateInvalidTypeExpressions(true);
    assertThat(assertThrows(IllegalArgumentException.class, () -> tagFilePossiblyFailing("0")))
        .hasMessageThat()
        .contains("type tagging requires that tolerateInvalidTypeExpressions is not set");
  }

  @Test
  public void extractType_primitives() throws Exception {
    assertThat(extractType("None")).isEqualTo(Types.NONE);
    assertThat(extractType("bool")).isEqualTo(Types.BOOL);
    assertThat(extractType("int")).isEqualTo(Types.INT);
    assertThat(extractType("float")).isEqualTo(Types.FLOAT);
    assertThat(extractType("str")).isEqualTo(Types.STR);

    assertExtractTypeFails("None[bool]", "'None' does not accept arguments");
    assertExtractTypeFails("bool[bool]", "'bool' does not accept arguments");
    assertExtractTypeFails("int[bool]", "'int' does not accept arguments");
    assertExtractTypeFails("float[bool]", "'float' does not accept arguments");
    assertExtractTypeFails("str[bool]", "'str' does not accept arguments");
  }

  @Test
  public void extractType_union() throws Exception {
    assertThat(extractType("int|bool")).isEqualTo(Types.union(Types.INT, Types.BOOL));
  }

  // These are also tests of the list, dict, and tuple type constructors, not just the TypeTagger.

  @Test
  public void extractType_list() throws Exception {
    assertThat(extractType("list[int]")).isEqualTo(Types.list(Types.INT));
    assertThat(extractType("list[list[int]]")).isEqualTo(Types.list(Types.list(Types.INT)));
    assertThat(extractType("list")).isEqualTo(Types.list(Types.ANY));

    assertExtractTypeFails("list[int, bool]", "list[] accepts exactly 1 argument but got 2");
    assertExtractTypeFails("list[[int]]", "unexpected expression '[int]'");
    assertExtractTypeFails("list[int, ...]", "in application to list, got '...', expected a type");
    assertExtractTypeFails("list[()]", "in application to list, got '()', expected a type");
  }

  @Test
  public void extractType_dict() throws Exception {
    assertThat(extractType("dict[int, str]")).isEqualTo(Types.dict(Types.INT, Types.STR));
    assertThat(extractType("dict[int, list[str]]"))
        .isEqualTo(Types.dict(Types.INT, Types.list(Types.STR)));
    assertThat(extractType("dict")).isEqualTo(Types.dict(Types.ANY, Types.ANY));

    assertExtractTypeFails("dict[int]", "dict[] accepts exactly 2 arguments but got 1");
    assertExtractTypeFails("dict[int, str, bool]", "dict[] accepts exactly 2 arguments but got 3");
  }

  @Test
  public void extractType_tuple() throws Exception {
    assertThat(extractType("tuple[()]")).isEqualTo(Types.EMPTY_TUPLE);
    assertThat(extractType("tuple[int]")).isEqualTo(Types.tuple(Types.INT));
    assertThat(extractType("tuple[int, str, bool]"))
        .isEqualTo(Types.tuple(Types.INT, Types.STR, Types.BOOL));
    assertThat(extractType("tuple[tuple[int, str], bool]"))
        .isEqualTo(Types.tuple(Types.tuple(Types.INT, Types.STR), Types.BOOL));
    assertThat(extractType("tuple[int, ...]")).isEqualTo(Types.homogeneousTuple(Types.INT));
    assertThat(extractType("tuple")).isEqualTo(Types.homogeneousTuple(Types.ANY));

    assertExtractTypeFails(
        "tuple[...]",
        "in application to tuple, '...' can only appear as the second of exactly 2 arguments, where"
            + " the first argument is a type");
    assertExtractTypeFails(
        "tuple[int, str, ...]",
        "in application to tuple, '...' can only appear as the second of exactly 2 arguments, where"
            + " the first argument is a type");
    assertExtractTypeFails(
        "tuple[(), int]",
        "in application to tuple, '()' can only appear if it is the only argument");
  }

  @Test
  public void extractType_struct() throws Exception {
    assertThat(extractType("struct[{}]")).isEqualTo(Types.struct(ImmutableMap.of()));
    assertThat(extractType("struct[{'foo': int, 'bar': list[str]}]"))
        .isEqualTo(Types.struct(ImmutableMap.of("foo", Types.INT, "bar", Types.list(Types.STR))));

    assertThat(extractType("struct")).isEqualTo(Types.ANY_STRUCT);
    assertThat(extractType("struct[{'foo': int}, ...]"))
        .isEqualTo(Types.partialStruct(ImmutableMap.of("foo", Types.INT)));

    assertExtractTypeFails("struct[...]", "in application to struct, got '...', expected a dict");
    assertExtractTypeFails(
        "struct[{'a': int}, int]",
        "in application to struct, got 'int' for optional argument #2, expected '...'");
    assertExtractTypeFails(
        "struct[{'a': int}, ..., ...]", "struct[] accepts at most 2 arguments but got 3");
    // Just like for eval-time dict literals, keys must be unique.
    assertExtractTypeFails(
        "struct[{'foo': int, 'foo': bool}]", "dictionary expression has duplicate key: \"foo\"");
  }

  @Test
  public void extractType_unknownIdentifier() throws Exception {
    assertExtractTypeFails("Foo", "name 'Foo' is not defined");
    assertExtractTypeFails("Foo[int]", "name 'Foo' is not defined");
  }

  @Test
  public void localCannotShadowPredeclaredType() throws Exception {
    assertInvalid(
        "local symbol 'int' cannot be used as a type",
        """
        def f():
            int = 123
            x : int
        """);
  }

  @Test
  public void nonTypeCannotBeUsedAsType() throws Exception {
    module = TestUtils.Module.withTypes("Foo", null);
    assertInvalid(
        "predeclared symbol 'Foo' cannot be used as a type",
        """
        x : Foo
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
    TypeTable typeTable =
        tagFilePossiblyFailing(
                """
                def f():
                    x : int
                    x : str
                """)
            .typeTable();
    assertThat(typeTable.ok()).isFalse();
    assertContainsError(
        typeTable.errors(), "3:5: type annotation on 'x' may only appear at its declaration");
    assertContainsError(typeTable.errors(), "2:5: 'x' previously declared here");
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
  public void tagFile_setsFunctionType_basic() throws Exception {
    Result result =
        tagFile(
            """
            def f(a : int, b = 1, *c : bool, d : str = "abc", e, **f : int) -> bool:
                pass
            """);
    Types.CallableType type = result.getType(getFirstStatement(DefStatement.class, result.file()));

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
  public void tagFile_setsFunctionType_omittedDetailsHandledCorrectly() throws Exception {
    Result result =
        tagFile(
            """
            def f(*a, **b):
                pass
            """);
    Types.CallableType type = result.getType(getFirstStatement(DefStatement.class, result.file()));

    assertThat(type).isNotNull();
    assertThat(type.getParameterNames()).isEmpty();
    assertThat(type.getParameterTypes()).isEmpty();
    assertThat(type.getVarargsType()).isEqualTo(Types.ANY);
    assertThat(type.getKwargsType()).isEqualTo(Types.ANY);
    assertThat(type.getReturnType()).isEqualTo(Types.ANY);

    result =
        tagFile(
            """
            def f():
                pass
            """);
    type = result.getType(getFirstStatement(DefStatement.class, result.file()));

    assertThat(type).isNotNull();
    assertThat(type.getVarargsType()).isNull();
    assertThat(type.getKwargsType()).isNull();
    assertThat(type.getReturnType()).isEqualTo(Types.ANY);
  }

  @Test
  public void tagFile_reachesInnerFunctions() throws Exception {
    Result result =
        tagFile(
            """
            def f():
                def g(a : int):
                    pass
            """);
    var outer = getFirstStatement(DefStatement.class, result.file());
    var inner = getFirstStatement(DefStatement.class, outer);
    Types.CallableType type = result.getType(inner);

    assertThat(type).isNotNull();
    assertThat(type.getParameterNames()).containsExactly("a");
    assertThat(type.getParameterTypes()).containsExactly(Types.INT);
  }

  @Test
  public void tagFile_setsFunctionType_onLambdas() throws Exception {
    Result result =
        tagFile(
            """
            lambda x: 123
            """);
    var stmt = getFirstStatement(ExpressionStatement.class, result.file());
    Types.CallableType type = result.getType((LambdaExpression) stmt.getExpression());

    assertThat(type).isNotNull();
    assertThat(type.getParameterNames()).containsExactly("x");
    assertThat(type.getParameterTypes()).containsExactly(Types.ANY);
    assertThat(type.getReturnType()).isEqualTo(Types.ANY);

    result =
        tagFile(
            """
            lambda x: lambda y: 123
            """);
    stmt = getFirstStatement(ExpressionStatement.class, result.file());
    type = result.getType((LambdaExpression) ((LambdaExpression) stmt.getExpression()).getBody());

    assertThat(type).isNotNull();
  }

  // No type is set for the callable created for evaluating a file.
  // (There's no equivalent test for evaluating an expression, since that callable is created
  // on-the-fly by Starlark#eval.)
  @Test
  public void tagFile_doesNotSetTypeOnStarlarkFileFunction() throws Exception {
    Result result = tagFile("pass");
    Types.CallableType type = result.getType(result.file().getResolvedFunction());

    assertThat(type).isNull();
  }

  @Test
  public void tagFile_setsBindingType_nullByDefault() throws Exception {
    Result result =
        tagFile(
            """
            x = 1
            """);
    var stmt = getFirstStatement(AssignmentStatement.class, result.file());
    StarlarkType type = result.getType((Identifier) stmt.getLHS());

    assertThat(type).isNull();
  }

  @Test
  public void tagFile_setsBindingType_var() throws Exception {
    Result result =
        tagFile(
            """
            x : int
            """);
    var stmt = getFirstStatement(VarStatement.class, result.file());
    StarlarkType type = result.getType(stmt.getIdentifier());

    assertThat(type).isEqualTo(Types.INT);
  }

  @Test
  public void tagFile_setsBindingType_assignment() throws Exception {
    options.allowToplevelRebinding(true);

    Result result =
        tagFile(
            """
            x : int = 5
            x = 6  # not clobbered by annotation-less reassignment
            """);
    var stmt = getFirstStatement(AssignmentStatement.class, result.file());
    StarlarkType type = result.getType(((Identifier) stmt.getLHS()));

    assertThat(type).isEqualTo(Types.INT);
  }

  @Test
  public void tagFile_setsBindingType_functionIdentifier() throws Exception {
    Result result =
        tagFile(
            """
            def f(x : int):
                pass
            """);
    var stmt = getFirstStatement(DefStatement.class, result.file());
    StarlarkType type = result.getType(stmt.getIdentifier());

    assertThat(type).isInstanceOf(Types.CallableType.class);
    assertThat(((Types.CallableType) type).getParameterTypeByPos(0)).isEqualTo(Types.INT);
  }

  @Test
  public void tagFile_setsBindingType_functionParams() throws Exception {
    Result result =
        tagFile(
            """
            def f(a : int, b = 1, *c : bool, d : str = "abc", e, **f : int) -> bool:
                pass
            """);
    var stmt = getFirstStatement(DefStatement.class, result.file());
    ArrayList<StarlarkType> bindingTypes = new ArrayList<>();
    for (var param : stmt.getParameters()) {
      bindingTypes.add(result.getType(param.getIdentifier()));
    }

    assertThat(bindingTypes)
        .containsExactly(Types.INT, Types.ANY, Types.BOOL, Types.STR, Types.ANY, Types.INT)
        .inOrder();
  }

  @Test
  public void tagFile_setsBindingType_lambdaParams() throws Exception {
    Result result =
        tagFile(
            """
            lambda x, y: 123
            """);
    var stmt = getFirstStatement(ExpressionStatement.class, result.file());
    var lambda = (LambdaExpression) stmt.getExpression();
    ArrayList<StarlarkType> bindingTypes = new ArrayList<>();
    for (var param : lambda.getParameters()) {
      bindingTypes.add(result.getType(param.getIdentifier()));
    }

    assertThat(bindingTypes).containsExactly(Types.ANY, Types.ANY).inOrder();
  }

  @Test
  public void tagFile_setsBindingType_insideFunctions() throws Exception {
    Result result =
        tagFile(
            """
            def f():
                x : int
            """);
    var stmt = getFirstStatement(DefStatement.class, result.file());
    StarlarkType type = result.getType(getFirstStatement(VarStatement.class, stmt).getIdentifier());

    assertThat(type).isEqualTo(Types.INT);
  }

  @Test
  public void tagFile_toleratesBareStarParam() throws Exception {
    tagFile(
        """
        def f(*, x):
            pass
        """);
  }

  @Test
  public void tagFile_toplevelUsesTypeSyntax() throws Exception {
    // <toplevel> is considered to use static type syntax if any part of the file uses static type
    // syntax.
    assertTopLevelUsesTypeSyntax(
            """
            # No type syntax anywhere.
            z = 1
            def f(x):
                return lambda y: x + 2
            f(z)
            """)
        .isFalse();

    assertTopLevelUsesTypeSyntax("type X = int").isTrue();
    assertTopLevelUsesTypeSyntax("x: int").isTrue();
    assertTopLevelUsesTypeSyntax("x: int = 1").isTrue();
    assertTopLevelUsesTypeSyntax("x = cast(int, 1)").isTrue();
    // nested lambda and def statements
    assertTopLevelUsesTypeSyntax("lambda x: cast(int, x)").isTrue();
    assertTopLevelUsesTypeSyntax(
            """
            def f(x: int):
                pass
            """)
        .isTrue();
    assertTopLevelUsesTypeSyntax(
            """
            def f(x) -> int:
                pass
            """)
        .isTrue();
    assertTopLevelUsesTypeSyntax(
            """
            def f(x):
                def g(y):
                    z: int = 42
                    return z + y
            """)
        .isTrue();
  }

  @Test
  public void tagFile_defStatementUsesTypeSyntax() throws Exception {
    // A def statement uses static type syntax if it has type annotations in its declarations on in
    // its body (including nested lambdas but not nested def statements).
    assertDefFunctionUsesTypeSyntax("f", "def f(x): return x").isFalse();
    assertDefFunctionUsesTypeSyntax("f", "def f(x) -> int: return 42").isTrue();
    assertDefFunctionUsesTypeSyntax("f", "def f(x: int): return x").isTrue();
    assertDefFunctionUsesTypeSyntax("f", "def f(x): return cast(int, x)").isTrue();

    // Nesting
    assertDefFunctionUsesTypeSyntax(
            "untyped_in_typed_toplevel",
            """
            X: int = 42
            def untyped_in_typed_toplevel(x):
                return X
            """)
        .isFalse();
    String typedInUntypedDef =
        """
        def untyped_with_nested_typed(x):
            def typed_nested_in_untyped(y: int) -> int:
                return cast(int, x) + y
            return typed_nested_in_untyped(x)
        """;
    assertDefFunctionUsesTypeSyntax("untyped_with_nested_typed", typedInUntypedDef).isFalse();
    assertDefFunctionUsesTypeSyntax("typed_nested_in_untyped", typedInUntypedDef).isTrue();
    assertDefFunctionUsesTypeSyntax(
            "untyped_with_nested_typed_lambda",
            """
            def untyped_with_nested_typed_lambda(x):
                return (lambda y: cast(int, y) + 42)(x)
            """)
        .isTrue();
    assertDefFunctionUsesTypeSyntax(
            "untyped_with_nested_untyped_def_with_nested_typed_lambda",
            """
            def untyped_with_nested_untyped_def_with_nested_typed_lambda(x):
                def nested(y):
                    return (lambda z: cast(int, z) + 42)(y)
                return (lambda w: w)(nested(x))
            """)
        .isFalse();
  }

  @Test
  public void loadStatement() throws Exception {
    loader = importName -> TestUtils.LoadableModule.of("typed", Types.INT, "untyped", Types.ANY);
    Result result = tagFile("load('//x:x.bzl', local_t = 'typed', local_u = 'untyped')");
    LoadStatement loadStmt = getFirstStatement(LoadStatement.class, result.file());

    assertThat(loadStmt.getBindings().stream().map(b -> result.getType(b.getLocalName())))
        .containsExactly(Types.INT, Types.ANY)
        .inOrder();
  }

  @Test
  public void loadStatement_requiresWorkingLoader() throws Exception {
    loader = null;
    assertInvalid(
        "load statements are not supported because no module loader has been defined",
        "load('//x:x.bzl', 'x')");
  }

  @Test
  public void loadStatement_requiresLoadableModule() throws Exception {
    loader = importName -> null;
    assertInvalid("module '//x:x.bzl' not found", "load('//x:x.bzl', 'x')");
  }

  @Test
  public void loadStatement_requiresExportedGlobal() throws Exception {
    loader = importName -> TestUtils.LoadableModule.of();
    assertInvalid("module '//x:x.bzl' does not contain symbol 'x'", "load('//x:x.bzl', 'x')");
  }

  @Test
  public void loadStatement_loadsTypeConstructor() throws Exception {
    loader =
        importName ->
            TestUtils.LoadableModule.ofTypesAndConstructors(
                "numeric", Types.ANY, Types.wrapType("numeric", Types.NUMERIC));
    Result result =
        tagFile(
            """
            load("//x:x.bzl", "numeric")
            x: numeric = 1
            """);
    assertThat(result.getType("x")).isEqualTo(Types.NUMERIC);
  }

  @Test
  public void loadedTypeConstructor_cannotBeRebound() throws Exception {
    loader =
        importName ->
            TestUtils.LoadableModule.ofTypesAndConstructors(
                "numeric", Types.ANY, Types.wrapType("numeric", Types.NUMERIC));
    options.allowToplevelRebinding(true).loadBindsGlobally(true);

    // TODO: #27370 - Arguably, these should all be allowed unless the loaded symbol is used in a
    // type expression in the file.
    assertInvalid(
        ":2:1: type 'numeric' redeclared",
        """
        load("//x:x.bzl", "numeric")
        type numeric = int
        """);

    assertInvalid(
        ":2:1: type 'numeric' redeclared",
        """
        load("//x:x.bzl", "numeric")
        numeric = 123
        """);

    assertInvalid(
        ":2:1: type 'numeric' redeclared",
        """
        load("//x:x.bzl", "numeric")
        foo, bar, numeric = 123, 456, 789
        """);

    assertInvalid(
        ":2:1: type 'numeric' redeclared",
        """
        load("//x:x.bzl", "numeric")
        def numeric(x):
            return cast(int|float, x)
        """);
  }

  @Test
  public void typeAlias() throws Exception {
    Result result =
        tagFile(
            """
            type int_or_str = int | str
            x: int_or_str

            type int_or_str_or_list[T] = int_or_str | list[T]
            y: int_or_str_or_list[float]

            type custom_struct[T, U] = struct[{'a': T, 'b': U}, ...]
            z: custom_struct[int, int_or_str_or_list[bool]]
            """);
    assertThat(result.getType("x")).isEqualTo(Types.union(Types.INT, Types.STR));
    assertThat(result.getType("y"))
        .isEqualTo(Types.union(Types.INT, Types.STR, Types.list(Types.FLOAT)));
    assertThat(result.getType("z"))
        .isEqualTo(
            Types.partialStruct(
                ImmutableMap.of(
                    "a",
                    Types.INT,
                    "b",
                    Types.union(Types.INT, Types.STR, Types.list(Types.BOOL)))));
  }

  @Test
  public void typeAlias_mayIgnoreParams() throws Exception {
    Result result =
        tagFile(
            """
            type int_or_str[T] = int | str  # T is ignored
            x: int_or_str[int]

            type optional_of_second_arg[T, U, V] = U | None  # T and V are ignored
            y: optional_of_second_arg[bool, str, float]
            """);
    assertThat(result.getType("x")).isEqualTo(Types.union(Types.INT, Types.STR));
    assertThat(result.getType("y")).isEqualTo(Types.union(Types.STR, Types.NONE));
  }

  @Test
  public void typeAlias_mayUseLoadedTypeConstructors() throws Exception {
    loader =
        importName ->
            TestUtils.LoadableModule.ofTypesAndConstructors(
                "numeric", Types.ANY, Types.wrapType("numeric", Types.NUMERIC));
    Result result =
        tagFile(
            """
            load("//x:x.bzl", "numeric")
            type numeric_or_str_or[T] = numeric | str | T
            x: numeric_or_str_or[bool]
            """);
    assertThat(result.getType("x")).isEqualTo(Types.union(Types.NUMERIC, Types.STR, Types.BOOL));
  }

  @Test
  public void typeAlias_requiresCorrectNumberOfTypeArgs() throws Exception {
    assertInvalid(
        "'int_or_str' does not accept arguments",
        """
        type int_or_str = int | str
        x: int_or_str[int]
        """);
    assertInvalid(
        "optional_list[] accepts exactly 1 argument but got 0",
        """
        type optional_list[T] = list[T] | None
        x: optional_list
        """);
    assertInvalid(
        "optional_dict[] accepts exactly 2 arguments but got 1",
        """
        type optional_dict[K, V] = dict[K, V] | None
        x: optional_dict[str]
        """);
    assertInvalid(
        "optional_dict[] accepts exactly 2 arguments but got 3",
        """
        type optional_dict[K, V] = dict[K, V] | None
        x: optional_dict[str, int, bool]
        """);
  }

  @Test
  public void typeAlias_requiresArgsToBeTypes() throws Exception {
    // TODO: #27370 - Should we relax this requirement? Ideally, we'd want to be able to query each
    // type constructor, given a particular arity, for the allowed classes of TypeConstructor.Arg it
    // allows for the n-th argument, and use that to determine the allowed arg kinds for the type
    // alias's type constructor.
    assertInvalid(
        "in application to optional_struct, got '{\"a\": int}', expected a type",
        """
        type optional_struct[T] = struct[T] | None
        x: optional_struct[{"a": int}]
        """);
    assertInvalid(
        "in application to optional_singleton_tuple, got '()', expected a type",
        """
        type optional_singleton_tuple[T] = tuple[T] | None
        x: optional_singleton_tuple[()]
        """);
  }

  @Test
  public void typeAlias_cannotBeUsedBeforeDefinition() throws Exception {
    assertInvalid(
        "name 'int_or_str' is not defined",
        """
        x: int_or_str
        type int_or_str = int | str
        """);
    assertInvalid(
        "name 'optional_list' is not defined",
        """
        x: optional_list[int]
        type optional_list[T] = list[T] | None
        """);
  }
}
