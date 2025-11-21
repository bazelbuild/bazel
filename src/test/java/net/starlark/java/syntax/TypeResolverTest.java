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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark type resolution. */
@RunWith(JUnit4.class)
public class TypeResolverTest {
  @Test
  public void resolveType_onPrimitiveTypes() throws Exception {
    assertThat(evalType("None")).isEqualTo(Types.NONE);
    assertThat(evalType("bool")).isEqualTo(Types.BOOL);
    assertThat(evalType("int")).isEqualTo(Types.INT);
    assertThat(evalType("float")).isEqualTo(Types.FLOAT);
    assertThat(evalType("str")).isEqualTo(Types.STR);
  }

  @Test
  public void resolveType_union() throws Exception {
    assertThat(evalType("int|bool")).isEqualTo(Types.union(Types.INT, Types.BOOL));
  }

  @Test
  public void resolveType_list() throws Exception {
    assertThat(evalType("list[int]")).isEqualTo(Types.list(Types.INT));
    assertThat(evalType("list[list[int]]")).isEqualTo(Types.list(Types.list(Types.INT)));

    var exception = assertThrows(SyntaxError.Exception.class, () -> evalType("list[int, bool]"));
    assertThat(exception).hasMessageThat().isEqualTo("list[] accepts exactly 1 argument but got 2");

    exception = assertThrows(SyntaxError.Exception.class, () -> evalType("list[[int]]"));
    assertThat(exception).hasMessageThat().isEqualTo("unexpected expression '[int]'");
  }

  @Test
  public void resolveType_dict() throws Exception {
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
  }

  @Test
  public void resolveType_unknownIdentifier() {
    SyntaxError.Exception e = assertThrows(SyntaxError.Exception.class, () -> evalType("Foo"));

    assertThat(e).hasMessageThat().isEqualTo("type 'Foo' is not defined");
  }

  @Test
  public void resolveType_badTypeApplications() {
    SyntaxError.Exception e =
        assertThrows(SyntaxError.Exception.class, () -> evalType("int[bool]"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("'int' is not a type constructor, cannot be applied to '[bool]'");

    e = assertThrows(SyntaxError.Exception.class, () -> evalType("Foo[int]"));
    assertThat(e).hasMessageThat().isEqualTo("type constructor 'Foo' is not defined");

    e = assertThrows(SyntaxError.Exception.class, () -> evalType("list"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected type arguments after the type constructor 'list'");
  }

  // TODO: #27728 - This is a test of CallableType, not the TypeResolver. Split into its own file,
  // either in this package or in types/.
  @Test
  public void callable_toSignatureString() {
    // ordinary only
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("a"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 1,
                    /* mandatoryParams= */ ImmutableSet.of("a"),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("(a: int) -> bool");
    // kwonly only
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("a"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of("a"),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("(*, a: int) -> bool");
    // positional-only only
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("x"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 1,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of(),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("([int], /) -> bool");
    // no params
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of(),
                    /* parameterTypes= */ ImmutableList.of(),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of(),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("() -> bool");
    // all kinds of params
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("x", "a", "b", "c", "d"),
                    /* parameterTypes= */ ImmutableList.of(
                        Types.BOOL, Types.INT, Types.FLOAT, Types.NONE, Types.ANY),
                    /* numPositionalOnlyParameters= */ 1,
                    /* numPositionalParameters= */ 3,
                    /* mandatoryParams= */ ImmutableSet.of("a", "c", "x"),
                    /* varargsType= */ Types.INT,
                    /* kwargsType= */ Types.INT,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo(
            "(bool, /, a: int, b: [float], *args: int, c: None, d: [Any], **kwargs: int) -> bool");
  }

  private StarlarkType evalType(String type) throws Exception {
    Expression typeExpr = Expression.parseTypeExpression(ParserInput.fromLines(type));
    // TODO: #27728 - When type resolution can consider non-universal types, use a better mock
    // module here that supports evalType().
    return TypeResolver.evalTypeExpression(typeExpr, Resolver.moduleWithPredeclared());
  }
}
