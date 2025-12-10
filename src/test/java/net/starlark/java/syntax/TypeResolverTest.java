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

import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark type resolution. */
@RunWith(JUnit4.class)
public class TypeResolverTest {

  private StarlarkType evalType(String type) throws Exception {
    Expression typeExpr = Expression.parseTypeExpression(ParserInput.fromLines(type));
    // TODO: #27728 - When type resolution can consider non-universal types, use a better mock
    // module here that supports evalType().
    return TypeResolver.evalTypeExpression(typeExpr, Resolver.moduleWithPredeclared());
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

    assertThat(e).hasMessageThat().isEqualTo("type 'Foo' is not defined");
  }

  @Test
  public void evalType_badTypeApplications() {
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
}
