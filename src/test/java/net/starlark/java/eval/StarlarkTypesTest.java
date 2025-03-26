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

package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Iterables;
import net.starlark.java.syntax.DefStatement;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark types. */
@RunWith(JUnit4.class)
public class StarlarkTypesTest {
  @Test
  public void evalType_onPrimitiveTypes() throws Exception {
    assertThat(evalType("None")).isEqualTo(Types.NONE);
    assertThat(evalType("bool")).isEqualTo(Types.BOOL);
    assertThat(evalType("int")).isEqualTo(Types.INT);
    assertThat(evalType("float")).isEqualTo(Types.FLOAT);
    assertThat(evalType("str")).isEqualTo(Types.STR);
  }

  @Test
  public void evalType_unknownIdentifier() {
    EvalException e = assertThrows(EvalException.class, () -> evalType("Foo"));

    assertThat(e).hasMessageThat().isEqualTo("type 'Foo' is not defined");
  }

  private Expression parseTypeExpr(String typeExpr) throws Exception {
    // Use a simple function definition to parse type expression
    ParserInput input = ParserInput.fromLines(String.format("def f() -> %s: pass", typeExpr));
    StarlarkFile file =
        StarlarkFile.parse(input, FileOptions.builder().allowTypeAnnotations(true).build());
    if (!file.ok()) {
      throw new SyntaxError.Exception(file.errors());
    }
    return ((DefStatement) Iterables.getOnlyElement(file.getStatements())).getReturnType();
  }

  private StarlarkType evalType(String type) throws Exception {
    return EvalTypes.evalType(Module.create(), parseTypeExpr(type));
  }
}
