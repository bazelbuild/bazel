// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static net.starlark.java.syntax.TestUtils.assertContainsError;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.SyntaxError;
import net.starlark.java.syntax.TypeConstructor;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integrated tests for static type checking of Starlark code.
 *
 * <p>The test suite {@code syntax/TypeCheckerTest.java} checks the behavior of the static type
 * checker and the base type definitions in the syntax package. In contrast, this suite checks the
 * overall process of static type checking on a Starlark program, using the production universal
 * types defined in the eval/ package. This includes for instance the machinery to generate type
 * information for {@link StarlarkBuiltin}s.
 */
@RunWith(JUnit4.class)
public final class StaticTypeCheckTest {

  @SuppressWarnings("FieldCanBeFinal")
  private FileOptions.Builder options =
      FileOptions.builder()
          .allowTypeSyntax(true)
          .resolveTypeSyntax(true)
          .staticTypeChecking(true)
          // This lets us construct simpler test cases without wrapper `def` statements.
          .allowToplevelRebinding(true);

  @SuppressWarnings("FieldCanBeFinal")
  private Module module = Module.create();

  private Program compile(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, options.build());
    return Program.compileFile(file, module);
  }

  private void assertValid(String... lines) {
    try {
      compile(lines);
    } catch (SyntaxError.Exception ex) {
      throw new AssertionError("Expected success, but got: " + ex.getMessage(), ex);
    }
  }

  private void assertInvalid(String message, String... lines) {
    SyntaxError.Exception ex = assertThrows(SyntaxError.Exception.class, () -> compile(lines));
    assertContainsError(ex.errors(), message);
  }

  @SuppressWarnings("UnusedMethod")
  private StarlarkType inferType(String expr) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(expr);
    Expression expression = Expression.parse(input, options.build());
    Program program = Program.compileExpr(expression, module, options.build());
    return program.getResolvedFunction().getFunctionType();
  }

  @Test
  public void typecheckSuccess() {
    assertValid("n = 123 + 123");
  }

  @Test
  public void typecheckFailure() {
    assertInvalid(
        "operator '+' cannot be applied to types 'int' and 'str'",
        """
        n = 123 + 'abc'
        """);
  }

  @Test
  public void unknownSymbolAsType() {
    assertInvalid(
        "name 'unknown' is not defined",
        """
        x : unknown
        """);
  }

  @Test
  public void nonTypeSymbolAsType() {
    assertInvalid(
        "universal symbol 'len' cannot be used as a type",
        """
        x : len
        """);
  }

  @Test
  public void noneAsType() {
    assertValid("x : None = None");

    assertInvalid(
        "cannot assign type 'int' to 'x' of type 'None'",
        """
        x : None = 123
        """);
  }

  @Test
  public void starlarkBuiltinAsType() {
    assertValid("x : list[int] = [123]");

    assertInvalid(
        "cannot assign type 'list[str]' to 'x' of type 'list[int]'",
        """
        x: list[int] = ["abc"]
        """);
  }

  @StarlarkBuiltin(name = "BadBodyTypeBuiltin")
  public static final class BadBodyTypeBuiltin implements StarlarkValue {
    @SuppressWarnings("DoNotCallSuggester")
    public static TypeConstructor getBaseTypeConstructor() {
      throw new RuntimeException("fail");
    }
  }

  @StarlarkBuiltin(name = "BadSignatureTypeBuiltin")
  public static final class BadSignatureTypeBuiltin implements StarlarkValue {
    @SuppressWarnings("DoNotCallSuggester")
    public TypeConstructor getBaseTypeConstructor() { // missing `static`
      throw new RuntimeException("fail");
    }
  }

  @StarlarkBuiltin(name = "MissingStaticMethodTypeBuiltin")
  public static final class MissingStaticMethodTypeBuiltin implements StarlarkValue {
    // no getBaseTypeConstructor()
  }

  public static final class DummyLibrary {
    @StarlarkMethod(name = "BadSignature", documented = false, isTypeConstructor = true)
    public BadSignatureTypeBuiltin badSignature() {
      return new BadSignatureTypeBuiltin();
    }

    @StarlarkMethod(name = "BadBody", documented = false, isTypeConstructor = true)
    public BadBodyTypeBuiltin badBody() {
      return new BadBodyTypeBuiltin();
    }

    @StarlarkMethod(name = "MissingStaticMethod", documented = false, isTypeConstructor = true)
    public MissingStaticMethodTypeBuiltin missingStaticMethod() {
      return new MissingStaticMethodTypeBuiltin();
    }
  }

  @Test
  public void starlarkBuiltinWithBadBaseTypeConstructor() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, new DummyLibrary());
    module = Module.withPredeclared(StarlarkSemantics.DEFAULT, env.buildOrThrow());

    var ex = assertThrows(IllegalArgumentException.class, () -> compile("x: BadSignature = None"));
    assertThat(ex)
        .hasMessageThat()
        .containsMatch(".*BadSignatureTypeBuiltin#getBaseTypeConstructor has an invalid signature");

    ex = assertThrows(IllegalArgumentException.class, () -> compile("x: BadBody = None"));
    assertThat(ex)
        .hasMessageThat()
        .containsMatch("Error invoking .*BadBodyTypeBuiltin#getBaseTypeConstructor");

    ex =
        assertThrows(
            IllegalArgumentException.class, () -> compile("x: MissingStaticMethod = None"));
    assertThat(ex)
        .hasMessageThat()
        .containsMatch("invalid type constructor proxy: .*MissingStaticMethodTypeBuiltin");
  }
}
