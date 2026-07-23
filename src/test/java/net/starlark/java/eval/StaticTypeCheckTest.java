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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkLibrary;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.SyntaxError;
import net.starlark.java.syntax.TypeChecker;
import net.starlark.java.syntax.TypeConstructor;
import net.starlark.java.syntax.TypeContext;
import net.starlark.java.syntax.TypeTable;
import net.starlark.java.syntax.TypeTagger;
import net.starlark.java.syntax.Types;
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
          // This lets us construct simpler test cases without wrapper `def` statements.
          .allowToplevelRebinding(true);

  @SuppressWarnings("FieldCanBeFinal")
  private StarlarkSemantics semantics =
      StarlarkSemantics.builder()
          .setBool(StarlarkSemantics.EXPERIMENTAL_STARLARK_STATIC_TYPE_CHECKING, true)
          .build();

  @SuppressWarnings("FieldCanBeFinal")
  private Module module = Module.create();

  @SuppressWarnings("FieldCanBeFinal")
  @Nullable
  private TypeTagger.Loader loader = null;

  private Program compile(String... lines) throws SyntaxError.Exception {
    Preconditions.checkArgument(lines.length > 0);
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, options.build());
    Program prog = Program.compileFile(file, module);
    TypeTable typeTable = TypeTagger.tagProgram(prog, module, loader);
    if (typeTable.ok()) {
      TypeChecker.checkProgram(prog, typeTable, module);
    }
    if (!typeTable.ok()) {
      throw new SyntaxError.Exception(typeTable.errors());
    }
    return prog.withTypeTable(typeTable);
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
    return program.getTypeTable().getType(program.getResolvedFunction());
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
        _unused: bool  # ensure file uses type syntax
        """);
  }

  @Test
  public void unknownSymbolAsType() {
    assertInvalid(
        "name 'unknown' is not defined",
        """
        x: unknown
        """);
  }

  @Test
  public void nonTypeSymbolAsType() {
    assertInvalid(
        "universal symbol 'len' cannot be used as a type",
        """
        x: len
        """);
  }

  @Test
  public void noneAsType() {
    assertValid("x: None = None");

    assertInvalid(
        "cannot assign type 'int' to 'x' of type 'None'",
        """
        x: None = 123
        """);
  }

  @Test
  public void starlarkBuiltinAsType() {
    assertValid("x: list[int] = [123]");

    assertInvalid(
        "cannot assign type 'list[str]' to 'x' of type 'list[int]'",
        """
        x: list[int] = ["abc"]
        """);
  }

  @StarlarkBuiltin(name = "BadBodyTypeBuiltin")
  public static final class BadBodyTypeBuiltin implements StarlarkValue {
    @SuppressWarnings("DoNotCallSuggester")
    public static TypeConstructor getAssociatedTypeConstructor() {
      throw new RuntimeException("fail");
    }
  }

  @StarlarkBuiltin(name = "BadSignatureTypeBuiltin")
  public static final class BadSignatureTypeBuiltin implements StarlarkValue {
    @SuppressWarnings("DoNotCallSuggester")
    public TypeConstructor getAssociatedTypeConstructor() { // missing `static`
      throw new RuntimeException("fail");
    }
  }

  @StarlarkBuiltin(name = "MissingStaticMethodTypeBuiltin")
  public static final class MissingStaticMethodTypeBuiltin implements StarlarkValue {
    // no getAssociatedTypeConstructor()

    // Override ensures that we don't generate a StarlarkBuiltinAutoType for this class.
    @Override
    public StarlarkType getStarlarkType(StarlarkSemantics semantics) {
      throw new UnsupportedOperationException("fail");
    }
  }

  @StarlarkLibrary
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
  public void starlarkBuiltinWithBadAssociatedTypeConstructor() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, new DummyLibrary());
    module = Module.withPredeclared(StarlarkSemantics.DEFAULT, env.buildOrThrow());

    var ex = assertThrows(IllegalArgumentException.class, () -> compile("x: BadSignature = None"));
    assertThat(ex)
        .hasMessageThat()
        .containsMatch(
            ".*BadSignatureTypeBuiltin#getAssociatedTypeConstructor has an invalid signature");

    ex = assertThrows(IllegalArgumentException.class, () -> compile("x: BadBody = None"));
    assertThat(ex)
        .hasMessageThat()
        .containsMatch("Error invoking .*BadBodyTypeBuiltin#getAssociatedTypeConstructor");

    ex =
        assertThrows(
            IllegalArgumentException.class, () -> compile("x: MissingStaticMethod = None"));
    assertThat(ex)
        .hasMessageThat()
        .containsMatch("invalid type constructor proxy: .*MissingStaticMethodTypeBuiltin");
  }

  @Test
  public void listMethods() {
    assertValid(
        """
        x: list[int]
        x.pop(0)
        """);

    assertInvalid(
        "in call to 'x.pop()', parameter 'i' got value of type 'str', want 'int'",
        """
        x: list[int]
        x.pop("abc")
        """);

    assertInvalid(
        "'x' of type 'list[int]' does not have field 'does_not_exist'",
        """
        x: list[int]
        x.does_not_exist
        """);
  }

  @Test
  public void dictMethods() {
    assertValid(
        """
        d: dict[str, int]
        v = d.get("a", 0)
        d.setdefault("b", 2)
        """);
  }

  @Test
  public void setMethods() {
    assertValid(
        """
        s: set[int]
        s.add(3)
        """);
  }

  @Test
  public void strMethods() {
    // Note that StringModule is special-cased to take the receiver string object as a separate
    // parameter to the Java method, yet it doesn't appear in the signature for type-checking
    // purposes.
    assertValid(
        """
        s: str
        s.startswith("abc")
        """);

    assertInvalid(
        "'s.startswith()' missing 1 required argument: sub",
        """
        s: str
        s.startswith()
        """);
  }

  @Test
  public void universalSymbolTypes() throws Exception {
    assertValid(
        """
        b: bool = True
        b = False
        n: None = None
        s: str = str(123)
        i: int = int(123)
        f: float = float(123)
        l: list = list()
        d: dict = dict()
        se: set = set()
        """);
    assertInvalid("cannot assign type 'bool' to 'x' of type 'str'", "x: str = True");
    assertInvalid("cannot assign type 'bool' to 'x' of type 'str'", "x: str = False");
    assertInvalid("cannot assign type 'None' to 'x' of type 'str'", "x: str = None");
    assertInvalid("cannot assign type 'str' to 'x' of type 'int'", "x: int = str(123)");
    assertInvalid("cannot assign type 'int' to 'x' of type 'str'", "x: str = int(123)");
    assertInvalid("cannot assign type 'float' to 'x' of type 'str'", "x: str = float(123)");
    assertInvalid("cannot assign type 'list[Any]' to 'x' of type 'str'", "x: str = list()");
    assertInvalid("cannot assign type 'dict[Any, Any]' to 'x' of type 'str'", "x: str = dict()");
    assertInvalid("cannot assign type 'set[Any]' to 'x' of type 'str'", "x: str = set()");
  }

  @Test
  public void predeclaredSymbolTypes() throws Exception {
    module =
        Module.withPredeclared(
            StarlarkSemantics.DEFAULT,
            ImmutableMap.of("PREDECLARED_INT", StarlarkInt.of(123), "PREDECLARED_STR", "abc"));
    assertValid(
        """
        x: int = PREDECLARED_INT
        y: str = PREDECLARED_STR
        """);
    assertInvalid("cannot assign type 'int' to 'x' of type 'str'", "x: str = PREDECLARED_INT");
    assertInvalid("cannot assign type 'str' to 'x' of type 'int'", "x: int = PREDECLARED_STR");
  }

  // No StarlarkBuiltin annotation.
  public static final class MyUnannotatedType implements StarlarkValue {
    @StarlarkMethod(name = "foo", doc = "...")
    public int foo() {
      return 123;
    }
  }

  @StarlarkBuiltin(name = "MyType")
  public static sealed class MyType implements StarlarkValue permits MyTypeSubclass {
    @StarlarkMethod(name = "foo", doc = "...")
    public int foo() {
      return 123;
    }
  }

  public static final class MyTypeSubclass extends MyType {}

  @StarlarkBuiltin(name = "MySelfCallType")
  public static final class MySelfCallType implements StarlarkValue {
    @StarlarkMethod(name = "MySelfCallType", doc = "...", selfCall = true)
    public int selfCall() {
      return 123;
    }

    @StarlarkMethod(name = "bar", doc = "...")
    public int bar() {
      return 123;
    }
  }

  @StarlarkBuiltin(name = "MyExplicitlyTypedType")
  public static final class MyExplicitlyTypedType implements StarlarkValue {
    @Override
    // Override causes no 'MyExplicitlyTypedType' type to be auto-generated.
    public StarlarkType getStarlarkType(StarlarkSemantics semantics) {
      return Types.ANY_STRUCT;
    }
  }

  @StarlarkBuiltin(name = "MyExplicitlyTypedSelfCallType")
  public static final class MyExplicitlyTypedSelfCallType implements StarlarkValue {
    @StarlarkMethod(name = "MyExplicitlyTypedSelfCallType", doc = "...", selfCall = true)
    public int selfCall() {
      return 123;
    }

    @Override
    // Override causes no 'MyExplicitlyTypedSelfCallType' type to be auto-generated.
    public StarlarkType getStarlarkType(StarlarkSemantics semantics) {
      return new StarlarkType() {
        @Override
        public String toString() {
          return "ExplicitlyTypedSelfCall";
        }

        @Override
        public ImmutableList<StarlarkType> getSupertypes(TypeContext context) {
          return ImmutableList.of(
              // Nullary callable returning int.
              Types.callable(
                  ImmutableList.of(),
                  ImmutableList.of(),
                  0,
                  0,
                  ImmutableSet.of(),
                  null,
                  null,
                  Types.INT));
        }
      };
    }
  }

  @Test
  public void predeclaredBuiltinTypes() throws Exception {
    module =
        Module.withPredeclared(
            StarlarkSemantics.DEFAULT,
            ImmutableMap.of(
                "my_type_value",
                new MyType(),
                "my_type_subclass_value",
                new MyTypeSubclass(),
                "my_self_call_value",
                new MySelfCallType(),
                "my_explicitly_typed_value",
                new MyExplicitlyTypedType(),
                "my_explicitly_typed_self_call_value",
                new MyExplicitlyTypedSelfCallType()));

    assertValid(
        """
        a: int = my_type_value.foo()
        b: int = my_type_subclass_value.foo()
        c: int = my_self_call_value()
        d: int = my_self_call_value.bar()
        e: int = my_explicitly_typed_value.some_field  # typed as struct-of-Any
        f: int = my_explicitly_typed_self_call_value()
        """);

    assertInvalid("cannot assign type 'MyType' to 'x' of type 'str'", "x: str = my_type_value");
    assertInvalid(
        "cannot assign type 'MySelfCallType' to 'x' of type 'str'", "x: str = my_self_call_value");
    assertInvalid("cannot assign type 'int' to 'x' of type 'str'", "x: str = my_self_call_value()");
    assertInvalid(
        "cannot assign type 'ExplicitlyTypedSelfCall' to 'x' of type 'str'",
        "x: str = my_explicitly_typed_self_call_value");
    assertInvalid(
        "cannot assign type 'int' to 'x' of type 'float'",
        "x: float = my_explicitly_typed_self_call_value()");
    assertInvalid(
        "cannot assign type 'struct' to 'x' of type 'str'", "x: str = my_explicitly_typed_value");

    assertInvalid(
        "'my_type_value' of type 'MyType' does not have field 'bar'",
        "_: str = my_type_value.bar()");
    assertInvalid("'my_type_value' is not callable; got type 'MyType'", "_: str = my_type_value()");
  }

  @Test
  public void unannotatedStarlarkValues_notAutoTyped() throws Exception {
    module =
        Module.withPredeclared(
            StarlarkSemantics.DEFAULT,
            ImmutableMap.of("unannotated_value", new MyUnannotatedType()));
    // MyUnannotatedType has no @StarlarkBuiltin-annotated ancestor, so there's no
    // StarlarkBuiltinAutoType generated for it; therefore, unannotated_value is typed as Object.
    assertInvalid("cannot assign type 'object' to 'x' of type 'str'", "x: str = unannotated_value");
  }

  @Test
  public void subclassesShareSameAutoType() throws Exception {
    module =
        Module.withPredeclared(
            StarlarkSemantics.DEFAULT,
            ImmutableMap.of(
                "my_type_value", new MyType(), "my_type_subclass_value", new MyTypeSubclass()));
    assertValid(
        """
        _: None  # ensure file uses type syntax
        list_a = [my_type_value]  # inferred as list[MyType]
        list_a[0] = my_type_subclass_value
        list_b = [my_type_subclass_value]  # also inferred as list[MyType]
        list_b[0] = my_type_value
        """);

    assertInvalid(
        "cannot assign type 'MyType' to 'x' of type 'str'", "x: str = my_type_subclass_value");
  }

  @StarlarkBuiltin(name = "SelfReferentialType")
  public static final class SelfReferentialType implements StarlarkValue {
    @StarlarkMethod(
        name = "self",
        doc = "...",
        parameters = {@Param(name = "x", doc = "...")})
    public SelfReferentialType self(SelfReferentialType x) {
      return x;
    }
  }

  @StarlarkBuiltin(name = "MutuallyReferentialTypeA")
  public static final class MutuallyReferentialTypeA implements StarlarkValue {
    @StarlarkMethod(
        name = "b",
        doc = "...",
        parameters = {@Param(name = "x", doc = "...")})
    public MutuallyReferentialTypeB b(MutuallyReferentialTypeB x) {
      return x;
    }
  }

  @StarlarkBuiltin(name = "MutuallyReferentialTypeB")
  public static final class MutuallyReferentialTypeB implements StarlarkValue {
    @StarlarkMethod(
        name = "a",
        doc = "...",
        parameters = {@Param(name = "x", doc = "...")})
    public MutuallyReferentialTypeA a(MutuallyReferentialTypeA x) {
      return x;
    }
  }

  @Test
  public void selfReferentialTypes() throws Exception {
    // Test types of @StalarkBuiltin-annotated classes whose methods depend on the type itself
    // (whether directly or transitively).
    module =
        Module.withPredeclared(
            StarlarkSemantics.DEFAULT,
            ImmutableMap.of(
                "self_ref", new SelfReferentialType(),
                "mutually_ref_a", new MutuallyReferentialTypeA(),
                "mutually_ref_b", new MutuallyReferentialTypeB()));
    assertValid(
        """
        x = self_ref.self(self_ref)
        y = mutually_ref_a.b(mutually_ref_b).a(mutually_ref_a)
        _unused: bool = True  # ensure file uses type syntax
        """);

    assertInvalid(
        "in call to 'self_ref.self()', parameter 'x' got value of type 'int', want"
            + " 'SelfReferentialType'",
        """
        self_ref.self(123)
        _unused: bool = True
        """);

    assertInvalid(
        "cannot assign type 'SelfReferentialType' to 'x' of type 'int'",
        """
        x: int = self_ref.self(self_ref)
        """);

    assertInvalid(
        "in call to 'mutually_ref_a.b()', parameter 'x' got value of type 'int', want"
            + " 'MutuallyReferentialTypeB'",
        """
        mutually_ref_a.b(123)
        _unused: bool = True
        """);

    assertInvalid(
        "cannot assign type 'MutuallyReferentialTypeB' to 'x' of type 'int'",
        """
        x: int = mutually_ref_a.b(mutually_ref_b)
        """);

    assertInvalid(
        "in call to 'mutually_ref_b.a()', parameter 'x' got value of type 'int', want"
            + " 'MutuallyReferentialTypeA'",
        """
        mutually_ref_b.a(123)
        _unused: bool = True
        """);

    assertInvalid(
        "cannot assign type 'MutuallyReferentialTypeA' to 'x' of type 'int'",
        """
        x: int = mutually_ref_b.a(mutually_ref_a)
        """);
  }

  @Test
  public void typeAlias_canBeExportedAndLoaded() throws Exception {
    Module depModule = Module.create();
    try (Mutability depMutability = Mutability.create("dep")) {
      StarlarkThread depThread = StarlarkThread.createTransient(depMutability, semantics);
      var unused =
          Starlark.execFile(
              ParserInput.fromLines(
                  """
                  type int_or_str = int | str
                  type optional_list_of[T] = list[T] | None
                  """),
              options.build(),
              depModule,
              depThread);
    }

    loader = name -> name.equals("dep.bzl") ? depModule : null;

    assertValid(
        """
        load("dep.bzl", "int_or_str", "optional_list_of")
        x: int_or_str = 123
        y: optional_list_of[int] = [123]
        """);

    assertInvalid(
        "cannot assign type 'bool' to 'x' of type 'int|str'",
        """
        load("dep.bzl", "int_or_str")
        x: int_or_str = False
        """);

    assertInvalid(
        "cannot assign type 'list[str]' to 'x' of type 'list[int]|None'",
        """
        load("dep.bzl", "optional_list_of")
        x: optional_list_of[int] = ["abc"]
        """);
  }

  @Test
  public void nonTypeConstructorLoadedValues_cannotBeUsedAsTypeConstructors() throws Exception {
    Module depModule = Module.create();
    try (Mutability depMutability = Mutability.create("dep")) {
      StarlarkThread depThread = StarlarkThread.createTransient(depMutability, semantics);
      var unused =
          Starlark.execFile(
              ParserInput.fromLines("not_a_type = 123"), options.build(), depModule, depThread);
    }

    loader = name -> name.equals("dep.bzl") ? depModule : null;

    assertInvalid(
        "local symbol 'not_a_type' cannot be used as a type",
        """
        load("dep.bzl", "not_a_type")
        x: not_a_type = 123
        """);
  }
}
