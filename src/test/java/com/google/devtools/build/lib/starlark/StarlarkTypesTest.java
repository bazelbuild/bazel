// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlark;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.DefaultInfo.DefaultInfoProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.SelectorValue;
import com.google.devtools.build.lib.skyframe.BzlCompileValue;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.function.Predicate;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.CallUtils;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.AssignmentStatement;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.TypeContext;
import net.starlark.java.syntax.Types;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for Starlark types. */
@RunWith(TestParameterInjector.class)
public class StarlarkTypesTest extends BuildViewTestCase {

  /**
   * Given a load label string for a .bzl file, retrieves the resulting {@link Module}. Any
   * exception is propagated.
   */
  private Module loadModule(String label) throws Exception {
    BzlLoadValue.Key key = BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked(label));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), key, /* keepGoing= */ false, reporter);
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key).getModule();
  }

  private Module loadModuleUnchecked(String label) {
    try {
      return loadModule(label);
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Writes the given Starlark code to a .bzl file in a new temporary directory, and creates an
   * empty BUILD file in the same directory in order to make the .bzl file loadable.
   *
   * @return the load label string of the .bzl file
   */
  private String writeTempBzlFile(String... lines) throws IOException {
    String tempDir = rootDirectory.createTempDirectory("test").getBaseName();
    scratch.file(tempDir + "/BUILD");
    scratch.file(tempDir + "/test.bzl", lines);
    return String.format("//%s:test.bzl", tempDir);
  }

  /**
   * Evaluates the given Starlark expression in a .bzl module (with an optional prefix containing
   * load() statements, etc.), and returns the type under which it would be exported by a module to
   * other .bzl files. (This is type is based on the dynamic type of the expression's value.)
   */
  private StarlarkType revealExportedType(String expr, String... prefixLines) throws Exception {
    StringBuilder lines = new StringBuilder();
    for (String prefixLine : prefixLines) {
      lines.append(prefixLine).append("\n");
    }
    lines.append("VAL = ").append(expr).append("\n");
    Module module = loadModule(writeTempBzlFile(lines.toString()));
    return module.getExportType("VAL");
  }

  /**
   * Parses the given Starlark type expression in a .bzl module (with an optional prefix for load()
   * statements, etc.), and returns its (type) value.
   */
  private StarlarkType parseType(String expr, String... prefixLines) throws Exception {
    StringBuilder lines = new StringBuilder();
    for (String prefixLine : prefixLines) {
      lines.append(prefixLine).append("\n");
    }
    // To parse the type expression, print it as part of a type alias statement, extract the
    // TypeConstructor from the alias's TypeConstructorValue, and apply the TypeConstructor with
    // zero arguments.
    lines.append("type TYPE = ").append(expr).append("\n");
    Module module = loadModule(writeTempBzlFile(lines.toString()));
    return module.getExportTypeConstructor("TYPE").createStarlarkType();
  }

  /**
   * Evaluates the given Starlark expression in a .bzl module (with an optional prefix for load()
   * statements, etc.), and returns the type that had been inferred for the expression by the static
   * type checker.
   *
   * <p>Even though this method returns the static type of the expression, it also evaluates the
   * .bzl code as a side effect, and throws an exception if there is a dynamic error.
   */
  private StarlarkType revealStaticType(String expr, String... prefixLines) throws Exception {
    StringBuilder lines = new StringBuilder();
    for (String prefixLine : prefixLines) {
      lines.append(prefixLine).append("\n");
    }
    lines.append("_: Any  # enable type checker\n");
    lines.append("VAL = ").append(expr).append("\n");
    String labelString = writeTempBzlFile(lines.toString());

    // BzlLoadFunction throws away the type checking results, so we cannot retrieve the type from
    // the Module directly. However, we still need the predeclared symbols that BzlLoadFunction
    // placed in the module for us.
    // As a side effect, this also verifies that the expression is dynamically valid.
    ImmutableMap<String, Object> predeclareds =
        ImmutableMap.copyOf(loadModule(labelString).getPredeclaredBindings());

    BzlCompileValue.Key key = BzlCompileValue.key(root, Label.parseCanonicalUnchecked(labelString));
    EvaluationResult<BzlCompileValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), key, /* keepGoing= */ false, reporter);
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    // Replicates how BzlLoadFunction invokes the type checker.
    Program program =
        Starlark.withTypeInfo(
            result.get(key).getProgram(),
            Module.withPredeclared(getStarlarkSemantics(), predeclareds),
            /* staticTypeChecking= */ true,
            this::loadModuleUnchecked);
    // Find the statically inferred type of `VAL` in the `VAL = <...>` assignment statement which is
    // the last line of the program.
    AssignmentStatement assign =
        (AssignmentStatement) program.getResolvedFunction().getBody().getLast();
    return program.getTypeTable().getType(((Identifier) assign.getLHS()).getBinding());
  }

  /** Checks that a .bzl file consisting of the given lines can be loaded. */
  private void checkValid(String... lines) throws Exception {
    String labelString = writeTempBzlFile(lines);
    var _ = loadModule(labelString);
  }

  /**
   * Checks that the a .bzl file consisting of the given lines fails to load with the given error.
   *
   * <p>This method removes {@link #failFastHandler} and adds events to the {@link #eventCollector}.
   */
  private void checkInvalid(String expectedError, String... lines) throws Exception {
    reporter.removeHandler(failFastHandler);
    int initialEventCount = eventCollector.count();
    String labelString = writeTempBzlFile(lines);
    BzlLoadValue.Key key = BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked(labelString));
    EvaluationResult<BzlLoadValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), key, /* keepGoing= */ false, reporter);
    assertThat(result.hasError()).isTrue();
    Iterable<Event> newEvents = Iterables.skip(eventCollector, initialEventCount);
    MoreAsserts.assertContainsEvent(newEvents, expectedError);
  }

  /** Asserts {@code t1} is assignable to {@code t2}. */
  private void assertLt(String context, StarlarkType t1, StarlarkType t2) {
    assertWithMessage(
            "%s%s%s (class %s) is expected to be assignable to %s (class %s)",
            context,
            context.isEmpty() ? "" : ": ",
            t1.typeRepr(),
            t1.getClass(),
            t2.typeRepr(),
            t2.getClass())
        .that(StarlarkType.assignableFrom(t2, t1, getTypeContext()))
        .isTrue();
  }

  private void assertLt(StarlarkType t1, StarlarkType t2) {
    assertLt("", t1, t2);
  }

  /** Asserts {@code t1} is *not* assignable to {@code t2}. */
  private void assertNotLt(String context, StarlarkType t1, StarlarkType t2) {
    assertWithMessage(
            "%s%s%s (class %s) is expected to be not assignable to %s (class %s)",
            context,
            context.isEmpty() ? "" : ": ",
            t1.typeRepr(),
            t1.getClass(),
            t2.typeRepr(),
            t2.getClass())
        .that(StarlarkType.assignableFrom(t2, t1, getTypeContext()))
        .isFalse();
  }

  private void assertNotLt(StarlarkType t1, StarlarkType t2) {
    assertNotLt("", t1, t2);
  }

  /** Asserts {@code t1} is assignable to {@code t2}, but not vice versa. */
  @SuppressWarnings("unused")
  private void assertStrictLt(String context, StarlarkType t1, StarlarkType t2) {
    assertLt(context, t1, t2);
    assertNotLt(context, t2, t1);
  }

  private void assertStrictLt(StarlarkType t1, StarlarkType t2) {
    assertLt(t1, t2);
    assertNotLt(t2, t1);
  }

  /** Asserts {@code t1} and {@code t2} are assignable in both directions. */
  private void assertLtAndGt(String context, StarlarkType t1, StarlarkType t2) {
    assertLt(context, t1, t2);
    assertLt(context, t2, t1);
  }

  private void assertLtAndGt(StarlarkType t1, StarlarkType t2) {
    assertLt(t1, t2);
    assertLt(t2, t1);
  }

  /** Asserts that the given types are *not* assignable in either direction. */
  private void assertIncomparable(String context, StarlarkType... types) {
    for (int i = 0; i < types.length - 1; i++) {
      for (int j = i + 1; j < types.length; j++) {
        assertNotLt(context, types[i], types[j]);
        assertNotLt(context, types[j], types[i]);
      }
    }
  }

  private void assertIncomparable(StarlarkType... types) {
    assertIncomparable("", types);
  }

  @StarlarkBuiltin(name = "TestStructApiImpl")
  private static final class TestStructApiImpl implements StructApi {
    private final int answer;

    private TestStructApiImpl(int answer) {
      this.answer = answer;
    }

    @StarlarkMethod(name = "answer", doc = "A field", structField = true)
    public int answer() {
      return answer;
    }

    @StarlarkMethod(
        name = "plus",
        doc = "Not a field",
        parameters = {
          @Param(name = "other"),
        })
    public TestStructApiImpl plus(TestStructApiImpl other) {
      return new TestStructApiImpl(this.answer + other.answer);
    }

    @StarlarkMethod(
        name = "plus_or",
        doc = "Not a field",
        parameters = {
          @Param(name = "other"),
        })
    public StructApi abstractPlus(StructApi other) {
      if (other instanceof TestStructApiImpl otherTestStruct) {
        return new TestStructApiImpl(this.answer + otherTestStruct.answer);
      } else {
        return other;
      }
    }
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addBzlToplevel("test_struct_api_impl", new TestStructApiImpl(42));
    return builder.build();
  }

  @Test
  public void experimentalStarlarkTypes_on_allowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=//test");
    scratch.file(
        "test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.bzl', 'f')");

    getTarget("//test:BUILD");

    assertNoEvents();
  }

  @Test
  public void experimentalStarlarkTypes_off_disallowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--noexperimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=//test");
    scratch.file(
        "test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.bzl', 'f')");

    checkLoadingPhaseError("//test:BUILD", "syntax error at ':': type annotations are disallowed");
    assertContainsEvent(
        "Type annotations syntax can be enabled with --experimental_starlark_type_syntax and/or"
            + " --experimental_starlark_types_allowed_paths.");
  }

  @Test
  public void experimentalStarlarkTypes_prohibitedInSclRegardlessOfFlag() throws Exception {
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.scl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.scl', 'f')");

    checkLoadingPhaseError("//test:BUILD", "syntax error at ':': type annotations are disallowed");
    assertContainsEvent("Type annotations are not permitted in .scl files.");
  }

  @Test
  public void starlarkTypesAllowedPath_notOnPath_disallowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=//main");
    scratch.file(
        "test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.bzl', 'f')");

    checkLoadingPhaseError("//test:BUILD", "syntax error at ':': type annotations are disallowed");
    assertContainsEvent(
        "Type annotations syntax can be enabled with --experimental_starlark_type_syntax and/or"
            + " --experimental_starlark_types_allowed_paths.");
  }

  @Test
  public void starlarkTypesAllowedPath_externalPath_allowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=@@r+//test");
    scratch.overwriteFile(
        "MODULE.bazel", "bazel_dep(name='r')", "local_path_override(module_name='r', path='/r')");
    scratch.file("/r/MODULE.bazel", "module(name='r')");
    scratch.file(
        "/r/test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("/r/test/BUILD", "load(':foo.bzl', 'f')");

    // Required since we have a new MODULE.bazel file.
    invalidatePackages(true);
    getTarget("@@r+//test:BUILD");

    assertNoEvents();
  }

  @Test
  public void typeResolverRunsByDefault() throws Exception {
    scratch.file(
        "test/foo.bzl",
        """
        def f():
            x = 1
            x : int
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        """);

    checkLoadingPhaseError(
        "//test:BUILD", "type annotation on 'x' may only appear at its declaration");
  }

  @Test
  public void typeResolverDoesRunWithOnlyDynamicTypeCheckingFlag() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax",
        "--noexperimental_starlark_static_type_checking",
        "--experimental_starlark_dynamic_type_checking");
    scratch.file(
        "test/foo.bzl",
        """
        def f():
            x = 1
            x : int
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        """);

    checkLoadingPhaseError(
        "//test:BUILD", "type annotation on 'x' may only appear at its declaration");
  }

  @Test
  public void staticTypeCheckingRunsByDefault() throws Exception {
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.bzl",
        """
        x: int = "a"
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "x")
        """);

    checkLoadingPhaseError("//test:BUILD", "cannot assign type 'str' to 'x' of type 'int'");
  }

  @Test
  public void dynamicTypeCheckingDoesNotRunByDefault() throws Exception {
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.bzl",
        """
        def f(x: int):
            pass
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        f("abc")
        """);

    getTarget("//test:BUILD");
    assertNoEvents();
  }

  @Test
  public void dynamicTypeCheckingDoesRunWithDynamicTypeCheckingFlag() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_dynamic_type_checking");
    scratch.file(
        "test/foo.bzl",
        """
        def f(x: int):
            pass
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        f("abc")
        """);

    reporter.removeHandler(failFastHandler);
    getTarget("//test:BUILD");
    assertContainsEvent("in call to f(), parameter 'x' got value of type 'str', want 'int'");
  }

  @Test
  public void structConstructor() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    assertThat(revealStaticType("struct(x = 1, y = 2)")).isEqualTo(Types.ANY_STRUCT);

    // Type is narrowed on export
    assertThat(revealExportedType("struct(x = 1, y = 2)"))
        .isEqualTo(Types.struct(ImmutableMap.of("x", Types.INT, "y", Types.INT)));

    // struct constructor (the singleton StructProvider instance) is a provider
    StarlarkType structSymbolType = revealStaticType("struct");
    StarlarkType anyProviderType = parseType("Provider");
    assertLtAndGt(structSymbolType, anyProviderType);
  }

  @Test
  public void structApiImplementations_assignableToStructType() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    assertThat(revealStaticType("test_struct_api_impl.plus(test_struct_api_impl)"))
        .isNotEqualTo(Types.ANY_STRUCT);

    checkValid(
        """
        good: struct[{"answer": int}] = test_struct_api_impl.plus(test_struct_api_impl)
        """);

    checkInvalid(
        "cannot assign type 'TestStructApiImpl' to 'bad' of type 'struct[{\"answer\": float}]'",
        """
        bad: struct[{"answer": float}] = test_struct_api_impl.plus(test_struct_api_impl)
        """);
  }

  @Test
  public void structApiItself_isAnyStruct() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    assertThat(revealStaticType("test_struct_api_impl.plus_or(struct())"))
        .isEqualTo(Types.ANY_STRUCT);
  }

  private void doTypeConstructorUsableTest(String pkgName, String typeExpr, String valueExpr)
      throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file(pkgName + "/lib/lib.bzl", "value = " + valueExpr);
    scratch.file(pkgName + "/lib/BUILD");

    checkValid(
        String.format(
            """
            load("//%s/lib:lib.bzl", "value")
            good: %s = value
            """,
            pkgName, typeExpr));

    checkInvalid(
        String.format("cannot assign type '%s' to 'bad' of type 'None'", typeExpr),
        String.format(
            """
            load("//%s/lib:lib.bzl", "value")
            bad: None = value
            """,
            pkgName));
  }

  @Test
  public void selfReferentialValue_exportedType() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    assertThat(
            revealExportedType(
                "self_referential_struct",
                """
                self_referential_struct = struct(data = [])
                self_referential_struct.data.append(self_referential_struct)
                """))
        .isEqualTo(Types.ANY_STRUCT);

    // A deeply nested cycle is caught only at the outermost level.
    assertThat(
            revealExportedType(
                "struct_with_deeply_nested_self_referential_field",
                """
                struct_with_deeply_nested_self_referential_field = struct(data = {"x": []})
                struct_with_deeply_nested_self_referential_field.data["x"].append(
                    struct_with_deeply_nested_self_referential_field.data["x"]
                )
                """))
        .isEqualTo(Types.ANY_STRUCT);

    assertThat(
            revealExportedType(
                "self_referential_list",
                """
                self_referential_list = []
                cast(list, self_referential_list).append(self_referential_list)
                """))
        .isEqualTo(Types.list(Types.ANY));

    assertThat(
            revealExportedType(
                "self_referential_dict",
                """
                self_referential_dict = {}
                cast(dict, self_referential_dict)["self"] = self_referential_dict
                """))
        .isEqualTo(Types.dict(Types.ANY, Types.ANY));
  }

  @Test
  public void autogeneratedTypeConstructors_usable() throws Exception {
    doTypeConstructorUsableTest("label", "Label", "Label('//foo:bar')");
    doTypeConstructorUsableTest("depset", "depset", "depset([1, 2])");
    doTypeConstructorUsableTest("exec_group", "exec_group", "exec_group()");
  }

  @Test
  public void builtinExtraTypes_usable() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    checkValid(
        """
        def impl(ctx: Ctx):
            f: File = ctx.actions.declare_file("out")
            ctx.actions.write(f, "content")
            a: Args = ctx.actions.args()
            r: Runfiles = ctx.runfiles()
            root: Root = f.root
            if ctx.attr.deps:
                t: Target = ctx.attr.deps[0]
            return []
        """);

    checkInvalid(
        "cannot assign type 'File' to 'bad' of type 'Ctx'",
        """
        def _impl(ctx: Ctx):
            f: File = ctx.actions.declare_file("out")
            # Wrong type assignment
            bad: Ctx = f
            return []
        """);
  }

  @Test
  public void select_basicUsage() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file(
        "lib/select.bzl",
        """
        select_of_int = select({"//cfg": 1, "//conditions:default": 0})
        select_of_str = select({"//cfg": "foo", "//conditions:default": "bar"})
        select_of_empty_list = select({"//cfg": [], "//conditions:default": []})
        select_of_str_list = select({"//cfg": ["a", "b"], "//conditions:default": ["c"]})
        select_of_potentially_empty_label_list = select({"//cfg": [Label("//foo:a")], "//conditions:default": []})
        select_of_potentially_empty_dict = select({"//cfg": {"a": Label("//foo:a")}, "//conditions:default": {}})
        select_of_label_or_none = select({"//cfg": Label("//foo:a"), "//conditions:default": None})
        """);
    scratch.file("lib/BUILD");

    StarlarkType labelType = parseType("Label");

    assertThat(
            revealStaticType(
                "select_of_int + select_of_int", "load('//lib:select.bzl', 'select_of_int')"))
        .isEqualTo(SelectorValue.Type.of(Types.INT));
    assertThat(
            revealStaticType(
                "select_of_str + 'hello'", "load('//lib:select.bzl', 'select_of_str')"))
        .isEqualTo(SelectorValue.Type.of(Types.STR));
    assertThat(
            revealStaticType(
                "['hello'] + select_of_empty_list + select_of_str_list + []",
                "load('//lib:select.bzl', 'select_of_empty_list', 'select_of_str_list')"))
        .isEqualTo(SelectorValue.Type.of(Types.list(Types.STR)));
    assertThat(
            revealStaticType(
                "[] + select_of_potentially_empty_label_list + [Label('//x')]",
                "load('//lib:select.bzl', 'select_of_potentially_empty_label_list')"))
        .isEqualTo(SelectorValue.Type.of(Types.list(labelType)));
    assertThat(
            revealStaticType(
                "select_of_potentially_empty_dict | {'b': Label('//foo:b')}",
                "load('//lib:select.bzl', 'select_of_potentially_empty_dict')"))
        .isEqualTo(SelectorValue.Type.of(Types.dict(Types.STR, labelType)));
    assertThat(
            revealStaticType(
                "select_of_label_or_none", "load('//lib:select.bzl', 'select_of_label_or_none')"))
        .isEqualTo(SelectorValue.Type.of(labelType));
  }

  @Test
  public void select_typeIsInterned() throws Exception {
    assertThat(SelectorValue.Type.of(Types.INT)).isSameInstanceAs(SelectorValue.Type.of(Types.INT));
  }

  @Test
  public void select_validBinaryOperator() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    scratch.file(
        "lib/lib.bzl",
        """
        select_of_int = select({"//cfg": 1, "//conditions:default": 0})
        select_of_str = select({"//cfg": "foo", "//conditions:default": "bar"})
        select_of_list_of_str = select({"//cfg": ["foo"], "//conditions:default": ["bar"]})
        select_of_list_of_label = select({"//cfg": [Label("//a")], "//conditions:default": [Label("//b")]})
        select_of_dict_of_str = select({"//cfg": {"a": "A"}, "//conditions:default": None})
        """);
    scratch.file("lib/BUILD");

    assertThat(revealStaticType("select_of_int + 1", "load('//lib:lib.bzl', 'select_of_int')"))
        .isEqualTo(parseType("select[int]"));
    assertThat(
            revealStaticType("'hello' + select_of_str", "load('//lib:lib.bzl', 'select_of_str')"))
        .isEqualTo(parseType("select[str]"));
    assertThat(
            revealStaticType(
                "select_of_list_of_str + select_of_list_of_label",
                "load('//lib:lib.bzl', 'select_of_list_of_str', 'select_of_list_of_label')"))
        .isEqualTo(parseType("select[list[str | Label]]"));
    assertThat(
            revealStaticType(
                "select_of_dict_of_str | {'y': Label('//foo')}",
                "load('//lib:lib.bzl', 'select_of_dict_of_str')"))
        .isEqualTo(parseType("select[dict[str, str | Label]]"));
  }

  @Test
  public void select_invalidBinaryOperator() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    scratch.file(
        "lib/lib.bzl",
        """
        select_of_int = select({"//cfg": 1, "//conditions:default": 0})
        select_of_str = select({"//cfg": "foo", "//conditions:default": "bar"})
        select_of_list_of_str = select({"//cfg": ["foo"], "//conditions:default": ["bar"]})
        select_of_dict = select({"//cfg": {"a": 1}, "//conditions:default": None})
        """);
    scratch.file("lib/BUILD");

    checkInvalid(
        "'+' cannot be applied to types 'select[int]' and 'float'",
        """
        load("//lib:lib.bzl", "select_of_int")
        bad = select_of_int + 1.5
        _: None = None  # enable type syntax
        """);

    checkInvalid(
        "'+' cannot be applied to types 'select[int]' and 'select[str]'",
        """
        load("//lib:lib.bzl", "select_of_int", "select_of_str")
        bad = select_of_int + select_of_str
        _: None = None  # enable type syntax
        """);

    checkInvalid(
        "'+' cannot be applied to types 'select[str]' and 'select[list[str]]'",
        """
        load("//lib:lib.bzl", "select_of_str", "select_of_list_of_str")
        bad = select_of_str + select_of_list_of_str
        _: None = None  # enable type syntax
        """);

    checkInvalid(
        "'|' cannot be applied to types 'select[list[str]]' and 'select[dict[str, int]]'",
        """
        load("//lib:lib.bzl", "select_of_list_of_str", "select_of_dict")
        bad = select_of_list_of_str | select_of_dict
        _: None = None  # enable type syntax
        """);
  }

  /** Fake type for testing; returns itself when added to the given type on the given side. */
  private static final class AdditiveType extends StarlarkType {
    private final String name;
    private final Predicate<StarlarkType> addsTo;
    private final StarlarkType result;
    private final boolean thisLeft;

    AdditiveType(
        String name, Predicate<StarlarkType> addsTo, StarlarkType result, boolean thisLeft) {
      this.name = name;
      this.addsTo = addsTo;
      this.result = result;
      this.thisLeft = thisLeft;
    }

    @Override
    public String typeRepr() {
      return name;
    }

    @Override
    public StarlarkType inferBinaryOperator(
        TokenKind operator, StarlarkType that, boolean thisLeft) {
      if (operator == TokenKind.PLUS && addsTo.test(that) && thisLeft == this.thisLeft) {
        return result;
      }
      return null;
    }
  }

  /**
   * Verifies that {@link StarlarkType#inferBinaryOperator} produces the given result when called
   * with selector types of the given arguments, or with the selector type of one argument and a
   * plain type of the other.
   */
  private void assertSelectBinaryOperator(
      StarlarkType lhsArgToSelec,
      TokenKind operator,
      StarlarkType rhsArgToSelect,
      StarlarkType expectedArgToSelect) {
    assertThat(
            StarlarkType.inferBinaryOperator(
                SelectorValue.Type.of(lhsArgToSelec),
                operator,
                SelectorValue.Type.of(rhsArgToSelect)))
        .isEqualTo(SelectorValue.Type.of(expectedArgToSelect));
    assertThat(
            StarlarkType.inferBinaryOperator(
                SelectorValue.Type.of(lhsArgToSelec), operator, rhsArgToSelect))
        .isEqualTo(SelectorValue.Type.of(expectedArgToSelect));
    assertThat(
            StarlarkType.inferBinaryOperator(
                lhsArgToSelec, operator, SelectorValue.Type.of(rhsArgToSelect)))
        .isEqualTo(SelectorValue.Type.of(expectedArgToSelect));
  }

  @Test
  public void select_inferBinaryOperator() throws Exception {
    // Basic cases
    assertThat(
            SelectorValue.Type.of(Types.INT).inferBinaryOperator(TokenKind.PLUS, Types.INT, true))
        .isEqualTo(SelectorValue.Type.of(Types.INT));
    assertThat(
            SelectorValue.Type.of(Types.list(Types.INT))
                .inferBinaryOperator(TokenKind.PLUS, Types.list(Types.STR), true))
        .isEqualTo(SelectorValue.Type.of(Types.list(Types.union(Types.INT, Types.STR))));

    // Distinct types which add to str on the left, producing int and float respectively.
    AdditiveType l1 = new AdditiveType("l1", Predicates.equalTo(Types.STR), Types.INT, true);
    AdditiveType l2 = new AdditiveType("l2", Predicates.equalTo(Types.STR), Types.FLOAT, true);
    StarlarkType l1or2 = Types.union(l1, l2);
    assertSelectBinaryOperator(l1or2, TokenKind.PLUS, Types.STR, Types.NUMERIC);

    // Type which adds to l1 or l2 on the right, producing bool.
    AdditiveType r1 = new AdditiveType("r1", t -> t.equals(l1) || t.equals(l2), Types.BOOL, false);
    StarlarkType r1orStr = Types.union(r1, Types.STR);
    StarlarkType boolOrNumeric = Types.union(Types.BOOL, Types.NUMERIC);
    assertSelectBinaryOperator(l1or2, TokenKind.PLUS, r1orStr, boolOrNumeric);

    // Special case: adding a select of a numeric type and a different numeric type is a dynamic
    // error, so we forbid it statically too.
    assertThat(
            SelectorValue.Type.of(Types.FLOAT)
                .inferBinaryOperator(TokenKind.PLUS, Types.INT, false))
        .isNull();
    assertThat(
            SelectorValue.Type.of(Types.INT)
                .inferBinaryOperator(TokenKind.PLUS, Types.FLOAT, false))
        .isNull();
  }

  @Test
  public void schemalessProvider() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    TypeContext typeContext = getTypeContext();

    scratch.file(
        "lib/provider.bzl",
        """
        MyInfo = provider()
        MyOtherInfo = provider()
        """);
    scratch.file("lib/BUILD");
    Module providerBzl = loadModule("//lib:provider.bzl");
    StarlarkType myInfoSymbolType = providerBzl.getExportType("MyInfo");
    StarlarkType myOtherInfoSymbolType = providerBzl.getExportType("MyOtherInfo");

    // Provider symbols can be used as provider symbols, as callables, or as types.
    assertThat(myInfoSymbolType.typeRepr()).isEqualTo("<Provider[MyInfo]>");
    assertLtAndGt(myInfoSymbolType, parseType("Provider"));
    assertStrictLt(myInfoSymbolType, Types.ANY_CALLABLE);
    assertStrictLt(myInfoSymbolType, Types.TYPE);
    assertIncomparable(myInfoSymbolType, myOtherInfoSymbolType);

    StarlarkType myInfoType = parseType("MyInfo", "load('//lib:provider.bzl', 'MyInfo')");
    assertThat(myInfoType.typeRepr()).isEqualTo("MyInfo");
    StarlarkType myOtherInfoType =
        parseType("MyOtherInfo", "load('//lib:provider.bzl', 'MyOtherInfo')");

    assertStrictLt(myInfoType, Types.ANY_STRUCT);
    assertIncomparable(myInfoType, myOtherInfoType);
    // Allow all field access.
    assertThat(myInfoType.getField("nonexistent field", typeContext)).isEqualTo(Types.ANY);

    StarlarkType instanceTypeStatically =
        revealStaticType("MyInfo(foobar = 'abc')", "load('//lib:provider.bzl', 'MyInfo')");
    assertThat(instanceTypeStatically).isEqualTo(myInfoType);

    StarlarkType instanceTypeDynamically =
        revealExportedType("MyInfo(foobar = 'abc')", "load('//lib:provider.bzl', 'MyInfo')");
    assertThat(instanceTypeDynamically).isEqualTo(myInfoType);
  }

  @Test
  public void schemafulProvider_noInit() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    TypeContext typeContext = getTypeContext();

    scratch.file(
        "lib/provider.bzl",
        """
        MyInfo = provider(fields = ["x", "y"])
        MyOtherInfo = provider(fields = ["x", "y"])
        """);
    scratch.file("lib/BUILD");
    Module providerBzl = loadModule("//lib:provider.bzl");
    StarlarkType myInfoSymbolType = providerBzl.getExportType("MyInfo");
    StarlarkType myOtherInfoSymbolType = providerBzl.getExportType("MyOtherInfo");

    // Provider symbols can be used as provider symbols, as callables, or as types.
    assertLtAndGt(myInfoSymbolType, parseType("Provider"));
    assertStrictLt(
        myInfoSymbolType,
        revealStaticType(
            "callable_with_MyInfo_signature",
            """
            load("//lib:provider.bzl", "MyInfo")

            def callable_with_MyInfo_signature(*, x=None, y=None) -> MyInfo:
                return MyInfo(x = x, y = y)
            """));
    assertStrictLt(myInfoSymbolType, Types.TYPE);
    assertIncomparable(myInfoSymbolType, myOtherInfoSymbolType);

    StarlarkType myInfoType = parseType("MyInfo", "load('//lib:provider.bzl', 'MyInfo')");
    StarlarkType myOtherInfoType =
        parseType("MyOtherInfo", "load('//lib:provider.bzl', 'MyOtherInfo')");

    assertStrictLt(myInfoType, Types.struct(ImmutableMap.of("x", Types.ANY, "y", Types.ANY)));
    assertIncomparable(myInfoType, myOtherInfoType);
    assertThat(myInfoType.getField("y", typeContext)).isEqualTo(Types.ANY);
    assertThat(myInfoType.getField("nonexistent field", typeContext)).isNull();

    StarlarkType instanceTypeStatically =
        revealStaticType("MyInfo(x = 1)", "load('//lib:provider.bzl', 'MyInfo')");
    assertThat(instanceTypeStatically).isEqualTo(myInfoType);

    StarlarkType instanceTypeDynamically =
        revealExportedType("MyInfo(x = 1)", "load('//lib:provider.bzl', 'MyInfo')");
    assertThat(instanceTypeDynamically).isEqualTo(myInfoType);
  }

  @Test
  public void schemafulProvider_withInit() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    TypeContext typeContext = getTypeContext();

    scratch.file(
        "lib/provider.bzl",
        """
        def init(a: int, *, b: str = "", **kwargs):
            return {"x": a, "y": b}

        MyInfo, _new_MyInfo = provider(fields = ["x", "y"], init = init)
        MyOtherInfo, _new_MyOtherInfo = provider(fields = ["x", "y"], init = init)
        """);
    scratch.file("lib/BUILD");
    Module providerBzl = loadModule("//lib:provider.bzl");
    StarlarkType myInfoSymbolType = providerBzl.getExportType("MyInfo");
    StarlarkType myOtherInfoSymbolType = providerBzl.getExportType("MyOtherInfo");

    // Provider symbols can be used as provider symbols, as callables, or as types.
    assertLtAndGt(myInfoSymbolType, parseType("Provider"));
    assertStrictLt(
        myInfoSymbolType,
        revealStaticType(
            "callable_with_MyInfo_signature",
            """
            load("//lib:provider.bzl", "MyInfo")

            def callable_with_MyInfo_signature(a: int, *, b: str = "", **kwargs) -> MyInfo:
                return MyInfo(a, b = b, **kwargs)
            """));
    assertStrictLt(myInfoSymbolType, Types.TYPE);
    assertIncomparable(myInfoSymbolType, myOtherInfoSymbolType);

    StarlarkType myInfoType = parseType("MyInfo", "load('//lib:provider.bzl', 'MyInfo')");
    StarlarkType myOtherInfoType =
        parseType("MyOtherInfo", "load('//lib:provider.bzl', 'MyOtherInfo')");

    assertIncomparable(myInfoType, myOtherInfoType);
    // Fields, not init params
    assertStrictLt(myInfoType, Types.struct(ImmutableMap.of("x", Types.ANY, "y", Types.ANY)));
    assertThat(myInfoType.getField("y", typeContext)).isEqualTo(Types.ANY);
    assertThat(myInfoType.getField("nonexistent field", typeContext)).isNull();

    StarlarkType instanceTypeStatically =
        revealStaticType("MyInfo(1)", "load('//lib:provider.bzl', 'MyInfo')");
    assertThat(instanceTypeStatically).isEqualTo(myInfoType);

    StarlarkType instanceTypeDynamically =
        revealExportedType("MyInfo(2)", "load('//lib:provider.bzl', 'MyInfo')");
    assertThat(instanceTypeDynamically).isEqualTo(myInfoType);
  }

  @Test
  public void defaultInfo() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    StarlarkType defaultInfoSymbolType = revealStaticType("DefaultInfo");
    assertThat(defaultInfoSymbolType.typeRepr()).isEqualTo("<Provider[DefaultInfo]>");
    assertStrictLt(defaultInfoSymbolType, Types.ANY_CALLABLE);
    assertStrictLt(defaultInfoSymbolType, Types.TYPE);
    assertLtAndGt(defaultInfoSymbolType, parseType("Provider"));

    StarlarkType defaultInfoType = parseType("DefaultInfo");
    assertThat(revealStaticType("DefaultInfo()")).isEqualTo(defaultInfoType);
    assertThat(revealExportedType("DefaultInfo()")).isEqualTo(defaultInfoType);
    assertThat(defaultInfoType.typeRepr()).isEqualTo("DefaultInfo");
    assertStrictLt(defaultInfoType, Types.ANY_STRUCT);
    assertThat(defaultInfoType.getField("files", getTypeContext()))
        .isEqualTo(parseType("depset | None"));
    assertThat(defaultInfoType.getField("nonexistent field", getTypeContext())).isNull();
  }

  @Test
  public void anyProvider() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    StarlarkType anyProviderType = parseType("Provider");

    assertThat(Starlark.getStarlarkTypeFromJavaType(ProviderApi.class, /* objectIsAny= */ false))
        .isEqualTo(anyProviderType);
    assertStrictLt(anyProviderType, Types.ANY_CALLABLE);
    assertStrictLt(anyProviderType, Types.TYPE);
  }

  @Test
  public void builtinProviders() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    StarlarkType anyProviderType = parseType("Provider");

    ImmutableSet<Class<?>> builtinProviderClasses =
        Classpath.findClasses("com/google/devtools/build").stream()
            .filter(BuiltinProvider.class::isAssignableFrom)
            .collect(toImmutableSet());
    assertThat(builtinProviderClasses).isNotEmpty();
    assertThat(builtinProviderClasses).contains(DefaultInfoProvider.class);
    for (Class<?> cls : builtinProviderClasses) {
      if (isCallableFromStarlark(cls) || cls == BuiltinProvider.class) {
        // Callable from Starlark -> exposed as a Provider (sub)type
        String messagePrefix = "For builtin provider " + cls;
        StarlarkType providerType = Starlark.getStarlarkTypeFromJavaType(cls, false);
        assertLtAndGt(messagePrefix, providerType, anyProviderType);
        assertStrictLt(messagePrefix, providerType, Types.ANY_CALLABLE);
        if (StarlarkAnnotations.getParentWithStarlarkBuiltin(cls) != ProviderApi.class) {
          // If a builtin provider has a custom @StarlarkBuiltin annotation, it's expected to have a
          // custom Starlark type too.
          assertWithMessage(messagePrefix).that(providerType).isNotEqualTo(anyProviderType);
          assertWithMessage(messagePrefix)
              .that(providerType.typeRepr())
              .matches("<Provider\\[.*\\]>");
        }
      } else {
        // Not callable from Starlark -> not typed as a Provider
        // Examples: AnalysisFailureInfoProvider, ActionsProvider.
        String messagePrefix =
            "If you just now added this subclass of BuiltinProvider, you should probably subclass"
                + " BuiltinProvider.NonCallable\n. For non-Starlark-callable builtin provider "
                + cls;
        StarlarkType nonProviderType = Starlark.getStarlarkTypeFromJavaType(cls, false);
        assertNotLt(messagePrefix, nonProviderType, anyProviderType);
        assertNotLt(messagePrefix, nonProviderType, Types.ANY_CALLABLE);
      }
    }
  }

  private boolean isCallableFromStarlark(Class<?> cls) {
    if (StarlarkCallable.class.isAssignableFrom(cls)) {
      return true;
    }
    if (CallUtils.getBuiltinManager(getStarlarkSemantics()).getSelfCallMethodType(cls) != null) {
      return true;
    }
    return false;
  }

  @Test
  public void nativeRule() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    TypeContext typeContext = getTypeContext();

    StarlarkType genruleType = revealStaticType("native.genrule");
    assertThat(genruleType).isEqualTo(revealExportedType("native.genrule"));
    assertThat(genruleType.typeRepr()).isEqualTo("rule");
    assertLt(genruleType, Types.ANY_CALLABLE);
    Types.CallableType genruleCallableType = Types.toCallableType(genruleType, typeContext);
    assertThat(genruleCallableType).isNotNull();
    // TODO: #23730 - add types for rule attributes and return type
  }

  @Test
  public void starlarkRule() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    TypeContext typeContext = getTypeContext();

    StarlarkType myRuleType =
        revealStaticType(
            """
            rule(
                implementation = lambda ctx: None,
                attrs = {
                    "srcs": attr.label_list(allow_files = True),
                    "deps": attr.label_list(),
                },
            )
            """);

    assertThat(myRuleType.typeRepr()).isEqualTo("rule");
    assertLt(myRuleType, Types.ANY_CALLABLE);
    Types.CallableType myRuleCallableType = Types.toCallableType(myRuleType, typeContext);
    assertThat(myRuleCallableType).isNotNull();
    // TODO: #23730 - add types for rule attributes and return type
  }

  @Test
  public void subrule() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    TypeContext typeContext = getTypeContext();

    StarlarkType subruleType =
        revealStaticType(
            """
            subrule(
                implementation = _subrule_impl,
                attrs = {"_tool": attr.label(allow_files = True, default = "//tool")},
            )
            """,
            """
            def _subrule_impl(ctx, positional_arg, *, _tool, **kwargs) -> list[int]:
                return [42]
            """);
    assertThat(subruleType.typeRepr()).isEqualTo("Subrule");
    assertLt(subruleType, Types.ANY_CALLABLE);
    Types.CallableType subruleCallableType = Types.toCallableType(subruleType, typeContext);
    assertThat(subruleCallableType).isNotNull();
    // TODO: #23730 - add types for subrule params and return type
  }

  @Test
  public void symbolicMacro() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    TypeContext typeContext = getTypeContext();

    StarlarkType macroType =
        revealStaticType(
            """
            macro(
                implementation = _macro_impl,
                attrs = {"deps": attr.label_list()},
            )
            """,
            """
            def _macro_impl(*, deps, visibility, **kwargs):
                pass
            """);
    assertThat(macroType.typeRepr()).isEqualTo("macro");
    assertLt(macroType, Types.ANY_CALLABLE);
    Types.CallableType macroCallableType = Types.toCallableType(macroType, typeContext);
    assertThat(macroCallableType).isNotNull();
    // TODO: #23730 - add types for macro attributes and return type
  }
}
