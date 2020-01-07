// Copyright 2006 The Bazel Authors. All Rights Reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.syntax.StarlarkThread.Extension;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of StarlarkThread. */
@RunWith(JUnit4.class)
public final class StarlarkThreadTest extends EvaluationTestCase {

  // Test the API directly
  @Test
  public void testLookupAndUpdate() throws Exception {
    assertThat(lookup("foo")).isNull();
    update("foo", "bar");
    assertThat(lookup("foo")).isEqualTo("bar");
  }

  @Test
  public void testDoubleUpdateSucceeds() throws Exception {
    assertThat(lookup("VERSION")).isNull();
    update("VERSION", 42);
    assertThat(lookup("VERSION")).isEqualTo(42);
    update("VERSION", 43);
    assertThat(lookup("VERSION")).isEqualTo(43);
  }

  // Test assign through interpreter, lookup through API:
  @Test
  public void testAssign() throws Exception {
    assertThat(lookup("foo")).isNull();
    exec("foo = 'bar'");
    assertThat(lookup("foo")).isEqualTo("bar");
  }

  // Test update through API, reference through interpreter:
  @Test
  public void testReference() throws Exception {
    setFailFast(false);
    SyntaxError e = assertThrows(SyntaxError.class, () -> eval("foo"));
    assertThat(e).hasMessageThat().isEqualTo("name 'foo' is not defined");
    update("foo", "bar");
    assertThat(eval("foo")).isEqualTo("bar");
  }

  // Test assign and reference through interpreter:
  @Test
  public void testAssignAndReference() throws Exception {
    SyntaxError e = assertThrows(SyntaxError.class, () -> eval("foo"));
    assertThat(e).hasMessageThat().isEqualTo("name 'foo' is not defined");
    exec("foo = 'bar'");
    assertThat(eval("foo")).isEqualTo("bar");
  }

  @Test
  public void testBuilderRequiresSemantics() throws Exception {
    try (Mutability mut = Mutability.create("test")) {
      IllegalArgumentException expected =
          assertThrows(IllegalArgumentException.class, () -> StarlarkThread.builder(mut).build());
      assertThat(expected)
          .hasMessageThat()
          .contains("must call either setSemantics or useDefaultSemantics");
    }
  }

  @Test
  public void testGetVariableNames() throws Exception {
    StarlarkThread thread;
    try (Mutability mut = Mutability.create("outer")) {
      thread =
          StarlarkThread.builder(mut)
              .useDefaultSemantics()
              .setGlobals(Module.createForBuiltins(Starlark.UNIVERSE))
              .build();
      thread.getGlobals().put("foo", "bar");
      thread.getGlobals().put("wiz", 3);
    }

    assertThat(thread.getVariableNames())
        .isEqualTo(
            Sets.newHashSet(
                "foo",
                "wiz",
                "False",
                "None",
                "True",
                "all",
                "any",
                "bool",
                "depset",
                "dict",
                "dir",
                "enumerate",
                "fail",
                "getattr",
                "hasattr",
                "hash",
                "int",
                "len",
                "list",
                "max",
                "min",
                "print",
                "range",
                "repr",
                "reversed",
                "select",
                "sorted",
                "str",
                "tuple",
                "type",
                "zip"));
  }

  @Test
  public void testBindToNullThrowsException() throws Exception {
    NullPointerException e =
        assertThrows(NullPointerException.class, () -> update("some_name", null));
    assertThat(e).hasMessageThat().isEqualTo("Module.put(some_name, null)");
  }

  @Test
  public void testFrozen() throws Exception {
    Module module;
    try (Mutability mutability = Mutability.create("testFrozen")) {
      // TODO(adonovan): make it simpler to construct a module without a thread,
      // and move this test to ModuleTest.
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .useDefaultSemantics()
              .setGlobals(Module.createForBuiltins(Starlark.UNIVERSE))
              .build();
      module = thread.getGlobals();
      module.put("x", 1);
      assertThat(module.lookup("x")).isEqualTo(1);
      module.put("y", 2);
      assertThat(module.lookup("y")).isEqualTo(2);
      assertThat(module.lookup("x")).isEqualTo(1);
      module.put("x", 3);
      assertThat(module.lookup("x")).isEqualTo(3);
    }

    // This update to an existing variable should fail because the environment was frozen.
    Mutability.MutabilityException ex =
        assertThrows(Mutability.MutabilityException.class, () -> module.put("x", 4));
    assertThat(ex).hasMessageThat().isEqualTo("trying to mutate a frozen object");

    // This update to a new variable should also fail because the environment was frozen.
    ex = assertThrows(Mutability.MutabilityException.class, () -> module.put("newvar", 5));
    assertThat(ex).hasMessageThat().isEqualTo("trying to mutate a frozen object");
  }

  @Test
  public void testBuiltinsCanBeShadowed() throws Exception {
    StarlarkThread thread = newStarlarkThreadWithSkylarkOptions();
    EvalUtils.exec(ParserInput.fromLines("True = 123"), thread);
    assertThat(thread.getGlobals().lookup("True")).isEqualTo(123);
  }

  @Test
  public void testVariableIsReferencedBeforeAssignment() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("global_var", 666);
    try {
      EvalUtils.exec(
          ParserInput.fromLines(
              "def foo(x): x += global_var; global_var = 36; return x", //
              "foo(1)"),
          thread);
      throw new AssertionError("failed to fail");
    } catch (EvalExceptionWithStackTrace e) {
      assertThat(e)
          .hasMessageThat()
          .contains("local variable 'global_var' is referenced before assignment.");
    }
  }

  @Test
  public void testVarOrderDeterminism() throws Exception {
    Mutability parentMutability = Mutability.create("parent env");
    StarlarkThread parentThread =
        StarlarkThread.builder(parentMutability).useDefaultSemantics().build();
    parentThread.getGlobals().put("a", 1);
    parentThread.getGlobals().put("c", 2);
    parentThread.getGlobals().put("b", 3);
    Module parentFrame = parentThread.getGlobals();
    parentMutability.freeze();
    Mutability mutability = Mutability.create("testing");
    StarlarkThread thread =
        StarlarkThread.builder(mutability).useDefaultSemantics().setGlobals(parentFrame).build();
    thread.getGlobals().put("x", 4);
    thread.getGlobals().put("z", 5);
    thread.getGlobals().put("y", 6);
    // The order just has to be deterministic, but for definiteness this test spells out the exact
    // order returned by the implementation: parent frame before current environment, and bindings
    // within a frame ordered by when they were added.
    assertThat(thread.getVariableNames()).containsExactly("a", "c", "b", "x", "z", "y").inOrder();
    assertThat(thread.getGlobals().getTransitiveBindings())
        .containsExactly("a", 1, "c", 2, "b", 3, "x", 4, "z", 5, "y", 6)
        .inOrder();
  }

  @Test
  public void testTransitiveHashCodeDeterminism() throws Exception {
    // As a proxy for determinism, test that changing the order of imports doesn't change the hash
    // code (within any one execution).
    Extension a = new Extension(ImmutableMap.of(), "a123");
    Extension b = new Extension(ImmutableMap.of(), "b456");
    Extension c = new Extension(ImmutableMap.of(), "c789");
    StarlarkThread thread1 =
        StarlarkThread.builder(Mutability.create("testing1"))
            .useDefaultSemantics()
            .setImportedExtensions(ImmutableMap.of("a", a, "b", b, "c", c))
            .setFileContentHashCode("z")
            .build();
    StarlarkThread thread2 =
        StarlarkThread.builder(Mutability.create("testing2"))
            .useDefaultSemantics()
            .setImportedExtensions(ImmutableMap.of("c", c, "b", b, "a", a))
            .setFileContentHashCode("z")
            .build();
    assertThat(thread1.getTransitiveContentHashCode())
        .isEqualTo(thread2.getTransitiveContentHashCode());
  }

  @Test
  public void testExtensionEqualityDebugging_RhsIsNull() {
    assertCheckStateFailsWithMessage(new Extension(ImmutableMap.of(), "abc"), null, "got a null");
  }

  @Test
  public void testExtensionEqualityDebugging_RhsHasBadType() {
    assertCheckStateFailsWithMessage(
        new Extension(ImmutableMap.of(), "abc"), 5, "got a java.lang.Integer");
  }

  @Test
  public void testExtensionEqualityDebugging_DifferentBindings() {
    assertCheckStateFailsWithMessage(
        new Extension(ImmutableMap.of("w", 1, "x", 2, "y", 3), "abc"),
        new Extension(ImmutableMap.of("y", 3, "z", 4), "abc"),
        "in this one but not given one: [w, x]; in given one but not this one: [z]");
  }

  @Test
  public void testExtensionEqualityDebugging_DifferentValues() {
    assertCheckStateFailsWithMessage(
        new Extension(ImmutableMap.of("x", 1, "y", "foo", "z", true), "abc"),
        new Extension(ImmutableMap.of("x", 2.0, "y", "foo", "z", false), "abc"),
        "bindings are unequal: x: this one has 1 (class java.lang.Integer, 1), but given one has "
            + "2.0 (class java.lang.Double, 2.0); z: this one has True (class java.lang.Boolean, "
            + "true), but given one has False (class java.lang.Boolean, false)");
  }

  @Test
  public void testExtensionEqualityDebugging_DifferentHashes() {
    assertCheckStateFailsWithMessage(
        new Extension(ImmutableMap.of(), "abc"),
        new Extension(ImmutableMap.of(), "xyz"),
        "transitive content hashes don't match: abc != xyz");
  }

  private static void assertCheckStateFailsWithMessage(
      Extension left, Object right, String substring) {
    IllegalStateException expected =
        assertThrows(IllegalStateException.class, () -> left.checkStateEquals(right));
    assertThat(expected).hasMessageThat().contains(substring);
  }
}
