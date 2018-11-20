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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of Environment. */
@RunWith(JUnit4.class)
public class EnvironmentTest extends EvaluationTestCase {

  @Override
  public Environment newEnvironment() {
    return newBuildEnvironment();
  }

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
    eval("foo = 'bar'");
    assertThat(lookup("foo")).isEqualTo("bar");
  }

  // Test update through API, reference through interpreter:
  @Test
  public void testReference() throws Exception {
    setFailFast(false);
    try {
      eval("foo");
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }
    update("foo", "bar");
    assertThat(eval("foo")).isEqualTo("bar");
  }

  // Test assign and reference through interpreter:
  @Test
  public void testAssignAndReference() throws Exception {
    setFailFast(false);
    try {
      eval("foo");
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }
    eval("foo = 'bar'");
    assertThat(eval("foo")).isEqualTo("bar");
  }

  @Test
  public void testBuilderRequiresSemantics() throws Exception {
    try (Mutability mut = Mutability.create("test")) {
      IllegalArgumentException expected =
          assertThrows(IllegalArgumentException.class, () -> Environment.builder(mut).build());
      assertThat(expected)
          .hasMessageThat()
          .contains("must call either setSemantics or useDefaultSemantics");
    }
  }

  @Test
  public void testGetVariableNames() throws Exception {
    Environment env;
    try (Mutability mut = Mutability.create("outer")) {
      env =
          Environment.builder(mut)
              .useDefaultSemantics()
              .setGlobals(Environment.DEFAULT_GLOBALS)
              .build()
              .update("foo", "bar")
              .update("wiz", 3);
    }

    assertThat(env.getVariableNames())
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
  public void testToString() throws Exception {
    update("subject", new StringLiteral("Hello, 'world'."));
    update("from", new StringLiteral("Java"));
    assertThat(getEnvironment().toString()).isEqualTo("<Environment[test]>");
  }

  @Test
  public void testBindToNullThrowsException() throws Exception {
    try {
      update("some_name", null);
      fail();
    } catch (NullPointerException e) {
      assertThat(e).hasMessage("trying to assign null to 'some_name'");
    }
  }

  @Test
  public void testFrozen() throws Exception {
    Environment env;
    try (Mutability mutability = Mutability.create("testFrozen")) {
      env =
          Environment.builder(mutability)
              .useDefaultSemantics()
              .setGlobals(Environment.DEFAULT_GLOBALS)
              .setEventHandler(Environment.FAIL_FAST_HANDLER)
              .build();
      env.update("x", 1);
      assertThat(env.moduleLookup("x")).isEqualTo(1);
      env.update("y", 2);
      assertThat(env.moduleLookup("y")).isEqualTo(2);
      assertThat(env.moduleLookup("x")).isEqualTo(1);
      env.update("x", 3);
      assertThat(env.moduleLookup("x")).isEqualTo(3);
    }
    try {
      // This update to an existing variable should fail because the environment was frozen.
      env.update("x", 4);
      throw new Exception("failed to fail"); // not an AssertionError like fail()
    } catch (AssertionError e) {
      assertThat(e).hasMessage("Can't update x to 4 in frozen environment");
    }
    try {
      // This update to a new variable should also fail because the environment was frozen.
      env.update("newvar", 5);
      throw new Exception("failed to fail"); // not an AssertionError like fail()
    } catch (AssertionError e) {
      assertThat(e).hasMessage("Can't update newvar to 5 in frozen environment");
    }
  }

  @Test
  public void testBuiltinsCanBeShadowed() throws Exception {
    Environment env =
        newEnvironmentWithSkylarkOptions("--incompatible_static_name_resolution=true")
            .setup("special_var", 42);
    BuildFileAST.eval(env, "special_var = 41");
    assertThat(env.moduleLookup("special_var")).isEqualTo(41);
  }

  @Test
  public void testVariableIsReferencedBeforeAssignment() throws Exception {
    Environment env = newSkylarkEnvironment().update("global_var", 666);
    try {
      BuildFileAST.eval(env, "def foo(x): x += global_var; global_var = 36; return x", "foo(1)");
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
    Environment parentEnv = Environment.builder(parentMutability).useDefaultSemantics().build();
    parentEnv.update("a", 1);
    parentEnv.update("c", 2);
    parentEnv.update("b", 3);
    Environment.GlobalFrame parentFrame = parentEnv.getGlobals();
    parentMutability.freeze();
    Mutability mutability = Mutability.create("testing");
    Environment env =
        Environment.builder(mutability).useDefaultSemantics().setGlobals(parentFrame).build();
    env.update("x", 4);
    env.update("z", 5);
    env.update("y", 6);
    // The order just has to be deterministic, but for definiteness this test spells out the exact
    // order returned by the implementation: parent frame before current environment, and bindings
    // within a frame ordered by when they were added.
    assertThat(env.getVariableNames()).containsExactly("a", "c", "b", "x", "z", "y").inOrder();
    assertThat(env.getGlobals().getTransitiveBindings())
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
    Environment env1 =
        Environment.builder(Mutability.create("testing1"))
            .useDefaultSemantics()
            .setImportedExtensions(ImmutableMap.of("a", a, "b", b, "c", c))
            .setFileContentHashCode("z")
            .build();
    Environment env2 =
        Environment.builder(Mutability.create("testing2"))
            .useDefaultSemantics()
            .setImportedExtensions(ImmutableMap.of("c", c, "b", b, "a", a))
            .setFileContentHashCode("z")
            .build();
    assertThat(env1.getTransitiveContentHashCode()).isEqualTo(env2.getTransitiveContentHashCode());
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
