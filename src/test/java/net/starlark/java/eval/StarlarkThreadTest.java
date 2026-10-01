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

package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.List;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of StarlarkThread. */
@RunWith(JUnit4.class)
public final class StarlarkThreadTest {

  private final EvaluationTestCase ev = new EvaluationTestCase();

  // Test the API directly
  @Test
  public void testLookupAndUpdate() throws Exception {
    assertThat(ev.lookup("foo")).isNull();
    ev.update("foo", "bar");
    assertThat(ev.lookup("foo")).isEqualTo("bar");
  }

  @Test
  public void testDoubleUpdateSucceeds() throws Exception {
    assertThat(ev.lookup("VERSION")).isNull();
    ev.update("VERSION", StarlarkInt.of(42));
    assertThat(ev.lookup("VERSION")).isEqualTo(StarlarkInt.of(42));
    ev.update("VERSION", StarlarkInt.of(43));
    assertThat(ev.lookup("VERSION")).isEqualTo(StarlarkInt.of(43));
  }

  // Test assign through interpreter, ev.lookup through API:
  @Test
  public void testAssign() throws Exception {
    assertThat(ev.lookup("foo")).isNull();
    ev.exec("foo = 'bar'");
    assertThat(ev.lookup("foo")).isEqualTo("bar");
  }

  // Test update through API, reference through interpreter:
  @Test
  public void testReference() throws Exception {
    SyntaxError.Exception e = assertThrows(SyntaxError.Exception.class, () -> ev.eval("foo"));
    assertThat(e).hasMessageThat().isEqualTo("name 'foo' is not defined");
    ev.update("foo", "bar");
    assertThat(ev.eval("foo")).isEqualTo("bar");
  }

  // Test assign and reference through interpreter:
  @Test
  public void testAssignAndReference() throws Exception {
    SyntaxError.Exception e = assertThrows(SyntaxError.Exception.class, () -> ev.eval("foo"));
    assertThat(e).hasMessageThat().isEqualTo("name 'foo' is not defined");
    ev.exec("foo = 'bar'");
    assertThat(ev.eval("foo")).isEqualTo("bar");
  }

  @Test
  public void testBindToNullThrowsException() throws Exception {
    NullPointerException e =
        assertThrows(NullPointerException.class, () -> ev.update("some_name", null));
    assertThat(e).hasMessageThat().isEqualTo("Module.setGlobal(some_name, null)");
  }

  @Test
  public void testUniverseCanBeShadowed() throws Exception {
    Module module = Module.create();
    try (Mutability mu = Mutability.create("test")) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      Starlark.execFile(ParserInput.fromLines("True = 123"), FileOptions.DEFAULT, module, thread);
    }
    assertThat(module.getGlobal("True")).isEqualTo(StarlarkInt.of(123));
  }

  @Test
  public void testVariableIsReferencedBeforeAssignment() throws Exception {
    ev.new Scenario()
        .testIfErrorContains(
            "local variable 'y' is referenced before assignment",
            "y = 1", // bind => y is global
            "def foo(x):",
            "  x += y", // fwd ref to local y
            "  y = 2", // binding => y is local
            "  return x",
            "foo(1)");
    ev.new Scenario()
        .testIfErrorContains(
            "global variable 'len' is referenced before assignment",
            "print(len)", // fwd ref to global len
            "len = 1"); // binding => len is local
  }

  @Test
  public void testReusedFrameHasFreshReturnValue() throws Exception {
    // Calls at the same depth reuse the same frame; a stale result must not leak into a function
    // that returns without a value.
    ev.exec("def f(): return 1", "def g(): return", "def h(): pass", "x = [f(), g(), f(), h()]");
    assertThat((List<?>) ev.lookup("x"))
        .containsExactly(StarlarkInt.of(1), Starlark.NONE, StarlarkInt.of(1), Starlark.NONE)
        .inOrder();
  }

  @Test
  public void testReusedFrameHasFreshErrorLocation() throws Exception {
    Module module = Module.create();
    try (Mutability mu = Mutability.create("test")) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, StarlarkSemantics.DEFAULT);
      Starlark.execFile(
          ParserInput.fromLines(
              "def fail1():", //
              "  return 1 // 0",
              "def fail2():",
              "  x = 1",
              "  return [][x]"),
          FileOptions.DEFAULT,
          module,
          thread);
      EvalException e1 =
          assertThrows(
              EvalException.class,
              () ->
                  Starlark.call(
                      thread, module.getGlobal("fail1"), ImmutableList.of(), ImmutableMap.of()));
      assertThat(e1.getCallStack().getLast().location.line()).isEqualTo(2);
      // The second call reuses the frame of the first; its error location must not be stale.
      EvalException e2 =
          assertThrows(
              EvalException.class,
              () ->
                  Starlark.call(
                      thread, module.getGlobal("fail2"), ImmutableList.of(), ImmutableMap.of()));
      assertThat(e2.getCallStack().getLast().name).isEqualTo("fail2");
      assertThat(e2.getCallStack().getLast().location.line()).isEqualTo(5);
      assertThat(thread.getCallStack()).isEmpty();
    }
  }
}
