// Copyright 2018 The Bazel Authors. All Rights Reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.Debug.ReadyToPause;
import com.google.devtools.build.lib.syntax.Debug.Stepping;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of debugging features of StarlarkThread. */
@RunWith(JUnit4.class)
public class StarlarkThreadDebuggingTest {

  // TODO(adonovan): rewrite these tests at a higher level.

  private static StarlarkThread newThread() {
    return new StarlarkThread(Mutability.create("test"), StarlarkSemantics.DEFAULT);
  }

  // Executes the definition of a trivial function f and returns the function value.
  private static StarlarkFunction defineFunc() throws Exception {
    Module module = Module.create();
    EvalUtils.exec(
        ParserInput.fromLines("def f(): pass"), FileOptions.DEFAULT, module, newThread());
    return (StarlarkFunction) module.getGlobal("f");
  }

  @Test
  public void testListFramesEmptyStack() throws Exception {
    StarlarkThread thread = newThread();
    assertThat(Debug.getCallStack(thread)).isEmpty();
    assertThat(thread.getCallStack()).isEmpty();
  }

  @Test
  public void testListFramesFromBuiltin() throws Exception {
    // f is a built-in that captures the stack using the Debugger API.
    Object[] result = {null, null};
    StarlarkCallable f =
        new StarlarkCallable() {
          @Override
          public String getName() {
            return "f";
          }

          @Override
          public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
            result[0] = Debug.getCallStack(thread);
            result[1] = thread.getCallStack();
            return Starlark.NONE;
          }

          @Override
          public Location getLocation() {
            return Location.fromFileLineColumn("builtin", 12, 0);
          }

          @Override
          public String toString() {
            return "<debug function>";
          }
        };

    // Set up global environment.
    Module module =
        Module.withPredeclared(StarlarkSemantics.DEFAULT, ImmutableMap.of("a", 1, "b", 2, "f", f));

    // Execute a small file that calls f.
    ParserInput input =
        ParserInput.fromString(
            "def g(a, y, z):\n" // shadows global a
                + "  f()\n"
                + "g(4, 5, 6)",
            "main.star");
    EvalUtils.exec(input, FileOptions.DEFAULT, module, newThread());

    @SuppressWarnings("unchecked")
    ImmutableList<Debug.Frame> stack = (ImmutableList<Debug.Frame>) result[0];

    // Check the stack captured by f.
    // We compare printed string forms, as it gives more informative assertion failures.
    StringBuilder buf = new StringBuilder();
    for (Debug.Frame fr : stack) {
      buf.append(
          String.format(
              "%s @ %s local=%s\n", fr.getFunction().getName(), fr.getLocation(), fr.getLocals()));
    }
    assertThat(buf.toString())
        .isEqualTo(
            ""
                // location is paren of g(4, 5, 6) call:
                + "<toplevel> @ main.star:3:2 local={}\n"
                // location is paren of "f()" call:
                + "g @ main.star:2:4 local={a=4, y=5, z=6}\n"
                // location is "current PC" in f.
                + "f @ builtin:12 local={}\n");

    // Same, with "lite" stack API.
    assertThat(result[1].toString()) // an ImmutableList<StarlarkThread.CallStackEntry>
        .isEqualTo("[<toplevel>@main.star:3:2, g@main.star:2:4, f@builtin:12]");

    // TODO(adonovan): more tests:
    // - a stack containing functions defined in different modules.
    // - changing environment at various program points within a function.
  }

  @Test
  public void testStepIntoFunction() throws Exception {
    StarlarkThread thread = newThread();

    ReadyToPause predicate = Debug.stepControl(thread, Stepping.INTO);
    thread.push(defineFunc());

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepIntoFallsBackToStepOver() {
    // test that when stepping into, we'll fall back to stopping at the next statement in the
    // current frame
    StarlarkThread thread = newThread();

    ReadyToPause predicate = Debug.stepControl(thread, Stepping.INTO);

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepIntoFallsBackToStepOut() throws Exception {
    // test that when stepping into, we'll fall back to stopping when exiting the current frame
    StarlarkThread thread = newThread();
    thread.push(defineFunc());

    ReadyToPause predicate = Debug.stepControl(thread, Stepping.INTO);
    thread.pop();

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOverFunction() throws Exception {
    StarlarkThread thread = newThread();

    ReadyToPause predicate = Debug.stepControl(thread, Stepping.OVER);
    thread.push(defineFunc());

    assertThat(predicate.test(thread)).isFalse();
    thread.pop();
    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOverFallsBackToStepOut() throws Exception {
    // test that when stepping over, we'll fall back to stopping when exiting the current frame
    StarlarkThread thread = newThread();
    thread.push(defineFunc());

    ReadyToPause predicate = Debug.stepControl(thread, Stepping.OVER);
    thread.pop();

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOutOfInnerFrame() throws Exception {
    StarlarkThread thread = newThread();
    thread.push(defineFunc());

    ReadyToPause predicate = Debug.stepControl(thread, Stepping.OUT);

    assertThat(predicate.test(thread)).isFalse();
    thread.pop();
    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOutOfOutermostFrame() {
    StarlarkThread thread = newThread();

    assertThat(Debug.stepControl(thread, Stepping.OUT)).isNull();
  }

  @Test
  public void testStepControlWithNoSteppingReturnsNull() {
    StarlarkThread thread = newThread();

    assertThat(Debug.stepControl(thread, Stepping.NONE)).isNull();
  }

  @Test
  public void testEvaluateVariableInScope() throws Exception {
    Module module = Module.withPredeclared(StarlarkSemantics.DEFAULT, ImmutableMap.of("a", 1));

    StarlarkThread thread = newThread();
    Object a = EvalUtils.exec(ParserInput.fromLines("a"), FileOptions.DEFAULT, module, thread);
    assertThat(a).isEqualTo(1);
  }

  @Test
  public void testEvaluateVariableNotInScopeFails() throws Exception {
    Module module = Module.create();

    SyntaxError.Exception e =
        assertThrows(
            SyntaxError.Exception.class,
            () ->
                EvalUtils.exec(
                    ParserInput.fromLines("b"), FileOptions.DEFAULT, module, newThread()));

    assertThat(e).hasMessageThat().isEqualTo("name 'b' is not defined");
  }

  @Test
  public void testEvaluateExpressionOnVariableInScope() throws Exception {
    StarlarkThread thread = newThread();
    Module module =
        Module.withPredeclared(
            StarlarkSemantics.DEFAULT, /*predeclared=*/ ImmutableMap.of("a", "string"));

    assertThat(
            EvalUtils.exec(
                ParserInput.fromLines("a.startswith('str')"), FileOptions.DEFAULT, module, thread))
        .isEqualTo(true);
    EvalUtils.exec(ParserInput.fromLines("a = 1"), FileOptions.DEFAULT, module, thread);
    assertThat(EvalUtils.exec(ParserInput.fromLines("a"), FileOptions.DEFAULT, module, thread))
        .isEqualTo(1);
  }
}
