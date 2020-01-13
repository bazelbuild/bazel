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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.StarlarkThread.ReadyToPause;
import com.google.devtools.build.lib.syntax.StarlarkThread.Stepping;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of debugging features of StarlarkThread. */
@RunWith(JUnit4.class)
public class StarlarkThreadDebuggingTest {

  // TODO(adonovan): rewrite these tests at a higher level.

  private static StarlarkThread newStarlarkThread() {
    Mutability mutability = Mutability.create("test");
    return StarlarkThread.builder(mutability).useDefaultSemantics().build();
  }

  // Executes the definition of a trivial function f in the specified thread,
  // and returns the function value.
  private static StarlarkFunction defineFunc(StarlarkThread thread) throws Exception {
    EvalUtils.exec(ParserInput.fromLines("def f(): pass"), thread);
    return (StarlarkFunction) thread.getGlobals().lookup("f");
  }

  @Test
  public void testListFramesEmptyStack() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    assertThat(thread.listFrames()).isEmpty();
  }

  @Test
  public void testListFramesFromBuiltin() throws Exception {
    // f is a built-in that captures the stack using the Debugger API.
    Object[] result = {null};
    StarlarkCallable f =
        new StarlarkCallable() {
          @Override
          public String getName() {
            return "f";
          }

          @Override
          public Object fastcall(
              StarlarkThread thread, Location loc, Object[] positional, Object[] named) {
            result[0] = thread.listFrames();
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
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", 1);
    thread.getGlobals().put("b", 2);
    thread.getGlobals().put("f", f);

    // Execute a small file that calls f.
    ParserInput input =
        ParserInput.create(
            "def g(a, y, z):\n" // shadows global a
                + "  f()\n"
                + "g(4, 5, 6)",
            "main.star");
    EvalUtils.exec(input, thread);

    @SuppressWarnings("unchecked")
    ImmutableList<DebugFrame> stack = (ImmutableList<DebugFrame>) result[0];

    // Check the stack captured by f.
    // We compare printed string forms, as it gives more informative assertion failures.
    StringBuilder buf = new StringBuilder();
    for (DebugFrame fr : stack) {
      buf.append(
          String.format(
              "%s @ %s local=%s global=%s\n",
              fr.functionName(), fr.location(), fr.lexicalFrameBindings(), fr.globalBindings()));
    }
    assertThat(buf.toString())
        .isEqualTo(
            // location is "current PC" in f.
            // Observe that the globals (module) of g is still in force,
            // even though logically a built-in such as f has no module.
            "f @ builtin:12 local={} global={a=1, b=2, f=<debug function>, g=g(a, y, z)}\n"
                // location is start of "f()" call:
                + "g @ main.star:2:3 local={a=4, y=5, z=6} global={a=1, b=2, f=<debug function>,"
                + " g=g(a, y, z)}\n"
                // location is start of g(4, 5, 6) call:
                + "<toplevel> @ main.star:3:1 local={} global={a=1, b=2, f=<debug function>,"
                + " g=g(a, y, z)}\n");

    // TODO(adonovan): more tests:
    // - a stack containing functions defined in different modules.
    // - changing environment at various program points within a function.
  }

  @Test
  public void testStepIntoFunction() throws Exception {
    StarlarkThread thread = newStarlarkThread();

    ReadyToPause predicate = thread.stepControl(Stepping.INTO);
    thread.push(defineFunc(thread));

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepIntoFallsBackToStepOver() {
    // test that when stepping into, we'll fall back to stopping at the next statement in the
    // current frame
    StarlarkThread thread = newStarlarkThread();

    ReadyToPause predicate = thread.stepControl(Stepping.INTO);

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepIntoFallsBackToStepOut() throws Exception {
    // test that when stepping into, we'll fall back to stopping when exiting the current frame
    StarlarkThread thread = newStarlarkThread();
    thread.push(defineFunc(thread));

    ReadyToPause predicate = thread.stepControl(Stepping.INTO);
    thread.pop();

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOverFunction() throws Exception {
    StarlarkThread thread = newStarlarkThread();

    ReadyToPause predicate = thread.stepControl(Stepping.OVER);
    thread.push(defineFunc(thread));

    assertThat(predicate.test(thread)).isFalse();
    thread.pop();
    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOverFallsBackToStepOut() throws Exception {
    // test that when stepping over, we'll fall back to stopping when exiting the current frame
    StarlarkThread thread = newStarlarkThread();
    thread.push(defineFunc(thread));

    ReadyToPause predicate = thread.stepControl(Stepping.OVER);
    thread.pop();

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOutOfInnerFrame() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.push(defineFunc(thread));

    ReadyToPause predicate = thread.stepControl(Stepping.OUT);

    assertThat(predicate.test(thread)).isFalse();
    thread.pop();
    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOutOfOutermostFrame() {
    StarlarkThread thread = newStarlarkThread();

    assertThat(thread.stepControl(Stepping.OUT)).isNull();
  }

  @Test
  public void testStepControlWithNoSteppingReturnsNull() {
    StarlarkThread thread = newStarlarkThread();

    assertThat(thread.stepControl(Stepping.NONE)).isNull();
  }

  @Test
  public void testEvaluateVariableInScope() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", 1);

    Object a = EvalUtils.execAndEvalOptionalFinalExpression(ParserInput.fromLines("a"), thread);
    assertThat(a).isEqualTo(1);
  }

  @Test
  public void testEvaluateVariableNotInScopeFails() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", 1);

    SyntaxError e =
        assertThrows(
            SyntaxError.class,
            () -> EvalUtils.execAndEvalOptionalFinalExpression(ParserInput.fromLines("b"), thread));

    assertThat(e).hasMessageThat().isEqualTo("name 'b' is not defined");
  }

  @Test
  public void testEvaluateExpressionOnVariableInScope() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", "string");

    assertThat(
            EvalUtils.execAndEvalOptionalFinalExpression(
                ParserInput.fromLines("a.startswith('str')"), thread))
        .isEqualTo(true);
    EvalUtils.exec(
        EvalUtils.parseAndValidateSkylark(ParserInput.fromLines("a = 1"), thread), thread);
    assertThat(EvalUtils.execAndEvalOptionalFinalExpression(ParserInput.fromLines("a"), thread))
        .isEqualTo(1);
  }
}
