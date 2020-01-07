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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.syntax.StarlarkThread.ReadyToPause;
import com.google.devtools.build.lib.syntax.StarlarkThread.Stepping;
import com.google.devtools.build.lib.vfs.PathFragment;
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
  public void testListFramesFromGlobalFrame() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", 1);
    thread.getGlobals().put("b", 2);
    thread.getGlobals().put("c", 3);

    ImmutableList<DebugFrame> frames = thread.listFrames(Location.BUILTIN);

    assertThat(frames).hasSize(1);
    assertThat(frames.get(0))
        .isEqualTo(
            DebugFrame.builder()
                .setFunctionName("<top level>")
                .setLocation(Location.BUILTIN)
                .setGlobalBindings(ImmutableMap.of("a", 1, "b", 2, "c", 3))
                .build());
  }

  @Test
  public void testListFramesFromChildFrame() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", 1);
    thread.getGlobals().put("b", 2);
    thread.getGlobals().put("c", 3);
    Location loc =
        Location.fromPathAndStartColumn(
            PathFragment.create("foo/bar"), 0, 0, new LineAndColumn(12, 0));
    StarlarkFunction f = defineFunc(thread);
    thread.push(f, loc);
    thread.updateLexical("a", 4); // shadow parent frame var
    thread.updateLexical("y", 5);
    thread.updateLexical("z", 6);

    ImmutableList<DebugFrame> frames = thread.listFrames(Location.BUILTIN);

    assertThat(frames).hasSize(2);
    assertThat(frames.get(0))
        .isEqualTo(
            DebugFrame.builder()
                .setFunctionName("f")
                .setLocation(Location.BUILTIN)
                .setLexicalFrameBindings(ImmutableMap.of("a", 4, "y", 5, "z", 6))
                .setGlobalBindings(ImmutableMap.of("a", 1, "b", 2, "c", 3, "f", f))
                .build());
    assertThat(frames.get(1))
        .isEqualTo(
            DebugFrame.builder()
                .setFunctionName("<top level>")
                .setLocation(loc)
                .setGlobalBindings(ImmutableMap.of("a", 1, "b", 2, "c", 3, "f", f))
                .build());
  }

  @Test
  public void testStepIntoFunction() throws Exception {
    StarlarkThread thread = newStarlarkThread();

    ReadyToPause predicate = thread.stepControl(Stepping.INTO);
    thread.push(defineFunc(thread), Location.BUILTIN);

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
    thread.push(defineFunc(thread), Location.BUILTIN);

    ReadyToPause predicate = thread.stepControl(Stepping.INTO);
    thread.pop();

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOverFunction() throws Exception {
    StarlarkThread thread = newStarlarkThread();

    ReadyToPause predicate = thread.stepControl(Stepping.OVER);
    thread.push(defineFunc(thread), Location.BUILTIN);

    assertThat(predicate.test(thread)).isFalse();
    thread.pop();
    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOverFallsBackToStepOut() throws Exception {
    // test that when stepping over, we'll fall back to stopping when exiting the current frame
    StarlarkThread thread = newStarlarkThread();
    thread.push(defineFunc(thread), Location.BUILTIN);

    ReadyToPause predicate = thread.stepControl(Stepping.OVER);
    thread.pop();

    assertThat(predicate.test(thread)).isTrue();
  }

  @Test
  public void testStepOutOfInnerFrame() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.push(defineFunc(thread), Location.BUILTIN);

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

    Object a = thread.debugEval(Expression.parse(ParserInput.fromLines("a")));
    assertThat(a).isEqualTo(1);
  }

  @Test
  public void testEvaluateVariableNotInScopeFails() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", 1);

    EvalException e =
        assertThrows(
            EvalException.class,
            () -> thread.debugEval(Expression.parse(ParserInput.fromLines("b"))));
    assertThat(e).hasMessageThat().isEqualTo("name 'b' is not defined");
  }

  @Test
  public void testEvaluateExpressionOnVariableInScope() throws Exception {
    StarlarkThread thread = newStarlarkThread();
    thread.getGlobals().put("a", "string");

    assertThat(thread.debugEval(Expression.parse(ParserInput.fromLines("a.startswith('str')"))))
        .isEqualTo(true);
    EvalUtils.exec(
        EvalUtils.parseAndValidateSkylark(ParserInput.fromLines("a = 1"), thread), thread);
    assertThat(thread.debugEval(Expression.parse(ParserInput.fromLines("a")))).isEqualTo(1);
  }
}
