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
import com.google.devtools.build.lib.syntax.Debuggable.ReadyToPause;
import com.google.devtools.build.lib.syntax.Debuggable.Stepping;
import com.google.devtools.build.lib.syntax.Environment.LexicalFrame;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests of {@link Environment}s implementation of {@link Debuggable}. */
@RunWith(JUnit4.class)
public class EnvironmentDebuggingTest {

  private static Environment newEnvironment() {
    Mutability mutability = Mutability.create("test");
    return Environment.builder(mutability).useDefaultSemantics().build();
  }

  /** Enter a dummy function scope with the given name, and the current environment's globals. */
  private static void enterFunctionScope(Environment env, String functionName, Location location) {
    FuncallExpression ast = new FuncallExpression(Identifier.of("test"), ImmutableList.of());
    ast.setLocation(location);
    env.enterScope(
        new BaseFunction(functionName) {},
        LexicalFrame.create(env.mutability()),
        ast,
        env.getGlobals());
  }

  @Test
  public void testListFramesFromGlobalFrame() throws Exception {
    Environment env = newEnvironment();
    env.update("a", 1);
    env.update("b", 2);
    env.update("c", 3);

    ImmutableList<DebugFrame> frames = env.listFrames(Location.BUILTIN);

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
    Environment env = newEnvironment();
    env.update("a", 1);
    env.update("b", 2);
    env.update("c", 3);
    Location funcallLocation =
        Location.fromPathAndStartColumn(
            PathFragment.create("foo/bar"), 0, 0, new LineAndColumn(12, 0));
    enterFunctionScope(env, "function", funcallLocation);
    env.update("a", 4); // shadow parent frame var
    env.update("y", 5);
    env.update("z", 6);

    ImmutableList<DebugFrame> frames = env.listFrames(Location.BUILTIN);

    assertThat(frames).hasSize(2);
    assertThat(frames.get(0))
        .isEqualTo(
            DebugFrame.builder()
                .setFunctionName("function")
                .setLocation(Location.BUILTIN)
                .setLexicalFrameBindings(ImmutableMap.of("a", 4, "y", 5, "z", 6))
                .setGlobalBindings(ImmutableMap.of("a", 1, "b", 2, "c", 3))
                .build());
    assertThat(frames.get(1))
        .isEqualTo(
            DebugFrame.builder()
                .setFunctionName("<top level>")
                .setLocation(funcallLocation)
                .setGlobalBindings(ImmutableMap.of("a", 1, "b", 2, "c", 3))
                .build());
  }

  @Test
  public void testStepIntoFunction() {
    Environment env = newEnvironment();

    ReadyToPause predicate = env.stepControl(Stepping.INTO);
    enterFunctionScope(env, "function", Location.BUILTIN);

    assertThat(predicate.test(env)).isTrue();
  }

  @Test
  public void testStepIntoFallsBackToStepOver() {
    // test that when stepping into, we'll fall back to stopping at the next statement in the
    // current frame
    Environment env = newEnvironment();

    ReadyToPause predicate = env.stepControl(Stepping.INTO);

    assertThat(predicate.test(env)).isTrue();
  }

  @Test
  public void testStepIntoFallsBackToStepOut() {
    // test that when stepping into, we'll fall back to stopping when exiting the current frame
    Environment env = newEnvironment();
    enterFunctionScope(env, "function", Location.BUILTIN);

    ReadyToPause predicate = env.stepControl(Stepping.INTO);
    env.exitScope();

    assertThat(predicate.test(env)).isTrue();
  }

  @Test
  public void testStepOverFunction() {
    Environment env = newEnvironment();

    ReadyToPause predicate = env.stepControl(Stepping.OVER);
    enterFunctionScope(env, "function", Location.BUILTIN);

    assertThat(predicate.test(env)).isFalse();
    env.exitScope();
    assertThat(predicate.test(env)).isTrue();
  }

  @Test
  public void testStepOverFallsBackToStepOut() {
    // test that when stepping over, we'll fall back to stopping when exiting the current frame
    Environment env = newEnvironment();
    enterFunctionScope(env, "function", Location.BUILTIN);

    ReadyToPause predicate = env.stepControl(Stepping.OVER);
    env.exitScope();

    assertThat(predicate.test(env)).isTrue();
  }

  @Test
  public void testStepOutOfInnerFrame() {
    Environment env = newEnvironment();
    enterFunctionScope(env, "function", Location.BUILTIN);

    ReadyToPause predicate = env.stepControl(Stepping.OUT);

    assertThat(predicate.test(env)).isFalse();
    env.exitScope();
    assertThat(predicate.test(env)).isTrue();
  }

  @Test
  public void testStepOutOfOutermostFrame() {
    Environment env = newEnvironment();

    assertThat(env.stepControl(Stepping.OUT)).isNull();
  }

  @Test
  public void testStepControlWithNoSteppingReturnsNull() {
    Environment env = newEnvironment();

    assertThat(env.stepControl(Stepping.NONE)).isNull();
  }

  @Test
  public void testEvaluateVariableInScope() throws Exception {
    Environment env = newEnvironment();
    env.update("a", 1);

    Object result = env.evaluate("a");

    assertThat(result).isEqualTo(1);
  }

  @Test
  public void testEvaluateVariableNotInScopeFails() throws Exception {
    Environment env = newEnvironment();
    env.update("a", 1);

    EvalException e = assertThrows(EvalException.class, () -> env.evaluate("b"));
    assertThat(e).hasMessageThat().isEqualTo("name 'b' is not defined");
  }

  @Test
  public void testEvaluateExpressionOnVariableInScope() throws Exception {
    Environment env = newEnvironment();
    env.update("a", "string");

    Object result = env.evaluate("a.startswith(\"str\")");

    assertThat(result).isEqualTo(Boolean.TRUE);
  }
}
