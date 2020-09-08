// Copyright 2020 The Bazel Authors. All rights reserved.
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for StringModule. */
// TODO(bazel-team): Migrate these test cases back to string_misc.sky, once either
// 1) --incompatible_string_replace_count has been flipped (#11244) and deleted, or 2) the
// standalone Starlark interpreter and tests gain the ability to run with semantics flags.
@RunWith(JUnit4.class)
public class StringModuleTest {

  private final EvaluationTestCase ev = new EvaluationTestCase();

  private void runReplaceTest(StarlarkSemantics semantics) throws Exception {
    ev.new Scenario(semantics)
        .testEval("'banana'.replace('a', 'o')", "'bonono'")
        .testEval("'banana'.replace('a', 'o', 2)", "'bonona'")
        .testEval("'banana'.replace('a', 'o', 0)", "'banana'")
        .testEval("'banana'.replace('a', 'e')", "'benene'")
        .testEval("'banana'.replace('a', '$()')", "'b$()n$()n$()'")
        .testEval("'banana'.replace('a', '$')", "'b$n$n$'")
        .testEval("'b$()n$()n$()'.replace('$()', '$($())')", "'b$($())n$($())n$($())'")
        .testEval("'banana'.replace('a', 'e', 2)", "'benena'")
        .testEval("'banana'.replace('a', 'e', 0)", "'banana'")
        .testEval("'banana'.replace('', '-')", "'-b-a-n-a-n-a-'")
        .testEval("'banana'.replace('', '-', 2)", "'-b-anana'")
        .testEval("'banana'.replace('', '-', 0)", "'banana'")
        .testEval("'banana'.replace('', '')", "'banana'")
        .testEval("'banana'.replace('a', '')", "'bnn'")
        .testEval("'banana'.replace('a', '', 2)", "'bnna'");
  }

  @Test
  public void testReplaceWithAndWithoutFlag() throws Exception {
    runReplaceTest(replaceCount(false));
    runReplaceTest(replaceCount(true));
  }

  @Test
  public void testReplaceIncompatibleFlag() throws Exception {
    // Test the scenario that changes with the incompatible flag
    ev.new Scenario(replaceCount(false))
        .testEval("'banana'.replace('a', 'o', -2)", "'banana'")
        .testEval("'banana'.replace('a', 'e', -1)", "'banana'")
        .testEval("'banana'.replace('a', 'e', -10)", "'banana'")
        .testEval("'banana'.replace('', '-', -2)", "'banana'");

    ev.new Scenario(replaceCount(true))
        .testEval("'banana'.replace('a', 'o', -2)", "'bonono'")
        .testEval("'banana'.replace('a', 'e', -1)", "'benene'")
        .testEval("'banana'.replace('a', 'e', -10)", "'benene'")
        .testEval("'banana'.replace('', '-', -2)", "'-b-a-n-a-n-a-'");
  }

  @Test
  public void testReplaceNoneCount() throws Exception {
    // Passing None as the max number of replacements is disallowed with the incompatible flag.
    ev.new Scenario(replaceCount(false)).testEval("'banana'.replace('a', 'e', None)", "'benene'");
    ev.new Scenario(replaceCount(true))
        .testIfErrorContains("Cannot pass a None count", "'banana'.replace('a', 'e', None)");
  }

  // Returns semantics for --incompatible_string_replace_count=x.
  private static final StarlarkSemantics replaceCount(boolean x) {
    return StarlarkSemantics.builder()
        .setBool(StarlarkSemantics.INCOMPATIBLE_STRING_REPLACE_COUNT, x)
        .build();
  }
}
