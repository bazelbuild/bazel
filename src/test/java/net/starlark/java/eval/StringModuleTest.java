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
// TODO(bazel-team): Migrate these test cases back to string_misc.sky.
@RunWith(JUnit4.class)
public class StringModuleTest {

  private final EvaluationTestCase ev = new EvaluationTestCase();

  @Test
  public void testReplace() throws Exception {
    ev.new Scenario()
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

    ev.new Scenario()
        .testEval("'banana'.replace('a', 'o', -2)", "'bonono'")
        .testEval("'banana'.replace('a', 'e', -1)", "'benene'")
        .testEval("'banana'.replace('a', 'e', -10)", "'benene'")
        .testEval("'banana'.replace('', '-', -2)", "'-b-a-n-a-n-a-'");

    ev.new Scenario()
        .testIfErrorContains(
            "parameter 'count' got value of type 'NoneType', want 'int'",
            "'banana'.replace('a', 'e', None)");
  }
}
