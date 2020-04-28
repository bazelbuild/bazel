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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.Arrays;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;
import org.junit.Test;

/**
 * Tests for SkylarkStringModule.
 */
@RunWith(Parameterized.class)
public class StringModuleTest extends EvaluationTestCase {

  @Parameters(name = "{index}: flag={0}")
  public static Iterable<? extends Object> data() {
      return Arrays.asList(
          "--incompatible_string_replace_count=false",
          "--incompatible_string_replace_count=true");
  }

  @Parameter
  public String flag;

  @Test
  public void testReplace() throws Exception {
      // Test that the behaviour is the same for the basic case both before
      // and after the incompatible change.
      new Scenario(flag)
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
  public void testReplaceIncompatibleFlag() throws Exception {
      // Test the scenario that changes with the incompatible flag
      new Scenario("--incompatible_string_replace_count=false")
        .testEval("'banana'.replace('a', 'o', -2)", "'banana'")
        .testEval("'banana'.replace('a', 'e', -1)", "'banana'")
        .testEval("'banana'.replace('a', 'e', -10)", "'banana'")
        .testEval("'banana'.replace('', '-', -2)", "'banana'");

      new Scenario("--incompatible_string_replace_count=true")
        .testEval("'banana'.replace('a', 'o', -2)", "'bonono'")
        .testEval("'banana'.replace('a', 'e', -1)", "'benene'")
        .testEval("'banana'.replace('a', 'e', -10)", "'benene'")
        .testEval("'banana'.replace('', '-', -2)", "'-b-a-n-a-n-a-'");
  }
}
