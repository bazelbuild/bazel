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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkStringModule.
 */
@RunWith(JUnit4.class)
public class StringModuleTest extends EvaluationTestCase {

  @Test
  public void testReplace() throws Exception {
      // Test that the behaviour is the same for the basic case both before
      // and after the incompatible change.
      new Scenario("--incompatible_string_replace_count=false")
        .testEval("\"banana\".replace(\"a\", \"o\")", "\"bonono\"")
        .testEval("\"banana\".replace(\"a\", \"o\", 2)", "\"bonona\"")
        .testEval("\"banana\".replace(\"a\", \"o\", 0)", "\"banana\"");

      new Scenario("--incompatible_string_replace_count=true")
        .testEval("\"banana\".replace(\"a\", \"o\")", "\"bonono\"")
        .testEval("\"banana\".replace(\"a\", \"o\", 2)", "\"bonona\"")
        .testEval("\"banana\".replace(\"a\", \"o\", 0)", "\"banana\"");
  }

  @Test
  public void testReplaceIncompatibleChange() throws Exception {
      new Scenario("--incompatible_string_replace_count=false")
        .testEval("\"banana\".replace(\"a\", \"o\", -2)", "\"banana\"");

      new Scenario("--incompatible_string_replace_count=true")
        .testEval("\"banana\".replace(\"a\", \"o\", -2)", "\"bonono\"");
  }
}
