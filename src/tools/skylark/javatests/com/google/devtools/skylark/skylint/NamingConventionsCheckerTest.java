// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.skylint;

import com.google.common.truth.Truth;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the lint done by {@link com.google.devtools.skylark.skylint.NamingConventionsChecker}. */
@RunWith(JUnit4.class)
public class NamingConventionsCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return NamingConventionsChecker.check(ast);
  }

  @Test
  public void testMessages() throws Exception {
    String errorMessage =
        findIssues(
                "badGlobalVariableName = 0",
                "def badFunctionName(BadParameterName):",
                "  BAD_LOCAL_VARIABLE_NAME = 1")
            .toString();
    Truth.assertThat(errorMessage)
        .contains("'badGlobalVariableName' should be lower_snake_case or UPPER_SNAKE_CASE");
    Truth.assertThat(errorMessage).contains("'badFunctionName' should be lower_snake_case");
    Truth.assertThat(errorMessage).contains("'BadParameterName' should be lower_snake_case");
    Truth.assertThat(errorMessage).contains("'BAD_LOCAL_VARIABLE_NAME' should be lower_snake_case");
  }

  @Test
  public void testNoIssues() throws Exception {
    Truth.assertThat(
            findIssues(
                "GOOD_GLOBAL_VARIABLE_NAME = 0",
                "def good_function_name(good_parameter_name):",
                "  good_local_variable_name = 1"))
        .isEmpty();
  }

  @Test
  public void testNoDuplicates() throws Exception {
    Truth.assertThat(findIssues("def foo():", "  BAD = 1", "  BAD += 1")).hasSize(1);
  }
}
