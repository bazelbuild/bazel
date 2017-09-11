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
  public void testBadLetterCase() throws Exception {
    String errorMessage =
        findIssues(
                "badGlobalVariableName = 0",
                "def BAD_FUNCTION_NAME(BadParameterName):",
                "  badLocalVariableName = 1")
            .toString();
    Truth.assertThat(errorMessage)
        .contains(
            "'badGlobalVariableName' should be lower_snake_case (for variables)"
                + " or UPPER_SNAKE_CASE (for constants)");
    Truth.assertThat(errorMessage).contains("'BAD_FUNCTION_NAME' should be lower_snake_case");
    Truth.assertThat(errorMessage).contains("'BadParameterName' should be lower_snake_case");
    Truth.assertThat(errorMessage)
        .contains(
            "'badLocalVariableName' should be lower_snake_case (for variables)"
                + " or UPPER_SNAKE_CASE (for constants)");
  }

  @Test
  public void testConfusingNames() throws Exception {
    String errorMessage = findIssues("O = 0", "def fail():", "  True = False").toString();
    Truth.assertThat(errorMessage)
        .contains(
            "never use 'l', 'I', or 'O' as names"
                + " (they're too easily confused with 'I', 'l', or '0')");
    Truth.assertThat(errorMessage)
        .contains("identifier 'fail' shadows a builtin; please pick a different name");
    Truth.assertThat(errorMessage)
        .contains("identifier 'True' shadows a builtin; please pick a different name");
  }

  @Test
  public void testUnderscoreNames() throws Exception {
    Truth.assertThat(findIssues("a, _ = (1, 2) # underscore to ignore assignment")).isEmpty();
    Truth.assertThat(findIssues("_ = 1", "print(_)").toString())
        .contains(
            ":2:7: don't use '_' as an identifier, only to ignore the result in an assignment");
    Truth.assertThat(findIssues("__ = 1").toString())
        .contains(
            ":1:1: identifier '__' consists only of underscores; please pick a different name");
  }

  @Test
  public void testNoIssues() throws Exception {
    Truth.assertThat(
            findIssues(
                "GOOD_GLOBAL_VARIABLE_NAME = 0",
                "def good_function_name(good_parameter_name):",
                "  GOOD_LOCAL_CONSTANT_NAME = 1"))
        .isEmpty();
  }

  @Test
  public void testProviderNameMustBeCamelCase() throws Exception {
    Truth.assertThat(findIssues("FooBar = provider()")).isEmpty();
    Truth.assertThat(findIssues("foo_bar = provider()").toString())
        .contains("provider name 'foo_bar' should be UpperCamelCase");
  }

  @Test
  public void testNoDuplicates() throws Exception {
    Truth.assertThat(findIssues("def foo():", "  badName = 1", "  badName += 1")).hasSize(1);
    Truth.assertThat(findIssues("def foo():", "  Bad = 1", "  [[] for Bad in []]")).hasSize(2);
  }
}
