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

/** Tests the lint done by {@link DocstringChecker}. */
@RunWith(JUnit4.class)
public class DocstringCheckerTests {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseSkylarkString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return DocstringChecker.check(ast);
  }

  @Test
  public void reportMissingDocString() throws Exception {
    String errorMessage =
        findIssues("# no module docstring", "def function():", "  pass # no function docstring")
            .toString();
    Truth.assertThat(errorMessage).contains(":2:1: file has no module docstring");
    Truth.assertThat(errorMessage).contains(":2:1: function 'function' has no docstring");
  }

  @Test
  public void dontReportExistingDocstrings() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\" This is a module docstring",
                "\n\"\"\"",
                "def function():",
                "  \"\"\" This is a function docstring\n\"\"\""))
        .isEmpty();
  }

  @Test
  public void dontReportPrivateFunctionWithoutDocstring() throws Exception {
    Truth.assertThat(
            findIssues(
                "\"\"\" Module docstring\n\"\"\"",
                "def _private_function():",
                "  pass # no docstring necessary for private functions"))
        .isEmpty();
  }
}
