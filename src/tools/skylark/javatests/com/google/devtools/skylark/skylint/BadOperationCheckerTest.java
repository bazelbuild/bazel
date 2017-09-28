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

@RunWith(JUnit4.class)
public class BadOperationCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return BadOperationChecker.check(ast);
  }

  @Test
  public void dictionaryLiteralPlusOperator() {
    Truth.assertThat(findIssues("{} + foo").toString())
        .contains(":1:1: '+' operator is deprecated and should not be used on dictionaries");
    Truth.assertThat(findIssues("foo + {}").toString())
        .contains(":1:1: '+' operator is deprecated and should not be used on dictionaries");
    Truth.assertThat(findIssues("foo += {}").toString())
        .contains(":1:1: '+' operator is deprecated and should not be used on dictionaries");
  }

  @Test
  public void dictionaryComprehensionPlusOperator() {
    Truth.assertThat(findIssues("{k:v for k,v in []} + foo").toString())
        .contains(":1:1: '+' operator is deprecated and should not be used on dictionaries");
    Truth.assertThat(findIssues("foo + {k:v for k,v in []}").toString())
        .contains(":1:1: '+' operator is deprecated and should not be used on dictionaries");
    Truth.assertThat(findIssues("foo += {k:v for k,v in []}").toString())
        .contains(":1:1: '+' operator is deprecated and should not be used on dictionaries");
  }

  @Test
  public void dictionaryPlusOperatorNested() {
    Truth.assertThat(findIssues("foo + ({} + bar)").toString())
        .contains(":1:7: '+' operator is deprecated and should not be used on dictionaries");
    Truth.assertThat(findIssues("foo + (bar + {})").toString())
        .contains(":1:7: '+' operator is deprecated and should not be used on dictionaries");
  }

  @Test
  public void plusOperatorNoIssue() {
    Truth.assertThat(findIssues("foo + bar")).isEmpty();
    Truth.assertThat(findIssues("foo += bar")).isEmpty();
  }
}
