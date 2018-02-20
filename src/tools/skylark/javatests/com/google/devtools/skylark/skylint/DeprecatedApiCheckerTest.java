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
public class DeprecatedApiCheckerTest {
  private static List<Issue> findIssues(String... lines) {
    String content = String.join("\n", lines);
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              throw new IllegalArgumentException(event.getMessage());
            },
            content);
    return DeprecatedApiChecker.check(ast);
  }

  @Test
  public void deprecatedCtxMethods() {
    Truth.assertThat(findIssues("ctx.action()").toString())
        .contains("1:1-1:10: This method is deprecated");
    Truth.assertThat(findIssues("ctx.empty_action").toString())
        .contains("1:1-1:16: This method is deprecated");
    Truth.assertThat(findIssues("ctx.default_provider()").toString())
        .contains("1:1-1:20: This method is deprecated");

    Truth.assertThat(findIssues("ctx.actions()")).isEmpty();
  }
}
