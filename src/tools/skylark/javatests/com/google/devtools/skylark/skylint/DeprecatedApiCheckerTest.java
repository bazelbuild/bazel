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
        .contains("1:1-1:10: ctx.action is deprecated: Use ctx.actions.run");
    Truth.assertThat(findIssues("ctx.empty_action").toString())
        .contains("1:1-1:16: ctx.empty_action is deprecated");
    Truth.assertThat(findIssues("ctx.default_provider()").toString())
        .contains("1:1-1:20: ctx.default_provider is deprecated");
    Truth.assertThat(findIssues("PACKAGE_NAME").toString())
        .contains("1:1-1:12: PACKAGE_NAME is deprecated");
    Truth.assertThat(findIssues("f = ctx.outputs.executable").toString())
        .contains("1:5-1:26: ctx.outputs.executable is deprecated");
    Truth.assertThat(findIssues("css_filetype = FileType(['.css'])").toString())
        .contains("1:16-1:23: FileType is deprecated");

    Truth.assertThat(findIssues("ctx.actions()")).isEmpty();
  }

  @Test
  public void testRuleImplReturnValue() {
    Truth.assertThat(
            findIssues("def _impl(ctx): return struct()", "x = rule(implementation=_impl)")
                .toString())
        .contains("1:17-1:31: Avoid using the legacy provider syntax.");

    Truth.assertThat(
            findIssues(
                    "def _impl(ctx):",
                    "  if True: return struct()",
                    "  return",
                    "x = rule(_impl, attrs = {})")
                .toString())
        .contains("2:12-2:26: Avoid using the legacy provider syntax.");

    Truth.assertThat(
            findIssues(
                "def _impl(): return struct()",
                "def _impl2(): return []",
                "x = rule(",
                "  implementation=_impl2,",
                ")"))
        .isEmpty();
  }
}
