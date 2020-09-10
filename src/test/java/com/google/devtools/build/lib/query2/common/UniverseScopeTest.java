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
package com.google.devtools.build.lib.query2.common;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.query2.engine.RdepsFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UniverseScope}. */
@RunWith(JUnit4.class)
public class UniverseScopeTest {
  @Test
  public void testInferFromQueryExpression() throws QuerySyntaxException {
    UniverseScope underTest = UniverseScope.INFER_FROM_QUERY_EXPRESSION;

    ImmutableMap<QueryExpression, ImmutableList<String>> cases =
        ImmutableMap.<QueryExpression, ImmutableList<String>>builder()
            .put(parse("//a/..."), ImmutableList.of("//a/..."))
            .put(parse("//a:a"), ImmutableList.of("//a:a"))
            .put(parse("set(//a:a //b/...)"), ImmutableList.of("//a:a", "//b/..."))
            .put(parse("//a:a + //b/..."), ImmutableList.of("//a:a", "//b/..."))
            .put(parse("let x = a/... in $x + a/... + b/..."), ImmutableList.of("a/...", "b/..."))
            .put(
                parse("rdeps(a:a, b:b) - rdeps(c:c, d:d)"),
                ImmutableList.of("a:a", "b:b", "c:c", "d:d"))
            .build();
    cases.forEach(
        (expr, expectedInferredTargetPatterns) -> {
          assertThat(underTest.getUniverseKey(expr, PathFragment.EMPTY_FRAGMENT).getPatterns())
              .isEqualTo(expectedInferredTargetPatterns);
        });
  }

  private static QueryExpression parse(String exprString) throws QuerySyntaxException {
    return QueryParser.parse(exprString, ImmutableMap.of("rdeps", new RdepsFunction()));
  }
}
