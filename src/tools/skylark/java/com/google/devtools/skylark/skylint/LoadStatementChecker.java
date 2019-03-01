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

import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.skylark.common.DocstringUtils;
import java.util.ArrayList;
import java.util.List;

/** Checks that load statements are at the top of a file (after the docstring). */
public class LoadStatementChecker {
  private static final String LOAD_AT_TOP_CATEGORY = "load-at-top";

  private LoadStatementChecker() {}

  public static List<Issue> check(BuildFileAST ast) {
    List<Issue> issues = new ArrayList<>();
    List<Statement> statements = ast.getStatements();
    int firstStatementIndex = DocstringUtils.extractDocstring(statements) == null ? 0 : 1;
    boolean loadStatementsExpected = true;
    for (int i = firstStatementIndex; i < statements.size(); i++) {
      Statement statement = statements.get(i);
      if (statement instanceof LoadStatement) {
        if (!loadStatementsExpected) {
          issues.add(
              Issue.create(
                  LOAD_AT_TOP_CATEGORY,
                  "load statement should be at the top of the file (after the docstring)",
                  statement.getLocation()));
        }
      } else {
        loadStatementsExpected = false;
      }
    }
    return issues;
  }
}
