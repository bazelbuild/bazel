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
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ExpressionStatement;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import java.util.ArrayList;
import java.util.List;

/** Checks the existence of docstrings. */
public class DocstringChecker extends SyntaxTreeVisitor {

  private final List<Issue> issues = new ArrayList<>();

  public static List<Issue> check(BuildFileAST ast) {
    DocstringChecker checker = new DocstringChecker();
    ast.accept(checker);
    return checker.issues;
  }

  @Override
  public void visit(BuildFileAST node) {
    String moduleDocstring = extractDocstring(node.getStatements());
    if (moduleDocstring == null) {
      issues.add(new Issue("file has no module docstring", node.getLocation()));
    }
    super.visit(node);
  }

  @Override
  public void visit(FunctionDefStatement node) {
    String functionDocstring = extractDocstring(node.getStatements());
    if (functionDocstring == null && !node.getIdentifier().getName().startsWith("_")) {
      issues.add(
          new Issue(
              "function '" + node.getIdentifier().getName() + "' has no docstring",
              node.getLocation()));
    }
  }

  private static String extractDocstring(List<Statement> statements) {
    if (statements.isEmpty()) {
      return null;
    }
    Statement statement = statements.get(0);
    if (statement instanceof ExpressionStatement) {
      Expression expr = ((ExpressionStatement) statement).getExpression();
      if (expr instanceof StringLiteral) {
        return ((StringLiteral) expr).getValue();
      }
    }
    return null;
  }
}
