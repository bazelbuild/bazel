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

import com.google.devtools.build.lib.syntax.AugmentedAssignmentStatement;
import com.google.devtools.build.lib.syntax.BinaryOperatorExpression;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.DictComprehension;
import com.google.devtools.build.lib.syntax.DictionaryLiteral;
import com.google.devtools.build.lib.syntax.Operator;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import java.util.ArrayList;
import java.util.List;

/** Checks for operations that are deprecated */
public class BadOperationChecker extends SyntaxTreeVisitor {
  private final List<Issue> issues = new ArrayList<>();

  private BadOperationChecker() {}

  public static List<Issue> check(BuildFileAST ast) {
    BadOperationChecker checker = new BadOperationChecker();
    checker.visit(ast);
    return checker.issues;
  }

  @Override
  public void visit(BinaryOperatorExpression node) {
    super.visit(node);
    if (node.getOperator() == Operator.PLUS) {
      if (node.getLhs() instanceof DictionaryLiteral
          || node.getLhs() instanceof DictComprehension
          || node.getRhs() instanceof DictionaryLiteral
          || node.getRhs() instanceof DictComprehension) {
        issues.add(
            new Issue(
                "'+' operator is deprecated and should not be used on dictionaries",
                node.getLocation()));
      }
    }
  }

  @Override
  public void visit(AugmentedAssignmentStatement node) {
    super.visit(node);
    if (node.getExpression() instanceof DictionaryLiteral
        || node.getExpression() instanceof DictComprehension) {
      issues.add(
          new Issue(
              "'+' operator is deprecated and should not be used on dictionaries",
              node.getLocation()));
    }
  }
}
