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
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.IfStatement;
import com.google.devtools.build.lib.syntax.IfStatement.ConditionalStatements;
import com.google.devtools.build.lib.syntax.ReturnStatement;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Performs lints related to control flow.
 *
 * <p>For now, it only checks that if a function returns a value in some execution paths, it does so
 * in all execution paths.
 */
// TODO(skylark-team): Check for unreachable statements
public class ControlFlowChecker extends SyntaxTreeVisitor {
  private final List<Issue> issues = new ArrayList<>();
  private ControlFlowInfo cf = new ControlFlowInfo();
  private List<ReturnStatement> returnStatementsWithoutValue = new ArrayList<>();

  public static List<Issue> check(BuildFileAST ast) {
    ControlFlowChecker checker = new ControlFlowChecker();
    checker.visit(ast);
    return checker.issues;
  }

  @Override
  public void visit(IfStatement node) {
    ControlFlowInfo saved = cf;
    boolean someBranchHasReturnWithValue = false;
    boolean someBranchHasReturnWithoutValue = false;
    boolean allBranchesReturnAlwaysExplicitly = true;
    Stream<List<Statement>> branches =
        Stream.concat(
            node.getThenBlocks().stream().map(ConditionalStatements::getStatements),
            Stream.of(node.getElseBlock()));
    for (List<Statement> branch : (Iterable<List<Statement>>) branches::iterator) {
      cf = new ControlFlowInfo();
      visitAll(branch);
      someBranchHasReturnWithValue |= cf.hasReturnWithValue;
      someBranchHasReturnWithoutValue |= cf.hasReturnWithoutValue;
      allBranchesReturnAlwaysExplicitly &= cf.returnsAlwaysExplicitly;
    }
    cf.hasReturnWithValue = saved.hasReturnWithValue || someBranchHasReturnWithValue;
    cf.hasReturnWithoutValue = saved.hasReturnWithoutValue || someBranchHasReturnWithoutValue;
    cf.returnsAlwaysExplicitly = saved.returnsAlwaysExplicitly || allBranchesReturnAlwaysExplicitly;
  }

  @Override
  public void visit(ReturnStatement node) {
    cf.returnsAlwaysExplicitly = true;
    if (node.getReturnExpression() != null) {
      cf.hasReturnWithValue = true;
    } else {
      cf.hasReturnWithoutValue = true;
      returnStatementsWithoutValue.add(node);
    }
  }

  @Override
  public void visit(ExpressionStatement node) {
    if (isFail(node.getExpression())) {
      cf.returnsAlwaysExplicitly = true;
    }
  }

  private boolean isFail(Expression expression) {
    if (expression instanceof FuncallExpression) {
      Expression function = ((FuncallExpression) expression).getFunction();
      return function instanceof Identifier && ((Identifier) function).getName().equals("fail");
    }
    return false;
  }

  @Override
  public void visit(FunctionDefStatement node) {
    cf = new ControlFlowInfo();
    returnStatementsWithoutValue = new ArrayList<>();
    super.visit(node);
    if (cf.hasReturnWithValue && (!cf.returnsAlwaysExplicitly || cf.hasReturnWithoutValue)) {
      issues.add(
          new Issue(
              "some but not all execution paths of '" + node.getIdentifier() + "' return a value",
              node.getLocation()));
      for (ReturnStatement returnStatement : returnStatementsWithoutValue) {
        issues.add(
            new Issue(
                "return value missing (you can `return None` if this is desired)",
                returnStatement.getLocation()));
      }
    }
  }

  private static class ControlFlowInfo {
    boolean hasReturnWithValue = false;
    boolean hasReturnWithoutValue = false;
    boolean returnsAlwaysExplicitly = false;
  }
}
