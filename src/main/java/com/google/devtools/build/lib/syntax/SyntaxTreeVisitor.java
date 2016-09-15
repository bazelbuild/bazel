// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.syntax.DictionaryLiteral.DictionaryEntryLiteral;
import com.google.devtools.build.lib.syntax.IfStatement.ConditionalStatements;

import java.util.List;

/**
 * A visitor for visiting the nodes in the syntax tree left to right, top to
 * bottom.
 */
public class SyntaxTreeVisitor {

  public void visit(ASTNode node) {
    // dispatch to the node specific method
    node.accept(this);
  }

  public void visitAll(List<? extends ASTNode> nodes) {
    for (ASTNode node : nodes) {
      visit(node);
    }
  }

  // node specific visit methods
  public void visit(Argument.Passed node) {
    visit(node.getValue());
  }

  public void visit(@SuppressWarnings("unused") Parameter<?, ?> node) {
    // leaf node (we need the function for overrides)
  }

  public void visit(BuildFileAST node) {
    visitAll(node.getStatements());
    visitAll(node.getComments());
  }

  public void visit(BinaryOperatorExpression node) {
    visit(node.getLhs());
    visit(node.getRhs());
  }

  public void visit(FuncallExpression node) {
    if (node.getObject() != null) {
      visit(node.getObject());
    }
    visit(node.getFunction());
    visitAll(node.getArguments());
  }

  public void visit(@SuppressWarnings("unused") Identifier node) {}

  public void visit(AbstractComprehension node) {
    visitAll(node.getOutputExpressions());

    for (ListComprehension.Clause clause : node.getClauses()) {
      if (clause.getLValue() != null) {
        visit(clause.getLValue());
      }
      visit(clause.getExpression());
    }
  }

  public void visit(ForStatement node) {
    visit(node.getVariable().getExpression());
    visit(node.getCollection());
    visitAll(node.block());
  }

  public void visit(LoadStatement node) {
    visitAll(node.getSymbols());
  }

  public void visit(ListLiteral node) {
    visitAll(node.getElements());
  }

  public void visit(@SuppressWarnings("unused") IntegerLiteral node) {}

  public void visit(@SuppressWarnings("unused") StringLiteral node) {}

  public void visit(LValue node) {
    visit(node.getExpression());
  }

  public void visit(AssignmentStatement node) {
    visit(node.getLValue());
    visit(node.getExpression());
  }

  public void visit(ExpressionStatement node) {
    visit(node.getExpression());
  }

  public void visit(IfStatement node) {
    visitAll(node.getThenBlocks());
    visitAll(node.getElseBlock());
  }

  public void visit(ConditionalStatements node) {
    visit(node.getCondition());
    visitAll(node.getStmts());
  }

  public void visit(FunctionDefStatement node) {
    visit(node.getIdent());
    visitAll(node.getParameters());
    visitAll(node.getStatements());
  }

  public void visit(ReturnStatement node) {
    visit(node.getReturnExpression());
  }

  public void visit(DictionaryLiteral node) {
    visitAll(node.getEntries());
  }

  public void visit(DictionaryEntryLiteral node) {
    visit(node.getKey());
    visit(node.getValue());
  }

  public void visit(NotExpression node) {
    visit(node.getExpression());
  }

  public void visit(DotExpression node) {
    visit(node.getObj());
    visit(node.getField());
  }

  public void visit(IndexExpression node) {
    visit(node.getObject());
    visit(node.getKey());
  }

  public void visit(SliceExpression node) {
    visit(node.getObject());
    visit(node.getStart());
    visit(node.getEnd());
    visit(node.getStep());
  }

  public void visit(@SuppressWarnings("unused") Comment node) {}

  public void visit(ConditionalExpression node) {
    visit(node.getThenCase());
    visit(node.getCondition());
    if (node.getElseCase() != null) {
      visit(node.getElseCase());
    }
  }
}
