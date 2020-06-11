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

import java.util.List;

/** A visitor for visiting the nodes of a syntax tree in lexical order. */
public class NodeVisitor {

  public void visit(Node node) {
    // dispatch to the node specific method
    node.accept(this);
  }

  // methods dealing with sequences of nodes
  public void visitAll(List<? extends Node> nodes) {
    for (Node node : nodes) {
      visit(node);
    }
  }

  /**
   * Visit a sequence ("block") of statements (e.g. an if branch, for block, function block etc.)
   *
   * This method allows subclasses to handle statement blocks more easily, like doing an action
   * after every statement in a block without having to override visit(...) for all statements.
   *
   * @param statements list of statements in the block
   */
  public void visitBlock(List<Statement> statements) {
    visitAll(statements);
  }

  // node-specific visit methods

  // All four subclasses of Argument are handled together.
  public void visit(Argument node) {
    visit(node.getValue());
  }

  // All four subclasses of Parameter are handled together.
  public void visit(Parameter node) {
    visit(node.getIdentifier());
    if (node.getDefaultValue() != null) {
      visit(node.getDefaultValue());
    }
  }

  public void visit(StarlarkFile node) {
    visitBlock(node.getStatements());
    visitAll(node.getComments());
  }

  public void visit(BinaryOperatorExpression node) {
    visit(node.getX());
    visit(node.getY());
  }

  public void visit(CallExpression node) {
    visit(node.getFunction());
    visitAll(node.getArguments());
  }

  public void visit(Identifier node) {}

  public void visit(Comprehension node) {
    for (Comprehension.Clause clause : node.getClauses()) {
      if (clause instanceof Comprehension.For) {
        visit((Comprehension.For) clause);
      } else {
        visit((Comprehension.If) clause);
      }
    }
    visit(node.getBody());
  }

  public void visit(Comprehension.For node) {
    visit(node.getVars());
    visit(node.getIterable());
  }

  public void visit(Comprehension.If node) {
    visit(node.getCondition());
  }

  public void visit(ForStatement node) {
    visit(node.getCollection());
    visit(node.getVars());
    visitBlock(node.getBody());
  }

  public void visit(LoadStatement node) {
    for (LoadStatement.Binding binding : node.getBindings()) {
      visit(binding.getLocalName());
    }
  }

  public void visit(ListExpression node) {
    visitAll(node.getElements());
  }

  public void visit(@SuppressWarnings("unused") IntegerLiteral node) {}

  public void visit(@SuppressWarnings("unused") StringLiteral node) {}

  public void visit(AssignmentStatement node) {
    visit(node.getRHS());
    visit(node.getLHS());
  }

  public void visit(ExpressionStatement node) {
    visit(node.getExpression());
  }

  public void visit(IfStatement node) {
    visit(node.getCondition());
    visitBlock(node.getThenBlock());
    if (node.getElseBlock() != null) {
      visitBlock(node.getElseBlock());
    }
  }

  public void visit(DefStatement node) {
    visit(node.getIdentifier());
    visitAll(node.getParameters());
    visitBlock(node.getBody());
  }

  public void visit(ReturnStatement node) {
    if (node.getResult() != null) {
      visit(node.getResult());
    }
  }

  public void visit(FlowStatement node) {}

  public void visit(DictExpression node) {
    visitAll(node.getEntries());
  }

  public void visit(DictExpression.Entry node) {
    visit(node.getKey());
    visit(node.getValue());
  }

  public void visit(UnaryOperatorExpression node) {
    visit(node.getX());
  }

  public void visit(DotExpression node) {
    visit(node.getObject());
    visit(node.getField());
  }

  public void visit(IndexExpression node) {
    visit(node.getObject());
    visit(node.getKey());
  }

  public void visit(SliceExpression node) {
    visit(node.getObject());
    if (node.getStart() != null) {
      visit(node.getStart());
    }
    if (node.getStop() != null) {
      visit(node.getStop());
    }
    if (node.getStep() != null) {
      visit(node.getStep());
    }
  }

  public void visit(@SuppressWarnings("unused") Comment node) {}

  public void visit(ConditionalExpression node) {
    visit(node.getCondition());
    visit(node.getThenCase());
    if (node.getElseCase() != null) {
      visit(node.getElseCase());
    }
  }
}
