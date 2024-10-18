// Copyright 2022 The Bazel Authors. All rights reserved.
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

package net.starlark.java.syntax;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

abstract class CoverageVisitor extends NodeVisitor {

  private static class FunctionFrame {

    final String name;
    int lambdaCount;

    FunctionFrame(String name) {
      this.name = name;
      this.lambdaCount = 0;
    }
  }

  private final List<FunctionFrame> functionStack = new ArrayList<>();

  CoverageVisitor() {
    functionStack.add(new FunctionFrame("<top-level>"));
  }

  /**
   * Called for every (possibly nested, possibly lambda) function.
   *
   * @param identifier         a human-readable identifier for the function that is unique within
   *                           the entire source file
   * @param defStatement       the {@link Node} representing the function definition
   * @param firstBodyStatement the {@link Node} representing the first statement of the function's
   *                           body, which can be used to track how often the function has been
   *                           executed
   */
  abstract protected void visitFunction(String identifier, Node defStatement,
      Node firstBodyStatement);

  /**
   * Called for every conditional jump to either one of two successor nodes depending on a
   * condition.
   * <p>
   * Note: Any conditional branch in Starlark always has two successors, never more.
   *
   * @param owner                   the {@code Node} at whose location the branch should be
   *                                reported
   * @param condition               the {@code Node} representing the branch condition
   * @param positiveUniqueSuccessor a {@code Node} that is executed if and only if the "positive"
   *                                branch has been taken (e.g., the if condition was satisfied or
   *                                the iterable in a for loop has more elements). The node must not
   *                                be executed in any other situation.
   * @param negativeUniqueSuccessor a {@code Node} that is executed if and only if the "negative"
   *                                branch has been taken (e.g., the if condition was not satisfied
   *                                or the iterable in a for loop contains no more elements). The
   *                                node must not be executed in any other situation. May be
   *                                {@code null}, in which case the branch has to be marked as
   *                                executed manually via a call to
   *                                {@link CoverageRecorder#recordVirtualJump(Node)} with the
   *                                argument {@code owner}.
   */
  abstract protected void visitBranch(Node owner, Node condition, Node positiveUniqueSuccessor,
      @Nullable Node negativeUniqueSuccessor);

  /**
   * Called for every {@code Node} that corresponds to executable code. If node A contains node B in
   * its lexical scope, then {@code visitCode(A)} is called before {@code visitCode(B)}.
   */
  abstract protected void visitCode(Node node);

  private String enterFunction(Identifier identifier) {
    String name = identifier != null
        ? identifier.getName()
        : "lambda " + functionStack.get(functionStack.size() - 1).lambdaCount++;
    functionStack.add(new FunctionFrame(name));
    return functionStack.stream().skip(1).map(f -> f.name).collect(Collectors.joining(" > "));
  }

  private void leaveFunction() {
    functionStack.remove(functionStack.size() - 1);
  }

  @Override
  final public void visit(Argument node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(Parameter node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(@Nullable Identifier node) {
    // node can be null in the case of an anonymous vararg parameter, e.g.:
    //  def f(..., *, ...): ...
    if (node != null) {
      visitCode(node);
    }
    super.visit(node);
  }

  @Override
  final public void visit(BinaryOperatorExpression node) {
    visitCode(node);
    if (node.getOperator() == TokenKind.AND || node.getOperator() == TokenKind.OR) {
      // Manually track the short-circuit case.
      visitBranch(node, node.getX(), node.getY(), null);
    }
    super.visit(node);
  }

  @Override
  final public void visit(CallExpression node) {
    visitCode(node);
    super.visit(node);
  }

  private Node getClauseCondition(Node clause) {
    return clause instanceof Comprehension.If
        ? ((Comprehension.If) clause).getCondition()
        : ((Comprehension.For) clause).getIterable();
  }

  private void visitClauseBranches(Node clause, Node successor) {
    Node condition = getClauseCondition(clause);
    visitBranch(clause, condition, successor, null);
  }

  @Override
  final public void visit(Comprehension node) {
    Comprehension.Clause lastClause = null;
    for (Comprehension.Clause clause : node.getClauses()) {
      visitCode(clause);
      if (lastClause != null) {
        visitClauseBranches(lastClause, getClauseCondition(clause));
      }
      lastClause = clause;
      visit(clause);
    }
    if (lastClause != null) {
      visitClauseBranches(lastClause, node.getBody());
    }
    visit(node.getBody());
  }

  @Override
  final public void visit(Comprehension.For node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(Comprehension.If node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(ForStatement node) {
    visitCode(node);
    visitBranch(node, node.getCollection(), node.getBody().get(0), null);
    super.visit(node);
  }

  @Override
  final public void visit(ListExpression node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(IntLiteral node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(FloatLiteral node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(StringLiteral node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(AssignmentStatement node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(ExpressionStatement node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(IfStatement node) {
    visitCode(node);
    visitBranch(node,
        node.getCondition(),
        node.getThenBlock().get(0),
        node.getElseBlock() != null ? node.getElseBlock().get(0) : null);
    super.visit(node);
  }

  @Override
  final public void visit(DefStatement node) {
    visitCode(node);
    visitFunction(enterFunction(node.getIdentifier()), node, node.getBody().get(0));
    super.visit(node);
    leaveFunction();
  }

  @Override
  final public void visit(ReturnStatement node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(FlowStatement node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(DictExpression node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(DictExpression.Entry node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(UnaryOperatorExpression node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(DotExpression node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(IndexExpression node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(LambdaExpression node) {
    visitCode(node);
    visitFunction(enterFunction(null), node, node.getBody());
    super.visit(node);
    leaveFunction();
  }

  @Override
  final public void visit(SliceExpression node) {
    visitCode(node);
    super.visit(node);
  }

  @Override
  final public void visit(ConditionalExpression node) {
    visitCode(node);
    visitBranch(node, node.getCondition(), node.getThenCase(), node.getElseCase());
    super.visit(node);
  }

  // The following functions intentionally do not call visitCode as their nodes do not correspond to
  // executable code or they already delegate to functions that do.

  @Override
  final public void visit(LoadStatement node) {
    super.visit(node);
  }

  @Override
  final public void visit(Comment node) {
    super.visit(node);
  }

  @Override
  final public void visit(Node node) {
    super.visit(node);
  }

  @Override
  final public void visit(StarlarkFile node) {
    super.visit(node);
  }

  @Override
  final public void visitAll(List<? extends Node> nodes) {
    super.visitAll(nodes);
  }

  @Override
  final public void visitBlock(List<Statement> statements) {
    super.visitBlock(statements);
  }
}
