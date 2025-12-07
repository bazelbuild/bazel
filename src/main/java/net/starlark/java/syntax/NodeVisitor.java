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
package net.starlark.java.syntax;

import java.util.List;

/**
 * A visitor for visiting the nodes of a syntax tree in lexical order (not evaluation order!).
 *
 * <p>Comments are *not* visited.
 *
 * <p>A subclass can change the traversal logic by setting {@link #skipNonSymbolIdentifiers}.
 *
 * <p>Typical usage is for a subclass to just override the {@code visit()} method overloads for the
 * nodes that are relevant to its business logic, and to rely on the default implementations in this
 * class to ensure traversal over the remaining node types. Overriding implementations should
 * remember to traverse children using either {@code super.visit()} on the current node, or explicit
 * calls to {@link #visit(Node)}, {@link #visitAll}, or {@link #visitBlock} on child fields.
 *
 * <p>Contrary to usual Java style, it is *not* recommended to strictly group all overloads of
 * {@code visit()} together, but rather to place helper methods for a specific node type next to its
 * associated {@code visit()} overload. Rationale: The benefit of this style rule is that a reader
 * can rely on the absence of an overload in the immediate vicinity of the method as evidence that
 * no such overload exists. But this isn't helpful in the context of the visitor pattern, where the
 * reader expects there to be many unrelated overloads.
 */
public class NodeVisitor {

  /**
   * If set, we only visit {@link Identifier}s that correspond to a definition or use of a symbol in
   * the current file. Specifically, this omits:
   *
   * <ul>
   *   <li>names of keyword arguments (but not names of keyword parameters!)
   *   <li>field names in dot expressions
   * </ul>
   *
   * <p>Note that {@code Identifier}s in such contexts have no {@link Binding} set for them by the
   * resolver.
   */
  protected boolean skipNonSymbolIdentifiers = false;

  // visit() overloads in this class are ordered by node type, first by category (misc / statement /
  // expression), then alphabetically within category. (Subclasses are not obliged to maintain the
  // same method ordering.)

  /** Entrypoint for visiting a node. Clients should avoid calling node-specific overloads. */
  public void visit(Node node) {
    // Double-dispatch pattern.
    node.accept(this);
  }

  // ==== Miscellaneous node types ====

  /**
   * Handles all four Argument node types uniformly. Subclasses should not add an overload for a
   * concrete Argument subclass; it won't be called.
   */
  public void visit(Argument node) {
    if (!skipNonSymbolIdentifiers && node instanceof Argument.Keyword keyword) {
      visit(keyword.getIdentifier());
    }
    visit(node.getValue());
  }

  /**
   * @deprecated Not supported.
   * @throws UnsupportedOperationException always.
   */
  @Deprecated
  public void visit(@SuppressWarnings("unused") Comment node) {
    // No reason we can't support this if we needed to.
    throw new UnsupportedOperationException("NodeVisitor does not support visiting comments");
  }

  // Clause and Entry are handled below next to Comprehension and Dict respectively.

  /**
   * Handles all four Parameter node types uniformly. Subclasses should not add an overload for a
   * concrete Parameter subclass; it won't be called.
   */
  public void visit(Parameter node) {
    // TODO(#27728): Visit type annotation.
    if (node.getIdentifier() != null) {
      visit(node.getIdentifier());
    }
    if (node.getDefaultValue() != null) {
      visit(node.getDefaultValue());
    }
  }

  public void visit(StarlarkFile node) {
    visitBlock(node.getStatements());
  }

  // ==== Statement nodes ====

  public void visit(AssignmentStatement node) {
    // TODO(#27728): Visit type annotation if present.
    visit(node.getLHS());
    visit(node.getRHS());
  }

  public void visit(ExpressionStatement node) {
    visit(node.getExpression());
  }

  public void visit(FlowStatement node) {}

  public void visit(ForStatement node) {
    visit(node.getVars());
    visit(node.getCollection());
    visitBlock(node.getBody());
  }

  public void visit(DefStatement node) {
    // TODO(#27728): Visit return type annotation.
    visit(node.getIdentifier());
    visitAll(node.getParameters());
    visitBlock(node.getBody());
  }

  public void visit(IfStatement node) {
    visit(node.getCondition());
    visitBlock(node.getThenBlock());
    if (node.getElseBlock() != null) {
      visitBlock(node.getElseBlock());
    }
  }

  public void visit(LoadStatement node) {
    for (LoadStatement.Binding binding : node.getBindings()) {
      visit(binding.getLocalName());
      // We don't visit the original name.
      //
      // Currently, our AST doesn't distinguish between the case when the local name is omitted,
      // versus the case where it is given explicitly and exactly matches the original name. This
      // means that, if we visited both names here, we would end up double-visiting something that
      // often only appears once in the program source.
      //
      // TODO(bazel-team): Disambiguate these cases in LoadStatement.Binding, then visit it here,
      // but ONLY if skipNonSymbolIdentifiers is not set. Mind that subclasses might need updating
      // to continue to avoid traversing the original name.
    }
  }

  public void visit(ReturnStatement node) {
    if (node.getResult() != null) {
      visit(node.getResult());
    }
  }

  public void visit(TypeAliasStatement node) {
    // TODO(#27728): visit children
  }

  public void visit(VarStatement node) {
    visit(node.getIdentifier());
    // TODO(#27728): visit type annotation
  }

  // ==== Expression nodes ====

  public void visit(BinaryOperatorExpression node) {
    visit(node.getX());
    visit(node.getY());
  }

  public void visit(CallExpression node) {
    visit(node.getFunction());
    visitAll(node.getArguments());
  }

  public void visit(CastExpression node) {
    // TODO(#27728): Visit type annotation.
    visit(node.getValue());
  }

  public void visit(Comprehension node) {
    visit(node.getBody());
    for (Comprehension.Clause clause : node.getClauses()) {
      if (clause instanceof Comprehension.For) {
        visit((Comprehension.For) clause);
      } else {
        visit((Comprehension.If) clause);
      }
    }
  }

  public void visit(Comprehension.For node) {
    visit(node.getVars());
    visit(node.getIterable());
  }

  public void visit(Comprehension.If node) {
    visit(node.getCondition());
  }

  public void visit(ConditionalExpression node) {
    visit(node.getThenCase());
    visit(node.getCondition());
    if (node.getElseCase() != null) {
      visit(node.getElseCase());
    }
  }

  public void visit(DictExpression node) {
    visitAll(node.getEntries());
  }

  public void visit(DictExpression.Entry node) {
    visit(node.getKey());
    visit(node.getValue());
  }

  public void visit(DotExpression node) {
    visit(node.getObject());
    if (!skipNonSymbolIdentifiers) {
      visit(node.getField());
    }
  }

  public void visit(Ellipsis node) {}

  public void visit(@SuppressWarnings("unused") FloatLiteral node) {}

  public void visit(Identifier node) {}

  public void visit(IndexExpression node) {
    visit(node.getObject());
    visit(node.getKey());
  }

  public void visit(@SuppressWarnings("unused") IntLiteral node) {}

  public void visit(IsInstanceExpression node) {
    visit(node.getValue());
    // TODO(#27728): Visit type annotation.
  }

  public void visit(LambdaExpression node) {
    visitAll(node.getParameters());
    visit(node.getBody());
  }

  public void visit(ListExpression node) {
    visitAll(node.getElements());
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

  public void visit(@SuppressWarnings("unused") StringLiteral node) {}

  public void visit(UnaryOperatorExpression node) {
    visit(node.getX());
  }

  public void visit(TypeApplication node) {
    visit(node.getConstructor());
    visitAll(node.getArguments());
  }

  // ==== Helpers for sequences of nodes ====

  /**
   * Visits a sequence of nodes (e.g. a list of arguments).
   *
   * <p>See {@link #visitBlock} for a common case.
   */
  // Final because this method is called across completely different categories of nodes, so it is
  // usually a mistake to attempt to override it.
  public final void visitAll(List<? extends Node> nodes) {
    for (Node node : nodes) {
      visit(node);
    }
  }

  /** Convenience/readability method for visiting a block of statements (e.g. an if branch). */
  // Previously this method was non-final and it was recommended to override it to perform an action
  // for every block. However, this is error-prone if the subclass inadvertently calls visitAll()
  // rather than visitBlock() in one of its other overrides. No one seems to be overriding either
  // method, so we made both of them final to avoid this potential problem.
  public final void visitBlock(List<Statement> statements) {
    visitAll(statements);
  }
}
