// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import static com.google.common.base.MoreObjects.firstNonNull;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.ConditionalExpressionTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.ParenthesizedTree;
import com.sun.source.tree.Tree.Kind;
import com.sun.source.tree.TypeCastTree;
import com.sun.source.tree.UnaryTree;
import com.sun.source.util.SimpleTreeVisitor;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCBlock;
import com.sun.tools.javac.tree.JCTree.JCExpression;
import com.sun.tools.javac.tree.JCTree.JCExpressionStatement;
import com.sun.tools.javac.tree.JCTree.JCIdent;
import com.sun.tools.javac.tree.JCTree.JCMethodDecl;
import com.sun.tools.javac.tree.JCTree.JCMethodInvocation;
import com.sun.tools.javac.tree.JCTree.JCStatement;
import com.sun.tools.javac.tree.JCTree.JCVariableDecl;
import com.sun.tools.javac.tree.TreeScanner;
import com.sun.tools.javac.util.List;
import com.sun.tools.javac.util.Name;

/**
 * Prunes AST nodes that are not required for header compilation.
 *
 * <p>Used by Turbine after parsing and before all subsequent phases to avoid
 * doing unnecessary work.
 */
public class TreePruner {

  /**
   * Prunes AST nodes that are not required for header compilation.
   *
   * <p>Specifically:
   *
   * <ul>
   * <li>method bodies
   * <li>class and instance initializer blocks
   * <li>initializers of definitely non-constant fields
   * </ul>
   */
  static void prune(JCTree tree) {
    tree.accept(PRUNING_VISITOR);
  }

  /** A {@link TreeScanner} that deletes method bodies and blocks from the AST. */
  private static final TreeScanner PRUNING_VISITOR =
      new TreeScanner() {

        @Override
        public void visitMethodDef(JCMethodDecl tree) {
          if (tree.body == null) {
            return;
          }
          if (tree.getReturnType() == null && delegatingConstructor(tree.body.stats)) {
            // if the first statement of a constructor declaration delegates to another
            // constructor, it needs to be preserved to satisfy checks in Resolve
            tree.body.stats = com.sun.tools.javac.util.List.of(tree.body.stats.get(0));
            return;
          }
          tree.body.stats = com.sun.tools.javac.util.List.nil();
        }

        @Override
        public void visitBlock(JCBlock tree) {
          tree.stats = List.nil();
        }

        @Override
        public void visitVarDef(JCVariableDecl tree) {
          if ((tree.mods.flags & Flags.ENUM) == Flags.ENUM) {
            // javac desugars enum constants into fields during parsing
            return;
          }
          // drop field initializers unless the field looks like a JLS §4.12.4 constant variable
          if (isConstantVariable(tree)) {
            return;
          }
          tree.init = null;
        }
      };

  private static boolean delegatingConstructor(List<JCStatement> stats) {
    if (stats.isEmpty()) {
      return false;
    }
    JCStatement stat = stats.get(0);
    if (stat.getKind() != Kind.EXPRESSION_STATEMENT) {
      return false;
    }
    JCExpression expr = ((JCExpressionStatement) stat).getExpression();
    if (expr.getKind() != Kind.METHOD_INVOCATION) {
      return false;
    }
    JCExpression select = ((JCMethodInvocation) expr).getMethodSelect();
    if (select.getKind() != Kind.IDENTIFIER) {
      return false;
    }
    Name name = ((JCIdent) select).getName();
    return name.contentEquals("this") || name.contentEquals("super");
  }

  private static boolean isConstantVariable(JCVariableDecl tree) {
    if ((tree.mods.flags & Flags.FINAL) != Flags.FINAL) {
      return false;
    }
    if (!constantType(tree.getType())) {
      return false;
    }
    if (tree.getInitializer() != null) {
      Boolean result = tree.getInitializer().accept(CONSTANT_VISITOR, null);
      if (result == null || !result) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns true iff the given tree could be the type name of a constant type.
   *
   * <p>This is a conservative over-approximation: an identifier named {@code String}
   * isn't necessarily a type name, but this is used at parse-time before types have
   * been attributed.
   */
  private static boolean constantType(JCTree tree) {
    switch (tree.getKind()) {
      case PRIMITIVE_TYPE:
        return true;
      case IDENTIFIER:
        return tree.toString().contentEquals("String");
      case MEMBER_SELECT:
        return tree.toString().contentEquals("java.lang.String");
      default:
        return false;
    }
  }

  /** A visitor that identifies JLS §15.28 constant expressions. */
  private static final SimpleTreeVisitor<Boolean, Void> CONSTANT_VISITOR =
      new SimpleTreeVisitor<Boolean, Void>(false) {

        @Override
        public Boolean visitConditionalExpression(ConditionalExpressionTree node, Void p) {
          return reduce(
              node.getCondition().accept(this, null),
              node.getTrueExpression().accept(this, null),
              node.getFalseExpression().accept(this, null));
        }

        @Override
        public Boolean visitParenthesized(ParenthesizedTree node, Void p) {
          return node.getExpression().accept(this, null);
        }

        @Override
        public Boolean visitUnary(UnaryTree node, Void p) {
          switch (node.getKind()) {
            case UNARY_PLUS:
            case UNARY_MINUS:
            case BITWISE_COMPLEMENT:
            case LOGICAL_COMPLEMENT:
              break;
            default:
              // non-constant unary expression
              return false;
          }
          return node.getExpression().accept(this, null);
        }

        @Override
        public Boolean visitBinary(BinaryTree node, Void p) {
          switch (node.getKind()) {
            case MULTIPLY:
            case DIVIDE:
            case REMAINDER:
            case PLUS:
            case MINUS:
            case LEFT_SHIFT:
            case RIGHT_SHIFT:
            case UNSIGNED_RIGHT_SHIFT:
            case LESS_THAN:
            case LESS_THAN_EQUAL:
            case GREATER_THAN:
            case GREATER_THAN_EQUAL:
            case AND:
            case XOR:
            case OR:
            case CONDITIONAL_AND:
            case CONDITIONAL_OR:
            case EQUAL_TO:
            case NOT_EQUAL_TO:
              break;
            default:
              // non-constant binary expression
              return false;
          }
          return reduce(
              node.getLeftOperand().accept(this, null), node.getRightOperand().accept(this, null));
        }

        @Override
        public Boolean visitTypeCast(TypeCastTree node, Void p) {
          return reduce(
              constantType((JCTree) node.getType()), node.getExpression().accept(this, null));
        }

        @Override
        public Boolean visitMemberSelect(MemberSelectTree node, Void p) {
          return node.getExpression().accept(this, null);
        }

        @Override
        public Boolean visitIdentifier(IdentifierTree node, Void p) {
          // Assume all variables are constant variables. This is a conservative assumption, but
          // it's the best we can do with only syntactic information.
          return true;
        }

        @Override
        public Boolean visitLiteral(LiteralTree node, Void unused) {
          switch (node.getKind()) {
            case STRING_LITERAL:
            case INT_LITERAL:
            case LONG_LITERAL:
            case FLOAT_LITERAL:
            case DOUBLE_LITERAL:
            case BOOLEAN_LITERAL:
            case CHAR_LITERAL:
              return true;
            default:
              return false;
          }
        }

        public boolean reduce(Boolean... bx) {
          boolean r = true;
          for (Boolean b : bx) {
            r &= firstNonNull(b, false);
          }
          return r;
        }
      };
}
