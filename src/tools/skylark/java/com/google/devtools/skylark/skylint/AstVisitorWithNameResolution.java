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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.AbstractComprehension;
import com.google.devtools.build.lib.syntax.AugmentedAssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.DotExpression;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.LValue;
import com.google.devtools.build.lib.syntax.ListComprehension;
import com.google.devtools.build.lib.syntax.ListLiteral;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Parameter;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import java.util.Collection;

/**
 * AST visitor that keeps track of which symbols are in scope.
 *
 * <p>The methods {@code enterBlock}, {@code exitBlock}, {@code declare} and {@code reassign} can be
 * overridden by a subclass to handle these "events" during AST traversal.
 */
public class AstVisitorWithNameResolution extends SyntaxTreeVisitor {
  protected Environment env;

  public AstVisitorWithNameResolution() {
    this(Environment.defaultBazel());
  }

  public AstVisitorWithNameResolution(Environment env) {
    this.env = env;
  }

  @Override
  public void visit(BuildFileAST node) {
    enterBlock();
    // First process all global symbols ...
    for (Statement stmt : node.getStatements()) {
      if (stmt instanceof FunctionDefStatement) {
        Identifier fun = ((FunctionDefStatement) stmt).getIdentifier();
        env.addFunction(fun.getName(), fun);
        declare(fun.getName(), fun);
      } else {
        visit(stmt);
      }
    }
    // ... then check the functions
    for (Statement stmt : node.getStatements()) {
      if (stmt instanceof FunctionDefStatement) {
        visit(stmt);
      }
    }
    visitAll(node.getComments());
    Preconditions.checkState(env.inGlobalBlock(), "didn't exit some blocks");
    exitBlock();
  }

  @Override
  public void visit(LoadStatement node) {
    for (LoadStatement.Binding binding : node.getBindings()) {
      String name = binding.getLocalName().getName();
      env.addImported(name, binding.getLocalName());
      declare(name, binding.getLocalName());
    }
  }

  @Override
  public void visit(Identifier identifier) {
    use(identifier);
  }

  @Override
  public void visit(LValue node) {
    initializeOrReassignLValue(node);
    visitLvalue(node.getExpression());
  }

  protected void visitLvalue(Expression expr) {
    if (expr instanceof Identifier) {
      super.visit((Identifier) expr); // don't call this.visit because it doesn't count as usage
    } else if (expr instanceof ListLiteral) {
      for (Expression e : ((ListLiteral) expr).getElements()) {
        visitLvalue(e);
      }
    } else {
      visit(expr);
    }
  }

  @Override
  public void visit(AugmentedAssignmentStatement node) {
    for (Identifier ident : node.getLValue().boundIdentifiers()) {
      use(ident);
    }
    super.visit(node);
  }

  @Override
  public void visit(FunctionDefStatement node) {
    // First visit the default values for parameters in the global environment ...
    for (Parameter<Expression, Expression> param : node.getParameters()) {
      Expression expr = param.getDefaultValue();
      if (expr != null) {
        visit(expr);
      }
    }
    // ... then visit everything else in the local environment
    enterBlock();
    for (Parameter<Expression, Expression> param : node.getParameters()) {
      String name = param.getName();
      if (name != null) {
        env.addParameter(name, param);
        declare(name, param);
      }
    }
    // The function identifier was already added to the globals before, so we skip it
    visitAll(node.getParameters());
    visitBlock(node.getStatements());
    exitBlock();
  }

  @Override
  public void visit(DotExpression node) {
    // Don't visit the identifier field because it's not a variable and would confuse the identifier
    // visitor
    visit(node.getObject());
  }

  @Override
  public void visit(AbstractComprehension node) {
    enterBlock();
    for (ListComprehension.Clause clause : node.getClauses()) {
      visit(clause.getExpression());
      LValue lvalue = clause.getLValue();
      if (lvalue != null) {
        Collection<Identifier> boundIdents = lvalue.boundIdentifiers();
        for (Identifier ident : boundIdents) {
          env.addIdentifier(ident.getName(), ident);
          declare(ident.getName(), ident);
        }
        visit(lvalue);
      }
    }
    visitAll(node.getOutputExpressions());
    exitBlock();
  }

  private void initializeOrReassignLValue(LValue lvalue) {
    Iterable<Identifier> identifiers = lvalue.boundIdentifiers();
    for (Identifier identifier : identifiers) {
      if (env.isDefinedInCurrentScope(identifier.getName())) {
        reassign(identifier);
      } else {
        env.addIdentifier(identifier.getName(), identifier);
        declare(identifier.getName(), identifier);
      }
    }
  }

  /**
   * Invoked when a symbol is defined during AST traversal.
   *
   * <p>This method is there to be overridden in subclasses, it doesn't do anything by itself.
   *
   * @param name name of the variable declared
   * @param node {@code ASTNode} where it was declared
   */
  void declare(String name, ASTNode node) {}

  /**
   * Invoked when a variable is reassigned during AST traversal.
   *
   * <p>This method is there to be overridden in subclasses, it doesn't do anything by itself.
   *
   * @param ident {@code Identifier} that was reassigned
   */
  void reassign(Identifier ident) {}

  /**
   * Invoked when a variable is used during AST traversal.
   *
   * <p>This method is there to be overridden in subclasses, it doesn't do anything by itself.
   *
   * @param ident {@code Identifier} that was reassigned
   */
  void use(Identifier ident) {}

  /**
   * Invoked when a lexical block is entered during AST traversal.
   *
   * <p>This method is there to be overridden in subclasses.
   */
  void enterBlock() {
    env.enterBlock();
  }

  /**
   * Invoked when a lexical block is entered during AST traversal.
   *
   * <p>This method is there to be overridden in subclasses.
   */
  void exitBlock() {
    env.exitBlock();
  }
}
