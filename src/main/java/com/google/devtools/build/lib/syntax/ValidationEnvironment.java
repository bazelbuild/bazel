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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A class for doing static checks on files, before evaluating them.
 *
 * <p>We implement the semantics discussed in
 * https://github.com/bazelbuild/proposals/blob/master/docs/2018-06-18-name-resolution.md
 *
 * <p>When a variable is defined, it is visible in the entire block. For example, a global variable
 * is visible in the entire file; a variable in a function is visible in the entire function block
 * (even on the lines before its first assignment).
 */
public final class ValidationEnvironment extends SyntaxTreeVisitor {

  enum Scope {
    /** Symbols defined inside a function or a comprehension. */
    Local("local"),
    /** Symbols defined at a module top-level, e.g. functions, loaded symbols. */
    Module("global"),
    /** Predefined symbols (builtins) */
    Universe("builtin");

    private final String qualifier;

    private Scope(String qualifier) {
      this.qualifier = qualifier;
    }

    public String getQualifier() {
      return qualifier;
    }
  }

  private static class Block {
    private final Set<String> variables = new HashSet<>();
    private final Scope scope;
    @Nullable private final Block parent;

    Block(Scope scope, @Nullable Block parent) {
      this.scope = scope;
      this.parent = parent;
    }
  }

  /**
   * We use an unchecked exception around EvalException because the SyntaxTreeVisitor doesn't let
   * visit methods throw checked exceptions. We might change that later.
   */
  private static class ValidationException extends RuntimeException {
    EvalException exception;

    ValidationException(EvalException e) {
      exception = e;
    }

    ValidationException(Location location, String message, String url) {
      exception = new EvalException(location, message, url);
    }

    ValidationException(Location location, String message) {
      exception = new EvalException(location, message);
    }
  }

  private final Environment env;
  private Block block;
  private int loopCount;
  /** In BUILD files, we have a slightly different behavior for legacy reasons. */
  private final boolean isBuildFile;

  /** Create a ValidationEnvironment for a given global Environment (containing builtins). */
  private ValidationEnvironment(Environment env, boolean isBuildFile) {
    Preconditions.checkArgument(env.isGlobal());
    this.env = env;
    this.isBuildFile = isBuildFile;
    block = new Block(Scope.Universe, null);
    Set<String> builtinVariables = env.getVariableNames();
    block.variables.addAll(builtinVariables);
  }

  /**
   * First pass: add all definitions to the current block. This is done because symbols are
   * sometimes used before their definition point (e.g. a functions are not necessarily declared in
   * order).
   */
  private void collectDefinitions(Iterable<Statement> stmts) {
    for (Statement stmt : stmts) {
      collectDefinitions(stmt);
    }
  }

  private void collectDefinitions(Statement stmt) {
    switch (stmt.kind()) {
      case ASSIGNMENT:
        collectDefinitions(((AssignmentStatement) stmt).getLValue());
        break;
      case AUGMENTED_ASSIGNMENT:
        collectDefinitions(((AugmentedAssignmentStatement) stmt).getLValue());
        break;
      case IF:
        IfStatement ifStmt = (IfStatement) stmt;
        for (IfStatement.ConditionalStatements cond : ifStmt.getThenBlocks()) {
          collectDefinitions(cond.getStatements());
        }
        collectDefinitions(ifStmt.getElseBlock());
        break;
      case FOR:
        ForStatement forStmt = (ForStatement) stmt;
        collectDefinitions(forStmt.getVariable());
        collectDefinitions(forStmt.getBlock());
        break;
      case FUNCTION_DEF:
        Identifier fctName = ((FunctionDefStatement) stmt).getIdentifier();
        declare(fctName.getName(), fctName.getLocation());
        break;
      case LOAD:
        for (LoadStatement.Binding binding : ((LoadStatement) stmt).getBindings()) {
          declare(binding.getLocalName().getName(), binding.getLocalName().getLocation());
        }
        break;
      case CONDITIONAL:
      case EXPRESSION:
      case FLOW:
      case PASS:
      case RETURN:
        // nothing to declare
    }
  }

  private void collectDefinitions(LValue left) {
    for (Identifier id : left.boundIdentifiers()) {
      declare(id.getName(), id.getLocation());
    }
  }

  private void validateLValue(Location loc, Expression expr) {
    if (expr instanceof IndexExpression) {
      visit(expr);
    } else if (expr instanceof ListLiteral) {
      for (Expression e : ((ListLiteral) expr).getElements()) {
        validateLValue(loc, e);
      }
    } else if (!(expr instanceof Identifier)) {
      throw new ValidationException(loc, "cannot assign to '" + expr + "'");
    }
  }

  @Override
  public void visit(Identifier node) {
    @Nullable Block b = blockThatDefines(node.getName());
    if (b == null) {
      // The identifier might not exist because it was restricted (hidden) by the current semantics.
      // If this is the case, output a more helpful error message than 'not found'.
      FlagGuardedValue result = env.getRestrictedBindings().get(node.getName());
      if (result != null) {
        throw new ValidationException(
            result.getEvalExceptionFromAttemptingAccess(
                node.getLocation(), env.getSemantics(), node.getName()));
      }
      throw new ValidationException(node.createInvalidIdentifierException(getAllSymbols()));
    }
    // TODO(laurentlb): In BUILD files, calling setScope will throw an exception. This happens
    // because some AST nodes are shared across multipe ASTs (due to the prelude file).
    if (!isBuildFile) {
      node.setScope(b.scope);
    }
  }

  @Override
  public void visit(LValue node) {
    validateLValue(node.getLocation(), node.getExpression());
  }

  @Override
  public void visit(ReturnStatement node) {
    if (block.scope != Scope.Local) {
      throw new ValidationException(
          node.getLocation(), "return statements must be inside a function");
    }
    super.visit(node);
  }

  @Override
  public void visit(ForStatement node) {
    loopCount++;
    super.visit(node);
    Preconditions.checkState(loopCount > 0);
    loopCount--;
  }

  @Override
  public void visit(FlowStatement node) {
    if (loopCount <= 0) {
      throw new ValidationException(
          node.getLocation(), node.getKind().getName() + " statement must be inside a for loop");
    }
    super.visit(node);
  }

  @Override
  public void visit(DotExpression node) {
    visit(node.getObject());
    // Do not visit the field.
  }

  @Override
  public void visit(AbstractComprehension node) {
    openBlock(Scope.Local);
    for (AbstractComprehension.Clause clause : node.getClauses()) {
      if (clause.getLValue() != null) {
        collectDefinitions(clause.getLValue());
      }
    }
    super.visit(node);
    closeBlock();
  }

  @Override
  public void visit(FunctionDefStatement node) {
    for (Parameter<Expression, Expression> param : node.getParameters()) {
      if (param.isOptional()) {
        visit(param.getDefaultValue());
      }
    }
    openBlock(Scope.Local);
    for (Parameter<Expression, Expression> param : node.getParameters()) {
      if (param.hasName()) {
        declare(param.getName(), param.getLocation());
      }
    }
    collectDefinitions(node.getStatements());
    visitAll(node.getStatements());
    closeBlock();
  }

  @Override
  public void visit(IfStatement node) {
    if (block.scope != Scope.Local) {
      throw new ValidationException(
          node.getLocation(),
          "if statements are not allowed at the top level. You may move it inside a function "
              + "or use an if expression (x if condition else y).");
    }
    super.visit(node);
  }

  @Override
  public void visit(AugmentedAssignmentStatement node) {
    if (node.getLValue().getExpression() instanceof ListLiteral) {
      throw new ValidationException(
          node.getLocation(), "cannot perform augmented assignment on a list or tuple expression");
    }
    // Other bad cases are handled when visiting the LValue node.
    super.visit(node);
  }

  /** Declare a variable and add it to the environment. */
  private void declare(String varname, Location location) {
    // TODO(laurentlb): Forbid reassignment in BUILD files.
    if (!isBuildFile && block.scope == Scope.Module && block.variables.contains(varname)) {
      // Symbols defined in the module scope cannot be reassigned.
      throw new ValidationException(
          location,
          String.format("Variable %s is read only", varname),
          "https://bazel.build/versions/master/docs/skylark/errors/read-only-variable.html");
    }
    block.variables.add(varname);
  }

  /** Returns the nearest Block that defines a symbol. */
  private Block blockThatDefines(String varname) {
    for (Block b = block; b != null; b = b.parent) {
      if (b.variables.contains(varname)) {
        return b;
      }
    }
    return null;
  }

  /** Returns the set of all accessible symbols (both local and global) */
  private Set<String> getAllSymbols() {
    Set<String> all = new HashSet<>();
    for (Block b = block; b != null; b = b.parent) {
      all.addAll(b.variables);
    }
    return all;
  }

  /** Throws ValidationException if a load() appears after another kind of statement. */
  private static void checkLoadAfterStatement(List<Statement> statements) {
    Location firstStatement = null;

    for (Statement statement : statements) {
      // Ignore string literals (e.g. docstrings).
      if (statement instanceof ExpressionStatement
          && ((ExpressionStatement) statement).getExpression() instanceof StringLiteral) {
        continue;
      }

      if (statement instanceof LoadStatement) {
        if (firstStatement == null) {
          continue;
        }
        throw new ValidationException(
            statement.getLocation(),
            "load() statements must be called before any other statement. "
                + "First non-load() statement appears at "
                + firstStatement
                + ". Use --incompatible_bzl_disallow_load_after_statement=false to temporarily "
                + "disable this check.");
      }

      if (firstStatement == null) {
        firstStatement = statement.getLocation();
      }
    }
  }

  /** Validates the AST and runs static checks. */
  private void validateAst(List<Statement> statements) {
    // Check that load() statements are on top.
    if (!isBuildFile && env.getSemantics().incompatibleBzlDisallowLoadAfterStatement()) {
      checkLoadAfterStatement(statements);
    }

    openBlock(Scope.Module);

    // Add each variable defined by statements, not including definitions that appear in
    // sub-scopes of the given statements (function bodies and comprehensions).
    collectDefinitions(statements);

    // Second pass: ensure that all symbols have been defined.
    visitAll(statements);
    closeBlock();
  }

  public static void validateAst(Environment env, List<Statement> statements) throws EvalException {
    try {
      ValidationEnvironment venv = new ValidationEnvironment(env, false);
      venv.validateAst(statements);
      // Check that no closeBlock was forgotten.
      Preconditions.checkState(venv.block.parent == null);
    } catch (ValidationException e) {
      throw e.exception;
    }
  }

  public static boolean validateAst(
      Environment env, List<Statement> statements, EventHandler eventHandler) {
    try {
      validateAst(env, statements);
      return true;
    } catch (EvalException e) {
      if (!e.isDueToIncompleteAST()) {
        eventHandler.handle(Event.error(e.getLocation(), e.getMessage()));
      }
      return false;
    }
  }

  /** Open a new lexical block that will contain the future declarations. */
  private void openBlock(Scope scope) {
    block = new Block(scope, block);
  }

  /** Close a lexical block (and lose all declarations it contained). */
  private void closeBlock() {
    block = Preconditions.checkNotNull(block.parent);
  }

  /**
   * Checks that the AST is using the restricted syntax.
   *
   * <p>Restricted syntax is used by Bazel BUILD files. It forbids function definitions, *args, and
   * **kwargs. This creates a better separation between code and data.
   */
  public static boolean checkBuildSyntax(
      List<Statement> statements, final EventHandler eventHandler, Environment env) {
    // Wrap the boolean inside an array so that the inner class can modify it.
    final boolean[] success = new boolean[] {true};

    if (env.getSemantics().incompatibleStaticNameResolutionInBuildFiles()) {
      ValidationEnvironment venv = new ValidationEnvironment(env, true);
      try {
        venv.validateAst(statements);
      } catch (ValidationException e) {
        eventHandler.handle(Event.error(e.exception.getLocation(), e.exception.getMessage()));
        return false;
      }
    }

    // TODO(laurentlb): Merge with the visitor above when possible (i.e. when BUILD files use it).
    SyntaxTreeVisitor checker =
        new SyntaxTreeVisitor() {

          private void error(ASTNode node, String message) {
            eventHandler.handle(Event.error(node.getLocation(), message));
            success[0] = false;
          }

          @Override
          public void visit(FunctionDefStatement node) {
            error(
                node,
                "function definitions are not allowed in BUILD files. You may move the function to "
                    + "a .bzl file and load it.");
          }

          @Override
          public void visit(ForStatement node) {
            error(
                node,
                "for statements are not allowed in BUILD files. You may inline the loop, move it "
                    + "to a function definition (in a .bzl file), or as a last resort use a list "
                    + "comprehension.");
          }

          @Override
          public void visit(IfStatement node) {
            error(
                node,
                "if statements are not allowed in BUILD files. You may move conditional logic to a "
                    + "function definition (in a .bzl file), or for simple cases use an if "
                    + "expression.");
          }

          @Override
          public void visit(FuncallExpression node) {
            for (Argument.Passed arg : node.getArguments()) {
              if (arg.isStarStar()) {
                error(
                    node,
                    "**kwargs arguments are not allowed in BUILD files. Pass the arguments in "
                        + "explicitly.");
              } else if (arg.isStar()) {
                error(
                    node,
                    "*args arguments are not allowed in BUILD files. Pass the arguments in "
                        + "explicitly.");
              }
            }
            if (env.getSemantics().incompatibleNoKwargsInBuildFiles()) {
              super.visit(node);
            }
          }
        };
    checker.visitAll(statements);
    return success[0];
  }
}
