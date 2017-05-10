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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

/**
 * An Environment for the semantic checking of Skylark files.
 *
 * @see Statement#validate
 * @see Expression#validate
 */
public final class ValidationEnvironment {

  private final ValidationEnvironment parent;

  private final Set<String> variables = new HashSet<>();

  private final Map<String, Location> variableLocations = new HashMap<>();

  private final Set<String> readOnlyVariables = new HashSet<>();

  private final SkylarkSemanticsOptions semantics;

  // A stack of variable-sets which are read only but can be assigned in different
  // branches of if-else statements.
  private final Stack<Set<String>> futureReadOnlyVariables = new Stack<>();

  /**
   * Create a ValidationEnvironment for a given global Environment.
   */
  public ValidationEnvironment(Environment env) {
    Preconditions.checkArgument(env.isGlobal());
    parent = null;
    Set<String> builtinVariables = env.getVariableNames();
    variables.addAll(builtinVariables);
    readOnlyVariables.addAll(builtinVariables);
    semantics = env.getSemantics();
  }

  /**
   * Creates a local ValidationEnvironment to validate user defined function bodies.
   */
  public ValidationEnvironment(ValidationEnvironment parent) {
    // Don't copy readOnlyVariables: Variables may shadow global values.
    this.parent = parent;
    semantics = parent.semantics;
  }

  /**
   * Returns true if this ValidationEnvironment is top level i.e. has no parent.
   */
  public boolean isTopLevel() {
    return parent == null;
  }

  /**
   * Declare a variable and add it to the environment.
   */
  public void declare(String varname, Location location)
      throws EvalException {
    checkReadonly(varname, location);
    if (parent == null) {  // top-level values are immutable
      readOnlyVariables.add(varname);
      if (!futureReadOnlyVariables.isEmpty()) {
        // Currently validating an if-else statement
        futureReadOnlyVariables.peek().add(varname);
      }
    }
    variables.add(varname);
    variableLocations.put(varname, location);
  }

  private void checkReadonly(String varname, Location location) throws EvalException {
    if (readOnlyVariables.contains(varname)) {
      throw new EvalException(
          location,
          String.format("Variable %s is read only", varname),
          "https://bazel.build/versions/master/docs/skylark/errors/read-only-variable.html");
    }
  }

  /**
   * Returns true if the symbol exists in the validation environment.
   */
  public boolean hasSymbolInEnvironment(String varname) {
    return variables.contains(varname)
        || (parent != null && topLevel().variables.contains(varname));
  }

  /** Returns the set of all accessible symbols (both local and global) */
  public Set<String> getAllSymbols() {
    Set<String> all = new HashSet<>();
    all.addAll(variables);
    if (parent != null) {
      all.addAll(parent.getAllSymbols());
    }
    return all;
  }

  private ValidationEnvironment topLevel() {
    return Preconditions.checkNotNull(parent == null ? this : parent);
  }

  /**
   * Starts a session with temporarily disabled readonly checking for variables between branches.
   * This is useful to validate control flows like if-else when we know that certain parts of the
   * code cannot both be executed.
   */
  public void startTemporarilyDisableReadonlyCheckSession() {
    futureReadOnlyVariables.add(new HashSet<String>());
  }

  /**
   * Finishes the session with temporarily disabled readonly checking.
   */
  public void finishTemporarilyDisableReadonlyCheckSession() {
    Set<String> variables = futureReadOnlyVariables.pop();
    readOnlyVariables.addAll(variables);
    if (!futureReadOnlyVariables.isEmpty()) {
      futureReadOnlyVariables.peek().addAll(variables);
    }
  }

  /**
   * Finishes a branch of temporarily disabled readonly checking.
   */
  public void finishTemporarilyDisableReadonlyCheckBranch() {
    readOnlyVariables.removeAll(futureReadOnlyVariables.peek());
  }

  /** Throws EvalException if a load() appears after another kind of statement. */
  private void checkLoadAfterStatement(List<Statement> statements) throws EvalException {
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
        throw new EvalException(
            statement.getLocation(),
            "load() statements must be called before any other statement. "
                + "First non-load() statement appears at "
                + firstStatement
                + ". Use --incompatible_bzl_disallow_load_after_statement to temporarily disable "
                + "this check.");
      }

      if (firstStatement == null) {
        firstStatement = statement.getLocation();
      }
    }
  }

  /**
   * Validates the AST and runs static checks.
   */
  public void validateAst(List<Statement> statements) throws EvalException {
    // Check that load() statements are on top.
    if (semantics.incompatibleBzlDisallowLoadAfterStatement) {
      checkLoadAfterStatement(statements);
    }

    // Add every function in the environment before validating. This is
    // necessary because functions may call other functions defined
    // later in the file.
    for (Statement statement : statements) {
      if (statement instanceof FunctionDefStatement) {
        FunctionDefStatement fct = (FunctionDefStatement) statement;
        declare(fct.getIdent().getName(), fct.getLocation());
      }
    }

    for (Statement statement : statements) {
      statement.validate(this);
    }
  }

  public boolean validateAst(List<Statement> statements, EventHandler eventHandler) {
    try {
      validateAst(statements);
      return true;
    } catch (EvalException e) {
      if (!e.isDueToIncompleteAST()) {
        eventHandler.handle(Event.error(e.getLocation(), e.getMessage()));
      }
      return false;
    }
  }
}
