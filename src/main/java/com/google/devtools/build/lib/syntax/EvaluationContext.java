// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.syntax.Environment.NoSuchVariableException;
import com.google.devtools.build.lib.vfs.Path;

import java.util.List;

import javax.annotation.Nullable;

/**
 * Context for the evaluation of programs.
 */
public final class EvaluationContext {

  @Nullable private EventHandler eventHandler;
  private Environment env;
  @Nullable private ValidationEnvironment validationEnv;
  private boolean parsePython;

  private EvaluationContext(EventHandler eventHandler, Environment env,
      @Nullable ValidationEnvironment validationEnv, boolean parsePython) {
    this.eventHandler = eventHandler;
    this.env = env;
    this.validationEnv = validationEnv;
    this.parsePython = parsePython;
  }

  /**
   * The fail fast handler, which throws a runtime exception whenever we encounter an error event.
   */
  public static final EventHandler FAIL_FAST_HANDLER = new EventHandler() {
      @Override
      public void handle(Event event) {
        Preconditions.checkArgument(
            !EventKind.ERRORS_AND_WARNINGS.contains(event.getKind()), event);
      }
    };

  public static EvaluationContext newBuildContext(EventHandler eventHandler, Environment env,
      boolean parsePython) {
    return new EvaluationContext(eventHandler, env, null, parsePython);
  }

  public static EvaluationContext newBuildContext(EventHandler eventHandler, Environment env) {
    return newBuildContext(eventHandler, env, false);
  }

  public static EvaluationContext newBuildContext(EventHandler eventHandler) {
    return newBuildContext(eventHandler, new Environment());
  }

  public static EvaluationContext newSkylarkContext(
      Environment env, ValidationEnvironment validationEnv) {
    return new EvaluationContext(env.getEventHandler(), env, validationEnv, false);
  }

  public static EvaluationContext newSkylarkContext(EventHandler eventHandler) {
    return newSkylarkContext(new SkylarkEnvironment(eventHandler), new ValidationEnvironment());
  }

  /** Base context for Skylark evaluation for internal use only, while initializing builtins */
  static final EvaluationContext SKYLARK_INITIALIZATION = newSkylarkContext(FAIL_FAST_HANDLER);

  @VisibleForTesting
  public Environment getEnvironment() {
    return env;
  }

  /** Mock package locator */
  private static final class EmptyPackageLocator implements CachingPackageLocator {
    @Override
    public Path getBuildFileForPackage(PackageIdentifier packageName) {
      return null;
    }
  }

  /** An empty package locator */
  private static final CachingPackageLocator EMPTY_PACKAGE_LOCATOR = new EmptyPackageLocator();

  /** Create a Lexer without a supporting file */
  @VisibleForTesting
  Lexer createLexer(String... input) {
    return new Lexer(ParserInputSource.create(Joiner.on("\n").join(input), null),
        eventHandler);
  }

  /** Is this a Skylark evaluation context? */
  public boolean isSkylark() {
    return env.isSkylark();
  }

  /** Parse a string without a supporting file, returning statements and comments */
  @VisibleForTesting
  Parser.ParseResult parseFileWithComments(String... input) {
    return isSkylark()
        ? Parser.parseFileForSkylark(createLexer(input), eventHandler, null, validationEnv)
        : Parser.parseFile(createLexer(input), eventHandler, EMPTY_PACKAGE_LOCATOR, parsePython);
  }

  /** Parse a string without a supporting file, returning statements only */
  @VisibleForTesting
  List<Statement> parseFile(String... input) {
    return parseFileWithComments(input).statements;
  }

  /** Parse an Expression from string without a supporting file */
  @VisibleForTesting
  Expression parseExpression(String... input) {
    return Parser.parseExpression(createLexer(input), eventHandler);
  }

  /** Evaluate an Expression */
  @VisibleForTesting
  Object evalExpression(Expression expression) throws EvalException, InterruptedException {
    return expression.eval(env);
  }

  /** Evaluate an Expression as parsed from String-s */
  Object evalExpression(String... input) throws EvalException, InterruptedException {
    return evalExpression(parseExpression(input));
  }

  /** Parse a build (not Skylark) Statement from string without a supporting file */
  @VisibleForTesting
  Statement parseStatement(String... input) {
    return Parser.parseStatement(createLexer(input), eventHandler);
  }

  /**
   * Evaluate a Statement
   * @param statement the Statement
   * @return the value of the evaluation, if it's an Expression, or else null
   */
  @Nullable private Object eval(Statement statement) throws EvalException, InterruptedException {
    if (statement instanceof ExpressionStatement) {
      return evalExpression(((ExpressionStatement) statement).getExpression());
    }
    statement.exec(env);
    return null;
  }

  /**
   * Evaluate a list of Statement-s
   * @return the value of the last statement if it's an Expression or else null
   */
  @Nullable private Object eval(List<Statement> statements)
      throws EvalException, InterruptedException {
    Object last = null;
    for (Statement statement : statements) {
      last = eval(statement);
    }
    return last;
  }

  /** Update a variable in the environment, in fluent style */
  public EvaluationContext update(String varname, Object value) throws EvalException {
    env.update(varname, value);
    if (validationEnv != null) {
      validationEnv.declare(varname, null);
    }
    return this;
  }

  /** Lookup a variable in the environment */
  public Object lookup(String varname) throws NoSuchVariableException {
    return env.lookup(varname);
  }

  /** Evaluate a series of statements */
  public Object eval(String... input) throws EvalException, InterruptedException {
    return eval(parseFile(input));
  }
}
