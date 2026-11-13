// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Resolves an unresolved {@link QueryAst} into an executable {@link QueryExpression} using a set of
 * available {@link QueryFunction} definitions.
 */
public final class QueryExpressionResolver {

  private final Map<String, QueryFunction> functions;
  private final Map<String, Integer> boundVariables = new HashMap<>();

  /** Resolves {@code ast} against the functions provided by {@code env}. */
  public static QueryExpression resolve(QueryAst ast, QueryEnvironment<?> env)
      throws QuerySyntaxException {
    HashMap<String, QueryFunction> functions = new HashMap<>();
    for (QueryFunction queryFunction : env.getFunctions()) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    return resolve(ast, functions);
  }

  /** Resolves {@code ast} against {@code functions}. */
  public static QueryExpression resolve(QueryAst ast, Map<String, QueryFunction> functions)
      throws QuerySyntaxException {
    return new QueryExpressionResolver(functions).resolveExpression(ast);
  }

  private QueryExpressionResolver(Map<String, QueryFunction> functions) {
    this.functions = functions;
  }

  private QueryExpression resolveExpression(QueryAst ast) throws QuerySyntaxException {
    return switch (ast) {
      case QueryAst.Word(String value) -> {
        QueryParser.validateTargetLiteral(value);
        yield new TargetLiteral(value);
      }
      case QueryAst.Parenthesized(QueryAst inner) -> resolveExpression(inner);
      case QueryAst.BinaryOperator(Lexer.TokenKind operator, ImmutableList<QueryAst> operands) -> {
        List<QueryExpression> resolvedOperands = new ArrayList<>(operands.size());
        for (QueryAst operand : operands) {
          resolvedOperands.add(resolveExpression(operand));
        }
        yield new BinaryOperatorExpression(operator, resolvedOperands);
      }
      case QueryAst.Let(String varName, QueryAst varExpr, QueryAst bodyExpr) -> {
        QueryExpression resolvedVarExpr = resolveExpression(varExpr);
        boundVariables.merge(varName, 1, Integer::sum);
        QueryExpression resolvedBodyExpr;
        try {
          resolvedBodyExpr = resolveExpression(bodyExpr);
        } finally {
          boundVariables.compute(varName, (k, v) -> (v == null || v <= 1) ? null : v - 1);
        }
        yield new LetExpression(varName, resolvedVarExpr, resolvedBodyExpr);
      }
      case QueryAst.Set(ImmutableList<QueryAst.Word> words) -> {
        List<TargetLiteral> targetLiterals = new ArrayList<>(words.size());
        for (QueryAst.Word word : words) {
          QueryParser.validateTargetLiteral(word.value());
          targetLiterals.add(new TargetLiteral(word.value()));
        }
        yield new SetExpression(targetLiterals);
      }
      case QueryAst.FunctionCall call -> resolveFunctionCall(call);
    };
  }

  private QueryExpression resolveFunctionCall(QueryAst.FunctionCall call)
      throws QuerySyntaxException {
    QueryFunction function = functions.get(call.functionName());
    if (function == null) {
      throw unknownFunctionError(call.functionName(), call.functionContextForErrorMessages());
    }
    List<Argument> resolvedArgs = new ArrayList<>();
    ImmutableList<QueryAst.FunctionArgument> rawArgs = call.arguments();
    int argsSeen = 0;
    for (ArgumentType type : function.getArgumentTypes()) {
      if (argsSeen >= rawArgs.size()) {
        if (argsSeen >= function.getMandatoryArguments()) {
          break;
        } else {
          throw tooFewArgumentsError(function, call.closeParenContextForErrorMessages());
        }
      }
      QueryAst.FunctionArgument rawArg = rawArgs.get(argsSeen);
      switch (type) {
        case EXPRESSION -> resolvedArgs.add(Argument.of(resolveExpression(rawArg.expr())));
        case WORD -> {
          if (!(rawArg.expr() instanceof QueryAst.Word(String argWord))) {
            throw new QuerySyntaxException(
                "syntax error at '" + rawArg.argContextForErrorMessages() + "'");
          }
          if (LetExpression.isValidVarReference(argWord)) {
            String varName = LetExpression.getNameFromReference(argWord);
            if (boundVariables.getOrDefault(varName, 0) > 0) {
              throw new QuerySyntaxException(
                  String.format(
                      "variable '$%s' cannot be used as an argument to function '%s'; "
                          + "variable references are only allowed in expression arguments",
                      varName, function.getName()));
            }
          }
          resolvedArgs.add(Argument.of(argWord));
        }
        case INTEGER -> {
          if (!(rawArg.expr() instanceof QueryAst.Word(String intString))) {
            throw new QuerySyntaxException(
                "syntax error at '" + rawArg.argContextForErrorMessages() + "'");
          }
          try {
            resolvedArgs.add(Argument.of(Integer.parseInt(intString)));
          } catch (
              @SuppressWarnings("UnusedException")
              NumberFormatException e) {
            throw new QuerySyntaxException("expected an integer literal: '" + intString + "'");
          }
        }
      }
      argsSeen++;
    }
    if (argsSeen < rawArgs.size()) {
      if (argsSeen == 0) {
        throw new QuerySyntaxException(
            "syntax error at '" + rawArgs.get(0).argContextForErrorMessages() + "'");
      }
      throw tooManyArgumentsError(
          function, rawArgs.get(argsSeen).separatorContextForErrorMessages());
    }
    return new FunctionExpression(function, resolvedArgs);
  }

  @CanIgnoreReturnValue
  private QuerySyntaxException unknownFunctionError(String functionName, String context)
      throws QuerySyntaxException {
    StringBuilder buf = new StringBuilder("unknown function '");
    buf.append(functionName);
    buf.append("' at '");
    buf.append(context);
    buf.append("'; expected one of ['");
    buf.append(functions.keySet().stream().sorted().collect(joining("', '")));
    buf.append("']");
    throw new QuerySyntaxException(buf.toString());
  }

  @CanIgnoreReturnValue
  private static QuerySyntaxException functionArgumentCountError(
      QueryFunction function, String description, String context) throws QuerySyntaxException {
    StringBuilder buf = new StringBuilder(description);
    buf.append(" arguments to function '");
    buf.append(function.getName());
    buf.append("' at '");
    buf.append(context);
    buf.append("'");
    throw new QuerySyntaxException(buf.toString());
  }

  @CanIgnoreReturnValue
  private static QuerySyntaxException tooFewArgumentsError(QueryFunction function, String context)
      throws QuerySyntaxException {
    throw functionArgumentCountError(function, "too few", context);
  }

  @CanIgnoreReturnValue
  private static QuerySyntaxException tooManyArgumentsError(QueryFunction function, String context)
      throws QuerySyntaxException {
    throw functionArgumentCountError(function, "too many", context);
  }
}
