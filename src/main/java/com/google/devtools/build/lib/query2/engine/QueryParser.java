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
package com.google.devtools.build.lib.query2.engine;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.query2.engine.Lexer.BINARY_OPERATORS;
import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.auto.value.AutoBuilder;
import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.Lexer.TokenKind;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;

/**
 * LL(1) recursive descent parser for the Blaze query language, revision 2.
 *
 * <p>In the grammar below, non-terminals are lowercase and terminals are uppercase, or character
 * literals.
 *
 * <pre>
 * expr ::= WORD
 *        | PLACEHOLDER
 *        | LET WORD = expr IN expr
 *        | '(' expr ')'
 *        | WORD '(' ( expr ( ',' expr ) * ) ? ')'
 *        | expr INTERSECT expr
 *        | expr '^' expr
 *        | expr UNION expr
 *        | expr '+' expr
 *        | expr EXCEPT expr
 *        | expr '-' expr
 *        | SET '(' WORD * ')'
 * </pre>
 */
public final class QueryParser {

  private static final int MAX_QUERY_EXPRESSION_LOG_CHARS = 1000;

  /**
   * Configuration options for {@link QueryParser}.
   *
   * @param allowPlaceholders whether unquoted {@code @(name)} tokens are parsed as {@link
   *     QueryAst.Placeholder} nodes
   * @param maxQueryDepth maximum allowed expression nesting depth. Unnested leaf tokens ({@code
   *     WORD} and {@code PLACEHOLDER}) have depth {@code 0}; each enclosing parenthesized
   *     expression, function call, {@code set(...)} expression, {@code let} binding, or distinct
   *     binary-operator group adds {@code 1} to the nesting depth.
   */
  public record ParserOptions(boolean allowPlaceholders, int maxQueryDepth) {
    /**
     * Default parser options ({@code allowPlaceholders = false}, {@code maxQueryDepth =
     * Integer.MAX_VALUE}).
     */
    public static final ParserOptions DEFAULT = builder().build();

    public ParserOptions {
      checkArgument(maxQueryDepth >= 0, "maxQueryDepth must be non-negative: %s", maxQueryDepth);
    }

    public static Builder builder() {
      return new AutoBuilder_QueryParser_ParserOptions_Builder()
          .setAllowPlaceholders(false)
          .setMaxQueryDepth(Integer.MAX_VALUE);
    }

    public Builder toBuilder() {
      return builder().setAllowPlaceholders(allowPlaceholders).setMaxQueryDepth(maxQueryDepth);
    }

    /** Builder for {@link ParserOptions}. */
    @AutoBuilder
    public abstract static class Builder {
      public abstract Builder setAllowPlaceholders(boolean allowPlaceholders);

      public abstract Builder setMaxQueryDepth(int maxQueryDepth);

      public abstract ParserOptions build();
    }
  }

  private record ParsedNode(QueryAst ast, int subtreeDepth) {}

  private Lexer.Token token; // current lookahead token
  private final List<Lexer.Token> tokens;
  private final ParserOptions options;
  private int tokenIndex = -1;

  /**
   * Scans and parses the specified query expression into an unresolved {@link QueryAst} using the
   * default {@link ParserOptions}.
   *
   * @throws QuerySyntaxException if the query has a syntax error.
   */
  public static QueryAst parse(String query) throws QuerySyntaxException {
    return parse(query, ParserOptions.DEFAULT);
  }

  /**
   * Scans and parses the specified query expression into an unresolved {@link QueryAst} using the
   * provided {@link ParserOptions}.
   */
  public static QueryAst parse(String query, ParserOptions options) throws QuerySyntaxException {
    checkNotNull(query);
    checkNotNull(options);
    List<Lexer.Token> tokens = Lexer.scan(query, options.allowPlaceholders());
    QueryParser parser = new QueryParser(tokens, options);
    QueryAst ast = parser.parseExpression(/* currentDepth= */ 0).ast();
    if (parser.token.kind != TokenKind.EOF) {
      String parsedPrefix =
          Ascii.truncate(
              query.substring(
                  tokens.get(0).startOffset, tokens.get(parser.tokenIndex - 1).endOffset),
              MAX_QUERY_EXPRESSION_LOG_CHARS,
              "[truncated]");
      throw new QuerySyntaxException(
          String.format(
              "unexpected token '%s' after query expression '%s'", parser.token, parsedPrefix));
    }
    return ast;
  }

  private QueryParser(List<Lexer.Token> tokens, ParserOptions options) {
    this.tokens = tokens;
    this.options = options;
    nextToken();
  }

  @CanIgnoreReturnValue
  private int incrementDepth(int currentDepth) throws QuerySyntaxException {
    if (currentDepth >= options.maxQueryDepth()) {
      throw maxDepthExceeded();
    }
    return currentDepth + 1;
  }

  private void checkSubtreeDepth(int ambientDepth, int subtreeDepth) throws QuerySyntaxException {
    if (subtreeDepth > options.maxQueryDepth() - ambientDepth) {
      throw maxDepthExceeded();
    }
  }

  private QuerySyntaxException maxDepthExceeded() {
    return new QuerySyntaxException(
        "query exceeds maximum nesting depth of " + options.maxQueryDepth());
  }

  /** Throws a syntax error exception. */
  @CanIgnoreReturnValue
  private QuerySyntaxException syntaxError(Lexer.Token token) throws QuerySyntaxException {
    String message = "premature end of input";
    if (token.kind != TokenKind.EOF) {
      message = "syntax error at '" + inputContextAt(tokenIndex) + "'";
    }
    throw new QuerySyntaxException(message);
  }

  private String inputContextAt(int startTokenIndex) {
    StringBuilder buf = new StringBuilder();
    String sep = "";
    for (int index = startTokenIndex,
            max = min(tokens.size() - 1, startTokenIndex + 3); // 3 tokens of context
        index < max;
        ++index) {
      buf.append(sep).append(tokens.get(index));
      sep = " ";
    }
    return buf.toString();
  }

  /**
   * Consumes the current token. If it is not of the specified (expected) kind, throws {@link
   * QuerySyntaxException}. Returns the value associated with the consumed token, if any.
   */
  @CanIgnoreReturnValue
  private String consume(TokenKind kind) throws QuerySyntaxException {
    if (token.kind != kind) {
      throw syntaxError(token);
    }
    String word = token.word;
    nextToken();
    return word;
  }

  private void nextToken() {
    if (token == null || token.kind != TokenKind.EOF) {
      tokenIndex++;
      token = tokens.get(tokenIndex);
    }
  }

  /**
   * <pre>
   * expr ::= primary
   *        | expr INTERSECT expr
   *        | expr '^' expr
   *        | expr UNION expr
   *        | expr '+' expr
   *        | expr EXCEPT expr
   *        | expr '-' expr
   * </pre>
   */
  private ParsedNode parseExpression(int currentDepth) throws QuerySyntaxException {
    // All operators are left-associative and of equal precedence.
    return parseBinaryOperatorTail(parsePrimary(currentDepth), currentDepth);
  }

  /**
   * <pre>
   * tail ::= ( <op> <primary> )*
   * </pre>
   *
   * <p>All operators have equal precedence. This factoring is required for left-associative binary
   * operators in LL(1).
   */
  private ParsedNode parseBinaryOperatorTail(ParsedNode lhs, int currentDepth)
      throws QuerySyntaxException {
    if (!BINARY_OPERATORS.contains(token.kind)) {
      return lhs;
    }

    int operandDepth = incrementDepth(currentDepth);
    int groupMaxOperandDepth = lhs.subtreeDepth();
    checkSubtreeDepth(operandDepth, groupMaxOperandDepth);

    List<QueryAst> operands = new ArrayList<>();
    operands.add(lhs.ast());
    TokenKind lastOperator = token.kind;

    while (BINARY_OPERATORS.contains(token.kind)) {
      TokenKind operator = token.kind;
      consume(operator);
      if (operator != lastOperator) {
        lhs =
            new ParsedNode(
                new QueryAst.BinaryOperator(lastOperator, operands),
                /* subtreeDepth= */ 1 + groupMaxOperandDepth);
        operands.clear();
        operands.add(lhs.ast());
        lastOperator = operator;
        groupMaxOperandDepth = lhs.subtreeDepth();
        checkSubtreeDepth(operandDepth, groupMaxOperandDepth);
      }
      ParsedNode rhs = parsePrimary(operandDepth);
      operands.add(rhs.ast());
      groupMaxOperandDepth = max(groupMaxOperandDepth, rhs.subtreeDepth());
    }
    return new ParsedNode(
        new QueryAst.BinaryOperator(lastOperator, operands),
        /* subtreeDepth= */ 1 + groupMaxOperandDepth);
  }

  /**
   * <pre>
   * primary ::= WORD
   *           | WORD '(' ( arg ( ',' arg ) * ) ? ')'
   *           | PLACEHOLDER
   *           | LET WORD = expr IN expr
   *           | '(' expr ')'
   *           | SET '(' WORD * ')'
   * </pre>
   */
  private ParsedNode parsePrimary(int currentDepth) throws QuerySyntaxException {
    return switch (token.kind) {
      case WORD -> {
        int wordTokenIndex = tokenIndex;
        String word = consume(TokenKind.WORD);
        if (token.kind == TokenKind.LPAREN) {
          yield parseFunctionCallRest(word, wordTokenIndex, currentDepth);
        }
        validateTargetLiteral(word);
        yield new ParsedNode(new QueryAst.Word(word), /* subtreeDepth= */ 0);
      }
      case PLACEHOLDER -> {
        Lexer.Token placeholderToken = token;
        nextToken();
        yield new ParsedNode(
            new QueryAst.Placeholder(
                placeholderToken.word, placeholderToken.startOffset, placeholderToken.endOffset),
            /* subtreeDepth= */ 0);
      }
      case LET -> {
        consume(TokenKind.LET);
        String name = consume(TokenKind.WORD);
        consume(TokenKind.EQUALS);
        int innerDepth = incrementDepth(currentDepth);
        ParsedNode varExpr = parseExpression(innerDepth);
        consume(TokenKind.IN);
        ParsedNode bodyExpr = parseExpression(innerDepth);
        yield new ParsedNode(
            new QueryAst.Let(name, varExpr.ast(), bodyExpr.ast()),
            /* subtreeDepth= */ 1 + max(varExpr.subtreeDepth(), bodyExpr.subtreeDepth()));
      }
      case LPAREN -> {
        consume(TokenKind.LPAREN);
        int innerDepth = incrementDepth(currentDepth);
        ParsedNode expr = parseExpression(innerDepth);
        consume(TokenKind.RPAREN);
        yield new ParsedNode(
            new QueryAst.Parenthesized(expr.ast()), /* subtreeDepth= */ 1 + expr.subtreeDepth());
      }
      case SET -> {
        nextToken();
        consume(TokenKind.LPAREN);
        incrementDepth(currentDepth);
        List<QueryAst.Word> words = new ArrayList<>();
        while (token.kind == TokenKind.WORD) {
          String word = consume(TokenKind.WORD);
          validateTargetLiteral(word);
          words.add(new QueryAst.Word(word));
        }
        consume(TokenKind.RPAREN);
        yield new ParsedNode(new QueryAst.Set(words), /* subtreeDepth= */ 1);
      }
      default -> throw syntaxError(token);
    };
  }

  private ParsedNode parseFunctionCallRest(
      String functionName, int functionTokenIndex, int currentDepth) throws QuerySyntaxException {
    String functionContextForErrorMessages = inputContextAt(functionTokenIndex);
    int lparenIndex = tokenIndex;
    consume(TokenKind.LPAREN);
    int argDepth = incrementDepth(currentDepth);
    List<QueryAst.FunctionArgument> args = new ArrayList<>();
    int maxArgSubtreeDepth = 0;
    if (token.kind != TokenKind.RPAREN) {
      int sepIndex = lparenIndex;
      while (true) {
        String separatorContextForErrorMessages = inputContextAt(sepIndex);
        String argContextForErrorMessages = inputContextAt(tokenIndex);
        ParsedNode argExpr = parseFunctionArgumentExpression(argDepth);
        maxArgSubtreeDepth = max(maxArgSubtreeDepth, argExpr.subtreeDepth());
        args.add(
            new QueryAst.FunctionArgument(
                argExpr.ast(), separatorContextForErrorMessages, argContextForErrorMessages));
        if (token.kind != TokenKind.COMMA) {
          break;
        }
        sepIndex = tokenIndex;
        consume(TokenKind.COMMA);
      }
    }
    String closeParenContextForErrorMessages = inputContextAt(tokenIndex);
    consume(TokenKind.RPAREN);
    return new ParsedNode(
        new QueryAst.FunctionCall(
            functionName,
            ImmutableList.copyOf(args),
            functionContextForErrorMessages,
            closeParenContextForErrorMessages),
        /* subtreeDepth= */ 1 + maxArgSubtreeDepth);
  }

  /**
   * Parses an argument inside a function call. A standalone {@code WORD} argument is not checked
   * against {@link #validateTargetLiteral} here because the function's parameter may have type
   * {@code WORD} or {@code INTEGER} rather than {@code EXPRESSION}.
   */
  private ParsedNode parseFunctionArgumentExpression(int currentDepth) throws QuerySyntaxException {
    if (token.kind == TokenKind.WORD) {
      int wordTokenIndex = tokenIndex;
      String word = consume(TokenKind.WORD);
      if (token.kind == TokenKind.LPAREN) {
        return parseBinaryOperatorTail(
            parseFunctionCallRest(word, wordTokenIndex, currentDepth), currentDepth);
      }
      if (!BINARY_OPERATORS.contains(token.kind)) {
        return new ParsedNode(new QueryAst.Word(word), /* subtreeDepth= */ 0);
      }
      validateTargetLiteral(word);
      return parseBinaryOperatorTail(
          new ParsedNode(new QueryAst.Word(word), /* subtreeDepth= */ 0), currentDepth);
    }
    return parseExpression(currentDepth);
  }

  /**
   * Unquoted words may not start with a hyphen or asterisk, even though relative target names may
   * start with those characters.
   */
  static void validateTargetLiteral(String word) throws QuerySyntaxException {
    if (word.startsWith("-") || word.startsWith("*")) {
      throw new QuerySyntaxException(
          "target literal must not begin with (" + word.charAt(0) + "): " + word);
    }
  }
}
