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

import static com.google.devtools.build.lib.query2.engine.Lexer.BINARY_OPERATORS;
import static java.lang.Math.min;

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

  private Lexer.Token token; // current lookahead token
  private final List<Lexer.Token> tokens;
  private int tokenIndex = -1;

  /** Scans and parses the specified query expression into an unresolved {@link QueryAst}. */
  public static QueryAst parse(String query) throws QuerySyntaxException {
    List<Lexer.Token> tokens = Lexer.scan(query);
    QueryParser parser = new QueryParser(tokens);
    QueryAst ast = parser.parseExpression();
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

  QueryParser(List<Lexer.Token> tokens) {
    this.tokens = tokens;
    nextToken();
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
  private QueryAst parseExpression() throws QuerySyntaxException {
    // All operators are left-associative and of equal precedence.
    return parseBinaryOperatorTail(parsePrimary());
  }

  /**
   * <pre>
   * tail ::= ( <op> <primary> )*
   * </pre>
   *
   * <p>All operators have equal precedence. This factoring is required for left-associative binary
   * operators in LL(1).
   */
  private QueryAst parseBinaryOperatorTail(QueryAst lhs) throws QuerySyntaxException {
    if (!BINARY_OPERATORS.contains(token.kind)) {
      return lhs;
    }

    List<QueryAst> operands = new ArrayList<>();
    operands.add(lhs);
    TokenKind lastOperator = token.kind;

    while (BINARY_OPERATORS.contains(token.kind)) {
      TokenKind operator = token.kind;
      consume(operator);
      if (operator != lastOperator) {
        lhs = new QueryAst.BinaryOperator(lastOperator, operands);
        operands.clear();
        operands.add(lhs);
        lastOperator = operator;
      }
      QueryAst rhs = parsePrimary();
      operands.add(rhs);
    }
    return new QueryAst.BinaryOperator(lastOperator, operands);
  }

  /**
   * <pre>
   * primary ::= WORD
   *           | WORD '(' ( arg ( ',' arg ) * ) ? ')'
   *           | LET WORD = expr IN expr
   *           | '(' expr ')'
   *           | SET '(' WORD * ')'
   * </pre>
   */
  private QueryAst parsePrimary() throws QuerySyntaxException {
    return switch (token.kind) {
      case WORD -> {
        int wordTokenIndex = tokenIndex;
        String word = consume(TokenKind.WORD);
        if (token.kind == TokenKind.LPAREN) {
          yield parseFunctionCallRest(word, wordTokenIndex);
        }
        validateTargetLiteral(word);
        yield new QueryAst.Word(word);
      }
      case LET -> {
        consume(TokenKind.LET);
        String name = consume(TokenKind.WORD);
        consume(TokenKind.EQUALS);
        QueryAst varExpr = parseExpression();
        consume(TokenKind.IN);
        QueryAst bodyExpr = parseExpression();
        yield new QueryAst.Let(name, varExpr, bodyExpr);
      }
      case LPAREN -> {
        consume(TokenKind.LPAREN);
        QueryAst expr = parseExpression();
        consume(TokenKind.RPAREN);
        yield new QueryAst.Parenthesized(expr);
      }
      case SET -> {
        nextToken();
        consume(TokenKind.LPAREN);
        List<QueryAst.Word> words = new ArrayList<>();
        while (token.kind == TokenKind.WORD) {
          String word = consume(TokenKind.WORD);
          validateTargetLiteral(word);
          words.add(new QueryAst.Word(word));
        }
        consume(TokenKind.RPAREN);
        yield new QueryAst.Set(words);
      }
      default -> throw syntaxError(token);
    };
  }

  private QueryAst.FunctionCall parseFunctionCallRest(String functionName, int functionTokenIndex)
      throws QuerySyntaxException {
    String functionContextForErrorMessages = inputContextAt(functionTokenIndex);
    int lparenIndex = tokenIndex;
    consume(TokenKind.LPAREN);
    List<QueryAst.FunctionArgument> args = new ArrayList<>();
    if (token.kind != TokenKind.RPAREN) {
      int sepIndex = lparenIndex;
      while (true) {
        String separatorContextForErrorMessages = inputContextAt(sepIndex);
        String argContextForErrorMessages = inputContextAt(tokenIndex);
        QueryAst argExpr = parseFunctionArgumentExpression();
        args.add(
            new QueryAst.FunctionArgument(
                argExpr, separatorContextForErrorMessages, argContextForErrorMessages));
        if (token.kind != TokenKind.COMMA) {
          break;
        }
        sepIndex = tokenIndex;
        consume(TokenKind.COMMA);
      }
    }
    String closeParenContextForErrorMessages = inputContextAt(tokenIndex);
    consume(TokenKind.RPAREN);
    return new QueryAst.FunctionCall(
        functionName,
        ImmutableList.copyOf(args),
        functionContextForErrorMessages,
        closeParenContextForErrorMessages);
  }

  /**
   * Parses an argument inside a function call. A standalone {@code WORD} argument is not checked
   * against {@link #validateTargetLiteral} here because the function's parameter may have type
   * {@code WORD} or {@code INTEGER} rather than {@code EXPRESSION}.
   */
  private QueryAst parseFunctionArgumentExpression() throws QuerySyntaxException {
    if (token.kind == TokenKind.WORD) {
      int wordTokenIndex = tokenIndex;
      String word = consume(TokenKind.WORD);
      if (token.kind == TokenKind.LPAREN) {
        return parseBinaryOperatorTail(parseFunctionCallRest(word, wordTokenIndex));
      }
      if (!BINARY_OPERATORS.contains(token.kind)) {
        return new QueryAst.Word(word);
      }
      validateTargetLiteral(word);
      return parseBinaryOperatorTail(new QueryAst.Word(word));
    }
    return parseExpression();
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
