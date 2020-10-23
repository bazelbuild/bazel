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

import com.google.devtools.build.lib.query2.engine.Lexer.TokenKind;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * LL(1) recursive descent parser for the Blaze query language, revision 2.
 *
 * In the grammar below, non-terminals are lowercase and terminals are
 * uppercase, or character literals.
 *
 * <pre>
 * expr ::= WORD
 *        | LET WORD = expr IN expr
 *        | '(' expr ')'
 *        | WORD '(' expr ( ',' expr ) * ')'
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

  private Lexer.Token token; // current lookahead token
  private final List<Lexer.Token> tokens;
  private final Iterator<Lexer.Token> tokenIterator;
  private final Map<String, QueryFunction> functions;

  /** Scan and parse the specified query expression. */
  public static QueryExpression parse(String query, QueryEnvironment<?> env)
      throws QuerySyntaxException {
    HashMap<String, QueryFunction> functions = new HashMap<>();
    for (QueryFunction queryFunction : env.getFunctions()) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    return parse(query, functions);
  }

  public static QueryExpression parse(String query, Map<String, QueryFunction> functions)
      throws QuerySyntaxException {
    QueryParser parser = new QueryParser(Lexer.scan(query), functions);
    QueryExpression expr = parser.parseExpression();
    if (parser.token.kind != TokenKind.EOF) {
      throw new QuerySyntaxException(
          String.format(
              "unexpected token '%s' after query expression '%s'",
              parser.token, expr.toTrunctatedString()));
    }
    return expr;
  }

  public QueryParser(List<Lexer.Token> tokens, Map<String, QueryFunction> functions) {
    this.functions = functions;
    this.tokens = tokens;
    this.tokenIterator = tokens.iterator();
    nextToken();
  }

  /** Returns an exception. Don't forget to throw it. */
  private QuerySyntaxException syntaxError(Lexer.Token token) {
    String message = "premature end of input";
    if (token.kind != TokenKind.EOF) {
      StringBuilder buf = new StringBuilder("syntax error at '");
      String sep = "";
      for (int index = tokens.indexOf(token),
               max = Math.min(tokens.size() - 1, index + 3); // 3 tokens of context
               index < max; ++index) {
        buf.append(sep).append(tokens.get(index));
        sep = " ";
      }
      buf.append("'");
      message = buf.toString();
    }
    return new QuerySyntaxException(message);
  }

  /**
   * Consumes the current token. If it is not of the specified (expected) kind, throws {@link
   * QuerySyntaxException}. Returns the value associated with the consumed token, if any.
   */
  private String consume(TokenKind kind) throws QuerySyntaxException {
    if (token.kind != kind) {
      throw syntaxError(token);
    }
    String word = token.word;
    nextToken();
    return word;
  }

  /**
   * Consumes the current token, which must be a WORD containing an integer literal. Returns that
   * integer, or throws a {@link QuerySyntaxException} otherwise.
   */
  private int consumeIntLiteral() throws QuerySyntaxException {
    String intString = consume(TokenKind.WORD);
    try {
      return Integer.parseInt(intString);
    } catch (NumberFormatException e) {
      throw new QuerySyntaxException("expected an integer literal: '" + intString + "'");
    }
  }

  private void nextToken() {
    if (token == null || token.kind != TokenKind.EOF) {
      token = tokenIterator.next();
    }
  }

  /**
   *
   *
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
  private QueryExpression parseExpression() throws QuerySyntaxException {
    // All operators are left-associative and of equal precedence.
    return parseBinaryOperatorTail(parsePrimary());
  }

  /**
   *
   *
   * <pre>
   * tail ::= ( <op> <primary> )*
   * </pre>
   *
   * <p>All operators have equal precedence. This factoring is required for left-associative binary
   * operators in LL(1).
   */
  private QueryExpression parseBinaryOperatorTail(QueryExpression lhs) throws QuerySyntaxException {
    if (!BINARY_OPERATORS.contains(token.kind)) {
      return lhs;
    }

    List<QueryExpression> operands = new ArrayList<>();
    operands.add(lhs);
    TokenKind lastOperator = token.kind;

    while (BINARY_OPERATORS.contains(token.kind)) {
      TokenKind operator = token.kind;
      consume(operator);
      if (operator != lastOperator) {
        lhs = new BinaryOperatorExpression(lastOperator, operands);
        operands.clear();
        operands.add(lhs);
        lastOperator = operator;
      }
      QueryExpression rhs = parsePrimary();
      operands.add(rhs);
    }
    return new BinaryOperatorExpression(lastOperator, operands);
  }

  /**
   *
   *
   * <pre>
   * primary ::= WORD
   *           | WORD '(' arg ( ',' arg ) * ')'
   *           | LET WORD = expr IN expr
   *           | '(' expr ')'
   *           | SET '(' WORD * ')' arg ::= expr
   *           | WORD
   *           | INT
   * </pre>
   */
  private QueryExpression parsePrimary() throws QuerySyntaxException {
    switch (token.kind) {
      case WORD: {
        String word = consume(TokenKind.WORD);
        if (token.kind == TokenKind.LPAREN) {
          QueryFunction function = functions.get(word);
          if (function == null) {
            throw syntaxError(token);
          }
          List<Argument> args = new ArrayList<>();
          TokenKind tokenKind = TokenKind.LPAREN;
          int argsSeen = 0;
          for (ArgumentType type : function.getArgumentTypes()) {
            if (token.kind == TokenKind.RPAREN && argsSeen >= function.getMandatoryArguments()) {
              break;
            }

            consume(tokenKind);
            tokenKind = TokenKind.COMMA;
            switch (type) {
              case EXPRESSION:
                args.add(Argument.of(parseExpression()));
                break;

              case WORD:
                args.add(Argument.of(consume(TokenKind.WORD)));
                break;

              case INTEGER:
                args.add(Argument.of(consumeIntLiteral()));
                break;

              default:
                throw new IllegalStateException();
            }

            argsSeen++;
          }

          consume(TokenKind.RPAREN);
          return new FunctionExpression(function, args);
        } else {
            return validateTargetLiteral(word);
        }
      }
      case LET: {
        consume(TokenKind.LET);
        String name = consume(TokenKind.WORD);
        consume(TokenKind.EQUALS);
        QueryExpression varExpr = parseExpression();
        consume(TokenKind.IN);
        QueryExpression bodyExpr = parseExpression();
        return new LetExpression(name, varExpr, bodyExpr);
      }
      case LPAREN: {
        consume(TokenKind.LPAREN);
        QueryExpression expr = parseExpression();
        consume(TokenKind.RPAREN);
        return expr;
      }
      case SET: {
        nextToken();
        consume(TokenKind.LPAREN);
        List<TargetLiteral> words = new ArrayList<>();
        while (token.kind == TokenKind.WORD) {
            words.add(validateTargetLiteral(consume(TokenKind.WORD)));
        }
        consume(TokenKind.RPAREN);
        return new SetExpression(words);
      }
      default:
        throw syntaxError(token);
    }
  }

  /**
   * Unquoted words may not start with a hyphen or asterisk, even though relative target names may
   * start with those characters.
   */
  private static TargetLiteral validateTargetLiteral(String word) throws QuerySyntaxException {
    if (word.startsWith("-") || word.startsWith("*")) {
      throw new QuerySyntaxException(
          "target literal must not begin with " + "(" + word.charAt(0) + "): " + word);
    }
    return new TargetLiteral(word);
  }
}
