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
final class QueryParser {

  private Lexer.Token token; // current lookahead token
  private final List<Lexer.Token> tokens;
  private final Iterator<Lexer.Token> tokenIterator;
  private final Map<String, QueryFunction> functions;

  /**
   * Scan and parse the specified query expression.
   */
  static QueryExpression parse(String query, QueryEnvironment<?> env) throws QueryException {
    QueryParser parser = new QueryParser(
        Lexer.scan(query.toCharArray()), env);
    QueryExpression expr = parser.parseExpression();
    if (parser.token.kind != TokenKind.EOF) {
      throw new QueryException("unexpected token '" + parser.token
                               + "' after query expression '" + expr +  "'");
    }
    return expr;
  }

  private QueryParser(List<Lexer.Token> tokens, QueryEnvironment<?> env) {
    // TODO(bazel-team): We only need QueryEnvironment#getFunctions, consider refactoring users of
    // QueryParser#parse to instead just pass in the set of functions to make testing, among other
    // things, simpler.
    this.functions = new HashMap<>();
    for (QueryFunction queryFunction : env.getFunctions()) {
      this.functions.put(queryFunction.getName(), queryFunction);
    }
    this.tokens = tokens;
    this.tokenIterator = tokens.iterator();
    nextToken();
  }

  /**
   * Returns an exception.  Don't forget to throw it.
   */
  private QueryException syntaxError(Lexer.Token token) {
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
    return new QueryException(message);
  }

  /**
   * Consumes the current token.  If it is not of the specified (expected)
   * kind, throws QueryException.  Returns the value associated with the
   * consumed token, if any.
   */
  private String consume(TokenKind kind) throws QueryException {
    if (token.kind != kind) {
      throw syntaxError(token);
    }
    String word = token.word;
    nextToken();
    return word;
  }

  /**
   * Consumes the current token, which must be a WORD containing an integer
   * literal.  Returns that integer, or throws a QueryException otherwise.
   */
  private int consumeIntLiteral() throws QueryException {
    String intString = consume(TokenKind.WORD);
    try {
      return Integer.parseInt(intString);
    } catch (NumberFormatException e) {
      throw new QueryException("expected an integer literal: '" + intString + "'");
    }
  }

  private void nextToken() {
    if (token == null || token.kind != TokenKind.EOF) {
      token = tokenIterator.next();
    }
  }

  /**
   * expr ::= primary
   *        | expr INTERSECT expr
   *        | expr '^' expr
   *        | expr UNION expr
   *        | expr '+' expr
   *        | expr EXCEPT expr
   *        | expr '-' expr
   */
  private QueryExpression parseExpression() throws QueryException {
    // All operators are left-associative and of equal precedence.
    return parseBinaryOperatorTail(parsePrimary());
  }

  /**
   * tail ::= ( <op> <primary> )*
   * All operators have equal precedence.
   * This factoring is required for left-associative binary operators in LL(1).
   */
  private QueryExpression parseBinaryOperatorTail(QueryExpression lhs) throws QueryException {
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
   * primary ::= WORD
   *           | LET WORD = expr IN expr
   *           | '(' expr ')'
   *           | WORD '(' expr ( ',' expr ) * ')'
   *           | DEPS '(' expr ')'
   *           | DEPS '(' expr ',' WORD ')'
   *           | RDEPS '(' expr ',' expr ')'
   *           | RDEPS '(' expr ',' expr ',' WORD ')'
   *           | SET '(' WORD * ')'
   */
  private QueryExpression parsePrimary() throws QueryException {
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
          return new TargetLiteral(word);
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
          words.add(new TargetLiteral(consume(TokenKind.WORD)));
        }
        consume(TokenKind.RPAREN);
        return new SetExpression(words);
      }
      default:
        throw syntaxError(token);
    }
  }
}
