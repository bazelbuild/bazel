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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Recursive descent parser for LL(2) BUILD language. Loosely based on Python 2 grammar. See
 * https://docs.python.org/2/reference/grammar.html
 */
// TODO(adonovan): break syntax->event.Event dependency.
@VisibleForTesting
final class Parser {

  /** Combines the parser result into a single value object. */
  static final class ParseResult {
    /** The statements (rules, basically) from the parsed file. */
    final List<Statement> statements;

    /** The comments from the parsed file. */
    final List<Comment> comments;

    /** Represents every statement in the file. */
    final Lexer.LexerLocation location;

    // Errors encountered during scanning or parsing.
    // These lists are ultimately owned by StarlarkFile.
    final List<Event> errors;
    final List<Event> stringEscapeEvents;

    ParseResult(
        List<Statement> statements,
        List<Comment> comments,
        Lexer.LexerLocation location,
        List<Event> errors,
        List<Event> stringEscapeEvents) {
      // No need to copy here; when the object is created, the parser instance is just about to go
      // out of scope and be garbage collected.
      this.statements = Preconditions.checkNotNull(statements);
      this.comments = Preconditions.checkNotNull(comments);
      this.location = location;
      this.errors = errors;
      this.stringEscapeEvents = stringEscapeEvents;
    }
  }

  private static final EnumSet<TokenKind> STATEMENT_TERMINATOR_SET =
      EnumSet.of(TokenKind.EOF, TokenKind.NEWLINE, TokenKind.SEMI);

  private static final EnumSet<TokenKind> LIST_TERMINATOR_SET =
      EnumSet.of(TokenKind.EOF, TokenKind.RBRACKET, TokenKind.SEMI);

  private static final EnumSet<TokenKind> DICT_TERMINATOR_SET =
      EnumSet.of(TokenKind.EOF, TokenKind.RBRACE, TokenKind.SEMI);

  private static final EnumSet<TokenKind> EXPR_LIST_TERMINATOR_SET =
      EnumSet.of(
          TokenKind.EOF,
          TokenKind.NEWLINE,
          TokenKind.EQUALS,
          TokenKind.RBRACE,
          TokenKind.RBRACKET,
          TokenKind.RPAREN,
          TokenKind.SEMI);

  private static final EnumSet<TokenKind> EXPR_TERMINATOR_SET =
      EnumSet.of(
          TokenKind.COLON,
          TokenKind.COMMA,
          TokenKind.EOF,
          TokenKind.FOR,
          TokenKind.MINUS,
          TokenKind.PERCENT,
          TokenKind.PLUS,
          TokenKind.RBRACKET,
          TokenKind.RPAREN,
          TokenKind.SLASH);

  /** Current lookahead token. May be mutated by the parser. */
  private Token token;

  private static final boolean DEBUGGING = false;

  private final Lexer lexer;
  private final List<Event> errors;

  // TODO(adonovan): opt: compute this by subtraction.
  private static final Map<TokenKind, TokenKind> augmentedAssignments =
      new ImmutableMap.Builder<TokenKind, TokenKind>()
          .put(TokenKind.PLUS_EQUALS, TokenKind.PLUS)
          .put(TokenKind.MINUS_EQUALS, TokenKind.MINUS)
          .put(TokenKind.STAR_EQUALS, TokenKind.STAR)
          .put(TokenKind.SLASH_EQUALS, TokenKind.SLASH)
          .put(TokenKind.SLASH_SLASH_EQUALS, TokenKind.SLASH_SLASH)
          .put(TokenKind.PERCENT_EQUALS, TokenKind.PERCENT)
          .put(TokenKind.AMPERSAND_EQUALS, TokenKind.AMPERSAND)
          .put(TokenKind.CARET_EQUALS, TokenKind.CARET)
          .put(TokenKind.PIPE_EQUALS, TokenKind.PIPE)
          .put(TokenKind.GREATER_GREATER_EQUALS, TokenKind.GREATER_GREATER)
          .put(TokenKind.LESS_LESS_EQUALS, TokenKind.LESS_LESS)
          .build();

  /**
   * Highest precedence goes last. Based on:
   * http://docs.python.org/2/reference/expressions.html#operator-precedence
   */
  private static final List<EnumSet<TokenKind>> operatorPrecedence =
      ImmutableList.of(
          EnumSet.of(TokenKind.OR),
          EnumSet.of(TokenKind.AND),
          EnumSet.of(TokenKind.NOT),
          EnumSet.of(
              TokenKind.EQUALS_EQUALS,
              TokenKind.NOT_EQUALS,
              TokenKind.LESS,
              TokenKind.LESS_EQUALS,
              TokenKind.GREATER,
              TokenKind.GREATER_EQUALS,
              TokenKind.IN,
              TokenKind.NOT_IN),
          EnumSet.of(TokenKind.PIPE),
          EnumSet.of(TokenKind.CARET),
          EnumSet.of(TokenKind.AMPERSAND),
          EnumSet.of(TokenKind.GREATER_GREATER, TokenKind.LESS_LESS),
          EnumSet.of(TokenKind.MINUS, TokenKind.PLUS),
          EnumSet.of(TokenKind.SLASH, TokenKind.SLASH_SLASH, TokenKind.STAR, TokenKind.PERCENT));

  private int errorsCount;
  private boolean recoveryMode;  // stop reporting errors until next statement

  // Intern string literals, as some files contain many literals for the same string.
  private final Map<String, String> stringInterner = new HashMap<>();

  private Parser(Lexer lexer, List<Event> errors) {
    this.lexer = lexer;
    this.errors = errors;
    nextToken();
  }

  private String intern(String s) {
    String prev = stringInterner.putIfAbsent(s, s);
    return prev != null ? prev : s;
  }

  private static Lexer.LexerLocation locationFromStatements(
      Lexer lexer, List<Statement> statements) {
    int start = 0;
    int end = 0;
    if (!statements.isEmpty()) {
      start = statements.get(0).getStartOffset();
      end = Iterables.getLast(statements).getEndOffset();
    }
    return lexer.createLocation(start, end);
  }

  // Main entry point for parsing a file.
  static ParseResult parseFile(ParserInput input) {
    List<Event> errors = new ArrayList<>();
    Lexer lexer = new Lexer(input, errors);
    Parser parser = new Parser(lexer, errors);
    List<Statement> statements;
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.STARLARK_PARSER, input.getFile())) {
      statements = parser.parseFileInput();
    }
    return new ParseResult(
        statements,
        lexer.getComments(),
        locationFromStatements(lexer, statements),
        errors,
        lexer.getStringEscapeEvents());
  }

  // stmt ::= simple_stmt
  //        | def_stmt
  //        | for_stmt
  //        | if_stmt
  private void parseStatement(List<Statement> list) {
    if (token.kind == TokenKind.DEF) {
      list.add(parseDefStatement());
    } else if (token.kind == TokenKind.IF) {
      list.add(parseIfStatement());
    } else if (token.kind == TokenKind.FOR) {
      list.add(parseForStatement());
    } else {
      parseSimpleStatement(list);
    }
  }

  /** Parses an expression, possibly followed by newline tokens. */
  static Expression parseExpression(ParserInput input) throws SyntaxError {
    List<Event> errors = new ArrayList<>();
    Lexer lexer = new Lexer(input, errors);
    Parser parser = new Parser(lexer, errors);
    Expression result = parser.parseExpression();
    while (parser.token.kind == TokenKind.NEWLINE) {
      parser.nextToken();
    }
    parser.expect(TokenKind.EOF);
    if (!errors.isEmpty()) {
      throw new SyntaxError(errors);
    }
    return result;
  }

  private Expression parseExpression() {
    return parseExpression(false);
  }

  // Equivalent to 'testlist' rule in Python grammar. It can parse every kind of
  // expression. In many cases, we need to use parseNonTupleExpression to avoid ambiguity:
  //   e.g. fct(x, y)  vs  fct((x, y))
  //
  // Tuples can have a trailing comma only when insideParens is true. This prevents bugs
  // where a one-element tuple is surprisingly created:
  //   e.g.  foo = f(x),
  private Expression parseExpression(boolean insideParens) {
    int start = token.left;
    Expression expression = parseNonTupleExpression();
    if (token.kind != TokenKind.COMMA) {
      return expression;
    }

    // It's a tuple
    List<Expression> tuple = parseExprList(insideParens);
    tuple.add(0, expression); // add the first expression to the front of the tuple
    return setLocation(
        new ListExpression(/*isTuple=*/ true, tuple), start, Iterables.getLast(tuple));
  }

  private void reportError(Location location, String message) {
    errorsCount++;
    // Limit the number of reported errors to avoid spamming output.
    if (errorsCount <= 5) {
      errors.add(Event.error(location, message));
    }
  }

  private void syntaxError(String message) {
    if (!recoveryMode) {
      String msg = token.kind == TokenKind.INDENT
          ? "indentation error"
          : "syntax error at '" + token + "': " + message;
      reportError(lexer.createLocation(token.left, token.right), msg);
      recoveryMode = true;
    }
  }

  /**
   * Consumes the current token. If it is not of the specified (expected)
   * kind, reports a syntax error.
   */
  private boolean expect(TokenKind kind) {
    boolean expected = token.kind == kind;
    if (!expected) {
      syntaxError("expected " + kind);
    }
    nextToken();
    return expected;
  }

  /**
   * Same as expect, but stop the recovery mode if the token was expected.
   */
  private void expectAndRecover(TokenKind kind) {
    if (expect(kind)) {
      recoveryMode = false;
    }
  }

  /**
   * Consume tokens past the first token that has a kind that is in the set of
   * terminatingTokens.
   * @param terminatingTokens
   * @return the end offset of the terminating token.
   */
  private int syncPast(EnumSet<TokenKind> terminatingTokens) {
    Preconditions.checkState(terminatingTokens.contains(TokenKind.EOF));
    while (!terminatingTokens.contains(token.kind)) {
      nextToken();
    }
    int end = token.right;
    // read past the synchronization token
    nextToken();
    return end;
  }

  /**
   * Consume tokens until we reach the first token that has a kind that is in
   * the set of terminatingTokens.
   * @param terminatingTokens
   * @return the end offset of the terminating token.
   */
  private int syncTo(EnumSet<TokenKind> terminatingTokens) {
    // EOF must be in the set to prevent an infinite loop
    Preconditions.checkState(terminatingTokens.contains(TokenKind.EOF));
    // read past the problematic token
    int previous = token.right;
    nextToken();
    int current = previous;
    while (!terminatingTokens.contains(token.kind)) {
      nextToken();
      previous = current;
      current = token.right;
    }
    return previous;
  }

  // Keywords that exist in Python and that we don't parse.
  private static final EnumSet<TokenKind> FORBIDDEN_KEYWORDS =
      EnumSet.of(
          TokenKind.AS,
          TokenKind.ASSERT,
          TokenKind.CLASS,
          TokenKind.DEL,
          TokenKind.EXCEPT,
          TokenKind.FINALLY,
          TokenKind.FROM,
          TokenKind.GLOBAL,
          TokenKind.IMPORT,
          TokenKind.IS,
          TokenKind.LAMBDA,
          TokenKind.NONLOCAL,
          TokenKind.RAISE,
          TokenKind.TRY,
          TokenKind.WITH,
          TokenKind.WHILE,
          TokenKind.YIELD);

  private void checkForbiddenKeywords() {
    if (!FORBIDDEN_KEYWORDS.contains(token.kind)) {
      return;
    }
    String error;
    switch (token.kind) {
      case ASSERT: error = "'assert' not supported, use 'fail' instead"; break;
      case DEL:
        error = "'del' not supported, use '.pop()' to delete an item from a dictionary or a list";
        break;
      case IMPORT: error = "'import' not supported, use 'load' instead"; break;
      case IS: error = "'is' not supported, use '==' instead"; break;
      case LAMBDA: error = "'lambda' not supported, declare a function instead"; break;
      case RAISE: error = "'raise' not supported, use 'fail' instead"; break;
      case TRY: error = "'try' not supported, all exceptions are fatal"; break;
      case WHILE: error = "'while' not supported, use 'for' instead"; break;
      default:
        error = "keyword '" + token.kind + "' not supported";
        break;
    }
    reportError(lexer.createLocation(token.left, token.right), error);
  }

  private void nextToken() {
    if (token == null || token.kind != TokenKind.EOF) {
      token = lexer.nextToken();
    }
    checkForbiddenKeywords();
    if (DEBUGGING) {
      System.err.print(token);
    }
  }

  private Identifier makeErrorExpression(int start, int end) {
    // It's tempting to define a dedicated BadExpression type,
    // but it is convenient for parseIdent to return an Identifier
    // even when it fails.
    return setLocation(Identifier.of("$error$"), start, end);
  }

  // Convenience wrapper method around Node.setLocation
  private <NodeT extends Node> NodeT setLocation(NodeT node, int startOffset, int endOffset) {
    return Node.setLocation(lexer.createLocation(startOffset, endOffset), node);
  }

  // Convenience method that uses end offset from the last node.
  private <NodeT extends Node> NodeT setLocation(NodeT node, int startOffset, Node lastNode) {
    Preconditions.checkNotNull(lastNode, "can't extract end offset from a null node");
    return setLocation(node, startOffset, lastNode.getEndOffset());
  }

  // arg ::= IDENTIFIER '=' nontupleexpr
  //       | expr
  //       | *args
  //       | **kwargs
  private Argument parseArgument() {
    final int start = token.left;
    Expression expr;
    // parse **expr
    if (token.kind == TokenKind.STAR_STAR) {
      nextToken();
      expr = parseNonTupleExpression();
      return setLocation(new Argument.StarStar(expr), start, expr);
    }
    // parse *expr
    if (token.kind == TokenKind.STAR) {
      nextToken();
      expr = parseNonTupleExpression();
      return setLocation(new Argument.Star(expr), start, expr);
    }

    expr = parseNonTupleExpression();
    if (expr instanceof Identifier) {
      // parse a named argument
      if (token.kind == TokenKind.EQUALS) {
        nextToken();
        Expression val = parseNonTupleExpression();
        return setLocation(new Argument.Keyword(((Identifier) expr), val), start, val);
      }
    }

    // parse a positional argument
    return setLocation(new Argument.Positional(expr), start, expr);
  }

  // arg ::= IDENTIFIER '=' nontupleexpr
  //       | IDENTIFIER
  private Parameter parseFunctionParameter() {
    int start = token.left;
    if (token.kind == TokenKind.STAR_STAR) { // kwarg
      nextToken();
      Identifier ident = parseIdent();
      return setLocation(new Parameter.StarStar(ident), start, ident);
    } else if (token.kind == TokenKind.STAR) { // stararg
      int end = token.right;
      nextToken();
      if (token.kind == TokenKind.IDENTIFIER) {
        Identifier ident = parseIdent();
        return setLocation(new Parameter.Star(ident), start, ident);
      } else {
        return setLocation(new Parameter.Star(null), start, end);
      }
    } else {
      Identifier ident = parseIdent();
      if (token.kind == TokenKind.EQUALS) { // there's a default value
        nextToken();
        Expression expr = parseNonTupleExpression();
        return setLocation(new Parameter.Optional(ident, expr), start, expr);
      } else {
        return setLocation(new Parameter.Mandatory(ident), start, ident);
      }
    }
  }

  // call_suffix ::= '(' arg_list? ')'
  private Expression parseCallSuffix(int start, Expression function) {
    ImmutableList<Argument> args = ImmutableList.of();
    expect(TokenKind.LPAREN);
    int end;
    if (token.kind == TokenKind.RPAREN) {
      end = token.right;
      nextToken(); // RPAREN
    } else {
      args = parseArguments(); // (includes optional trailing comma)
      end = token.right;
      expect(TokenKind.RPAREN);
    }
    return setLocation(new CallExpression(function, args), start, end);
  }

  // Parse a list of call arguments.
  //
  // arg_list ::= ( (arg ',')* arg ','? )?
  private ImmutableList<Argument> parseArguments() {
    boolean hasArgs = false;
    boolean hasStarStar = false;
    ImmutableList.Builder<Argument> list = ImmutableList.builder();

    while (token.kind != TokenKind.RPAREN && token.kind != TokenKind.EOF) {
      if (hasArgs) {
        expect(TokenKind.COMMA);
        // The list may end with a comma.
        if (token.kind == TokenKind.RPAREN) {
          break;
        }
      }
      if (hasStarStar) {
        // TODO(adonovan): move this to validation pass too.
        reportError(
            lexer.createLocation(token.left, token.right),
            "unexpected tokens after **kwargs argument");
        break;
      }
      Argument arg = parseArgument();
      hasArgs = true;
      if (arg instanceof Argument.StarStar) { // TODO(adonovan): not Star too? verify.
        hasStarStar = true;
      }
      list.add(arg);
    }
    ImmutableList<Argument> args = list.build();
    validateArguments(args); // TODO(adonovan): move to validation pass.
    return args;
  }

  // TODO(adonovan): move all this to validator, since we have to check it again there.
  private void validateArguments(List<Argument> arguments) {
    int i = 0;
    int len = arguments.size();

    while (i < len && arguments.get(i) instanceof Argument.Positional) {
      i++;
    }

    while (i < len && arguments.get(i) instanceof Argument.Keyword) {
      i++;
    }

    if (i < len && arguments.get(i) instanceof Argument.Star) {
      i++;
    }

    if (i < len && arguments.get(i) instanceof Argument.StarStar) {
      i++;
    }

    // If there's no argument left, everything is correct.
    if (i == len) {
      return;
    }

    Argument arg = arguments.get(i);
    Location loc = arg.getStartLocation();
    if (arg instanceof Argument.Positional) {
      reportError(loc, "positional argument is misplaced (positional arguments come first)");
      return;
    }

    if (arg instanceof Argument.Keyword) {
      reportError(
          loc,
          "keyword argument is misplaced (keyword arguments must be before any *arg or **kwarg)");
      return;
    }

    if (i < len && arg instanceof Argument.Star) {
      reportError(loc, "*arg argument is misplaced");
      return;
    }

    if (i < len && arg instanceof Argument.StarStar) {
      reportError(loc, "**kwarg argument is misplaced (there can be only one)");
      return;
    }
  }

  // selector_suffix ::= '.' IDENTIFIER
  private Expression parseSelectorSuffix(int start, Expression receiver) {
    expect(TokenKind.DOT);
    if (token.kind == TokenKind.IDENTIFIER) {
      Identifier ident = parseIdent();
      return setLocation(new DotExpression(receiver, ident), start, ident);
    } else {
      syntaxError("expected identifier after dot");
      int end = syncTo(EXPR_TERMINATOR_SET);
      return makeErrorExpression(start, end);
    }
  }

  // expr_list parses a comma-separated list of expression. It assumes that the
  // first expression was already parsed, so it starts with a comma.
  // It is used to parse tuples and list elements.
  // expr_list ::= ( ',' expr )* ','?
  private List<Expression> parseExprList(boolean trailingColonAllowed) {
    List<Expression> list = new ArrayList<>();
    //  terminating tokens for an expression list
    while (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      if (EXPR_LIST_TERMINATOR_SET.contains(token.kind)) {
        if (!trailingColonAllowed) {
          reportError(
              lexer.createLocation(token.left, token.right),
              "Trailing comma is allowed only in parenthesized tuples.");
        }
        break;
      }
      list.add(parseNonTupleExpression());
    }
    return list;
  }

  // dict_entry_list ::= ( (dict_entry ',')* dict_entry ','? )?
  private List<DictExpression.Entry> parseDictEntryList() {
    List<DictExpression.Entry> list = new ArrayList<>();
    // the terminating token for a dict entry list
    while (token.kind != TokenKind.RBRACE) {
      list.add(parseDictEntry());
      if (token.kind == TokenKind.COMMA) {
        nextToken();
      } else {
        break;
      }
    }
    return list;
  }

  // dict_entry ::= nontupleexpr ':' nontupleexpr
  private DictExpression.Entry parseDictEntry() {
    int start = token.left;
    Expression key = parseNonTupleExpression();
    expect(TokenKind.COLON);
    Expression value = parseNonTupleExpression();
    return setLocation(new DictExpression.Entry(key, value), start, value);
  }

  /**
   * Parse a String literal value, e.g. "str".
   */
  private StringLiteral parseStringLiteral() {
    Preconditions.checkState(token.kind == TokenKind.STRING);
    int end = token.right;
    StringLiteral literal =
        setLocation(new StringLiteral(intern((String) token.value)), token.left, end);
    nextToken();
    if (token.kind == TokenKind.STRING) {
      reportError(lexer.createLocation(end, token.left),
          "Implicit string concatenation is forbidden, use the + operator");
    }
    return literal;
  }

  //  primary ::= INTEGER
  //            | STRING
  //            | IDENTIFIER
  //            | list_expression
  //            | '(' ')'                    // a tuple with zero elements
  //            | '(' expr ')'               // a parenthesized expression
  //            | dict_expression
  //            | '-' primary_with_suffix
  private Expression parsePrimary() {
    int start = token.left;
    switch (token.kind) {
      case INT:
        {
          IntegerLiteral literal = new IntegerLiteral((Integer) token.value);
          setLocation(literal, start, token.right);
          nextToken();
          return literal;
        }
      case STRING:
        return parseStringLiteral();
      case IDENTIFIER:
        return parseIdent();
      case LBRACKET: // it's a list
        return parseListMaker();
      case LBRACE: // it's a dictionary
        return parseDictExpression();
      case LPAREN:
        {
          nextToken();
          // check for the empty tuple literal
          if (token.kind == TokenKind.RPAREN) {
            ListExpression tuple = new ListExpression(/*isTuple=*/ true, ImmutableList.of());
            setLocation(tuple, start, token.right);
            nextToken();
            return tuple;
          }
          // parse the first expression
          Expression expression = parseExpression(true);
          setLocation(expression, start, token.right);
          if (token.kind == TokenKind.RPAREN) {
            nextToken();
            return expression;
          }
          expect(TokenKind.RPAREN);
          int end = syncTo(EXPR_TERMINATOR_SET);
          return makeErrorExpression(start, end);
        }
      case MINUS:
        {
          nextToken();
          Expression expr = parsePrimaryWithSuffix();
          UnaryOperatorExpression minus = new UnaryOperatorExpression(TokenKind.MINUS, expr);
          return setLocation(minus, start, expr);
        }
      case PLUS:
        {
          nextToken();
          Expression expr = parsePrimaryWithSuffix();
          UnaryOperatorExpression plus = new UnaryOperatorExpression(TokenKind.PLUS, expr);
          return setLocation(plus, start, expr);
        }
      case TILDE:
        {
          nextToken();
          Expression expr = parsePrimaryWithSuffix();
          UnaryOperatorExpression tilde = new UnaryOperatorExpression(TokenKind.TILDE, expr);
          return setLocation(tilde, start, expr);
        }
      default:
        {
          syntaxError("expected expression");
          int end = syncTo(EXPR_TERMINATOR_SET);
          return makeErrorExpression(start, end);
        }
    }
  }

  // primary_with_suffix ::= primary (selector_suffix | substring_suffix | call_suffix)*
  private Expression parsePrimaryWithSuffix() {
    int start = token.left;
    Expression receiver = parsePrimary();
    while (true) {
      if (token.kind == TokenKind.DOT) {
        receiver = parseSelectorSuffix(start, receiver);
      } else if (token.kind == TokenKind.LBRACKET) {
        receiver = parseSubstringSuffix(start, receiver);
      } else if (token.kind == TokenKind.LPAREN) {
        receiver = parseCallSuffix(start, receiver);
      } else {
        break;
      }
    }
    return receiver;
  }

  // substring_suffix ::= '[' expression? ':' expression?  ':' expression? ']'
  //                    | '[' expression? ':' expression? ']'
  //                    | '[' expression ']'
  private Expression parseSubstringSuffix(int start, Expression receiver) {
    Expression startExpr;

    expect(TokenKind.LBRACKET);
    if (token.kind == TokenKind.COLON) {
      startExpr = null;
    } else {
      startExpr = parseExpression();
    }
    // This is an index/key access
    if (token.kind == TokenKind.RBRACKET) {
      Expression expr = setLocation(new IndexExpression(receiver, startExpr), start, token.right);
      expect(TokenKind.RBRACKET);
      return expr;
    }
    // This is a slice (or substring)
    Expression endExpr = parseSliceArgument();
    Expression stepExpr = parseSliceArgument();
    Expression expr =
        setLocation(
            new SliceExpression(receiver, startExpr, endExpr, stepExpr), start, token.right);
    expect(TokenKind.RBRACKET);
    return expr;
  }

  /**
   * Parses {@code [':' [expr]]} which can either be the end or the step argument of a slice
   * operation. If no such expression is found, this method returns null.
   */
  private @Nullable Expression parseSliceArgument() {
    // There has to be a colon before any end or slice argument.
    // However, if the next token thereafter is another colon or a right bracket, no argument value
    // was specified.
    if (token.kind == TokenKind.COLON) {
      expect(TokenKind.COLON);
      if (token.kind != TokenKind.COLON && token.kind != TokenKind.RBRACKET) {
        return parseNonTupleExpression();
      }
    }
    return null;
  }

  // Equivalent to 'exprlist' rule in Python grammar.
  // loop_variables ::= primary_with_suffix ( ',' primary_with_suffix )* ','?
  private Expression parseForLoopVariables() {
    // We cannot reuse parseExpression because it would parse the 'in' operator.
    // e.g.  "for i in e: pass"  -> we want to parse only "i" here.
    int start = token.left;
    Expression e1 = parsePrimaryWithSuffix();
    if (token.kind != TokenKind.COMMA) {
      return e1;
    }

    // It's a tuple
    List<Expression> tuple = new ArrayList<>();
    tuple.add(e1);
    while (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      if (EXPR_LIST_TERMINATOR_SET.contains(token.kind)) {
        break;
      }
      tuple.add(parsePrimaryWithSuffix());
    }
    return setLocation(
        new ListExpression(/*isTuple=*/ true, tuple), start, Iterables.getLast(tuple));
  }

  // comprehension_suffix ::= 'FOR' loop_variables 'IN' expr comprehension_suffix
  //                        | 'IF' expr comprehension_suffix
  //                        | ']' | '}'
  private Expression parseComprehensionSuffix(Node body, TokenKind closingBracket, int offset) {
    ImmutableList.Builder<Comprehension.Clause> clauses = ImmutableList.builder();
    while (true) {
      if (token.kind == TokenKind.FOR) {
        nextToken();
        Expression vars = parseForLoopVariables();
        expect(TokenKind.IN);
        // The expression cannot be a ternary expression ('x if y else z') due to
        // conflicts in Python grammar ('if' is used by the comprehension).
        Expression seq = parseNonTupleExpression(0);
        clauses.add(new Comprehension.For(vars, seq));
      } else if (token.kind == TokenKind.IF) {
        nextToken();
        // [x for x in li if 1, 2]  # parse error
        // [x for x in li if (1, 2)]  # ok
        Expression cond = parseNonTupleExpression(0);
        clauses.add(new Comprehension.If(cond));
      } else if (token.kind == closingBracket) {
        break;
      } else {
        syntaxError("expected '" + closingBracket + "', 'for' or 'if'");
        syncPast(LIST_TERMINATOR_SET);
        return makeErrorExpression(offset, token.right);
      }
    }
    boolean isDict = closingBracket == TokenKind.RBRACE;
    Comprehension comp = new Comprehension(isDict, body, clauses.build());
    setLocation(comp, offset, token.right);
    nextToken();
    return comp;
  }

  // list_maker ::= '[' ']'
  //               |'[' expr ']'
  //               |'[' expr expr_list ']'
  //               |'[' expr comprehension_suffix ']'
  private Expression parseListMaker() {
    int start = token.left;
    expect(TokenKind.LBRACKET);
    if (token.kind == TokenKind.RBRACKET) { // empty List
      ListExpression list = new ListExpression(/*isTuple=*/ false, ImmutableList.of());
      setLocation(list, start, token.right);
      nextToken();
      return list;
    }
    Expression expression = parseNonTupleExpression();
    Preconditions.checkNotNull(expression,
        "null element in list in AST at %s:%s", token.left, token.right);
    switch (token.kind) {
      case RBRACKET: // singleton list
        {
          ListExpression list =
              new ListExpression(/*isTuple=*/ false, ImmutableList.of(expression));
          setLocation(list, start, token.right);
          nextToken();
          return list;
        }
      case FOR:
        { // list comprehension
          return parseComprehensionSuffix(expression, TokenKind.RBRACKET, start);
        }
      case COMMA:
        {
          List<Expression> elems = parseExprList(true);
          Preconditions.checkState(
              !elems.contains(null),
              "null element in list in AST at %s:%s",
              token.left,
              token.right);
          elems.add(0, expression); // TODO(adonovan): opt: don't do this
          if (token.kind == TokenKind.RBRACKET) {
            ListExpression list = new ListExpression(/*isTuple=*/ false, elems);
            setLocation(list, start, token.right);
            nextToken();
            return list;
          }
          expect(TokenKind.RBRACKET);
          int end = syncPast(LIST_TERMINATOR_SET);
          return makeErrorExpression(start, end);
        }
      default:
        {
          syntaxError("expected ',', 'for' or ']'");
          int end = syncPast(LIST_TERMINATOR_SET);
          return makeErrorExpression(start, end);
        }
    }
  }

  // dict_expression ::= '{' '}'
  //                    |'{' dict_entry_list '}'
  //                    |'{' dict_entry comprehension_suffix '}'
  private Expression parseDictExpression() {
    int start = token.left;
    expect(TokenKind.LBRACE);
    if (token.kind == TokenKind.RBRACE) { // empty Dict
      DictExpression literal = new DictExpression(ImmutableList.of());
      setLocation(literal, start, token.right);
      nextToken();
      return literal;
    }
    DictExpression.Entry entry = parseDictEntry();
    if (token.kind == TokenKind.FOR) {
      // Dict comprehension
      return parseComprehensionSuffix(entry, TokenKind.RBRACE, start);
    }
    List<DictExpression.Entry> entries = new ArrayList<>();
    entries.add(entry);
    if (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      entries.addAll(parseDictEntryList());
    }
    if (token.kind == TokenKind.RBRACE) {
      DictExpression dict = new DictExpression(entries);
      setLocation(dict, start, token.right);
      nextToken();
      return dict;
    }
    expect(TokenKind.RBRACE);
    int end = syncPast(DICT_TERMINATOR_SET);
    return makeErrorExpression(start, end);
  }

  private Identifier parseIdent() {
    if (token.kind != TokenKind.IDENTIFIER) {
      expect(TokenKind.IDENTIFIER);
      return makeErrorExpression(token.left, token.right);
    }
    Identifier ident = Identifier.of(((String) token.value));
    setLocation(ident, token.left, token.right);
    nextToken();
    return ident;
  }

  // binop_expression ::= binop_expression OP binop_expression
  //                    | parsePrimaryWithSuffix
  // This function takes care of precedence between operators (see operatorPrecedence for
  // the order), and it assumes left-to-right associativity.
  private Expression parseBinOpExpression(int prec) {
    int start = token.left;
    Expression expr = parseNonTupleExpression(prec + 1);
    // The loop is not strictly needed, but it prevents risks of stack overflow. Depth is
    // limited to number of different precedence levels (operatorPrecedence.size()).
    TokenKind lastOp = null;
    for (;;) {
      if (token.kind == TokenKind.NOT) {
        // If NOT appears when we expect a binary operator, it must be followed by IN.
        // Since the code expects every operator to be a single token, we push a NOT_IN token.
        expect(TokenKind.NOT);
        if (token.kind != TokenKind.IN) {
          syntaxError("expected 'in'");
        }
        token.kind = TokenKind.NOT_IN;
      }

      TokenKind op = token.kind;
      if (!operatorPrecedence.get(prec).contains(op)) {
        return expr;
      }

      // Operator '==' and other operators of the same precedence (e.g. '<', 'in')
      // are not associative.
      if (lastOp != null && operatorPrecedence.get(prec).contains(TokenKind.EQUALS_EQUALS)) {
        reportError(
            lexer.createLocation(token.left, token.right),
            String.format(
                "Operator '%s' is not associative with operator '%s'. Use parens.", lastOp, op));
      }

      nextToken();
      Expression secondary = parseNonTupleExpression(prec + 1);
      expr = optimizeBinOpExpression(expr, op, secondary);
      setLocation(expr, start, secondary);
      lastOp = op;
    }
  }

  // Optimize binary expressions.
  // string literal + string literal can be concatenated into one string literal
  // so we don't have to do the expensive string concatenation at runtime.
  private Expression optimizeBinOpExpression(Expression x, TokenKind op, Expression y) {
    if (op == TokenKind.PLUS) {
      if (x instanceof StringLiteral && y instanceof StringLiteral) {
        StringLiteral left = (StringLiteral) x;
        StringLiteral right = (StringLiteral) y;
        return new StringLiteral(intern(left.getValue() + right.getValue()));
      }
    }
    return new BinaryOperatorExpression(x, op, y);
  }

  // Equivalent to 'test' rule in Python grammar.
  private Expression parseNonTupleExpression() {
    int start = token.left;
    Expression expr = parseNonTupleExpression(0);
    if (token.kind == TokenKind.IF) {
      nextToken();
      Expression condition = parseNonTupleExpression(0);
      if (token.kind == TokenKind.ELSE) {
        nextToken();
        Expression elseClause = parseNonTupleExpression();
        return setLocation(new ConditionalExpression(expr, condition, elseClause),
            start, elseClause);
      } else {
        reportError(lexer.createLocation(start, token.left),
            "missing else clause in conditional expression or semicolon before if");
        return expr; // Try to recover from error: drop the if and the expression after it. Ouch.
      }
    }
    return expr;
  }

  private Expression parseNonTupleExpression(int prec) {
    if (prec >= operatorPrecedence.size()) {
      return parsePrimaryWithSuffix();
    }
    if (token.kind == TokenKind.NOT && operatorPrecedence.get(prec).contains(TokenKind.NOT)) {
      return parseNotExpression(prec);
    }
    return parseBinOpExpression(prec);
  }

  // not_expr :== 'not' expr
  private Expression parseNotExpression(int prec) {
    int start = token.left;
    expect(TokenKind.NOT);
    Expression expression = parseNonTupleExpression(prec);
    UnaryOperatorExpression not = new UnaryOperatorExpression(TokenKind.NOT, expression);
    return setLocation(not, start, expression);
  }

  // file_input ::= ('\n' | stmt)* EOF
  private List<Statement> parseFileInput() {
    List<Statement> list =  new ArrayList<>();
    while (token.kind != TokenKind.EOF) {
      if (token.kind == TokenKind.NEWLINE) {
        expectAndRecover(TokenKind.NEWLINE);
      } else if (recoveryMode) {
        // If there was a parse error, we want to recover here
        // before starting a new top-level statement.
        syncTo(STATEMENT_TERMINATOR_SET);
        recoveryMode = false;
      } else {
        parseStatement(list);
      }
    }
    return list;
  }

  // load '(' STRING (COMMA [IDENTIFIER EQUALS] STRING)+ COMMA? ')'
  private Statement parseLoadStatement() {
    int start = token.left;
    expect(TokenKind.LOAD);
    expect(TokenKind.LPAREN);
    if (token.kind != TokenKind.STRING) {
      StringLiteral module = setLocation(new StringLiteral(""), token.left, token.right);
      expect(TokenKind.STRING);
      return setLocation(new LoadStatement(module, ImmutableList.of()), start, token.right);
    }

    StringLiteral importString = parseStringLiteral();
    if (token.kind == TokenKind.RPAREN) {
      syntaxError("expected at least one symbol to load");
      return setLocation(new LoadStatement(importString, ImmutableList.of()), start, token.right);
    }
    expect(TokenKind.COMMA);

    ImmutableList.Builder<LoadStatement.Binding> bindings = ImmutableList.builder();
    // At least one symbol is required.
    parseLoadSymbol(bindings);
    while (token.kind != TokenKind.RPAREN && token.kind != TokenKind.EOF) {
      // A trailing comma is permitted after the last symbol.
      expect(TokenKind.COMMA);
      if (token.kind == TokenKind.RPAREN) {
        break;
      }
      parseLoadSymbol(bindings);
    }

    LoadStatement stmt =
        setLocation(new LoadStatement(importString, bindings.build()), start, token.right);
    expect(TokenKind.RPAREN);
    return stmt;
  }

  /**
   * Parses the next symbol argument of a load statement and puts it into the output map.
   *
   * <p>The symbol is either "name" (STRING) or name = "declared" (IDENTIFIER EQUALS STRING). If no
   * alias is used, "name" and "declared" will be identical. "Declared" refers to the original name
   * in the Bazel file that should be loaded, while "name" will be the key of the entry in the map.
   */
  private void parseLoadSymbol(ImmutableList.Builder<LoadStatement.Binding> symbols) {
    if (token.kind != TokenKind.STRING && token.kind != TokenKind.IDENTIFIER) {
      syntaxError("expected either a literal string or an identifier");
      return;
    }

    String name = (String) token.value;
    Identifier local = setLocation(Identifier.of(name), token.left, token.right);

    Identifier original;
    if (token.kind == TokenKind.STRING) {
      // load(..., "name")
      original = local;
    } else {
      // load(..., local = "orig")
      expect(TokenKind.IDENTIFIER);
      expect(TokenKind.EQUALS);
      if (token.kind != TokenKind.STRING) {
        syntaxError("expected string");
        return;
      }
      original = setLocation(Identifier.of((String) token.value), token.left, token.right);
    }
    nextToken();
    symbols.add(new LoadStatement.Binding(local, original));
  }

  // simple_stmt ::= small_stmt (';' small_stmt)* ';'? NEWLINE
  private void parseSimpleStatement(List<Statement> list) {
    list.add(parseSmallStatement());

    while (token.kind == TokenKind.SEMI) {
      nextToken();
      if (token.kind == TokenKind.NEWLINE) {
        break;
      }
      list.add(parseSmallStatement());
    }
    expectAndRecover(TokenKind.NEWLINE);
  }

  //     small_stmt ::= assign_stmt
  //                  | expr
  //                  | load_stmt
  //                  | return_stmt
  //                  | BREAK | CONTINUE | PASS
  //     assign_stmt ::= expr ('=' | augassign) expr
  //     augassign ::= '+=' | '-=' | '*=' | '/=' | '%=' | '//=' | '&=' | '|=' | '^=' |'<<=' | '>>='
  private Statement parseSmallStatement() {
    int start = token.left;
    if (token.kind == TokenKind.RETURN) {
      return parseReturnStatement();
    } else if (token.kind == TokenKind.BREAK
        || token.kind == TokenKind.CONTINUE
        || token.kind == TokenKind.PASS) {
      TokenKind kind = token.kind;
      int end = token.right;
      expect(kind);
      return setLocation(new FlowStatement(kind), start, end);
    } else if (token.kind == TokenKind.LOAD) {
      return parseLoadStatement();
    }

    Expression lhs = parseExpression();

    // lhs = rhs  or  lhs += rhs
    TokenKind op = augmentedAssignments.get(token.kind);
    if (token.kind == TokenKind.EQUALS || op != null) {
      nextToken();
      Expression rhs = parseExpression();
      // op == null for ordinary assignment.
      return setLocation(new AssignmentStatement(lhs, op, rhs), start, rhs);
    } else {
      return setLocation(new ExpressionStatement(lhs), start, lhs);
    }
  }

  // if_stmt ::= IF expr ':' suite [ELIF expr ':' suite]* [ELSE ':' suite]?
  private IfStatement parseIfStatement() {
    List<Integer> startOffsets = new ArrayList<>();
    startOffsets.add(token.left);
    expect(TokenKind.IF);
    Expression cond = parseNonTupleExpression();
    expect(TokenKind.COLON);
    List<Statement> body = parseSuite();
    IfStatement ifStmt = new IfStatement(TokenKind.IF, cond, body);
    IfStatement tail = ifStmt;
    while (token.kind == TokenKind.ELIF) {
      startOffsets.add(token.left);
      expect(TokenKind.ELIF);
      cond = parseNonTupleExpression();
      expect(TokenKind.COLON);
      body = parseSuite();
      IfStatement elif = new IfStatement(TokenKind.ELIF, cond, body);
      tail.setElseBlock(ImmutableList.of(elif));
      tail = elif;
    }
    if (token.kind == TokenKind.ELSE) {
      expect(TokenKind.ELSE);
      expect(TokenKind.COLON);
      body = parseSuite();
      tail.setElseBlock(body);
    }

    // Because locations are allocated and stored redundantly rather
    // than computed on demand from token offsets in the tree, we must
    // wait till the end of the chain, after all setElseBlock calls,
    // before setting the end location of each IfStatement.
    // Body may be empty after a parse error.
    int end = (body.isEmpty() ? tail.getCondition() : Iterables.getLast(body)).getEndOffset();
    IfStatement s = ifStmt;
    setLocation(s, startOffsets.get(0), end);
    for (int i = 1; i < startOffsets.size(); i++) {
      s = (IfStatement) s.elseBlock.get(0);
      setLocation(s, startOffsets.get(i), end);
    }

    return ifStmt;
  }

  // for_stmt ::= FOR IDENTIFIER IN expr ':' suite
  private ForStatement parseForStatement() {
    int start = token.left;
    expect(TokenKind.FOR);
    Expression lhs = parseForLoopVariables();
    expect(TokenKind.IN);
    Expression collection = parseExpression();
    expect(TokenKind.COLON);
    List<Statement> block = parseSuite();
    ForStatement stmt = new ForStatement(lhs, collection, block);
    int end = block.isEmpty() ? token.left : Iterables.getLast(block).getEndOffset();
    return setLocation(stmt, start, end);
  }

  // def_stmt ::= DEF IDENTIFIER '(' arguments ')' ':' suite
  private DefStatement parseDefStatement() {
    int start = token.left;
    expect(TokenKind.DEF);
    Identifier ident = parseIdent();
    expect(TokenKind.LPAREN);
    ImmutableList<Parameter> params = parseParameters();

    FunctionSignature signature;
    try {
      signature = FunctionSignature.fromParameters(params);
    } catch (FunctionSignature.SignatureException e) {
      reportError(e.getParameter().getStartLocation(), e.getMessage());
      // bogus empty signature
      signature = FunctionSignature.of();
    }

    expect(TokenKind.RPAREN);
    expect(TokenKind.COLON);
    ImmutableList<Statement> block = ImmutableList.copyOf(parseSuite());
    DefStatement stmt = new DefStatement(ident, params, signature, block);
    int end = block.isEmpty() ? token.left : Iterables.getLast(block).getEndOffset();
    return setLocation(stmt, start, end);
  }

  // Parse a list of function parameters.
  //
  // This parser does minimal validation: it ensures the proper python use of the comma (that can
  // terminate before a star but not after) and the fact that **kwargs must appear last. It does
  // not validate further ordering constraints. This validation happens in the validator pass.
  private ImmutableList<Parameter> parseParameters() {
    boolean hasParam = false;
    boolean hasStarStar = false;
    ImmutableList.Builder<Parameter> list = ImmutableList.builder();

    while (token.kind != TokenKind.RPAREN && token.kind != TokenKind.EOF) {
      if (hasParam) {
        expect(TokenKind.COMMA);
        // The list may end with a comma.
        if (token.kind == TokenKind.RPAREN) {
          break;
        }
      }
      if (hasStarStar) {
        // TODO(adonovan): move this to validation pass too.
        reportError(lexer.createLocation(token.left, token.right), "unexpected tokens after kwarg");
        break;
      }

      Parameter param = parseFunctionParameter();
      hasParam = true;
      if (param instanceof Parameter.StarStar) { // TODO(adonovan): not Star too? verify.
        hasStarStar = true;
      }
      list.add(param);
    }
    return list.build();
  }

  // suite is typically what follows a colon (e.g. after def or for).
  // suite ::= simple_stmt
  //         | NEWLINE INDENT stmt+ OUTDENT
  private List<Statement> parseSuite() {
    List<Statement> list = new ArrayList<>();
    if (token.kind == TokenKind.NEWLINE) {
      expect(TokenKind.NEWLINE);
      if (token.kind != TokenKind.INDENT) {
        reportError(lexer.createLocation(token.left, token.right),
                    "expected an indented block");
        return list;
      }
      expect(TokenKind.INDENT);
      while (token.kind != TokenKind.OUTDENT && token.kind != TokenKind.EOF) {
        parseStatement(list);
      }
      expectAndRecover(TokenKind.OUTDENT);
    } else {
      parseSimpleStatement(list);
    }
    return list;
  }

  // return_stmt ::= RETURN [expr]
  private ReturnStatement parseReturnStatement() {
    int start = token.left;
    int end = token.right;
    expect(TokenKind.RETURN);

    Expression expression = null;
    if (!STATEMENT_TERMINATOR_SET.contains(token.kind)) {
      expression = parseExpression();
      end = expression.getEndOffset();
    }
    return setLocation(new ReturnStatement(expression), start, end);
  }

}
