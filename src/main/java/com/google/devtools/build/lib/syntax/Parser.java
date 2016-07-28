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

import static com.google.devtools.build.lib.syntax.Parser.ParsingMode.BUILD;
import static com.google.devtools.build.lib.syntax.Parser.ParsingMode.PYTHON;
import static com.google.devtools.build.lib.syntax.Parser.ParsingMode.SKYLARK;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.DictionaryLiteral.DictionaryEntryLiteral;
import com.google.devtools.build.lib.syntax.IfStatement.ConditionalStatements;
import com.google.devtools.build.lib.syntax.SkylarkImports.SkylarkImportSyntaxException;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Recursive descent parser for LL(2) BUILD language.
 * Loosely based on Python 2 grammar.
 * See https://docs.python.org/2/reference/grammar.html
 */
@VisibleForTesting
public class Parser {

  /**
   * Combines the parser result into a single value object.
   */
  public static final class ParseResult {
    /** The statements (rules, basically) from the parsed file. */
    public final List<Statement> statements;

    /** The comments from the parsed file. */
    public final List<Comment> comments;

    /** Represents every statement in the file. */
    public final Location location;

    /** Whether the file contained any errors. */
    public final boolean containsErrors;

    public ParseResult(List<Statement> statements, List<Comment> comments, Location location,
        boolean containsErrors) {
      // No need to copy here; when the object is created, the parser instance is just about to go
      // out of scope and be garbage collected.
      this.statements = Preconditions.checkNotNull(statements);
      this.comments = Preconditions.checkNotNull(comments);
      this.location = location;
      this.containsErrors = containsErrors;
    }
  }

  /**
   * ParsingMode is used to select which features the parser should accept.
   */
  public enum ParsingMode {
    /** Used for parsing BUILD files */
    BUILD,
    /** Used for parsing .bzl files */
    SKYLARK,
    /** Used for syntax checking, ignoring all Python blocks (e.g. def, class, try) */
    PYTHON,
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

  private static final EnumSet<TokenKind> BLOCK_STARTING_SET =
      EnumSet.of(
          TokenKind.CLASS,
          TokenKind.DEF,
          TokenKind.ELSE,
          TokenKind.FOR,
          TokenKind.IF,
          TokenKind.TRY);

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

  /**
   * Keywords that are forbidden in both Skylark and BUILD parsing modes.
   *
   * <p>(Mapping: token -> human-readable string description)
   */
  private static final ImmutableMap<TokenKind, String> ILLEGAL_BLOCK_KEYWORDS =
      ImmutableMap.of(TokenKind.CLASS, "Class definition", TokenKind.TRY, "Try statement");

  private Token token; // current lookahead token
  private Token pushedToken = null; // used to implement LL(2)

  private static final boolean DEBUGGING = false;

  private final Lexer lexer;
  private final EventHandler eventHandler;
  private final List<Comment> comments;
  private final ParsingMode parsingMode;

  private static final Map<TokenKind, Operator> binaryOperators =
      new ImmutableMap.Builder<TokenKind, Operator>()
          .put(TokenKind.AND, Operator.AND)
          .put(TokenKind.EQUALS_EQUALS, Operator.EQUALS_EQUALS)
          .put(TokenKind.GREATER, Operator.GREATER)
          .put(TokenKind.GREATER_EQUALS, Operator.GREATER_EQUALS)
          .put(TokenKind.IN, Operator.IN)
          .put(TokenKind.LESS, Operator.LESS)
          .put(TokenKind.LESS_EQUALS, Operator.LESS_EQUALS)
          .put(TokenKind.MINUS, Operator.MINUS)
          .put(TokenKind.NOT_EQUALS, Operator.NOT_EQUALS)
          .put(TokenKind.NOT_IN, Operator.NOT_IN)
          .put(TokenKind.OR, Operator.OR)
          .put(TokenKind.PERCENT, Operator.PERCENT)
          .put(TokenKind.SLASH, Operator.DIVIDE)
          .put(TokenKind.PLUS, Operator.PLUS)
          .put(TokenKind.PIPE, Operator.PIPE)
          .put(TokenKind.STAR, Operator.MULT)
          .build();

  // TODO(bazel-team): add support for |=
  private static final Map<TokenKind, Operator> augmentedAssignmentMethods =
      new ImmutableMap.Builder<TokenKind, Operator>()
          .put(TokenKind.PLUS_EQUALS, Operator.PLUS)
          .put(TokenKind.MINUS_EQUALS, Operator.MINUS)
          .put(TokenKind.STAR_EQUALS, Operator.MULT)
          .put(TokenKind.SLASH_EQUALS, Operator.DIVIDE)
          .put(TokenKind.PERCENT_EQUALS, Operator.PERCENT)
          .build();

  /** Highest precedence goes last.
   *  Based on: http://docs.python.org/2/reference/expressions.html#operator-precedence
   **/
  private static final List<EnumSet<Operator>> operatorPrecedence = ImmutableList.of(
      EnumSet.of(Operator.OR),
      EnumSet.of(Operator.AND),
      EnumSet.of(Operator.NOT),
      EnumSet.of(Operator.EQUALS_EQUALS, Operator.NOT_EQUALS, Operator.LESS, Operator.LESS_EQUALS,
          Operator.GREATER, Operator.GREATER_EQUALS, Operator.IN, Operator.NOT_IN),
      EnumSet.of(Operator.PIPE),
      EnumSet.of(Operator.MINUS, Operator.PLUS),
      EnumSet.of(Operator.DIVIDE, Operator.MULT, Operator.PERCENT));

  private Iterator<Token> tokens = null;
  private int errorsCount;
  private boolean recoveryMode;  // stop reporting errors until next statement

  private Parser(Lexer lexer, EventHandler eventHandler, ParsingMode parsingMode) {
    this.lexer = lexer;
    this.eventHandler = eventHandler;
    this.parsingMode = parsingMode;
    this.tokens = lexer.getTokens().iterator();
    this.comments = new ArrayList<>();
    nextToken();
  }

  private static Location locationFromStatements(Lexer lexer, List<Statement> statements) {
    if (!statements.isEmpty()) {
      return lexer.createLocation(
          statements.get(0).getLocation().getStartOffset(),
          statements.get(statements.size() - 1).getLocation().getEndOffset());
    } else {
      return Location.fromPathFragment(lexer.getFilename());
    }
  }

  /**
   * Entry-point to parser that parses a build file with comments.  All errors
   * encountered during parsing are reported via "reporter".
   */
  public static ParseResult parseFile(
      ParserInputSource input, EventHandler eventHandler, boolean parsePython) {
    Lexer lexer = new Lexer(input, eventHandler, parsePython);
    ParsingMode parsingMode = parsePython ? PYTHON : BUILD;
    Parser parser = new Parser(lexer, eventHandler, parsingMode);
    List<Statement> statements = parser.parseFileInput();
    return new ParseResult(statements, parser.comments, locationFromStatements(lexer, statements),
        parser.errorsCount > 0 || lexer.containsErrors());
  }

  /**
   * Entry-point to parser that parses a build file with comments.  All errors
   * encountered during parsing are reported via "reporter".  Enable Skylark extensions
   * that are not part of the core BUILD language.
   */
  public static ParseResult parseFileForSkylark(
      ParserInputSource input,
      EventHandler eventHandler,
      @Nullable ValidationEnvironment validationEnvironment) {
    Lexer lexer = new Lexer(input, eventHandler, false);
    Parser parser = new Parser(lexer, eventHandler, SKYLARK);
    List<Statement> statements = parser.parseFileInput();
    boolean hasSemanticalErrors = false;
    try {
      if (validationEnvironment != null) {
        validationEnvironment.validateAst(statements);
      }
    } catch (EvalException e) {
      // Do not report errors caused by a previous parsing error, as it has already been reported.
      if (!e.isDueToIncompleteAST()) {
        eventHandler.handle(Event.error(e.getLocation(), e.getMessage()));
      }
      hasSemanticalErrors = true;
    }
    return new ParseResult(statements, parser.comments, locationFromStatements(lexer, statements),
        parser.errorsCount > 0 || lexer.containsErrors() || hasSemanticalErrors);
  }

  /**
   * Entry-point to parser that parses an expression.  All errors encountered
   * during parsing are reported via "reporter".  The expression may be followed
   * by newline tokens.
   */
  @VisibleForTesting
  public static Expression parseExpression(ParserInputSource input, EventHandler eventHandler) {
    Lexer lexer = new Lexer(input, eventHandler, false);
    Parser parser = new Parser(lexer, eventHandler, null);
    Expression result = parser.parseExpression();
    while (parser.token.kind == TokenKind.NEWLINE) {
      parser.nextToken();
    }
    parser.expect(TokenKind.EOF);
    return result;
  }

  private void reportError(Location location, String message) {
    errorsCount++;
    // Limit the number of reported errors to avoid spamming output.
    if (errorsCount <= 5) {
      eventHandler.handle(Event.error(location, message));
    }
  }

  private void syntaxError(Token token, String message) {
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
      syntaxError(token, "expected " + kind.getPrettyName());
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
   * teminatingTokens.
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
   * the set of teminatingTokens.
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
      EnumSet.of(TokenKind.AS, TokenKind.ASSERT,
          TokenKind.DEL, TokenKind.EXCEPT, TokenKind.FINALLY, TokenKind.FROM, TokenKind.GLOBAL,
          TokenKind.IMPORT, TokenKind.IS, TokenKind.LAMBDA, TokenKind.NONLOCAL, TokenKind.RAISE,
          TokenKind.TRY, TokenKind.WITH, TokenKind.WHILE, TokenKind.YIELD);

  private void checkForbiddenKeywords(Token token) {
    if (parsingMode == PYTHON || !FORBIDDEN_KEYWORDS.contains(token.kind)) {
      return;
    }
    String error;
    switch (token.kind) {
      case ASSERT: error = "'assert' not supported, use 'fail' instead"; break;
      case TRY: error = "'try' not supported, all exceptions are fatal"; break;
      case IMPORT: error = "'import' not supported, use 'load' instead"; break;
      case IS: error = "'is' not supported, use '==' instead"; break;
      case LAMBDA: error = "'lambda' not supported, declare a function instead"; break;
      case RAISE: error = "'raise' not supported, use 'fail' instead"; break;
      case WHILE: error = "'while' not supported, use 'for' instead"; break;
      default: error = "keyword '" + token.kind.getPrettyName() + "' not supported"; break;
    }
    reportError(lexer.createLocation(token.left, token.right), error);
  }

  private void nextToken() {
    if (pushedToken != null) {
      token = pushedToken;
      pushedToken = null;
    } else {
      if (token == null || token.kind != TokenKind.EOF) {
        token = tokens.next();
        // transparently handle comment tokens
        while (token.kind == TokenKind.COMMENT) {
          makeComment(token);
          token = tokens.next();
        }
      }
    }
    checkForbiddenKeywords(token);
    if (DEBUGGING) {
      System.err.print(token);
    }
  }

  private void pushToken(Token tokenToPush) {
    if (pushedToken != null) {
      throw new IllegalStateException("Exceeded LL(2) lookahead!");
    }
    pushedToken = token;
    token = tokenToPush;
  }

  // create an error expression
  private Identifier makeErrorExpression(int start, int end) {
    return setLocation(new Identifier("$error$"), start, end);
  }

  // Convenience wrapper around ASTNode.setLocation that returns the node.
  private <NODE extends ASTNode> NODE setLocation(NODE node, Location location) {
    return ASTNode.<NODE>setLocation(location, node);
  }

  // Another convenience wrapper method around ASTNode.setLocation
  private <NODE extends ASTNode> NODE setLocation(NODE node, int startOffset, int endOffset) {
    return setLocation(node, lexer.createLocation(startOffset, endOffset));
  }

  // Convenience method that uses end offset from the last node.
  private <NODE extends ASTNode> NODE setLocation(NODE node, int startOffset, ASTNode lastNode) {
    Preconditions.checkNotNull(lastNode, "can't extract end offset from a null node");
    Preconditions.checkNotNull(lastNode.getLocation(), "lastNode doesn't have a location");
    return setLocation(node, startOffset, lastNode.getLocation().getEndOffset());
  }

  // create a funcall expression
  private Expression makeFuncallExpression(Expression receiver, Identifier function,
                                           List<Argument.Passed> args,
                                           int start, int end) {
    if (function.getLocation() == null) {
      function = setLocation(function, start, end);
    }
    return setLocation(new FuncallExpression(receiver, function, args), start, end);
  }

  // arg ::= IDENTIFIER '=' nontupleexpr
  //       | expr
  //       | *args       (only in Skylark mode)
  //       | **kwargs    (only in Skylark mode)
  // To keep BUILD files declarative and easy to process, *args and **kwargs
  // arguments are allowed only in Skylark mode.
  private Argument.Passed parseFuncallArgument() {
    final int start = token.left;
    // parse **expr
    if (token.kind == TokenKind.STAR_STAR) {
      if (parsingMode != SKYLARK) {
        reportError(
            lexer.createLocation(token.left, token.right),
            "**kwargs arguments are not allowed in BUILD files");
      }
      nextToken();
      Expression expr = parseNonTupleExpression();
      return setLocation(new Argument.StarStar(expr), start, expr);
    }
    // parse *expr
    if (token.kind == TokenKind.STAR) {
      if (parsingMode != SKYLARK) {
        reportError(
            lexer.createLocation(token.left, token.right),
            "*args arguments are not allowed in BUILD files");
      }
      nextToken();
      Expression expr = parseNonTupleExpression();
      return setLocation(new Argument.Star(expr), start, expr);
    }
    // parse keyword = expr
    if (token.kind == TokenKind.IDENTIFIER) {
      Token identToken = token;
      String name = (String) token.value;
      nextToken();
      if (token.kind == TokenKind.EQUALS) { // it's a named argument
        nextToken();
        Expression expr = parseNonTupleExpression();
        return setLocation(new Argument.Keyword(name, expr), start, expr);
      } else { // oops, back up!
        pushToken(identToken);
      }
    }
    // parse a positional argument
    Expression expr = parseNonTupleExpression();
    return setLocation(new Argument.Positional(expr), start, expr);
  }

  // arg ::= IDENTIFIER '=' nontupleexpr
  //       | IDENTIFIER
  private Parameter<Expression, Expression> parseFunctionParameter() {
    // TODO(bazel-team): optionally support type annotations
    int start = token.left;
    if (token.kind == TokenKind.STAR_STAR) { // kwarg
      nextToken();
      Identifier ident = parseIdent();
      return setLocation(new Parameter.StarStar<Expression, Expression>(
          ident.getName()), start, ident);
    } else if (token.kind == TokenKind.STAR) { // stararg
      int end = token.right;
      nextToken();
      if (token.kind == TokenKind.IDENTIFIER) {
        Identifier ident = parseIdent();
        return setLocation(new Parameter.Star<Expression, Expression>(ident.getName()),
            start, ident);
      } else {
        return setLocation(new Parameter.Star<Expression, Expression>(null), start, end);
      }
    } else {
      Identifier ident = parseIdent();
      if (token.kind == TokenKind.EQUALS) { // there's a default value
        nextToken();
        Expression expr = parseNonTupleExpression();
        return setLocation(new Parameter.Optional<Expression, Expression>(
            ident.getName(), expr), start, expr);
      } else {
        return setLocation(new Parameter.Mandatory<Expression, Expression>(
            ident.getName()), start, ident);
      }
    }
  }

  // funcall_suffix ::= '(' arg_list? ')'
  private Expression parseFuncallSuffix(int start, Expression receiver, Identifier function) {
    List<Argument.Passed> args = Collections.emptyList();
    expect(TokenKind.LPAREN);
    int end;
    if (token.kind == TokenKind.RPAREN) {
      end = token.right;
      nextToken(); // RPAREN
    } else {
      args = parseFuncallArguments(); // (includes optional trailing comma)
      end = token.right;
      expect(TokenKind.RPAREN);
    }
    return makeFuncallExpression(receiver, function, args, start, end);
  }

  // selector_suffix ::= '.' IDENTIFIER
  //                    |'.' IDENTIFIER funcall_suffix
  private Expression parseSelectorSuffix(int start, Expression receiver) {
    expect(TokenKind.DOT);
    if (token.kind == TokenKind.IDENTIFIER) {
      Identifier ident = parseIdent();
      if (token.kind == TokenKind.LPAREN) {
        return parseFuncallSuffix(start, receiver, ident);
      } else {
        return setLocation(new DotExpression(receiver, ident), start, token.right);
      }
    } else {
      syntaxError(token, "expected identifier after dot");
      int end = syncTo(EXPR_TERMINATOR_SET);
      return makeErrorExpression(start, end);
    }
  }

  // arg_list ::= ( (arg ',')* arg ','? )?
  private List<Argument.Passed> parseFuncallArguments() {
    List<Argument.Passed> arguments =
        parseFunctionArguments(new Supplier<Argument.Passed>() {
              @Override public Argument.Passed get() {
                return parseFuncallArgument();
              }
            });
    try {
      Argument.validateFuncallArguments(arguments);
    } catch (Argument.ArgumentException e) {
      reportError(lexer.createLocation(token.left, token.right), e.getMessage());
    }
    return arguments;
  }

  // expr_list parses a comma-separated list of expression. It assumes that the
  // first expression was already parsed, so it starts with a comma.
  // It is used to parse tuples and list elements.
  // expr_list ::= ( ',' expr )* ','?
  private List<Expression> parseExprList() {
    List<Expression> list = new ArrayList<>();
    //  terminating tokens for an expression list
    while (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      if (EXPR_LIST_TERMINATOR_SET.contains(token.kind)) {
        break;
      }
      list.add(parseNonTupleExpression());
    }
    return list;
  }

  // dict_entry_list ::= ( (dict_entry ',')* dict_entry ','? )?
  private List<DictionaryEntryLiteral> parseDictEntryList() {
    List<DictionaryEntryLiteral> list = new ArrayList<>();
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
  private DictionaryEntryLiteral parseDictEntry() {
    int start = token.left;
    Expression key = parseNonTupleExpression();
    expect(TokenKind.COLON);
    Expression value = parseNonTupleExpression();
    return setLocation(new DictionaryEntryLiteral(key, value), start, value);
  }

  /**
   * Parse a String literal value, e.g. "str".
   */
  private StringLiteral parseStringLiteral() {
    Preconditions.checkState(token.kind == TokenKind.STRING);
    int end = token.right;
    char quoteChar = lexer.charAt(token.left);
    StringLiteral literal =
        setLocation(new StringLiteral((String) token.value, quoteChar), token.left, end);

    nextToken();
    if (token.kind == TokenKind.STRING) {
      reportError(lexer.createLocation(end, token.left),
          "Implicit string concatenation is forbidden, use the + operator");
    }
    return literal;
  }

  //  primary ::= INTEGER
  //            | STRING
  //            | STRING '.' IDENTIFIER funcall_suffix
  //            | IDENTIFIER
  //            | IDENTIFIER funcall_suffix
  //            | IDENTIFIER '.' selector_suffix
  //            | list_expression
  //            | '(' ')'                    // a tuple with zero elements
  //            | '(' expr ')'               // a parenthesized expression
  //            | dict_expression
  //            | '-' primary_with_suffix
  private Expression parsePrimary() {
    int start = token.left;
    switch (token.kind) {
      case INT: {
        IntegerLiteral literal = new IntegerLiteral((Integer) token.value);
        setLocation(literal, start, token.right);
        nextToken();
        return literal;
      }
      case STRING: {
        return parseStringLiteral();
      }
      case IDENTIFIER: {
        Identifier ident = parseIdent();
        if (token.kind == TokenKind.LPAREN) { // it's a function application
          return parseFuncallSuffix(start, null, ident);
        } else {
          return ident;
        }
      }
      case LBRACKET: { // it's a list
        return parseListMaker();
      }
      case LBRACE: { // it's a dictionary
        return parseDictExpression();
      }
      case LPAREN: {
        nextToken();
        // check for the empty tuple literal
        if (token.kind == TokenKind.RPAREN) {
          ListLiteral literal =
              ListLiteral.makeTuple(Collections.<Expression>emptyList());
          setLocation(literal, start, token.right);
          nextToken();
          return literal;
        }
        // parse the first expression
        Expression expression = parseExpression();
        setLocation(expression, start, token.right);
        if (token.kind == TokenKind.RPAREN) {
          nextToken();
          return expression;
        }
        expect(TokenKind.RPAREN);
        int end = syncTo(EXPR_TERMINATOR_SET);
        return makeErrorExpression(start, end);
      }
      case MINUS: {
        nextToken();

        List<Argument.Passed> args = new ArrayList<>();
        Expression expr = parsePrimaryWithSuffix();
        args.add(setLocation(new Argument.Positional(expr), start, expr));
        return makeFuncallExpression(null, new Identifier("-"), args,
                                     start, token.right);
      }
      default: {
        syntaxError(token, "expected expression");
        int end = syncTo(EXPR_TERMINATOR_SET);
        return makeErrorExpression(start, end);
      }
    }
  }

  // primary_with_suffix ::= primary selector_suffix*
  //                       | primary substring_suffix
  private Expression parsePrimaryWithSuffix() {
    int start = token.left;
    Expression receiver = parsePrimary();
    while (true) {
      if (token.kind == TokenKind.DOT) {
        receiver = parseSelectorSuffix(start, receiver);
      } else if (token.kind == TokenKind.LBRACKET) {
        receiver = parseSubstringSuffix(start, receiver);
      } else {
        break;
      }
    }
    return receiver;
  }

  // substring_suffix ::= '[' expression? ':' expression?  ':' expression? ']'
  private Expression parseSubstringSuffix(int start, Expression receiver) {
    List<Argument.Passed> args = new ArrayList<>();
    Expression startExpr;

    expect(TokenKind.LBRACKET);
    int loc1 = token.left;
    if (token.kind == TokenKind.COLON) {
      startExpr = setLocation(new Identifier("None"), token.left, token.right);
    } else {
      startExpr = parseExpression();
    }
    args.add(setLocation(new Argument.Positional(startExpr), loc1, startExpr));
    // This is a dictionary access
    if (token.kind == TokenKind.RBRACKET) {
      expect(TokenKind.RBRACKET);
      return makeFuncallExpression(receiver, new Identifier("$index"), args,
                                   start, token.right);
    }
    // This is a slice (or substring)
    args.add(parseSliceArgument(new Identifier("None")));
    args.add(parseSliceArgument(new IntegerLiteral(1)));
    expect(TokenKind.RBRACKET);
    return makeFuncallExpression(receiver, new Identifier("$slice"), args,
                                 start, token.right);
  }

  /**
   * Parses {@code [':' [expr]]} which can either be the end or the step argument of a slice
   * operation. If no such expression is found, this method returns an argument that represents
   * {@code defaultValue}.
   */
  private Argument.Positional parseSliceArgument(Expression defaultValue) {
    Expression explicitArg = getSliceEndOrStepExpression();
    Expression argValue =
        (explicitArg == null) ? setLocation(defaultValue, token.left, token.right) : explicitArg;
    return setLocation(new Argument.Positional(argValue), token.left, argValue);
  }

  private Expression getSliceEndOrStepExpression() {
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
    return setLocation(ListLiteral.makeTuple(tuple), start, token.right);
  }

  // comprehension_suffix ::= 'FOR' loop_variables 'IN' expr comprehension_suffix
  //                        | 'IF' expr comprehension_suffix
  //                        | ']'
  private Expression parseComprehensionSuffix(
      AbstractComprehension comprehension, TokenKind closingBracket) {
    while (true) {
      if (token.kind == TokenKind.FOR) {
        nextToken();
        Expression loopVar = parseForLoopVariables();
        expect(TokenKind.IN);
        // The expression cannot be a ternary expression ('x if y else z') due to
        // conflicts in Python grammar ('if' is used by the comprehension).
        Expression listExpression = parseNonTupleExpression(0);
        comprehension.addFor(loopVar, listExpression);
      } else if (token.kind == TokenKind.IF) {
        nextToken();
        comprehension.addIf(parseExpression());
      } else if (token.kind == closingBracket) {
        nextToken();
        return comprehension;
      } else {
        syntaxError(token, "expected '" + closingBracket.getPrettyName() + "', 'for' or 'if'");
        syncPast(LIST_TERMINATOR_SET);
        return makeErrorExpression(token.left, token.right);
      }
    }
  }

  // list_maker ::= '[' ']'
  //               |'[' expr ']'
  //               |'[' expr expr_list ']'
  //               |'[' expr ('FOR' loop_variables 'IN' expr)+ ']'
  private Expression parseListMaker() {
    int start = token.left;
    expect(TokenKind.LBRACKET);
    if (token.kind == TokenKind.RBRACKET) { // empty List
      ListLiteral literal = ListLiteral.emptyList();
      setLocation(literal, start, token.right);
      nextToken();
      return literal;
    }
    Expression expression = parseNonTupleExpression();
    Preconditions.checkNotNull(expression,
        "null element in list in AST at %s:%s", token.left, token.right);
    switch (token.kind) {
      case RBRACKET: { // singleton List
        ListLiteral literal = ListLiteral.makeList(Collections.singletonList(expression));
        setLocation(literal, start, token.right);
        nextToken();
        return literal;
      }
      case FOR:
        { // list comprehension
          Expression result =
              parseComprehensionSuffix(new ListComprehension(expression), TokenKind.RBRACKET);
          return setLocation(result, start, token.right);
        }
      case COMMA: {
        List<Expression> list = parseExprList();
        Preconditions.checkState(!list.contains(null),
            "null element in list in AST at %s:%s", token.left, token.right);
        list.add(0, expression);
        if (token.kind == TokenKind.RBRACKET) {
          ListLiteral literal = ListLiteral.makeList(list);
          setLocation(literal, start, token.right);
          nextToken();
          return literal;
        }
        expect(TokenKind.RBRACKET);
        int end = syncPast(LIST_TERMINATOR_SET);
        return makeErrorExpression(start, end);
      }
      default: {
        syntaxError(token, "expected ',', 'for' or ']'");
        int end = syncPast(LIST_TERMINATOR_SET);
        return makeErrorExpression(start, end);
      }
    }
  }

  // dict_expression ::= '{' '}'
  //                    |'{' dict_entry_list '}'
  //                    |'{' dict_entry 'FOR' loop_variables 'IN' expr '}'
  private Expression parseDictExpression() {
    int start = token.left;
    expect(TokenKind.LBRACE);
    if (token.kind == TokenKind.RBRACE) { // empty Dict
      DictionaryLiteral literal = DictionaryLiteral.emptyDict();
      setLocation(literal, start, token.right);
      nextToken();
      return literal;
    }
    DictionaryEntryLiteral entry = parseDictEntry();
    if (token.kind == TokenKind.FOR) {
      // Dict comprehension
      Expression result = parseComprehensionSuffix(
          new DictComprehension(entry.getKey(), entry.getValue()), TokenKind.RBRACE);
      return setLocation(result, start, token.right);
    }
    List<DictionaryEntryLiteral> entries = new ArrayList<>();
    entries.add(entry);
    if (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      entries.addAll(parseDictEntryList());
    }
    if (token.kind == TokenKind.RBRACE) {
      DictionaryLiteral literal = new DictionaryLiteral(entries);
      setLocation(literal, start, token.right);
      nextToken();
      return literal;
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
    Identifier ident = new Identifier(((String) token.value));
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
    for (;;) {

      if (token.kind == TokenKind.NOT) {
        // If NOT appears when we expect a binary operator, it must be followed by IN.
        // Since the code expects every operator to be a single token, we push a NOT_IN token.
        expect(TokenKind.NOT);
        expect(TokenKind.IN);
        pushToken(new Token(TokenKind.NOT_IN, token.left, token.right));
      }

      if (!binaryOperators.containsKey(token.kind)) {
        return expr;
      }
      Operator operator = binaryOperators.get(token.kind);
      if (!operatorPrecedence.get(prec).contains(operator)) {
        return expr;
      }
      nextToken();
      Expression secondary = parseNonTupleExpression(prec + 1);
      expr = optimizeBinOpExpression(operator, expr, secondary);
      setLocation(expr, start, secondary);
    }
  }

  // Optimize binary expressions.
  // string literal + string literal can be concatenated into one string literal
  // so we don't have to do the expensive string concatenation at runtime.
  private Expression optimizeBinOpExpression(
      Operator operator, Expression expr, Expression secondary) {
    if (operator == Operator.PLUS) {
      if (expr instanceof StringLiteral && secondary instanceof StringLiteral) {
        StringLiteral left = (StringLiteral) expr;
        StringLiteral right = (StringLiteral) secondary;
        if (left.getQuoteChar() == right.getQuoteChar()) {
          return new StringLiteral(left.getValue() + right.getValue(), left.getQuoteChar());
        }
      }
    }
    return new BinaryOperatorExpression(operator, expr, secondary);
  }

  // Equivalent to 'testlist' rule in Python grammar. It can parse every
  // kind of expression.
  // In many cases, we need to use parseNonTupleExpression to avoid ambiguity
  // e.g.  fct(x, y)  vs  fct((x, y))
  private Expression parseExpression() {
    int start = token.left;
    Expression expression = parseNonTupleExpression();
    if (token.kind != TokenKind.COMMA) {
      return expression;
    }

    // It's a tuple
    List<Expression> tuple = parseExprList();
    tuple.add(0, expression);  // add the first expression to the front of the tuple
    return setLocation(ListLiteral.makeTuple(tuple), start, token.right);
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
    if (token.kind == TokenKind.NOT && operatorPrecedence.get(prec).contains(Operator.NOT)) {
      return parseNotExpression(prec);
    }
    return parseBinOpExpression(prec);
  }

  // not_expr :== 'not' expr
  private Expression parseNotExpression(int prec) {
    int start = token.left;
    expect(TokenKind.NOT);
    Expression expression = parseNonTupleExpression(prec + 1);
    NotExpression notExpression = new NotExpression(expression);
    return setLocation(notExpression, start, token.right);
  }

  // file_input ::= ('\n' | stmt)* EOF
  private List<Statement> parseFileInput() {
    long startTime = Profiler.nanoTimeMaybe();
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
        parseTopLevelStatement(list);
      }
    }
    Profiler.instance().logSimpleTask(startTime, ProfilerTask.SKYLARK_PARSER, "");
    return list;
  }

  // load '(' STRING (COMMA [IDENTIFIER EQUALS] STRING)* COMMA? ')'
  private void parseLoad(List<Statement> list) {
    int start = token.left;
    if (token.kind != TokenKind.STRING) {
      expect(TokenKind.STRING);
      return;
    }

    StringLiteral importString = parseStringLiteral();
    expect(TokenKind.COMMA);

    Map<Identifier, String> symbols = new HashMap<>();
    parseLoadSymbol(symbols); // At least one symbol is required

    while (token.kind != TokenKind.RPAREN && token.kind != TokenKind.EOF) {
      expect(TokenKind.COMMA);
      if (token.kind == TokenKind.RPAREN) {
        break;
      }

      parseLoadSymbol(symbols);
    }
    expect(TokenKind.RPAREN);

    SkylarkImport imp;
    try {
      imp = SkylarkImports.create(importString.getValue());
      LoadStatement stmt = new LoadStatement(imp, importString.getLocation(), symbols);
      list.add(setLocation(stmt, start, token.left));
    } catch (SkylarkImportSyntaxException e) {
      String msg = "Load statement parameter '" + importString + "' is invalid. "
          + e.getMessage();
      reportError(importString.getLocation(), msg);
    }
  }

  /**
   * Parses the next symbol argument of a load statement and puts it into the output map.
   *
   * <p> The symbol is either "name" (STRING) or name = "declared" (IDENTIFIER EQUALS STRING).
   * If no alias is used, "name" and "declared" will be identical. "Declared" refers to the
   * original name in the Bazel file that should be loaded, while "name" will be the key of the
   * entry in the map.
   */
  private void parseLoadSymbol(Map<Identifier, String> symbols) {
    Token nameToken, declaredToken;

    if (token.kind == TokenKind.STRING) {
      nameToken = token;
      declaredToken = nameToken;
    } else {
      if (token.kind != TokenKind.IDENTIFIER) {
        syntaxError(token, "Expected either a literal string or an identifier");
      }

      nameToken = token;

      expect(TokenKind.IDENTIFIER);
      expect(TokenKind.EQUALS);

      declaredToken = token;
    }

    expect(TokenKind.STRING);

    try {
      Identifier identifier = new Identifier(nameToken.value.toString());

      if (symbols.containsKey(identifier)) {
        syntaxError(
            nameToken, String.format("Identifier '%s' is used more than once",
                identifier.getName()));
      } else {
        symbols.put(
            setLocation(identifier, nameToken.left, nameToken.right),
            declaredToken.value.toString());
      }
    } catch (NullPointerException npe) {
      // This means that the value of at least one token is null. In this case, the previous
      // expect() call has already logged an error.
    }
  }

  private void parseTopLevelStatement(List<Statement> list) {
    // In Python grammar, there is no "top-level statement" and imports are
    // considered as "small statements". We are a bit stricter than Python here.
    // Check if there is an include
    if (token.kind == TokenKind.IDENTIFIER) {
      Token identToken = token;
      Identifier ident = parseIdent();

      if (ident.getName().equals("load") && token.kind == TokenKind.LPAREN) {
        expect(TokenKind.LPAREN);
        parseLoad(list);
        return;
      }
      pushToken(identToken); // push the ident back to parse it as a statement
    }
    parseStatement(list, true);
  }

  // small_stmt | 'pass'
  private void parseSmallStatementOrPass(List<Statement> list) {
    if (token.kind == TokenKind.PASS) {
      // Skip the token, don't add it to the list.
      // It has no existence in the AST.
      expect(TokenKind.PASS);
    } else {
      list.add(parseSmallStatement());
    }
  }

  // simple_stmt ::= small_stmt (';' small_stmt)* ';'? NEWLINE
  private void parseSimpleStatement(List<Statement> list) {
    parseSmallStatementOrPass(list);

    while (token.kind == TokenKind.SEMI) {
      nextToken();
      if (token.kind == TokenKind.NEWLINE) {
        break;
      }
      parseSmallStatementOrPass(list);
    }
    expectAndRecover(TokenKind.NEWLINE);
  }

  //     small_stmt ::= assign_stmt
  //                  | expr
  //                  | RETURN expr
  //                  | flow_stmt
  //     assign_stmt ::= expr ('=' | augassign) expr
  //     augassign ::= ('+=' )
  // Note that these are in Python, but not implemented here (at least for now):
  // '-=' | '*=' | '/=' | '%=' | '&=' | '|=' | '^=' |'<<=' | '>>=' | '**=' | '//='
  // Semantic difference from Python:
  // In Skylark, x += y is simple syntactic sugar for x = x + y.
  // In Python, x += y is more or less equivalent to x = x + y, but if a method is defined
  // on x.__iadd__(y), then it takes precedence, and in the case of lists it side-effects
  // the original list (it doesn't do that on tuples); if no such method is defined it falls back
  // to the x.__add__(y) method that backs x + y. In Skylark, we don't support this side-effect.
  // Note also that there is a special casing to translate 'ident[key] = value'
  // to 'ident = ident + {key: value}'. This is needed to support the pure version of Python-like
  // dictionary assignment syntax.
  private Statement parseSmallStatement() {
    int start = token.left;
    if (token.kind == TokenKind.RETURN) {
      return parseReturnStatement();
    } else if ((parsingMode == SKYLARK)
        && (token.kind == TokenKind.BREAK || token.kind == TokenKind.CONTINUE)) {
      return parseFlowStatement(token.kind);
    }
    Expression expression = parseExpression();
    if (token.kind == TokenKind.EQUALS) {
      nextToken();
      Expression rvalue = parseExpression();
      return setLocation(new AssignmentStatement(expression, rvalue), start, rvalue);
    } else if (augmentedAssignmentMethods.containsKey(token.kind)) {
      Operator operator = augmentedAssignmentMethods.get(token.kind);
      nextToken();
      Expression operand = parseExpression();
      int end = operand.getLocation().getEndOffset();
      return setLocation(new AssignmentStatement(expression,
               setLocation(new BinaryOperatorExpression(
                   operator, expression, operand), start, end)),
               start, end);
    } else {
      return setLocation(new ExpressionStatement(expression), start, expression);
    }
  }

  // if_stmt ::= IF expr ':' suite [ELIF expr ':' suite]* [ELSE ':' suite]?
  private IfStatement parseIfStatement() {
    int start = token.left;
    List<ConditionalStatements> thenBlocks = new ArrayList<>();
    thenBlocks.add(parseConditionalStatements(TokenKind.IF));
    while (token.kind == TokenKind.ELIF) {
      thenBlocks.add(parseConditionalStatements(TokenKind.ELIF));
    }
    List<Statement> elseBlock;
    if (token.kind == TokenKind.ELSE) {
      expect(TokenKind.ELSE);
      expect(TokenKind.COLON);
      elseBlock = parseSuite();
    } else {
      elseBlock = ImmutableList.of();
    }
    return setLocation(new IfStatement(thenBlocks, elseBlock), start, token.right);
  }

  // cond_stmts ::= [EL]IF expr ':' suite
  private ConditionalStatements parseConditionalStatements(TokenKind tokenKind) {
    int start = token.left;
    expect(tokenKind);
    Expression expr = parseNonTupleExpression();
    expect(TokenKind.COLON);
    List<Statement> thenBlock = parseSuite();
    ConditionalStatements stmt = new ConditionalStatements(expr, thenBlock);
    return setLocation(stmt, start, token.right);
  }

  // for_stmt ::= FOR IDENTIFIER IN expr ':' suite
  private void parseForStatement(List<Statement> list) {
    int start = token.left;
    expect(TokenKind.FOR);
    Expression loopVar = parseForLoopVariables();
    expect(TokenKind.IN);
    Expression collection = parseExpression();
    expect(TokenKind.COLON);
    List<Statement> block = parseSuite();
    Statement stmt = new ForStatement(loopVar, collection, block);
    list.add(setLocation(stmt, start, token.right));
  }

  // def foo(bar1, bar2):
  private void parseFunctionDefStatement(List<Statement> list) {
    int start = token.left;
    expect(TokenKind.DEF);
    Identifier ident = parseIdent();
    expect(TokenKind.LPAREN);
    List<Parameter<Expression, Expression>> params = parseParameters();
    FunctionSignature.WithValues<Expression, Expression> signature = functionSignature(params);
    expect(TokenKind.RPAREN);
    expect(TokenKind.COLON);
    List<Statement> block = parseSuite();
    FunctionDefStatement stmt = new FunctionDefStatement(ident, params, signature, block);
    list.add(setLocation(stmt, start, token.right));
  }

  private FunctionSignature.WithValues<Expression, Expression> functionSignature(
      List<Parameter<Expression, Expression>> parameters) {
    try {
      return FunctionSignature.WithValues.<Expression, Expression>of(parameters);
    } catch (FunctionSignature.SignatureException e) {
      reportError(e.getParameter().getLocation(), e.getMessage());
      // return bogus empty signature
      return FunctionSignature.WithValues.<Expression, Expression>create(FunctionSignature.of());
    }
  }

  private List<Parameter<Expression, Expression>> parseParameters() {
    return parseFunctionArguments(
        new Supplier<Parameter<Expression, Expression>>() {
          @Override public Parameter<Expression, Expression> get() {
            return parseFunctionParameter();
          }
        });
  }

  /**
   * Parse a list of Argument-s. The arguments can be of class Argument.Passed or Parameter,
   * as returned by the Supplier parseArgument (that, taking no argument, must be closed over
   * the mutable input data structures).
   *
   * <p>This parser does minimal validation: it ensures the proper python use of the comma (that
   * can terminate before a star but not after) and the fact that a **kwarg must appear last.
   * It does NOT validate further ordering constraints for a {@code List<Argument.Passed>}, such as
   * all positional preceding keyword arguments in a call, nor does it check the more subtle
   * constraints for Parameter-s. This validation must happen afterwards in an appropriate method.
   */
  private <V extends Argument> ImmutableList<V>
      parseFunctionArguments(Supplier<V> parseArgument) {
    boolean hasArg = false;
    boolean hasStar = false;
    boolean hasStarStar = false;
    ArrayList<V> arguments = new ArrayList<>();

    while (token.kind != TokenKind.RPAREN && token.kind != TokenKind.EOF) {
      if (hasStarStar) {
        reportError(lexer.createLocation(token.left, token.right),
            "unexpected tokens after kwarg");
        break;
      }
      if (hasArg) {
        expect(TokenKind.COMMA);
      }
      if (token.kind == TokenKind.RPAREN && !hasStar) {
        // list can end with a COMMA if there is neither * nor **
        break;
      }
      V arg = parseArgument.get();
      hasArg = true;
      if (arg.isStar()) {
        hasStar = true;
      } else if (arg.isStarStar()) {
        hasStarStar = true;
      }
      arguments.add(arg);
    }
    return ImmutableList.copyOf(arguments);
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
        parseStatement(list, false);
      }
      expectAndRecover(TokenKind.OUTDENT);
    } else {
      parseSimpleStatement(list);
    }
    return list;
  }

  // skipSuite does not check that the code is syntactically correct, it
  // just skips based on indentation levels.
  private void skipSuite() {
    if (token.kind == TokenKind.NEWLINE) {
      expect(TokenKind.NEWLINE);
      if (token.kind != TokenKind.INDENT) {
        reportError(lexer.createLocation(token.left, token.right),
                    "expected an indented block");
        return;
      }
      expect(TokenKind.INDENT);

      // Don't try to parse all the Python syntax, just skip the block
      // until the corresponding outdent token.
      int depth = 1;
      while (depth > 0) {
        // Because of the way the lexer works, this should never happen
        Preconditions.checkState(token.kind != TokenKind.EOF);

        if (token.kind == TokenKind.INDENT) {
          depth++;
        }
        if (token.kind == TokenKind.OUTDENT) {
          depth--;
        }
        nextToken();
      }

    } else {
      // the block ends at the newline token
      // e.g.  if x == 3: print "three"
      syncTo(STATEMENT_TERMINATOR_SET);
    }
  }

  // stmt ::= simple_stmt
  //        | compound_stmt
  private void parseStatement(List<Statement> list, boolean isTopLevel) {
    if (token.kind == TokenKind.DEF && parsingMode == SKYLARK) {
      if (!isTopLevel) {
        reportError(lexer.createLocation(token.left, token.right),
            "nested functions are not allowed. Move the function to top-level");
      }
      parseFunctionDefStatement(list);
    } else if (token.kind == TokenKind.IF && parsingMode == SKYLARK) {
      list.add(parseIfStatement());
    } else if (token.kind == TokenKind.FOR && parsingMode == SKYLARK) {
      if (isTopLevel) {
        reportError(
            lexer.createLocation(token.left, token.right),
            "for loops are not allowed on top-level. Put it into a function");
      }
      parseForStatement(list);
    } else if (BLOCK_STARTING_SET.contains(token.kind)) {
      skipBlock();
    } else {
      parseSimpleStatement(list);
    }
  }

  // flow_stmt ::= break_stmt | continue_stmt
  private FlowStatement parseFlowStatement(TokenKind kind) {
    int start = token.left;
    expect(kind);
    FlowStatement.Kind flowKind =
        kind == TokenKind.BREAK ? FlowStatement.Kind.BREAK : FlowStatement.Kind.CONTINUE;
    return setLocation(new FlowStatement(flowKind), start, token.right);
  }

  // return_stmt ::= RETURN [expr]
  private ReturnStatement parseReturnStatement() {
    int start = token.left;
    int end = token.right;
    expect(TokenKind.RETURN);

    Expression expression;
    if (STATEMENT_TERMINATOR_SET.contains(token.kind)) {
        // this None makes the AST not correspond to the source exactly anymore
        expression = new Identifier("None");
        setLocation(expression, start, end);
    } else {
        expression = parseExpression();
    }
    return setLocation(new ReturnStatement(expression), start, expression);
  }

  // block ::= ('if' | 'for' | 'class' | 'try' | 'def') expr ':' suite
  private void skipBlock() {
    int start = token.left;
    Token blockToken = token;
    syncTo(EnumSet.of(TokenKind.COLON, TokenKind.EOF)); // skip over expression or name
    if (blockToken.kind == TokenKind.ELSE && parsingMode == SKYLARK) {
      reportError(
          lexer.createLocation(blockToken.left, blockToken.right),
          "syntax error at 'else': not allowed here.");
    } else if (parsingMode != PYTHON) {
      String msg =
          ILLEGAL_BLOCK_KEYWORDS.containsKey(blockToken.kind)
              ? String.format("%ss are not supported.", ILLEGAL_BLOCK_KEYWORDS.get(blockToken.kind))
              : "This is not supported in BUILD files. Move the block to a .bzl file and load it";
      reportError(
          lexer.createLocation(start, token.right),
          String.format("syntax error at '%s': %s", blockToken, msg));
    }
    expect(TokenKind.COLON);
    skipSuite();
  }

  // create a comment node
  private void makeComment(Token token) {
    comments.add(setLocation(new Comment((String) token.value), token.left, token.right));
  }
}
