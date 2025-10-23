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

package net.starlark.java.syntax;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** Parser is a recursive-descent parser for Starlark. */
final class Parser {

  /** Combines the parser result into a single value object. */
  static final class ParseResult {
    // Maps char offsets in the file to Locations.
    final FileLocations locs;

    /** The top-level statements of the parsed file. */
    final ImmutableList<Statement> statements;

    /** The comments from the parsed file. */
    final ImmutableList<Comment> comments;

    // Errors encountered during scanning or parsing.
    // These lists are ultimately owned by StarlarkFile.
    final List<SyntaxError> errors;

    private ParseResult(
        FileLocations locs,
        ImmutableList<Statement> statements,
        ImmutableList<Comment> comments,
        List<SyntaxError> errors) {
      this.locs = locs;
      // No need to copy here; when the object is created, the parser instance is just about to go
      // out of scope and be garbage collected.
      this.statements = Preconditions.checkNotNull(statements);
      this.comments = Preconditions.checkNotNull(comments);
      this.errors = errors;
    }
  }

  private static final EnumSet<TokenKind> STATEMENT_TERMINATOR_SET =
      EnumSet.of(TokenKind.EOF, TokenKind.NEWLINE, TokenKind.DOC_COMMENT_TRAILING, TokenKind.SEMI);

  private static final EnumSet<TokenKind> LIST_TERMINATOR_SET =
      EnumSet.of(TokenKind.EOF, TokenKind.RBRACKET, TokenKind.SEMI);

  private static final EnumSet<TokenKind> DICT_TERMINATOR_SET =
      EnumSet.of(TokenKind.EOF, TokenKind.RBRACE, TokenKind.SEMI);

  private static final EnumSet<TokenKind> EXPR_LIST_TERMINATOR_SET =
      EnumSet.of(
          TokenKind.EOF,
          TokenKind.NEWLINE,
          TokenKind.DOC_COMMENT_TRAILING,
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

  /** "type" is a keyword iff it precedes an identifier (such as in a type alias expression). */
  private static final String TYPE_SOFT_KEYWORD = "type";

  /** Current lookahead token. May be mutated by the parser. */
  private final Lexer token; // token.kind is a prettier alias for lexer.kind

  private final FileOptions options;

  private static final boolean DEBUGGING = false;

  private final Lexer lexer;
  private final FileLocations locs;
  private final List<SyntaxError> errors;

  /**
   * Doc comment block which may need to be attached to the next assignment statement. Set to null
   * after parsing a statement. *Not* necessarily set to null after a blank or non-doc comment line;
   * so should be accessed via {@link #getDocCommentBlockOnPreviousLine}.
   */
  private DocComments mostRecentDocCommentBlock = null;

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
          .buildOrThrow();

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
  private boolean recoveryMode; // stop reporting errors until next statement

  // Intern string literals, as some files contain many literals for the same string.
  //
  // Ideally we would move this to the lexer, where we already do interning of identifiers. However,
  // the parser has a special case optimization for concatenation of string literals, which the
  // lexer can't handle.
  private final Map<String, String> stringInterner = new HashMap<>();

  private Parser(Lexer lexer, List<SyntaxError> errors, FileOptions options) {
    this.lexer = lexer;
    this.locs = lexer.locs;
    this.errors = errors;
    this.token = lexer;
    this.options = options;
    nextToken();
  }

  private String intern(String s) {
    String prev = stringInterner.putIfAbsent(s, s);
    return prev != null ? prev : s;
  }

  // Returns a token's string form as used in error messages.
  private static String tokenString(TokenKind kind, @Nullable Object value) {
    return kind == TokenKind.STRING
        ? "\"" + value + "\"" // TODO(adonovan): do proper quotation
        : value == null ? kind.toString() : value.toString();
  }

  // Main entry point for parsing a file.
  static ParseResult parseFile(ParserInput input, FileOptions options) {
    List<SyntaxError> errors = new ArrayList<>();
    Lexer lexer = new Lexer(input, errors, options);
    Parser parser = new Parser(lexer, errors, options);

    StarlarkFile.ParseProfiler profiler = Parser.profiler;
    long profileStartNanos = profiler != null ? profiler.start() : -1;
    try {
      ImmutableList<Statement> statements = parser.parseFileInput();
      return new ParseResult(lexer.locs, statements, lexer.getComments(), errors);
    } finally {
      if (profileStartNanos != -1) {
        profiler.end(profileStartNanos, input.getFile());
      }
    }
  }

  @Nullable static StarlarkFile.ParseProfiler profiler;

  // stmt = simple_stmt
  //      | def_stmt
  //      | for_stmt
  //      | if_stmt
  private void parseStatement(ImmutableList.Builder<Statement> list) {
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

  // Saves the last doc comment block, so that it may be attached to the next assignment.
  private void maybeParseDocCommentBlock() {
    while (token.kind == TokenKind.DOC_COMMENT_BLOCK) {
      mostRecentDocCommentBlock = (DocComments) token.value;
      nextToken();
    }
  }

  @Nullable
  private DocComments getDocCommentBlockOnPreviousLine(int line) {
    if (mostRecentDocCommentBlock != null
        && mostRecentDocCommentBlock.getEndLocation().line() + 1 == line) {
      return mostRecentDocCommentBlock;
    }
    return null;
  }

  /** Parses an expression, possibly preceded or followed by comments or whitespace. */
  static Expression parseExpression(ParserInput input, FileOptions options)
      throws SyntaxError.Exception {
    return parseValueOrTypeExpr(input, options, /* isTypeExpr= */ false);
  }

  /** Parses a type expression, possibly preceded or followed by comments or whitespace. */
  static Expression parseTypeExpression(ParserInput input, FileOptions options)
      throws SyntaxError.Exception {
    return parseValueOrTypeExpr(input, options, /* isTypeExpr= */ true);
  }

  private static Expression parseValueOrTypeExpr(
      ParserInput input, FileOptions options, boolean isTypeExpr) throws SyntaxError.Exception {
    List<SyntaxError> errors = new ArrayList<>();
    Lexer lexer = new Lexer(input, errors, options);
    Parser parser = new Parser(lexer, errors, options);
    Expression result = null;
    try {
      // Skip preceding doc comments (no-ops for an expression).
      while (parser.token.kind == TokenKind.DOC_COMMENT_BLOCK) {
        parser.nextToken();
      }
      result = isTypeExpr ? parser.parseTypeExprWithFallback() : parser.parseExpr();
      // Skip following doc comments and newlines (no-ops for an expression).
      while (parser.token.kind == TokenKind.NEWLINE
          || parser.token.kind == TokenKind.DOC_COMMENT_BLOCK
          || parser.token.kind == TokenKind.DOC_COMMENT_TRAILING) {
        parser.nextToken();
      }
      parser.expect(TokenKind.EOF);
    } catch (StackOverflowError ex) {
      // See rationale at parseFile.
      parser.reportError(
          lexer.end,
          "internal error: stack overflow while parsing Starlark expression <<%s>>. Please report"
              + " the bug.\n"
              + "%s",
          new String(input.getContent()),
          Throwables.getStackTraceAsString(ex));
    }
    if (!errors.isEmpty()) {
      throw new SyntaxError.Exception(errors);
    }
    return result;
  }

  // Parses every kind of expression, including unparenthesized tuples.
  //
  // In Python the corresponding grammar production is called `expressions` (or previously, in
  // Python 3.8 and older, `testlist`).
  //
  // In many cases we need to use parseTest() in place of parseExpr() to avoid ambiguity, e.g.:
  //
  //   f(x, y)  vs  f((x, y))
  //
  // Unlike Python, a trailing comma is disallowed in an unparenthesized tuple.
  // This prevents bugs where a one-element tuple is surprisingly created, e.g.:
  //
  //   foo = f(x),
  private Expression parseExpr() {
    Expression e = parseTest();
    if (token.kind != TokenKind.COMMA) {
      return e;
    }

    // unparenthesized tuple
    ImmutableList.Builder<Expression> elems = ImmutableList.builder();
    elems.add(e);
    parseExprList(elems, /* trailingCommaAllowed= */ false);
    return new ListExpression(locs, /* isTuple= */ true, -1, elems.build(), -1);
  }

  @FormatMethod
  private void reportError(int offset, String format, Object... args) {
    errorsCount++;
    // Limit the number of reported errors to avoid spamming output.
    if (errorsCount <= 5) {
      Location location = locs.getLocation(offset);
      errors.add(new SyntaxError(location, String.format(format, args)));
    }
  }

  private void syntaxError(String message) {
    syntaxError(token.start, token.kind, token.value, message);
  }

  private void syntaxError(int offset, TokenKind tokenKind, Object tokenValue, String message) {
    if (!recoveryMode) {
      if (tokenKind == TokenKind.INDENT) {
        reportError(offset, "indentation error");
      } else {
        reportError(
            offset, "syntax error at '%s': %s", tokenString(tokenKind, tokenValue), message);
      }
      recoveryMode = true;
    }
  }

  // Consumes the current token and returns its position, like nextToken.
  // Reports a syntax error if the new token is not of the expected kind.
  private int expect(TokenKind kind) {
    if (token.kind != kind) {
      syntaxError("expected " + kind);
    }
    return nextToken();
  }

  // Like expect, but stops recovery mode if the token was expected.
  private int expectAndRecover(TokenKind kind) {
    if (token.kind != kind) {
      syntaxError("expected " + kind);
    } else {
      recoveryMode = false;
    }
    return nextToken();
  }

  // Consumes tokens past the first token belonging to terminatingTokens.
  // It returns the end offset of the terminating token.
  // TODO(adonovan): always used with makeErrorExpression. Combine and simplify.
  private int syncPast(EnumSet<TokenKind> terminatingTokens) {
    Preconditions.checkState(terminatingTokens.contains(TokenKind.EOF));
    while (!terminatingTokens.contains(token.kind)) {
      nextToken();
    }
    int end = token.end;
    // read past the synchronization token
    nextToken();
    return end;
  }

  /**
   * Consume tokens until we reach the first token that has a kind that is in the set of
   * terminatingTokens.
   *
   * @param terminatingTokens
   * @return the end offset of the terminating token.
   */
  private int syncTo(EnumSet<TokenKind> terminatingTokens) {
    // EOF must be in the set to prevent an infinite loop
    Preconditions.checkState(terminatingTokens.contains(TokenKind.EOF));
    // read past the problematic token
    int previous = token.end;
    nextToken();
    int current = previous;
    while (!terminatingTokens.contains(token.kind)) {
      nextToken();
      previous = current;
      current = token.end;
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
    reportError(
        token.start,
        "%s",
        switch (token.kind) {
          case ASSERT -> "'assert' not supported, use 'fail' instead";
          case DEL ->
              "'del' not supported, use '.pop()' to delete an item from a dictionary or a list";
          case IMPORT -> "'import' not supported, use 'load' instead";
          case IS -> "'is' not supported, use '==' instead";
          case RAISE -> "'raise' not supported, use 'fail' instead";
          case TRY -> "'try' not supported, all exceptions are fatal";
          case WHILE -> "'while' not supported, use 'for' instead";
          default -> "keyword '" + token.kind + "' not supported";
        });
  }

  private int nextToken() {
    int prev = token.start;
    if (token.kind != TokenKind.EOF) {
      lexer.nextToken();
    }
    checkForbiddenKeywords();
    // TODO(adonovan): move this to lexer so we see the first token too.
    if (DEBUGGING) {
      System.err.print(tokenString(token.kind, token.value));
    }
    return prev;
  }

  // Returns an "Identifier" whose content is the input from start to end.
  private Identifier makeErrorExpression(int start, int end) {
    // It's tempting to define a dedicated BadExpression type,
    // but it is convenient for parseIdent to return an Identifier
    // even when it fails.
    return new Identifier(locs, lexer.bufferSlice(start, end), start);
  }

  // arg = IDENTIFIER '=' test
  //     | expr
  //     | *args
  //     | **kwargs
  private Argument parseArgument() {
    Expression expr;

    // parse **expr
    if (token.kind == TokenKind.STAR_STAR) {
      int starStarOffset = nextToken();
      expr = parseTest();
      return new Argument.StarStar(locs, starStarOffset, expr);
    }

    // parse *expr
    if (token.kind == TokenKind.STAR) {
      int starOffset = nextToken();
      expr = parseTest();
      return new Argument.Star(locs, starOffset, expr);
    }

    // IDENTIFIER  or  IDENTIFIER = test
    expr = parseTest();
    if (expr instanceof Identifier id) {
      // parse a named argument
      if (token.kind == TokenKind.EQUALS) {
        nextToken();
        Expression arg = parseTest();
        return new Argument.Keyword(locs, id, arg);
      }
    }

    // parse a positional argument
    return new Argument.Positional(locs, expr);
  }

  // arg = IDENTIFIER [':' TypeExpr] [ '=' test ]
  //     | * [IDENTIFIER [':' TypeExpr]]
  //     | ** IDENTIFIER [':' TypeExpr]
  // Type annotations are only available on def statements (not lambdas)
  private Parameter parseParameter(boolean defStatement) {
    Expression type = null;

    // **kwargs
    if (token.kind == TokenKind.STAR_STAR) {
      int starStarOffset = nextToken();
      Identifier id = parseIdent();
      if (defStatement) {
        type = maybeParseTypeAnnotationAfter(TokenKind.COLON);
      }
      return new Parameter.StarStar(locs, starStarOffset, id, type);
    }

    // * or *args
    if (token.kind == TokenKind.STAR) {
      int starOffset = nextToken();
      if (token.kind == TokenKind.IDENTIFIER) {
        Identifier id = parseIdent();
        if (defStatement) {
          type = maybeParseTypeAnnotationAfter(TokenKind.COLON);
        }
        return new Parameter.Star(locs, starOffset, id, type);
      }
      return new Parameter.Star(locs, starOffset, null, null);
    }

    // name
    Identifier id = parseIdent();

    // name: type
    if (defStatement) {
      type = maybeParseTypeAnnotationAfter(TokenKind.COLON);
    }

    // name=default
    if (token.kind == TokenKind.EQUALS) {
      nextToken(); // TODO: save token pos?
      Expression expr = parseTest();
      return new Parameter.Optional(locs, id, type, expr);
    }

    return new Parameter.Mandatory(locs, id, type);
  }

  // call_suffix = '(' arg_list? ')'
  private Expression parseCallSuffix(Expression fn) {
    ImmutableList<Argument> args = ImmutableList.of();
    int lparenOffset = expect(TokenKind.LPAREN);
    if (token.kind != TokenKind.RPAREN) {
      args = parseArguments(); // (includes optional trailing comma)
    }
    int rparenOffset = expect(TokenKind.RPAREN);
    return new CallExpression(locs, fn, locs.getLocation(lparenOffset), args, rparenOffset);
  }

  // Parse a list of call arguments.
  //
  // arg_list = ( (arg ',')* arg ','? )?
  private ImmutableList<Argument> parseArguments() {
    boolean seenArg = false;
    ImmutableList.Builder<Argument> list = ImmutableList.builder();
    while (token.kind != TokenKind.RPAREN && token.kind != TokenKind.EOF) {
      if (seenArg) {
        // f(expr for vars in expr) -- Python generator expression?
        if (token.kind == TokenKind.FOR) {
          syntaxError("Starlark does not support Python-style generator expressions");
        }
        expect(TokenKind.COMMA);
        // If nonempty, the list may end with a comma.
        if (token.kind == TokenKind.RPAREN) {
          break;
        }
      }
      list.add(parseArgument());
      seenArg = true;
    }
    return list.build();
  }

  // selector_suffix = '.' IDENTIFIER
  private Expression parseSelectorSuffix(Expression e) {
    int dotOffset = expect(TokenKind.DOT);
    if (token.kind == TokenKind.IDENTIFIER) {
      Identifier id = parseIdent();
      return new DotExpression(locs, e, dotOffset, id);
    }

    syntaxError("expected identifier after dot");
    syncTo(EXPR_TERMINATOR_SET);
    return e;
  }

  // expr_list parses a comma-separated list of expression. It assumes that the
  // first expression was already parsed, so it starts with a comma.
  // It is used to parse tuples and list elements.
  //
  // expr_list = ( ',' expr )* ','?
  private void parseExprList(ImmutableList.Builder<Expression> list, boolean trailingCommaAllowed) {
    //  terminating tokens for an expression list
    while (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      if (EXPR_LIST_TERMINATOR_SET.contains(token.kind)) {
        if (!trailingCommaAllowed) {
          reportError(token.start, "Trailing comma is allowed only in parenthesized tuples.");
        }
        break;
      }
      list.add(parseTest());
    }
  }

  // dict_entry_list = ( (dict_entry ',')* dict_entry ','? )?
  private List<DictExpression.Entry> parseDictEntryList() {
    ImmutableList.Builder<DictExpression.Entry> list = ImmutableList.builder();
    // the terminating token for a dict entry list
    while (token.kind != TokenKind.RBRACE) {
      list.add(parseDictEntry());
      if (token.kind == TokenKind.COMMA) {
        nextToken();
      } else {
        break;
      }
    }
    return list.build();
  }

  // dict_entry = test ':' test
  private DictExpression.Entry parseDictEntry() {
    Expression key = parseTest();
    int colonOffset = expect(TokenKind.COLON);
    Expression value = parseTest();
    return new DictExpression.Entry(locs, key, colonOffset, value);
  }

  // expr = STRING
  private StringLiteral parseStringLiteral() {
    Preconditions.checkState(token.kind == TokenKind.STRING);
    StringLiteral literal =
        new StringLiteral(locs, token.start, intern((String) token.value), token.end);
    nextToken();
    if (token.kind == TokenKind.STRING) {
      reportError(token.start, "Implicit string concatenation is forbidden, use the + operator");
    }
    return literal;
  }

  //  primary = INT
  //          | FLOAT
  //          | STRING
  //          | IDENTIFIER
  //          | list_expression
  //          | '(' ')'                    // a tuple with zero elements
  //          | '(' expr ')'               // a parenthesized expression
  //          | dict_expression
  //          | '-' primary_with_suffix
  private Expression parsePrimary() {
    switch (token.kind) {
      case INT:
        {
          IntLiteral literal =
              new IntLiteral(locs, token.getRaw(), token.start, (Number) token.value);
          nextToken();
          return literal;
        }

      case FLOAT:
        {
          FloatLiteral literal =
              new FloatLiteral(locs, token.getRaw(), token.start, (double) token.value);
          nextToken();
          return literal;
        }

      case STRING:
        return parseStringLiteral();

      case IDENTIFIER:
        return parseIdent();

      case LBRACKET: // [...]
        return parseListMaker();

      case LBRACE: // {...}
        return parseDictExpression();

      case LPAREN:
        {
          int lparenOffset = nextToken();

          // empty tuple: ()
          if (token.kind == TokenKind.RPAREN) {
            int rparen = nextToken();
            return new ListExpression(
                locs, /* isTuple= */ true, lparenOffset, ImmutableList.of(), rparen);
          }

          Expression e = parseTest();

          // parenthesized expression: (e)
          // TODO(adonovan): materialize paren expressions (for fidelity).
          if (token.kind == TokenKind.RPAREN) {
            nextToken();
            return e;
          }

          // non-empty tuple: (e,) or (e, ..., e)
          if (token.kind == TokenKind.COMMA) {
            ImmutableList.Builder<Expression> elems = ImmutableList.builder();
            elems.add(e);
            parseExprList(elems, /* trailingCommaAllowed= */ true);
            int rparenOffset = expect(TokenKind.RPAREN);
            return new ListExpression(
                locs, /* isTuple= */ true, lparenOffset, elems.build(), rparenOffset);
          }

          // (expr for vars in expr) -- Python generator expression?
          if (token.kind == TokenKind.FOR) {
            syntaxError("Starlark does not support Python-style generator expressions");
          }

          expect(TokenKind.RPAREN);
          int end = syncTo(EXPR_TERMINATOR_SET);
          return makeErrorExpression(lparenOffset, end);
        }

      case MINUS:
      case PLUS:
      case TILDE:
        {
          TokenKind op = token.kind;
          int offset = nextToken();
          Expression x = parsePrimaryWithSuffix();
          return new UnaryOperatorExpression(locs, op, offset, x);
        }

      default:
        {
          int start = token.start;
          syntaxError("expected expression");
          int end = syncTo(EXPR_TERMINATOR_SET);
          return makeErrorExpression(start, end);
        }
    }
  }

  // primary_with_suffix = primary (selector_suffix | slice_suffix | call_suffix)*
  private Expression parsePrimaryWithSuffix() {
    Expression e = parsePrimary();
    while (true) {
      if (token.kind == TokenKind.DOT) {
        e = parseSelectorSuffix(e);
      } else if (token.kind == TokenKind.LBRACKET) {
        e = parseSliceSuffix(e);
      } else if (token.kind == TokenKind.LPAREN) {
        e = parseCallSuffix(e);
      } else {
        return e;
      }
    }
  }

  // slice_suffix = '[' expr? ':' expr?  ':' expr? ']'
  //              | '[' expr? ':' expr? ']'
  //              | '[' expr ']'
  private Expression parseSliceSuffix(Expression e) {
    int lbracketOffset = expect(TokenKind.LBRACKET);
    Expression start = null;
    Expression end = null;
    Expression step = null;

    if (token.kind != TokenKind.COLON) {
      start = parseExpr();

      // index x[i]
      if (token.kind == TokenKind.RBRACKET) {
        int rbracketOffset = expect(TokenKind.RBRACKET);
        return new IndexExpression(locs, e, lbracketOffset, start, rbracketOffset);
      }
    }

    // slice or substring x[i:j] or x[i:j:k]
    expect(TokenKind.COLON);
    if (token.kind != TokenKind.COLON && token.kind != TokenKind.RBRACKET) {
      end = parseTest();
    }
    if (token.kind == TokenKind.COLON) {
      expect(TokenKind.COLON);
      if (token.kind != TokenKind.RBRACKET) {
        step = parseTest();
      }
    }
    int rbracketOffset = expect(TokenKind.RBRACKET);
    return new SliceExpression(locs, e, lbracketOffset, start, end, step, rbracketOffset);
  }

  // Equivalent to 'exprlist' rule in Python grammar.
  // loop_variables = primary_with_suffix ( ',' primary_with_suffix )* ','?
  private Expression parseForLoopVariables() {
    // We cannot reuse parseExpr because it would parse the 'in' operator.
    // e.g.  "for i in e: pass"  -> we want to parse only "i" here.
    Expression e1 = parsePrimaryWithSuffix();
    if (token.kind != TokenKind.COMMA) {
      return e1;
    }

    // unparenthesized tuple
    ImmutableList.Builder<Expression> elems = ImmutableList.builder();
    elems.add(e1);
    while (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      if (EXPR_LIST_TERMINATOR_SET.contains(token.kind)) {
        break;
      }
      elems.add(parsePrimaryWithSuffix());
    }
    return new ListExpression(locs, /* isTuple= */ true, -1, elems.build(), -1);
  }

  // comprehension_suffix = 'FOR' loop_variables 'IN' expr comprehension_suffix
  //                      | 'IF' expr comprehension_suffix
  //                      | ']' | '}'
  private Expression parseComprehensionSuffix(int loffset, Node body, TokenKind closingBracket) {
    ImmutableList.Builder<Comprehension.Clause> clauses = ImmutableList.builder();
    while (true) {
      if (token.kind == TokenKind.FOR) {
        int forOffset = nextToken();
        Expression vars = parseForLoopVariables();
        expect(TokenKind.IN);
        // The expression cannot be a ternary expression ('x if y else z') due to
        // conflicts in Python grammar ('if' is used by the comprehension).
        Expression seq = parseTest(0);
        clauses.add(new Comprehension.For(locs, forOffset, vars, seq));
      } else if (token.kind == TokenKind.IF) {
        int ifOffset = nextToken();
        // [x for x in li if 1, 2]  # parse error
        // [x for x in li if (1, 2)]  # ok
        Expression cond = parseTestNoCond();
        clauses.add(new Comprehension.If(locs, ifOffset, cond));
      } else if (token.kind == closingBracket) {
        break;
      } else {
        syntaxError("expected '" + closingBracket + "', 'for' or 'if'");
        int end = syncPast(LIST_TERMINATOR_SET);
        return makeErrorExpression(loffset, end);
      }
    }

    boolean isDict = closingBracket == TokenKind.RBRACE;
    int roffset = expect(closingBracket);
    return new Comprehension(locs, isDict, loffset, body, clauses.build(), roffset);
  }

  // list_maker = '[' ']'
  //            | '[' expr ']'
  //            | '[' expr expr_list ']'
  //            | '[' expr comprehension_suffix ']'
  private Expression parseListMaker() {
    int lbracketOffset = expect(TokenKind.LBRACKET);
    if (token.kind == TokenKind.RBRACKET) { // empty List
      int rbracketOffset = nextToken();
      return new ListExpression(
          locs, /* isTuple= */ false, lbracketOffset, ImmutableList.of(), rbracketOffset);
    }

    Expression expression = parseTest();
    switch (token.kind) {
      case RBRACKET:
        // [e], singleton list
        {
          int rbracketOffset = nextToken();
          return new ListExpression(
              locs,
              /* isTuple= */ false,
              lbracketOffset,
              ImmutableList.of(expression),
              rbracketOffset);
        }

      case FOR:
        // [e for x in y], list comprehension
        return parseComprehensionSuffix(lbracketOffset, expression, TokenKind.RBRACKET);

      case COMMA:
        // [e, ...], list expression
        {
          ImmutableList.Builder<Expression> elems = ImmutableList.builder();
          elems.add(expression);
          parseExprList(elems, /* trailingCommaAllowed= */ true);
          if (token.kind == TokenKind.RBRACKET) {
            int rbracketOffset = nextToken();
            return new ListExpression(
                locs, /* isTuple= */ false, lbracketOffset, elems.build(), rbracketOffset);
          }

          expect(TokenKind.RBRACKET);
          int end = syncPast(LIST_TERMINATOR_SET);
          return makeErrorExpression(lbracketOffset, end);
        }

      default:
        {
          syntaxError("expected ',', 'for' or ']'");
          int end = syncPast(LIST_TERMINATOR_SET);
          return makeErrorExpression(lbracketOffset, end);
        }
    }
  }

  // dict_expression = '{' '}'
  //                 | '{' dict_entry_list '}'
  //                 | '{' dict_entry comprehension_suffix '}'
  private Expression parseDictExpression() {
    int lbraceOffset = expect(TokenKind.LBRACE);
    if (token.kind == TokenKind.RBRACE) { // empty Dict
      int rbraceOffset = nextToken();
      return new DictExpression(locs, lbraceOffset, ImmutableList.of(), rbraceOffset);
    }

    DictExpression.Entry entry = parseDictEntry();
    if (token.kind == TokenKind.FOR) {
      // Dict comprehension
      return parseComprehensionSuffix(lbraceOffset, entry, TokenKind.RBRACE);
    }

    ImmutableList.Builder<DictExpression.Entry> entries = ImmutableList.builder();
    entries.add(entry);
    if (token.kind == TokenKind.COMMA) {
      expect(TokenKind.COMMA);
      entries.addAll(parseDictEntryList());
    }
    if (token.kind == TokenKind.RBRACE) {
      int rbraceOffset = nextToken();
      return new DictExpression(locs, lbraceOffset, entries.build(), rbraceOffset);
    }

    expect(TokenKind.RBRACE);
    int end = syncPast(DICT_TERMINATOR_SET);
    return makeErrorExpression(lbraceOffset, end);
  }

  private Identifier parseIdent() {
    if (token.kind != TokenKind.IDENTIFIER) {
      int start = token.start;
      int end = expect(TokenKind.IDENTIFIER);
      return makeErrorExpression(start, end);
    }

    String name = (String) token.value;
    int offset = nextToken();
    return new Identifier(locs, name, offset);
  }

  // binop_expression = binop_expression OP binop_expression
  //                  | parsePrimaryWithSuffix
  // This function takes care of precedence between operators (see operatorPrecedence for
  // the order), and it assumes left-to-right associativity.
  private Expression parseBinOpExpression(int prec) {
    Expression x = parseTest(prec + 1);
    // The loop is not strictly needed, but it prevents risks of stack overflow. Depth is
    // limited to number of different precedence levels (operatorPrecedence.size()).
    TokenKind lastOp = null;
    for (; ; ) {
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
        return x;
      }

      // Operator '==' and other operators of the same precedence (e.g. '<', 'in')
      // are not associative.
      if (lastOp != null && operatorPrecedence.get(prec).contains(TokenKind.EQUALS_EQUALS)) {
        reportError(
            token.start,
            "Operator '%s' is not associative with operator '%s'. Use parens.",
            lastOp,
            op);
      }

      int opOffset = nextToken();
      Expression y = parseTest(prec + 1);
      x = optimizeBinOpExpression(x, op, opOffset, y);
      lastOp = op;
    }
  }

  // Optimize binary expressions.
  // string literal + string literal can be concatenated into one string literal
  // so we don't have to do the expensive string concatenation at runtime.
  private Expression optimizeBinOpExpression(
      Expression x, TokenKind op, int opOffset, Expression y) {
    if (op == TokenKind.PLUS && x instanceof StringLiteral && y instanceof StringLiteral) {
      return new StringLiteral(
          locs,
          x.getStartOffset(),
          intern(((StringLiteral) x).getValue() + ((StringLiteral) y).getValue()),
          y.getEndOffset());
    }
    return new BinaryOperatorExpression(locs, x, op, opOffset, y);
  }

  /**
   * Returns true if type syntax is allowed. Otherwise, reports a syntax error for the given offset
   * and token kind and value, and returns false.
   */
  @CanIgnoreReturnValue
  private boolean checkAllowTypeSyntax(int offset, TokenKind tokenKind, Object tokenValue) {
    if (options.allowTypeSyntax()) {
      return true;
    } else {
      syntaxError(
          offset,
          tokenKind,
          tokenValue,
          "type annotations are disallowed. Enable them with --experimental_starlark_type_syntax "
              + "and/or --experimental_starlark_types_allowed_paths.");
      return false;
    }
  }

  @Nullable
  private Expression maybeParseTypeAnnotationAfter(TokenKind expectedToken) {
    if (token.kind == expectedToken && checkAllowTypeSyntax(token.start, token.kind, token.value)) {
      nextToken();
      return parseTypeExprWithFallback();
    }
    return null;
  }

  // Hook for parsing either a structured type expression, or an unstructured arbitrary expression
  // (except for unparenthesized tuples). The latter is useless for type checking but allows the
  // parser to never fail on parsing a type annotation it doesn't recognize (e.g. supported by a
  // future version of Bazel), so long as it's valid expression syntax.
  private Expression parseTypeExprWithFallback() {
    if (options.allowArbitraryTypeExpressions()) {
      // parseTest, because allowing unparenthesized tuples here would consume subsequent params in
      // function signatures.
      return parseTest();
    } else {
      return parseTypeExpr();
    }
  }

  // TypeExpr = TypeAtom {'|' TypeAtom}.
  // TypeAtom = identifier [TypeArguments].
  private Expression parseTypeExpr() {
    if (token.kind != TokenKind.IDENTIFIER) {
      int start = token.start;
      syntaxError("expected a type");
      int end = syncTo(EXPR_TERMINATOR_SET);
      return makeErrorExpression(start, end);
    }
    Identifier typeOrConstructor = parseIdent();
    Expression expr;
    if (token.kind == TokenKind.LBRACKET) {
      expr = parseTypeApplication(typeOrConstructor);
    } else {
      expr = typeOrConstructor;
    }
    while (token.kind == TokenKind.PIPE) {
      int opOffset = nextToken();
      Identifier secondTypeOrConstructor = parseIdent();
      Expression y;
      if (token.kind == TokenKind.LBRACKET) {
        y = parseTypeApplication(secondTypeOrConstructor);
      } else {
        y = secondTypeOrConstructor;
      }
      expr = new BinaryOperatorExpression(locs, expr, TokenKind.PIPE, opOffset, y);
    }
    return expr;
  }

  // TypeArgument = TypeExpr | ListOfTypes | DictOfTypes | string
  private Expression parseTypeArgument() {
    switch (token.kind) {
      case LBRACKET: // [...]
        return parseTypeList();
      case LBRACE: // {...}
        return parseTypeDict();
      case STRING:
        return parseStringLiteral();
      default:
    }
    if (token.kind != TokenKind.IDENTIFIER) {
      int start = token.start;
      syntaxError("expected a type argument");
      int end = syncTo(EXPR_TERMINATOR_SET);
      return makeErrorExpression(start, end);
    }
    return parseTypeExpr();
  }

  // ListOfTypes = '[' [TypeArgument {',' TypeArgument} [',']] ']'.
  private Expression parseTypeList() {
    int lbracketOffset = expect(TokenKind.LBRACKET);
    ImmutableList.Builder<Expression> elems = ImmutableList.builder();
    if (token.kind != TokenKind.RBRACKET) {
      elems.add(parseTypeArgument());
    }
    while (token.kind != TokenKind.RBRACKET && token.kind != TokenKind.EOF) {
      expect(TokenKind.COMMA);
      if (token.kind == TokenKind.RBRACKET) {
        break;
      }
      elems.add(parseTypeArgument());
    }
    int rbracketOffset = nextToken();
    return new ListExpression(
        locs, /* isTuple= */ false, lbracketOffset, elems.build(), rbracketOffset);
  }

  // TypeEntry = string ':' TypeArgument .
  private DictExpression.Entry parseTypeDictEntry() {
    Expression key = parseStringLiteral();
    int colonOffset = expect(TokenKind.COLON);
    Expression value = parseTypeArgument();
    return new DictExpression.Entry(locs, key, colonOffset, value);
  }

  // DictOfTypes = '{' [TypeEntry {',' TypeEntry} [',']] '}' .
  private Expression parseTypeDict() {
    int lbraceOffset = expect(TokenKind.LBRACE);

    ImmutableList.Builder<DictExpression.Entry> entries = ImmutableList.builder();
    if (token.kind != TokenKind.RBRACE) {
      entries.add(parseTypeDictEntry());
    }
    while (token.kind != TokenKind.RBRACE && token.kind != TokenKind.EOF) {
      expect(TokenKind.COMMA);
      if (token.kind == TokenKind.RBRACE) {
        break;
      }
      entries.add(parseTypeDictEntry());
    }

    int rbraceOffset = nextToken();
    return new DictExpression(locs, lbraceOffset, entries.build(), rbraceOffset);
  }

  // TypeArguments = '[' TypeArgument {',' TypeArgument} ']'.
  private Expression parseTypeApplication(Identifier constructor) {
    expect(TokenKind.LBRACKET);
    ImmutableList.Builder<Expression> args = ImmutableList.builder();
    if (token.kind != TokenKind.RBRACKET) {
      args.add(parseTypeArgument());
    }
    while (token.kind != TokenKind.RBRACKET && token.kind != TokenKind.EOF) {
      expect(TokenKind.COMMA);
      args.add(parseTypeArgument());
    }
    int rbracketOffset = expect(TokenKind.RBRACKET);
    return new TypeApplication(locs, constructor, args.build(), rbracketOffset);
  }

  private static boolean isTypeSoftKeyword(Node node) {
    return node instanceof Identifier id && id.getName().equals(TYPE_SOFT_KEYWORD);
  }

  // type_alias_stmt = 'type' type_alias_stmt_tail
  // type_alias_stmt_tail = identifier optional_type_params '=' TypeExpr
  //
  // This method assumes that 'type' has already been consumed to produce typeSoftKeywordNode.
  private Statement parseTypeAliasStatementTail(Node typeSoftKeywordNode) {
    Preconditions.checkArgument(isTypeSoftKeyword(typeSoftKeywordNode));
    int startOffset = typeSoftKeywordNode.getStartOffset();
    // For user-friendliness, mark the error as if it was detected at 'type'
    checkAllowTypeSyntax(startOffset, TokenKind.IDENTIFIER, TYPE_SOFT_KEYWORD);
    Identifier identifier = parseIdent();
    ImmutableList<Identifier> parameters = parseOptionalTypeParameters();
    expect(TokenKind.EQUALS);
    Expression definition = parseTypeExprWithFallback();
    return new TypeAliasStatement(locs, startOffset, identifier, parameters, definition);
  }

  // optional_type_params = ['[' identifier {',' identifier} [','] ']']
  //
  // For syntactic compatibility with Python, the list of identifiers in optional_type_params cannot
  // contain duplicates; duplicate identifiers are treated as a syntax error.
  //
  // If the optional_type_params is absent (in other words, if the initial token is not '['), this
  // method returns an empty list. (Note that if optional_type_params is present, it must contain at
  // least one identifier.)
  private ImmutableList<Identifier> parseOptionalTypeParameters() {
    if (token.kind == TokenKind.LBRACKET) {
      checkAllowTypeSyntax(token.start, token.kind, token.value);
      nextToken();
      ImmutableList.Builder<Identifier> parameters = ImmutableList.builder();
      Set<String> uniqueParameterNames = new HashSet<>();
      parameters.add(parseTypeParameter(uniqueParameterNames));
      while (token.kind != TokenKind.RBRACKET && token.kind != TokenKind.EOF) {
        expect(TokenKind.COMMA);
        if (token.kind == TokenKind.RBRACKET) {
          break;
        }
        parameters.add(parseTypeParameter(uniqueParameterNames));
      }
      expect(TokenKind.RBRACKET);
      return parameters.build();
    } else {
      return ImmutableList.of();
    }
  }

  private Identifier parseTypeParameter(Set<String> uniqueParameterNames) {
    int tokenStart = token.start;
    TokenKind tokenKind = token.kind;
    Object tokenValue = token.value;
    Identifier ident = parseIdent();
    // If parseIdent() encountered a syntax error, Identifier.isValid(param.getName()) would be
    // false, and in that case, there's no need to check for the param's uniqueness.
    if (Identifier.isValid(ident.getName()) && !uniqueParameterNames.add(ident.getName())) {
      syntaxError(tokenStart, tokenKind, tokenValue, "duplicate type parameter");
    }
    return ident;
  }

  // Parses any expression except for an unparenthesized tuple.
  //
  // In Python the corresponding grammar production is called `expression` (or previously, in
  // Python 3.8 and older, `test`).
  private Expression parseTest() {
    int start = token.start;
    if (token.kind == TokenKind.LAMBDA) {
      return parseLambda(/* allowCond= */ true);
    }

    Expression expr = parseTest(0);
    if (token.kind == TokenKind.IF) {
      nextToken();
      Expression condition = parseTest(0);
      if (token.kind == TokenKind.ELSE) {
        nextToken();
        Expression elseClause = parseTest();
        return new ConditionalExpression(locs, expr, condition, elseClause);
      } else {
        reportError(start, "missing else clause in conditional expression or semicolon before if");
        return expr; // Try to recover from error: drop the if and the expression after it. Ouch.
      }
    }
    return expr;
  }

  private Expression parseTest(int prec) {
    if (prec >= operatorPrecedence.size()) {
      return parsePrimaryWithSuffix();
    }
    if (token.kind == TokenKind.NOT && operatorPrecedence.get(prec).contains(TokenKind.NOT)) {
      return parseNotExpression(prec);
    }
    return parseBinOpExpression(prec);
  }

  // parseLambda parses a lambda expression.
  // The allowCond flag allows the body to be an 'a if b else c' conditional.
  private LambdaExpression parseLambda(boolean allowCond) {
    int lambdaOffset = expect(TokenKind.LAMBDA);
    ImmutableList<Parameter> params = parseParameters(/* defStatement= */ false);
    expect(TokenKind.COLON);
    Expression body = allowCond ? parseTest() : parseTestNoCond();
    return new LambdaExpression(locs, lambdaOffset, params, body);
  }

  // parseTestNoCond parses a single-component expression without
  // consuming a trailing 'if expr else expr'.
  private Expression parseTestNoCond() {
    if (token.kind == TokenKind.LAMBDA) {
      return parseLambda(/* allowCond= */ false);
    }
    return parseTest(0);
  }

  // not_expr = 'not' expr
  private Expression parseNotExpression(int prec) {
    int notOffset = expect(TokenKind.NOT);
    Expression x = parseTest(prec);
    return new UnaryOperatorExpression(locs, TokenKind.NOT, notOffset, x);
  }

  // file_input = EOF
  //            | ('\n' | DOC_COMMENT_BLOCK | stmt)* '\n' EOF
  // The terminating newline is injected by the lexer even if not present in the input.
  private ImmutableList<Statement> parseFileInput() {
    ImmutableList.Builder<Statement> list = ImmutableList.builder();
    try {
      while (token.kind != TokenKind.EOF) {
        if (token.kind == TokenKind.NEWLINE) {
          expectAndRecover(TokenKind.NEWLINE);
        } else if (recoveryMode) {
          // If there was a parse error, we want to recover here
          // before starting a new top-level statement.
          syncTo(STATEMENT_TERMINATOR_SET);
          recoveryMode = false;
        } else {
          maybeParseDocCommentBlock();
          if (token.kind == TokenKind.EOF) {
            break;
          }
          parseStatement(list);
        }
      }
    } catch (StackOverflowError ex) {
      // JVM threads have very limited stack, and deeply nested inputs can
      // easily cause the parser to consume all available stack. It is hard
      // to anticipate all the possible recursions in the parser, especially
      // when considering error recovery. Consider a long list of dicts:
      // even if the intended parse tree has a depth of only two,
      // if each dict contains a syntax error, the parser will go into recovery
      // and may discard each dict's closing '}', turning a shallow tree
      // into a deep one (see b/157470754).
      //
      // So, for robustness, the parser treats StackOverflowError as a parse
      // error, exhorting the user to report a bug.
      reportError(
          token.end,
          "internal error: stack overflow in Starlark parser. Please report the bug and include"
              + " the text of %s.\n"
              + "%s",
          locs.file(),
          Throwables.getStackTraceAsString(ex));
    }
    return list.build();
  }

  // load '(' STRING (COMMA [IDENTIFIER EQUALS] STRING)+ COMMA? ')'
  private Statement parseLoadStatement() {
    int loadOffset = expect(TokenKind.LOAD);
    expect(TokenKind.LPAREN);
    if (token.kind != TokenKind.STRING) {
      // error: module is not a string literal.
      StringLiteral module = new StringLiteral(locs, token.start, "", token.end);
      expect(TokenKind.STRING);
      return new LoadStatement(locs, loadOffset, module, ImmutableList.of(), token.end);
    }

    StringLiteral module = parseStringLiteral();
    if (token.kind == TokenKind.RPAREN) {
      syntaxError("expected at least one symbol to load");
      return new LoadStatement(locs, loadOffset, module, ImmutableList.of(), token.end);
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

    int rparen = expect(TokenKind.RPAREN);
    return new LoadStatement(locs, loadOffset, module, bindings.build(), rparen);
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
    int nameOffset = token.start + (token.kind == TokenKind.STRING ? 1 : 0);
    Identifier local = new Identifier(locs, name, nameOffset);

    Identifier original;
    if (token.kind == TokenKind.STRING) {
      // load(..., "name")
      original = local;
    } else {
      // load(..., local = "orig")
      // The name "orig" is morally an identifier but, for legacy reasons (specifically,
      // a partial implementation of Starlark embedded in a Python interpreter used by
      // tests of Blaze), it must be a quoted string literal.
      expect(TokenKind.IDENTIFIER);
      expect(TokenKind.EQUALS);
      if (token.kind != TokenKind.STRING) {
        syntaxError("expected string");
        return;
      }
      original = new Identifier(locs, (String) token.value, token.start + 1);
    }
    nextToken();
    symbols.add(new LoadStatement.Binding(local, original));
  }

  // simple_stmt = small_stmt (';' small_stmt)* ';'? DOC_COMMENT_TRAILING? NEWLINE
  // Note that the DOC_COMMENT_TRAILING will be absorbed by the first small_stmt iff it is an
  // assign_stmt and there are no other tokens between it and the DOC_COMMENT_TRAILING.
  private void parseSimpleStatement(ImmutableList.Builder<Statement> list) {
    list.add(parseSmallStatement());
    mostRecentDocCommentBlock = null;

    while (token.kind == TokenKind.SEMI) {
      nextToken();
      if (token.kind == TokenKind.NEWLINE || token.kind == TokenKind.DOC_COMMENT_TRAILING) {
        break;
      }
      list.add(parseSmallStatement());
    }
    if (token.kind == TokenKind.DOC_COMMENT_TRAILING) {
      // Absorb trailing doc comments that weren't attached to an assignment.
      nextToken();
    }
    expectAndRecover(TokenKind.NEWLINE);
  }

  //     small_stmt = assign_stmt
  //                | type_alias_stmt
  //                | expr
  //                | load_stmt
  //                | return_stmt
  //                | BREAK | CONTINUE | PASS
  //
  //     assign_stmt = expr ('=' | augassign) expr DOC_COMMENT_TRAILING?
  //
  //     augassign = '+=' | '-=' | '*=' | '/=' | '%=' | '//=' | '&=' | '|=' | '^=' |'<<=' | '>>='
  private Statement parseSmallStatement() {
    // return
    if (token.kind == TokenKind.RETURN) {
      return parseReturnStatement();
    }

    // control flow
    if (token.kind == TokenKind.BREAK
        || token.kind == TokenKind.CONTINUE
        || token.kind == TokenKind.PASS) {
      TokenKind kind = token.kind;
      int offset = nextToken();
      return new FlowStatement(locs, kind, offset);
    }

    // load
    if (token.kind == TokenKind.LOAD) {
      return parseLoadStatement();
    }

    Expression lhs = parseExpr();

    // type alias; this is the only context in which the identifier `type` may be followed by
    // another identifier.
    if (token.kind == TokenKind.IDENTIFIER && isTypeSoftKeyword(lhs)) {
      return parseTypeAliasStatementTail(lhs);
    }

    // lhs = rhs  or  lhs += rhs
    TokenKind op = augmentedAssignments.get(token.kind);
    if (token.kind == TokenKind.EQUALS || op != null) {
      int opOffset = nextToken();
      Expression rhs = parseExpr();
      @Nullable
      DocComments docComments = getDocCommentBlockOnPreviousLine(lhs.getStartLocation().line());
      if (token.kind == TokenKind.DOC_COMMENT_TRAILING) {
        // Use trailing doc comment if it exists; it overrides the preceding doc comment block.
        docComments = (DocComments) token.value;
        nextToken();
      }
      // op == null for ordinary assignment. TODO(adonovan): represent as EQUALS.
      return new AssignmentStatement(locs, lhs, op, opOffset, rhs, docComments);
    } else {
      return new ExpressionStatement(locs, lhs);
    }
  }

  // if_stmt = IF expr ':' suite [ELIF expr ':' suite]* [ELSE ':' suite]?
  private IfStatement parseIfStatement() {
    int ifOffset = expect(TokenKind.IF);
    Expression cond = parseTest();
    expect(TokenKind.COLON);
    ImmutableList<Statement> body = parseSuite();
    IfStatement ifStmt = new IfStatement(locs, TokenKind.IF, ifOffset, cond, body);
    IfStatement tail = ifStmt;
    while (token.kind == TokenKind.ELIF) {
      int elifOffset = expect(TokenKind.ELIF);
      cond = parseTest();
      expect(TokenKind.COLON);
      body = parseSuite();
      IfStatement elif = new IfStatement(locs, TokenKind.ELIF, elifOffset, cond, body);
      tail.setElseBlock(ImmutableList.of(elif));
      tail = elif;
    }
    if (token.kind == TokenKind.ELSE) {
      expect(TokenKind.ELSE);
      expect(TokenKind.COLON);
      body = parseSuite();
      tail.setElseBlock(body);
    }
    return ifStmt;
  }

  // for_stmt = FOR IDENTIFIER IN expr ':' suite
  private ForStatement parseForStatement() {
    int forOffset = expect(TokenKind.FOR);
    Expression vars = parseForLoopVariables();
    expect(TokenKind.IN);
    Expression collection = parseExpr();
    expect(TokenKind.COLON);
    ImmutableList<Statement> body = parseSuite();
    return new ForStatement(locs, forOffset, vars, collection, body);
  }

  // def_stmt = DEF IDENTIFIER optional_type_parameters '(' arguments ')' ['->' TypeExpr] ':' suite
  private DefStatement parseDefStatement() {
    int defOffset = expect(TokenKind.DEF);
    Identifier ident = parseIdent();
    ImmutableList<Identifier> typeParams = parseOptionalTypeParameters();
    expect(TokenKind.LPAREN);
    ImmutableList<Parameter> params = parseParameters(/* defStatement= */ true);
    expect(TokenKind.RPAREN);
    Expression returnType = maybeParseTypeAnnotationAfter(TokenKind.RARROW);
    expect(TokenKind.COLON);
    ImmutableList<Statement> block = parseSuite();
    return new DefStatement(locs, defOffset, ident, typeParams, params, returnType, block);
  }

  // Parse a list of function parameters.
  // Validation of parameter ordering and uniqueness is the job of the Resolver.
  private ImmutableList<Parameter> parseParameters(boolean defStatement) {
    boolean hasParam = false;
    ImmutableList.Builder<Parameter> list = ImmutableList.builder();

    while (token.kind != TokenKind.RPAREN
        && token.kind != TokenKind.COLON
        && token.kind != TokenKind.EOF) {
      if (hasParam) {
        expect(TokenKind.COMMA);
        // The list may end with a comma.
        if (token.kind == TokenKind.RPAREN) {
          break;
        }
      }
      Parameter param = parseParameter(defStatement);
      hasParam = true;
      list.add(param);
    }
    return list.build();
  }

  // suite is typically what follows a colon (e.g. after def or for).
  // suite = simple_stmt
  //       | DOC_COMMENT_TRAILING? NEWLINE DOC_COMMENT_BLOCK? INDENT (stmt DOC_COMMENT_BLOCK?)+ \
  //         OUTDENT
  private ImmutableList<Statement> parseSuite() {
    ImmutableList.Builder<Statement> list = ImmutableList.builder();
    if (token.kind == TokenKind.DOC_COMMENT_TRAILING) {
      nextToken();
    }
    if (token.kind == TokenKind.NEWLINE) {
      expect(TokenKind.NEWLINE);
      maybeParseDocCommentBlock();
      if (token.kind != TokenKind.INDENT) {
        reportError(token.start, "expected an indented block");
        return list.build();
      }
      expect(TokenKind.INDENT);
      while (token.kind != TokenKind.OUTDENT && token.kind != TokenKind.EOF) {
        parseStatement(list);
        // Note that on the final loop iteration, we may encounter a doc comment block that will
        // need to be attached to the (dedented) assignment statement after the end of the suite.
        maybeParseDocCommentBlock();
      }
      expectAndRecover(TokenKind.OUTDENT);
    } else {
      parseSimpleStatement(list);
    }
    return list.build();
  }

  // return_stmt = RETURN [expr]
  private ReturnStatement parseReturnStatement() {
    int returnOffset = expect(TokenKind.RETURN);

    Expression result = null;
    if (!STATEMENT_TERMINATOR_SET.contains(token.kind)) {
      result = parseExpr();
    }
    return new ReturnStatement(locs, returnOffset, result);
  }
}
