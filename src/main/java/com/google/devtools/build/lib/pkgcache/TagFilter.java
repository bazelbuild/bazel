package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.common.options.OptionsParsingException;
import net.starlark.java.syntax.BinaryOperatorExpression;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.UnaryOperatorExpression;

/**
 * A helper object that converts a tag filter string to an Expression.
 */
public class TagFilter {

  private final String originalString;
  private final Expression expression;

  /**
   * @param input either a boolean expression (e.g. "tag1 or tag2") or a comma-separated list (e.g. "tag1,-tag2")
   */
  public TagFilter(String input) throws OptionsParsingException {
    originalString = input;

    if (input.isEmpty()) {
      expression = null;
    } else {
      expression = parse(input);
      validate(expression);
    }
  }

  private Expression parse(String input) throws OptionsParsingException {
    if (isCommaSeparatedListSyntax(input)) {
      input = convertCommaSeparatedListToBooleanExpression(input);
    }

    try {
      return Expression.parse(ParserInput.fromLines(input));
    } catch (SyntaxError.Exception e) {
      throw new OptionsParsingException("Failed to parse expression: " + e.getMessage() + " input: " + input, e);
    }
  }

  private boolean isCommaSeparatedListSyntax(String input) {
    return input.contains(",") || input.contains("-");
  }

  /**
   * Converts a string containing tags separated by comma to a boolean formula.
   */
  private String convertCommaSeparatedListToBooleanExpression(String input) {
    return input.replaceAll(" *, *-", " and not ")
            .replaceAll(" *, *", " or ")
            .replace("-", " not ").trim();
  }

  /**
   * Throws an OptionsParsingException if this expression does not follow the grammar:
   * expr = IDENT | expr 'or' expr | expr 'and' expr | 'not' expr | '(' expr ')'
   */
  private void validate(Expression expression) throws OptionsParsingException {
    switch (expression.kind()) {
      case IDENTIFIER:
        break;
      case BINARY_OPERATOR:
        BinaryOperatorExpression boe = (BinaryOperatorExpression) expression;
        if (boe.getOperator() != TokenKind.OR && boe.getOperator() != TokenKind.AND) {
          throw new OptionsParsingException(String.format("invalid Boolean operator: %s (want 'and' or 'or')",
                  boe.getOperator()));
        }
        validate(boe.getX());
        validate(boe.getY());
        break;
      case UNARY_OPERATOR:
        UnaryOperatorExpression uoe = (UnaryOperatorExpression) expression;
        if (uoe.getOperator() != TokenKind.NOT) {
          throw new OptionsParsingException(String.format("invalid Boolean operator: %s (want 'not')",
                  uoe.getOperator()));
        }
        validate(uoe.getX());
        break;
      default:
        throw new OptionsParsingException("invalid Boolean operator: " + expression.kind());
    }
  }

  public Expression getExpression() {
    return expression;
  }

  @Override
  public int hashCode() {
    return originalString.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof TagFilter)) {
      return false;
    }
    TagFilter other = (TagFilter) o;
    return this.originalString.equals(other.originalString);
  }
}
