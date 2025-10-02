// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen.starlark;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.StarlarkDocumentationProcessor.Category;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.syntax.BinaryOperatorExpression;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.TypeApplication;

/**
 * Parses a type expression in a Starlark docstring into HTML text with links to documentation for
 * built-in types.
 *
 * <p>This class exists in its current form only because the type resolver in Bazel's Starlark
 * interpreter is (1) disabled by default and (2) doesn't support providers yet. But in future, we
 * shouldn't need type expressions inside docstrings at all; the desired end state should be to
 * resolve type annotations in actual Starlark code, and then serialize the StarlarkType-s in
 * ModuleInfoExtractor into Stardoc protos.
 */
public final class TypeParser {
  private final ImmutableMap<String, Category> typeIdentifierToCategory;

  public TypeParser(Map<String, Category> typeIdentifierToCategory) {
    this.typeIdentifierToCategory = ImmutableMap.copyOf(typeIdentifierToCategory);
  }

  /**
   * The type expression and remainder parts of a docstring for a parameter, return type, or
   * provider field.
   */
  public static record TypedDocstring(String typeExpression, String remainder) {
    // Assume that type expressions cannot contain '(' or ')', so we can extract them using a regex.
    private static final Pattern TYPED_DOCSTRING_PATTERN =
        Pattern.compile("^\\(([^\\)]+)\\):?\\s*(.*)$");

    /**
     * Splits a docstring into type expression and remainder parts. Specifically, we expect a
     * docstring of the form {@code "(" <type expression> ")" <separator> <remainder>}, where {@code
     * <separator>} is an optional ':' followed by 0 or more whitespace characters; for example,
     * {@code "(string): Some free text about this parameter"}.
     */
    // TODO(arostovtsev): fix ModoleInfoExtractor to also support the `param: (int) ...` form in
    // docstrings.
    public static TypedDocstring of(String docstring) {
      Matcher matcher = TYPED_DOCSTRING_PATTERN.matcher(docstring);
      if (matcher.matches()) {
        return new TypedDocstring(matcher.group(1), matcher.group(2));
      } else {
        return new TypedDocstring("", docstring);
      }
    }
  }

  public String getHtml(String typeExpression, String fallback) throws EvalException {
    if (typeExpression.isEmpty()) {
      return fallback;
    }
    Expression parsedTypeExpression;
    try {
      parsedTypeExpression = Expression.parseTypeExpression(ParserInput.fromLines(typeExpression));
    } catch (SyntaxError.Exception ex) {
      throw Starlark.errorf("Failed to parse type expression '%s': %s", typeExpression, ex);
    }
    return emitHtml(new StringBuilder("<code>"), parsedTypeExpression).append("</code>").toString();
  }

  public String getHtml(String typeExpression) throws EvalException {
    return getHtml(typeExpression, /* fallback= */ "");
  }

  @CanIgnoreReturnValue
  private StringBuilder emitHtml(StringBuilder sb, Expression typeExpression) throws EvalException {
    if (typeExpression instanceof Identifier ident) {
      String url = getUrl(ident.getName(), typeIdentifierToCategory);
      if (url.isEmpty()) {
        return sb.append(ident.getName());
      } else {
        return sb.append(
            String.format(
                "<a class=\"anchor\" href=\"%s\">%s</a>",
                getUrl(ident.getName(), typeIdentifierToCategory), ident.getName()));
      }
    } else if (typeExpression instanceof BinaryOperatorExpression expr) {
      if (expr.getOperator() != TokenKind.PIPE) {
        throw Starlark.errorf(
            "Unexpected operator '%s' in type expression '%s'", expr.getOperator(), typeExpression);
      }
      emitHtml(sb, expr.getX());
      sb.append(" | ");
      return emitHtml(sb, expr.getY());
    } else if (typeExpression instanceof TypeApplication typeApp) {
      Identifier constructor = typeApp.getConstructor();
      emitHtml(sb, constructor);
      sb.append("[");
      boolean first = true;
      for (Expression arg : typeApp.getArguments()) {
        if (first) {
          first = false;
        } else {
          sb.append(", ");
        }
        emitHtml(sb, arg);
      }
      return sb.append("]");
    }
    throw Starlark.errorf("Unsupported type expression '%s'", typeExpression);
  }

  private static String getUrl(String name, Map<String, Category> docPages) {
    switch (name) {
      case "None":
        return "";
      case "Collection":
      case "Sequence":
      case "Mapping":
        return "https://github.com/bazelbuild/starlark/blob/master/spec.md#collection-types";
      case "Callable":
        return "https://github.com/bazelbuild/starlark/blob/master/spec.md#functions";
      default:
        if (docPages.containsKey(name)) {
          return String.format("../%s/%s.html", docPages.get(name).getPath(), name);
        } else {
          return "";
        }
    }
  }
}
