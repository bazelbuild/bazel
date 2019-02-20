// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.skylint;

import static com.google.devtools.skylark.common.DocstringUtils.extractDocstring;

import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Parameter;
import com.google.devtools.build.lib.syntax.ReturnStatement;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import com.google.devtools.skylark.common.DocstringUtils;
import com.google.devtools.skylark.common.DocstringUtils.DocstringInfo;
import com.google.devtools.skylark.common.DocstringUtils.DocstringParseError;
import com.google.devtools.skylark.common.DocstringUtils.ParameterDoc;
import com.google.devtools.skylark.common.LocationRange;
import com.google.devtools.skylark.common.LocationRange.Location;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/** Checks the existence of docstrings. */
public class DocstringChecker extends SyntaxTreeVisitor {
  private static final String MISSING_MODULE_DOCSTRING_CATEGORY = "missing-module-docstring";
  private static final String MISSING_FUNCTION_DOCSTRING_CATEGORY = "missing-function-docstring";
  private static final String INCONSISTENT_DOCSTRING_CATEGORY = "inconsistent-docstring";
  private static final String BAD_DOCSTRING_FORMAT_CATEGORY = "bad-docstring-format";
  private static final String ARGS_ARGUMENTS_DOCSTRING_CATEGORY = "args-arguments-docstring";
  /** If a function is at least this many statements long, a docstring is required. */
  private static final int FUNCTION_LENGTH_DOCSTRING_THRESHOLD = 5;

  private final List<Issue> issues = new ArrayList<>();
  private boolean containsReturnWithValue = false;

  public static List<Issue> check(BuildFileAST ast) {
    DocstringChecker checker = new DocstringChecker();
    ast.accept(checker);
    return checker.issues;
  }

  @Override
  public void visit(BuildFileAST node) {
    StringLiteral moduleDocstring = extractDocstring(node.getStatements());
    if (moduleDocstring == null) {
      // The reported location starts on the first line since that's where the docstring is expected
      Location start = new Location(1, 1);
      // This location is invalid if the file is empty but this edge case is not worth the trouble.
      Location end = new Location(2, 1);
      LocationRange range = new LocationRange(start, end);
      issues.add(
          new Issue(MISSING_MODULE_DOCSTRING_CATEGORY, "file has no module docstring", range));
    } else {
      List<DocstringParseError> errors = new ArrayList<>();
      DocstringUtils.parseDocstring(moduleDocstring, errors);
      for (DocstringParseError error : errors) {
        issues.add(docstringParseErrorToIssue(moduleDocstring, error));
      }
    }
    super.visit(node);
  }

  @Override
  public void visit(ReturnStatement node) {
    if (node.getReturnExpression() != null) {
      containsReturnWithValue = true;
    }
  }

  @Override
  public void visit(FunctionDefStatement node) {
    containsReturnWithValue = false;
    super.visit(node);
    StringLiteral functionDocstring = extractDocstring(node.getStatements());
    if (functionDocstring == null
        && !node.getIdentifier().getName().startsWith("_")
        && countNestedStatements(node) >= FUNCTION_LENGTH_DOCSTRING_THRESHOLD) {
      Location start = Location.from(node.getLocation().getStartLineAndColumn());
      Location end;
      if (node.getStatements().isEmpty()) {
        // empty statement suites cannot come from the parser yet we should handle this gracefully:
        end = Location.from(node.getLocation().getEndLineAndColumn());
      } else {
        LineAndColumn lac = node.getStatements().get(0).getLocation().getStartLineAndColumn();
        end = new Location(lac.getLine(), lac.getColumn() - 1); // right before the first statement
      }
      String name = node.getIdentifier().getName();
      issues.add(
          new Issue(
              MISSING_FUNCTION_DOCSTRING_CATEGORY,
              "function '"
                  + name
                  + "' has no docstring"
                  + " (if this function is intended to be private,"
                  + " the name should start with an underscore: '_"
                  + name
                  + "')",
              new LocationRange(start, end)));
    }
    if (functionDocstring == null) {
      return;
    }
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = DocstringUtils.parseDocstring(functionDocstring, errors);
    for (DocstringParseError error : errors) {
      issues.add(docstringParseErrorToIssue(functionDocstring, error));
    }
    if (!info.isSingleLineDocstring()) {
      checkMultilineFunctionDocstring(
          node, functionDocstring, info, containsReturnWithValue, issues);
    }
    if (info.getArgumentsLocation() != null) {
      int lineOffset = functionDocstring.getLocation().getStartLine() - 1;
      issues.add(
          new Issue(
              ARGS_ARGUMENTS_DOCSTRING_CATEGORY,
              "Prefer 'Args:' to 'Arguments:' when documenting function arguments.",
              new LocationRange(
                  new Location(
                      info.getArgumentsLocation().start.line + lineOffset,
                      info.getArgumentsLocation().start.column),
                  new Location(
                      info.getArgumentsLocation().end.line + lineOffset,
                      info.getArgumentsLocation().end.column))));
    }
  }

  private static class StatementCounter extends SyntaxTreeVisitor {
    public int count = 0;

    @Override
    public void visitBlock(List<Statement> statements) {
      count += statements.size();
    }
  }

  private static int countNestedStatements(ASTNode node) {
    StatementCounter counter = new StatementCounter();
    counter.visit(node);
    return counter.count;
  }

  private static void checkMultilineFunctionDocstring(
      FunctionDefStatement functionDef,
      StringLiteral docstringLiteral,
      DocstringInfo docstring,
      boolean functionReturnsWithValue,
      List<Issue> issues) {
    if (functionReturnsWithValue && docstring.getReturns().isEmpty()) {
      issues.add(
          Issue.create(
              INCONSISTENT_DOCSTRING_CATEGORY,
              "incomplete docstring: the return value is not documented"
                  + " (no 'Returns:' section found)",
              docstringLiteral.getLocation()));
    }
    List<String> documentedParams = new ArrayList<>();
    for (ParameterDoc param : docstring.getParameters()) {
      documentedParams.add(param.getParameterName());
    }
    List<String> declaredParams = new ArrayList<>();
    for (Parameter<Expression, Expression> param : functionDef.getParameters()) {
      if (param.getName() != null) {
        String name = param.getName();
        if (param.isStar()) {
          name = "*" + name;
        }
        if (param.isStarStar()) {
          name = "**" + name;
        }
        declaredParams.add(name);
      }
    }
    checkParamListsMatch(docstringLiteral, documentedParams, declaredParams, issues);
  }

  private static void checkParamListsMatch(
      StringLiteral docstringLiteral,
      List<String> documentedParams,
      List<String> declaredParams,
      List<Issue> issues) {
    if (documentedParams.isEmpty() && !declaredParams.isEmpty()) {
      StringBuilder message =
          new StringBuilder("incomplete docstring: the function parameters are not documented")
              .append(" (no 'Args:' section found)\n")
              .append("The parameter documentation should look like this:\n\n")
              .append("Args:\n");
      for (String param : declaredParams) {
        message.append("  ").append(param).append(": ...\n");
      }
      message.append("\n");
      issues.add(
          Issue.create(
              INCONSISTENT_DOCSTRING_CATEGORY, message.toString(), docstringLiteral.getLocation()));
      return;
    }
    for (String param : declaredParams) {
      if (!documentedParams.contains(param)) {
        issues.add(
            Issue.create(
                INCONSISTENT_DOCSTRING_CATEGORY,
                "incomplete docstring: parameter '" + param + "' not documented",
                docstringLiteral.getLocation()));
      }
    }
    for (String param : documentedParams) {
      if (!declaredParams.contains(param)) {
        issues.add(
            Issue.create(
                INCONSISTENT_DOCSTRING_CATEGORY,
                "inconsistent docstring: parameter '"
                    + param
                    + "' appears in docstring but not in function signature",
                docstringLiteral.getLocation()));
      }
    }
    if (new LinkedHashSet<>(declaredParams).equals(new LinkedHashSet<>(documentedParams))
        && !declaredParams.equals(documentedParams)) {
      String message =
          "inconsistent docstring: order of parameters differs from function signature\n"
              + "Declaration order:   "
              + String.join(", ", declaredParams)
              + "\n"
              + "Documentation order: "
              + String.join(", ", documentedParams)
              + "\n";
      issues.add(
          Issue.create(INCONSISTENT_DOCSTRING_CATEGORY, message, docstringLiteral.getLocation()));
    }
  }

  private Issue docstringParseErrorToIssue(StringLiteral docstring, DocstringParseError error) {
    int startLine = docstring.getLocation().getStartLine() + error.getLineNumber() - 1;
    int startColumn;
    if (error.getLineNumber() == 1) {
      // The Skylark AST does not expose whether the string literal was a triple-quoted string, so
      // we just assume the most common case: triple-quoted docstrings.
      // There's also the possibility of a raw string (r'''docstring'''), in which case we would
      // have to add 4 to the column instead of 3.
      // TODO(skylark-team): Clean this up once the AST contains more information.
      startColumn = docstring.getLocation().getStartLineAndColumn().getColumn() + 3;
    } else {
      startColumn = 1;
    }
    Location start = new Location(startLine, startColumn);
    Location end = new Location(startLine, Math.max(1, startColumn + error.getLine().length() - 1));
    return new Issue(
        BAD_DOCSTRING_FORMAT_CATEGORY,
        "bad docstring format: " + error.getMessage(),
        new LocationRange(start, end));
  }
}
