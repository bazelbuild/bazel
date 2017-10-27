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

import static com.google.devtools.skylark.skylint.DocstringUtils.extractDocstring;

import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Parameter;
import com.google.devtools.build.lib.syntax.ReturnStatement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import com.google.devtools.skylark.skylint.DocstringUtils.DocstringInfo;
import com.google.devtools.skylark.skylint.DocstringUtils.DocstringParseError;
import com.google.devtools.skylark.skylint.DocstringUtils.ParameterDoc;
import com.google.devtools.skylark.skylint.LocationRange.Location;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/** Checks the existence of docstrings. */
public class DocstringChecker extends SyntaxTreeVisitor {
  private static final String MISSING_DOCSTRING_CATEGORY = "missing-docstring";
  private static final String INCONSISTENT_DOCSTRING_CATEGORY = "inconsistent-docstring";
  private static final String BAD_DOCSTRING_FORMAT_CATEGORY = "bad-docstring-format";

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
      issues.add(new Issue(MISSING_DOCSTRING_CATEGORY, "file has no module docstring", range));
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
    if (functionDocstring == null && !node.getIdentifier().getName().startsWith("_")) {
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
              MISSING_DOCSTRING_CATEGORY,
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
  }

  private static void checkMultilineFunctionDocstring(
      FunctionDefStatement functionDef,
      StringLiteral docstringLiteral,
      DocstringInfo docstring,
      boolean functionReturnsWithValue,
      List<Issue> issues) {
    if (functionReturnsWithValue && docstring.returns.isEmpty()) {
      issues.add(
          Issue.create(
              INCONSISTENT_DOCSTRING_CATEGORY,
              "incomplete docstring: the return value is not documented"
                  + " (no 'Returns:' section found)",
              docstringLiteral.getLocation()));
    }
    List<String> documentedParams = new ArrayList<>();
    for (ParameterDoc param : docstring.parameters) {
      documentedParams.add(param.parameterName);
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
    int startLine = docstring.getLocation().getStartLine() + error.lineNumber - 1;
    int startColumn;
    if (error.lineNumber == 1) {
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
    Location end = new Location(startLine, Math.max(1, startColumn + error.line.length() - 1));
    return new Issue(
        BAD_DOCSTRING_FORMAT_CATEGORY,
        "bad docstring format: " + error.message,
        new LocationRange(start, end));
  }
}
