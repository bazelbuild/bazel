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
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/** Checks the existence of docstrings. */
public class DocstringChecker extends SyntaxTreeVisitor {
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
      issues.add(new Issue("file has no module docstring", node.getLocation()));
    } else {
      List<DocstringParseError> errors = new ArrayList<>();
      parseDocstring(moduleDocstring, errors);
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
      issues.add(
          new Issue(
              "function '" + node.getIdentifier().getName() + "' has no docstring",
              node.getLocation()));
    }
    if (functionDocstring == null) {
      return;
    }
    List<DocstringParseError> errors = new ArrayList<>();
    DocstringInfo info = parseDocstring(functionDocstring, errors);
    for (DocstringParseError error : errors) {
      issues.add(docstringParseErrorToIssue(functionDocstring, error));
    }
    if (!info.isSingleLineDocstring()) {
      checkMultilineFunctionDocstring(
          node, functionDocstring, info, containsReturnWithValue, issues);
    }
  }

  private static DocstringInfo parseDocstring(
      StringLiteral functionDocstring, List<DocstringParseError> errors) {
    int indentation = functionDocstring.getLocation().getStartLineAndColumn().getColumn() - 1;
    return DocstringUtils.parseDocstring(functionDocstring.getValue(), indentation, errors);
  }

  private static void checkMultilineFunctionDocstring(
      FunctionDefStatement functionDef,
      StringLiteral docstringLiteral,
      DocstringInfo docstring,
      boolean functionReturnsWithValue,
      List<Issue> issues) {
    if (functionReturnsWithValue && docstring.returns.isEmpty()) {
      issues.add(
          new Issue(
              "incomplete docstring: the return value is not documented",
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
      issues.add(
          new Issue(
              "incomplete docstring: the function parameters are not documented",
              docstringLiteral.getLocation()));
      return;
    }
    for (String param : declaredParams) {
      if (!documentedParams.contains(param)) {
        issues.add(
            new Issue(
                "incomplete docstring: parameter '" + param + "' not documented",
                docstringLiteral.getLocation()));
      }
    }
    for (String param : documentedParams) {
      if (!declaredParams.contains(param)) {
        issues.add(
            new Issue(
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
      issues.add(new Issue(message, docstringLiteral.getLocation()));
    }
  }

  private Issue docstringParseErrorToIssue(StringLiteral docstring, DocstringParseError error) {
    LinterLocation loc =
        new LinterLocation(docstring.getLocation().getStartLine() + error.lineNumber - 1, 1);
    return new Issue("invalid docstring format: " + error.message, loc);
  }
}
