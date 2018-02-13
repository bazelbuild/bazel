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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.syntax.Argument;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.ListLiteral;
import com.google.devtools.build.lib.syntax.StringLiteral;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Checks the adherence to Skylark best practices for recursive globs.
 *
 * <p>Recursive globs should be used sparingly and not for files containing source code. This check
 * flags incorrect usage of recurisive globs, for languages known to be problematic.
 */
public class NativeRecursiveGlobChecker extends AstVisitorWithNameResolution {

  /* List of instances of glob(**) we found. */
  private final List<Issue> issues = new ArrayList<>();

  /* List of known variables we've encountered for finding indirect use of glob(**) */
  private final Map<Identifier, Expression> vars = new HashMap<>();

  /* List of variables that were found in globs, but had not yet been resolved in SkyLark
   * processing.  Example:
   *
   * native.glob([my_var])
   *
   * ...
   *
   * my_var = "**\/*.java"
   */
  private final Set<Identifier> waitingFor = new HashSet<>();

  private static final String BAD_RECURSIVE_GLOB = "bad-recursive-glob";

  public static List<Issue> check(BuildFileAST ast) {
    NativeRecursiveGlobChecker checker = new NativeRecursiveGlobChecker();
    checker.visit(ast);
    return checker.issues;
  }

  private void evaluateInclude(Expression exp) {
    if (exp.kind() == Expression.Kind.STRING_LITERAL) {
      StringLiteral str = (StringLiteral) exp;
      String value = str.getValue();
      if (value.contains("**") && value.endsWith("*.java")) {
        issues.add(
            Issue.create(
                BAD_RECURSIVE_GLOB,
                "go/build-style#globs "
                    + "Do not use recursive globs for Java source files. glob() on multiple "
                    + "directories is error prone and can cause serious maintenance problems for "
                    + "BUILD files.", exp.getLocation()));
      }
    } else if (exp.kind() == Expression.Kind.IDENTIFIER) {
      Identifier id = (Identifier) exp;
      if (vars.containsKey(id)) {
        evaluateInclude(vars.get(id));
      } else {
        waitingFor.add(id);
      }
    }
  }

  @Override
  public void visit(FuncallExpression node) {
    if (node.getFunction().toString().equals("glob")) {
      Argument.Passed include = null;
      int index = 0;
      List<Argument.Passed> args = node.getArguments();
      for (Argument.Passed a : args) {
        if (index == 0 && a.isPositional()) {
          include = a;
          break;
        } else if (index > 1
            && a.isKeyword()
            && (a.getName() != null && a.getName().equals("include"))) {
          include = a;
          break;
        }
        index++;
      }
      if (include != null && include.getValue().kind() == Expression.Kind.LIST_LITERAL) {
        ListLiteral list = (ListLiteral) include.getValue();
        for (Expression exp : list.getElements()) {
          evaluateInclude(exp);
        }
      }
    }
    super.visit(node);
  }

  @Override
  public void visit(AssignmentStatement node) {
    super.visit(node);
    ImmutableSet<Identifier> lvalues = node.getLValue().boundIdentifiers();
    if (lvalues.size() != 1) {
      return;
    }
    Identifier ident = Iterables.getOnlyElement(lvalues);
    vars.put(ident, node.getExpression());

    if (waitingFor.contains(ident)) {
      evaluateInclude(node.getExpression());
    }
  }
}
