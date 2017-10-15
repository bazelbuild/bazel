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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.LValue;
import com.google.devtools.build.lib.syntax.Parameter;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Checks the adherence to Skylark naming conventions.
 *
 * <p>The convention is that functions, parameters and local variables should be lower_snake_case.
 */
// TODO(skylark-team): Also check for single-letter variable names that are easy to confuse (I, l,
// O)
// TODO(skylark-team): Allow CamelCase for providers, e.g. FooBar = provider(...)
// TODO(skylark-team): Local variables shouldn't start with an underscore, except '_' itself
// TODO(skylark-team): Check that UPPERCASE_VARIABLES are never mutated
public class NamingConventionsChecker extends SyntaxTreeVisitor {

  private final List<Issue> issues = new ArrayList<>();
  private boolean insideFunction = false;
  private boolean insideLvalue = false;
  // TODO(skylark-team): Store more symbol information than just the name (e.g. global/local)
  private final Set<String> alreadyReportedIdentifiers = new HashSet<>();

  public static List<Issue> check(BuildFileAST ast) {
    NamingConventionsChecker checker = new NamingConventionsChecker();
    checker.visit(ast);
    return checker.issues;
  }

  private void addAlreadyReportedIdentifier(String name) {
    alreadyReportedIdentifiers.add(name);
  }

  private void checkSnakeCase(String name, Location location) {
    if (!isSnakeCase(name) && !alreadyReportedIdentifiers.contains(name)) {
      issues.add(
          new Issue(
              "identifier '" + name + "' should be lower_snake_case or UPPER_SNAKE_CASE",
              location));
      addAlreadyReportedIdentifier(name);
    }
  }

  private void checkLowerSnakeCase(String name, Location location) {
    if (!isLowerSnakeCase(name) && !alreadyReportedIdentifiers.contains(name)) {
      issues.add(new Issue("identifier '" + name + "' should be lower_snake_case", location));
      addAlreadyReportedIdentifier(name);
    }
  }

  @Override
  public void visit(Identifier node) {
    // TODO(skylark-team): Maybe the lvalue contains a global variable?
    if (insideLvalue && insideFunction) {
      checkLowerSnakeCase(node.getName(), node.getLocation());
    } else if (insideLvalue) {
      checkSnakeCase(node.getName(), node.getLocation());
    }
  }

  @Override
  public void visit(Parameter<Expression, Expression> node) {
    String name = node.getName();
    if (name != null) {
      checkLowerSnakeCase(name, node.getLocation());
    }
  }

  @Override
  public void visit(FunctionDefStatement node) {
    insideFunction = true;
    Identifier funcIdent = node.getIdentifier();
    checkLowerSnakeCase(funcIdent.getName(), funcIdent.getLocation());
    super.visit(node);
    insideFunction = false;
    // TODO(skylark-team): Don't delete global variables
    alreadyReportedIdentifiers.clear();
  }

  @Override
  public void visit(LValue node) {
    insideLvalue = true;
    super.visit(node);
    insideLvalue = false;
  }

  private static boolean isSnakeCase(String name) {
    return isUpperSnakeCase(name) || isLowerSnakeCase(name);
  }

  private static boolean isUpperSnakeCase(String name) {
    return name.equals(name.toUpperCase());
  }

  private static boolean isLowerSnakeCase(String name) {
    return name.equals(name.toLowerCase());
  }
}
