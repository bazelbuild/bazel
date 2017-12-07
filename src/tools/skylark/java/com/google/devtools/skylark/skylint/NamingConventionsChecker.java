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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.skylark.skylint.Environment.NameInfo;
import com.google.devtools.skylark.skylint.Environment.NameInfo.Kind;
import java.util.ArrayList;
import java.util.List;

/**
 * Checks the adherence to Skylark naming conventions.
 *
 * <ul>
 *   <li>Functions and parameters should be lower_snake_case and all other identifiers should be
 *       lower_snake_case (for variables) or UPPER_SNAKE_CASE (for constants).
 *   <li>Providers are required to be UpperCamelCase. A variable FooBar is considered a provider if
 *       it appears in an assignment of the form "FooBar = provider(...)".
 *   <li>Shadowing of builtins (e.g. "True = False", "def fail()") is not allowed.
 *   <li>The single-letter variable names 'O', 'l', 'I' are disallowed since they're easy to
 *       confuse.
 *   <li>Multi-underscore names ('__', '___', etc.) are disallowed.
 *   <li>Single-underscore names may only be written to, as in "a, _ = tuple". They may not be read,
 *       as in "f(_)".
 * </ul>
 */
// TODO(skylark-team): Check that UPPERCASE_VARIABLES are never mutated
public class NamingConventionsChecker extends AstVisitorWithNameResolution {
  private static final String NAME_WITH_WRONG_CASE_CATEGORY = "name-with-wrong-case";
  private static final String PROVIDER_NAME_ENDS_IN_INFO_CATEGORY = "provider-name-suffix";
  private static final String CONFUSING_NAME_CATEGORY = "confusing-name";
  private static final ImmutableList<String> CONFUSING_NAMES = ImmutableList.of("O", "I", "l");
  private static final ImmutableSet<String> BUILTIN_NAMES;

  private final List<Issue> issues = new ArrayList<>();

  static {
    Environment env = Environment.defaultBazel();
    BUILTIN_NAMES =
        env.getNameIdsInCurrentBlock()
            .stream()
            .map(id -> env.getNameInfo(id).name)
            .collect(ImmutableSet.toImmutableSet());
  }

  public static List<Issue> check(BuildFileAST ast) {
    NamingConventionsChecker checker = new NamingConventionsChecker();
    checker.visit(ast);
    return checker.issues;
  }

  @Override
  public void visit(AssignmentStatement node) {
    // Check for the pattern "FooBar = provider(...)" because CamelCase for provider names is OK
    Expression lvalue = node.getLValue().getExpression();
    Expression rhs = node.getExpression();
    if (lvalue instanceof Identifier && rhs instanceof FuncallExpression) {
      Expression function = ((FuncallExpression) rhs).getFunction();
      if (function instanceof Identifier && ((Identifier) function).getName().equals("provider")) {
        checkProviderName(((Identifier) lvalue).getName(), lvalue.getLocation());
        visit(rhs);
        return;
      }
    }
    super.visit(node);
  }

  private void checkSnakeCase(String name, Location location) {
    if (!isSnakeCase(name)) {
      issues.add(
          Issue.create(
              NAME_WITH_WRONG_CASE_CATEGORY,
              "identifier '"
                  + name
                  + "' should be lower_snake_case (for variables)"
                  + " or UPPER_SNAKE_CASE (for constants)",
              location));
    }
  }

  private void checkLowerSnakeCase(String name, Location location) {
    if (!isLowerSnakeCase(name)) {
      issues.add(
          Issue.create(
              NAME_WITH_WRONG_CASE_CATEGORY,
              "identifier '" + name + "' should be lower_snake_case",
              location));
    }
  }

  private void checkProviderName(String name, Location location) {
    if (!isUpperCamelCase(name)) {
      issues.add(
          Issue.create(
              NAME_WITH_WRONG_CASE_CATEGORY,
              "provider name '" + name + "' should be UpperCamelCase",
              location));
    }
    if (!name.endsWith("Info")) {
      issues.add(
          Issue.create(
              PROVIDER_NAME_ENDS_IN_INFO_CATEGORY,
              "provider name '" + name + "' should end in the suffix 'Info'",
              location)
          );
    }
  }

  private void checkNameNotConfusing(String name, Location location) {
    if (CONFUSING_NAMES.contains(name)) {
      issues.add(
          Issue.create(
              CONFUSING_NAME_CATEGORY,
              "never use 'l', 'I', or 'O' as names "
                  + "(they're too easily confused with 'I', 'l', or '0')",
              location));
    }
    if (BUILTIN_NAMES.contains(name)) {
      issues.add(
          Issue.create(
              CONFUSING_NAME_CATEGORY,
              "identifier '" + name + "' shadows a builtin; please pick a different name",
              location));
    }
    if (name.chars().allMatch(c -> c == '_') && name.length() >= 2) {
      issues.add(
          Issue.create(
              CONFUSING_NAME_CATEGORY,
              "identifier '"
                  + name
                  + "' consists only of underscores; please pick a different name",
              location));
    }
  }

  @Override
  void use(Identifier identifier) {
    if (identifier.getName().equals("_")) {
      issues.add(
          Issue.create(
              CONFUSING_NAME_CATEGORY,
              "don't use '_' as an identifier, only to ignore the result in an assignment",
              identifier.getLocation()));
    }
  }

  @Override
  void declare(String name, ASTNode node) {
    NameInfo nameInfo = env.resolveExistingName(name);
    if (nameInfo.kind == Kind.IMPORTED) {
      // Users may not have control over imported names, so ignore them:
      return;
    }
    checkNameNotConfusing(name, node.getLocation());
    if (nameInfo.kind == Kind.PARAMETER || nameInfo.kind == Kind.FUNCTION) {
      checkLowerSnakeCase(nameInfo.name, node.getLocation());
    } else {
      checkSnakeCase(nameInfo.name, node.getLocation());
    }
  }

  private static boolean isUpperCamelCase(String name) {
    if (name.startsWith("_")) {
      name = name.substring(1); // private providers are allowed
    }
    return !name.contains("_") && Character.isUpperCase(name.charAt(0));
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
