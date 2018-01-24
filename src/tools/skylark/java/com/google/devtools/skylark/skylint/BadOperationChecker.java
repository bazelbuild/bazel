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
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.AugmentedAssignmentStatement;
import com.google.devtools.build.lib.syntax.BinaryOperatorExpression;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.DictComprehension;
import com.google.devtools.build.lib.syntax.DictionaryLiteral;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.Operator;
import com.google.devtools.skylark.skylint.Environment.NameInfo;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Checks for operations that are deprecated */
public class BadOperationChecker extends AstVisitorWithNameResolution {
  private static final String DEPRECATED_PLUS_DEPSET_CATEGORY = "deprecated-plus-depset";
  private static final String DEPRECATED_PLUS_DICT_CATEGORY = "deprecated-plus-dict";
  private static final String DEPRECATED_PIPE_CATEGORY = "deprecated-pipe-dict";

  private final List<Issue> issues = new ArrayList<>();

  /**
   * Set of variables that (we assume) are depsets.
   * We consider x a depset variable if it appears in a statement of form `x = expr`, where
   * expr is either a call to `depset()` or else is itself a depset variable.
   */
  private final Set<Integer> depsetVariables = Sets.newHashSet();

  private BadOperationChecker() {}

  public static List<Issue> check(BuildFileAST ast) {
    BadOperationChecker checker = new BadOperationChecker();
    checker.visit(ast);
    return checker.issues;
  }

  /** Use heuristic to guess if a node is an expression of type depset. */
  private boolean isDepset(ASTNode node) {
    if (node instanceof Identifier) {
      NameInfo name = env.resolveName(((Identifier) node).getName());
      return name != null && depsetVariables.contains(name.id);
    }

    if (node instanceof FuncallExpression) {
      Expression function = ((FuncallExpression) node).getFunction();
      return function instanceof Identifier && ((Identifier) function).getName().equals("depset");
    }

    return false;
  }

  @Override
  public void visit(BinaryOperatorExpression node) {
    super.visit(node);
    if (node.getOperator() == Operator.PLUS) {
      if (node.getLhs() instanceof DictionaryLiteral
          || node.getLhs() instanceof DictComprehension
          || node.getRhs() instanceof DictionaryLiteral
          || node.getRhs() instanceof DictComprehension) {
        issues.add(
            Issue.create(
                DEPRECATED_PLUS_DICT_CATEGORY,
                "'+' operator is deprecated and should not be used on dictionaries",
                node.getLocation()));
      }

      if (isDepset(node.getLhs()) || isDepset(node.getRhs())) {
        issues.add(
            Issue.create(
                DEPRECATED_PLUS_DEPSET_CATEGORY,
                "'+' operator is deprecated and should not be used on depsets",
                node.getLocation()));
      }

    } else if (node.getOperator() == Operator.PIPE) {
      issues.add(
          Issue.create(
              DEPRECATED_PIPE_CATEGORY,
              "'|' operator is deprecated and should not be used. "
                  + "See https://docs.bazel.build/versions/master/skylark/depsets.html "
                  + "for the recommended use of depsets.",
              node.getLocation()));
    }
  }

  @Override
  public void visit(AugmentedAssignmentStatement node) {
    super.visit(node);
    if (node.getExpression() instanceof DictionaryLiteral
        || node.getExpression() instanceof DictComprehension) {
      issues.add(
          Issue.create(
              DEPRECATED_PLUS_DICT_CATEGORY,
              "'+=' operator is deprecated and should not be used on dictionaries",
              node.getLocation()));
    }

    Identifier ident = Iterables.getOnlyElement(node.getLValue().boundIdentifiers());
    if (isDepset(ident) || isDepset(node.getExpression())) {
      issues.add(
          Issue.create(
              DEPRECATED_PLUS_DEPSET_CATEGORY,
              "'+' operator is deprecated and should not be used on depsets",
              node.getLocation()));
    }
  }

  @Override
  public void visit(AssignmentStatement node) {
    super.visit(node);
    ImmutableSet<Identifier> lvalues = node.getLValue().boundIdentifiers();
    if (lvalues.size() != 1) {
      return;
    }
    Identifier ident = Iterables.getOnlyElement(lvalues);
    if (isDepset(node.getExpression())) {
      depsetVariables.add(env.resolveName(ident.getName()).id);
    }
  }
}
