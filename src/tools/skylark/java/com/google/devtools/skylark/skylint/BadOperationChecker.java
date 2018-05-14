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
import com.google.common.collect.Maps;
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
import java.util.Map;
import javax.annotation.Nullable;

/** Checks for operations that are deprecated */
public class BadOperationChecker extends AstVisitorWithNameResolution {
  private static final String DEPRECATED_PLUS_DEPSET_CATEGORY = "deprecated-plus-depset";
  private static final String DEPRECATED_PLUS_DICT_CATEGORY = "deprecated-plus-dict";
  private static final String DEPRECATED_PIPE_CATEGORY = "deprecated-pipe-dict";
  private static final String DEPRECATED_DIVISION_CATEGORY = "deprecated-division";

  private final List<Issue> issues = new ArrayList<>();

  enum NodeType {
    DEPSET,
    DICT
  }

  /**
   * An inferred map of variables mapped to their type.
   * We consider x an 'inferred-type' variable if it appears in a statement of form `x = expr`,
   * where expr is either a trivially-typed constructor (e.g. 'depset()') or is itself an
   * inferred-type variable.
   *
   * In principle, it's possible that a variable is reassigned to a different type and that
   * this map is therefore inaccurate. In practice, however, that's a fairly uncommon and
   * error-prone pattern.
   */
  private final Map<Integer, NodeType> inferredTypeVariables = Maps.newHashMap();

  private BadOperationChecker() {}

  public static List<Issue> check(BuildFileAST ast) {
    BadOperationChecker checker = new BadOperationChecker();
    checker.visit(ast);
    return checker.issues;
  }

  /**
   * If the given node is an inferred-type {@link NodeType}, return that type.
   * Otherwise return null.
   */
  @Nullable
  private NodeType getInferredTypeOrNull(ASTNode node) {
    if (node instanceof Identifier) {
      NameInfo name = env.resolveName(((Identifier) node).getName());
      return name != null ? inferredTypeVariables.get(name.id) : null;
    }

    if (node instanceof FuncallExpression) {
      Expression function = ((FuncallExpression) node).getFunction();
      if (function instanceof Identifier) {
        if (((Identifier) function).getName().equals("depset")) {
          return NodeType.DEPSET;
        } else if (((Identifier) function).getName().equals("dict")) {
          return NodeType.DICT;
        }
      }
    }
    if (node instanceof DictionaryLiteral || node instanceof DictComprehension) {
      return NodeType.DICT;
    }
    return null;
  }

  @Override
  public void visit(BinaryOperatorExpression node) {
    super.visit(node);
    NodeType lhsNodeType = getInferredTypeOrNull(node.getLhs());
    NodeType rhsNodeType = getInferredTypeOrNull(node.getRhs());

    if (node.getOperator() == Operator.PLUS) {
      if (lhsNodeType == NodeType.DICT || rhsNodeType == NodeType.DICT) {
        issues.add(
            Issue.create(
                DEPRECATED_PLUS_DICT_CATEGORY,
                "'+' operator is deprecated and should not be used on dictionaries",
                node.getLocation()));
      }
      if (lhsNodeType == NodeType.DEPSET || rhsNodeType == NodeType.DEPSET) {
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
    } else if (node.getOperator() == Operator.DIVIDE) {
      issues.add(
          Issue.create(
              DEPRECATED_DIVISION_CATEGORY,
              "'/' operator is deprecated and should not be used. "
                  + "Use '//' instead (like in Python for floor division).",
              node.getLocation()));
    }
  }

  @Override
  public void visit(AugmentedAssignmentStatement node) {
    super.visit(node);
    Identifier ident = Iterables.getOnlyElement(node.getLValue().boundIdentifiers());
    NodeType identType = getInferredTypeOrNull(ident);
    NodeType expressionType = getInferredTypeOrNull(node.getExpression());
    if (identType == NodeType.DICT || expressionType == NodeType.DICT) {
      issues.add(
          Issue.create(
              DEPRECATED_PLUS_DICT_CATEGORY,
              "'+=' operator is deprecated and should not be used on dictionaries",
              node.getLocation()));
    }
    if (identType == NodeType.DEPSET || expressionType == NodeType.DEPSET) {
      issues.add(
          Issue.create(
              DEPRECATED_PLUS_DEPSET_CATEGORY,
              "'+=' operator is deprecated and should not be used on depsets",
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
    NodeType inferredType = getInferredTypeOrNull(node.getExpression());

    if (inferredType != null) {
      inferredTypeVariables.put(env.resolveName(ident.getName()).id, inferredType);
    }
  }
}
