// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.engine;

import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;

/**
 * A {@link QueryExpressionVisitor} that cheaply estimates the size of a {@link QueryExpression}
 * without stringify-ing it.
 */
public class TotalWeightQueryExpressionVisitor implements QueryExpressionVisitor<Long, Void> {

  @Override
  public Long visit(TargetLiteral targetLiteral, Void context) {
    return (long) targetLiteral.getPattern().length();
  }

  @Override
  public Long visit(BinaryOperatorExpression binaryOperatorExpression, Void context) {
    long totalWeight = 0L;
    for (QueryExpression operand : binaryOperatorExpression.getOperands()) {
      totalWeight += operand.accept(this);
    }
    return totalWeight;
  }

  @Override
  public Long visit(FunctionExpression functionExpression, Void context) {
    long totalWeight = 0L;
    for (Argument arg : functionExpression.getArgs()) {
      totalWeight +=
          switch (arg.getType()) {
            case WORD -> arg.getWord().length();
            case INTEGER -> 1L;
            case EXPRESSION -> arg.getExpression().accept(this);
          };
    }
    return totalWeight;
  }

  @Override
  public Long visit(LetExpression letExpression, Void context) {
    return letExpression.getVarName().length()
        + letExpression.getVarExpr().accept(this)
        + letExpression.getBodyExpr().accept(this);
  }

  @Override
  public Long visit(SetExpression setExpression, Void context) {
    long totalWeight = 0L;
    for (TargetLiteral word : setExpression.getWords()) {
      totalWeight += word.getPattern().length();
    }
    return totalWeight;
  }
}
