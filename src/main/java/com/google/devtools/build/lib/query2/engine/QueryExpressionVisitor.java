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
package com.google.devtools.build.lib.query2.engine;

/** Provides interfaces to visit all {@link QueryExpression}s with a context object. */
public interface QueryExpressionVisitor<T, C> {
  T visit(TargetLiteral targetLiteral, C context);

  T visit(BinaryOperatorExpression binaryOperatorExpression, C context);

  T visit(FunctionExpression functionExpression, C context);

  T visit(LetExpression letExpression, C context);

  T visit(SetExpression setExpression, C context);
}
