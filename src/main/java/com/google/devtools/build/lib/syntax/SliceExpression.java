// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import javax.annotation.Nullable;

/** Syntax node for a slice expression, {@code object[start:stop:step]}. */
public final class SliceExpression extends Expression {

  private final Expression object;
  @Nullable private final Expression start;
  @Nullable private final Expression stop;
  @Nullable private final Expression step;

  SliceExpression(Expression object, Expression start, Expression stop, Expression step) {
    this.object = object;
    this.start = start;
    this.stop = stop;
    this.step = step;
  }

  public Expression getObject() {
    return object;
  }

  @Nullable
  public Expression getStart() {
    return start;
  }

  @Nullable
  public Expression getStop() {
    return stop;
  }

  @Nullable
  public Expression getStep() {
    return step;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.SLICE;
  }
}
