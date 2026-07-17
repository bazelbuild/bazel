// Copyright 2020 The Bazel Authors. All rights reserved.
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
package net.starlark.java.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import javax.annotation.Nullable;

/** A LambdaExpression ({@code lambda params: body}) denotes an anonymous function. */
public final class LambdaExpression extends Expression {

  private final int lambdaOffset; // offset of 'lambda' token
  private final ImmutableList<Parameter> parameters;
  private final Expression body;

  // set by resolver
  @Nullable private Resolver.Function resolved;

  LambdaExpression(
      FileLocations locs, int lambdaOffset, ImmutableList<Parameter> parameters, Expression body) {
    super(locs, Kind.LAMBDA);
    this.lambdaOffset = lambdaOffset;
    this.parameters = Preconditions.checkNotNull(parameters);
    this.body = Preconditions.checkNotNull(body);
  }

  public ImmutableList<Parameter> getParameters() {
    return parameters;
  }

  public Expression getBody() {
    return body;
  }

  /** Returns information about the resolved function. Set by the resolver. */
  @Nullable
  public Resolver.Function getResolvedFunction() {
    return resolved;
  }

  void setResolvedFunction(Resolver.Function resolved) {
    this.resolved = resolved;
  }

  @Override
  public int getStartOffset() {
    return lambdaOffset;
  }

  @Override
  public int getEndOffset() {
    return body.getEndOffset();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
