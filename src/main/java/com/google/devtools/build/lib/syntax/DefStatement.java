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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

/** Syntax node for a 'def' statement, which defines a function. */
public final class DefStatement extends Statement {

  private final int defOffset;
  private final Identifier identifier;
  private final FunctionSignature signature;
  private final ImmutableList<Statement> body; // non-empty if well formed
  private final ImmutableList<Parameter> parameters;

  DefStatement(
      FileLocations locs,
      int defOffset,
      Identifier identifier,
      ImmutableList<Parameter> parameters,
      FunctionSignature signature,
      ImmutableList<Statement> body) {
    super(locs);
    this.defOffset = defOffset;
    this.identifier = identifier;
    this.parameters = Preconditions.checkNotNull(parameters);
    this.signature = Preconditions.checkNotNull(signature);
    this.body = Preconditions.checkNotNull(body);
  }

  @Override
  public String toString() {
    // "def f(...): \n"
    StringBuilder buf = new StringBuilder();
    new NodePrinter(buf).printDefSignature(this);
    buf.append(" ...\n");
    return buf.toString();
  }

  public Identifier getIdentifier() {
    return identifier;
  }

  // TODO(adonovan): rename to getBody.
  public ImmutableList<Statement> getStatements() {
    return body;
  }

  public ImmutableList<Parameter> getParameters() {
    return parameters;
  }

  FunctionSignature getSignature() {
    return signature;
  }

  @Override
  public int getStartOffset() {
    return defOffset;
  }

  @Override
  public int getEndOffset() {
    return body.isEmpty()
        ? identifier.getEndOffset() // wrong, but tree is ill formed
        : body.get(body.size() - 1).getEndOffset();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.DEF;
  }
}
