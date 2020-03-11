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

  private final Identifier identifier;
  private final FunctionSignature signature;
  private final ImmutableList<Statement> statements;
  private final ImmutableList<Parameter> parameters;

  DefStatement(
      Identifier identifier,
      ImmutableList<Parameter> parameters,
      FunctionSignature signature,
      ImmutableList<Statement> statements) {
    this.identifier = identifier;
    this.parameters = Preconditions.checkNotNull(parameters);
    this.signature = signature;
    this.statements = Preconditions.checkNotNull(statements);
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

  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public ImmutableList<Parameter> getParameters() {
    return parameters;
  }

  public FunctionSignature getSignature() {
    return signature;
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
