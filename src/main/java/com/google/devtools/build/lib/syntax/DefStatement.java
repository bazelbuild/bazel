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

import com.google.common.collect.ImmutableList;
import java.io.IOException;

/** Syntax node for a 'def' statement, which defines a function. */
public final class DefStatement extends Statement {

  private final Identifier identifier;
  private final FunctionSignature signature;
  private final ImmutableList<Statement> statements;
  private final ImmutableList<Parameter> parameters;

  DefStatement(
      Identifier identifier,
      Iterable<Parameter> parameters,
      FunctionSignature signature,
      Iterable<Statement> statements) {
    this.identifier = identifier;
    this.parameters = ImmutableList.copyOf(parameters);
    this.signature = signature;
    this.statements = ImmutableList.copyOf(statements);
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    appendSignatureTo(buffer);
    buffer.append('\n');
    printSuite(buffer, statements, indentLevel);
  }

  @Override
  public String toString() {
    // "def f(...): \n"
    StringBuilder buf = new StringBuilder();
    try {
      appendSignatureTo(buf);
    } catch (IOException ex) {
      throw new IllegalStateException(ex); // can't fail (StringBuilder)
    }
    buf.append(" ...\n");
    return buf.toString();
  }

  // Appends "def f(a, ..., z):" to the buffer.
  private void appendSignatureTo(Appendable buf) throws IOException {
    buf.append("def ");
    identifier.prettyPrint(buf);
    buf.append('(');
    String sep = "";
    for (Parameter param : parameters) {
      buf.append(sep);
      param.prettyPrint(buf);
      sep = ", ";
    }
    buf.append("):");
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
    return Kind.FUNCTION_DEF;
  }
}
