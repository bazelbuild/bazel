// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;

/** Represents a type alias statement in the Starlark AST. */
public final class TypeAliasStatement extends Statement {
  private final int startOffset;
  private final Identifier identifier;
  private final ImmutableList<Identifier> parameters;
  private final Expression definition;

  TypeAliasStatement(
      FileLocations locs,
      int startOffset,
      Identifier identifier,
      ImmutableList<Identifier> parameters,
      Expression definition) {
    super(locs, Kind.TYPE_ALIAS);
    this.startOffset = startOffset;
    this.identifier = identifier;
    this.parameters = parameters;
    this.definition = definition;
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append("type ");
    buf.append(identifier.getName());
    if (!parameters.isEmpty()) {
      buf.append('[');
      ListExpression.appendNodes(buf, parameters);
      buf.append(']');
    }
    buf.append(" = ...\n");
    return buf.toString();
  }

  public Identifier getIdentifier() {
    return identifier;
  }

  public ImmutableList<Identifier> getParameters() {
    return parameters;
  }

  public Expression getDefinition() {
    return definition;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Note that this is the start offset of the statement's {@code type} keyword.
   */
  @Override
  public int getStartOffset() {
    return startOffset;
  }

  @Override
  public int getEndOffset() {
    return definition.getEndOffset();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
