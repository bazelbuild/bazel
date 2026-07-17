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

/** Syntax node for isinstance() expressions. */
public final class IsInstanceExpression extends Expression {
  private final int startOffset;
  private final Expression value;
  private final Expression type;
  private final int rparenOffset;

  IsInstanceExpression(
      FileLocations locs, int startOffset, Expression value, Expression type, int rparenOffset) {
    super(locs, Kind.ISINSTANCE);
    this.startOffset = startOffset;
    this.value = value;
    this.type = type;
    this.rparenOffset = rparenOffset;
  }

  @Override
  public int getStartOffset() {
    return startOffset;
  }

  @Override
  public int getEndOffset() {
    return rparenOffset + 1;
  }

  public Expression getValue() {
    return value;
  }

  public Expression getType() {
    return type;
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append("isinstance(");
    buf.append(value);
    buf.append(", ");
    buf.append(type);
    buf.append(')');
    return buf.toString();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
