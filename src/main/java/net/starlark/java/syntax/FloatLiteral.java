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

/** Syntax node for a float literal. */
public final class FloatLiteral extends Expression {
  private final String raw;
  private final int tokenOffset;
  private final double value;

  FloatLiteral(FileLocations locs, String raw, int tokenOffset, double value) {
    super(locs, Kind.FLOAT_LITERAL);
    this.raw = raw;
    this.tokenOffset = tokenOffset;
    this.value = value;
  }

  /** Returns the value denoted by this literal. */
  public double getValue() {
    return value;
  }

  /** Returns the raw source text of the literal. */
  public String getRaw() {
    return raw;
  }

  @Override
  public int getStartOffset() {
    return tokenOffset;
  }

  @Override
  public int getEndOffset() {
    return tokenOffset + raw.length();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
