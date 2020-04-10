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


/** Syntax node for an integer literal. */
public final class IntegerLiteral extends Expression {
  private final String raw;
  private final int tokenOffset;
  private final int value;

  IntegerLiteral(FileLocations locs, String raw, int tokenOffset, int value) {
    super(locs);
    this.raw = raw;
    this.tokenOffset = tokenOffset;
    this.value = value;
  }

  public int getValue() {
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

  @Override
  public Kind kind() {
    return Kind.INTEGER_LITERAL;
  }
}
