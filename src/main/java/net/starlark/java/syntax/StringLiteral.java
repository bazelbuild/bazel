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
package net.starlark.java.syntax;

/** Syntax node for a string literal. */
public final class StringLiteral extends Expression {

  // See skyframe.serialization.StringLiteralCodec for custom serialization logic.

  private final int startOffset;
  private final String value;
  private final int endOffset;

  StringLiteral(FileLocations locs, int startOffset, String value, int endOffset) {
    super(locs);
    this.startOffset = startOffset;
    this.value = value;
    this.endOffset = endOffset;
  }

  /** Returns the value denoted by the string literal */
  public String getValue() {
    return value;
  }

  public Location getLocation() {
    return locs.getLocation(startOffset);
  }

  @Override
  public int getStartOffset() {
    return startOffset;
  }

  @Override
  public int getEndOffset() {
    // TODO(adonovan): when we switch to compilation,
    // making syntax trees ephemeral, we can afford to
    // record the raw literal. This becomes:
    //   return startOffset + raw.length().
    return endOffset;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.STRING_LITERAL;
  }

  // -- hooks to support Skyframe serialization without creating a dependency --

  /** Returns an opaque serializable object that may be passed to {@link #fromSerialization}. */
  public Object getFileLocations() {
    return locs;
  }

  /** Constructs a StringLiteral from its serialized components. */
  public static StringLiteral fromSerialization(
      Object fileLocations, int startOffset, String value, int endOffset) {
    return new StringLiteral((FileLocations) fileLocations, startOffset, value, endOffset);
  }
}
