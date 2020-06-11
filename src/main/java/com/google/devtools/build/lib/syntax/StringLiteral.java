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

import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Syntax node for a string literal. */
public final class StringLiteral extends Expression {

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

  static final class StringLiteralCodec implements ObjectCodec<StringLiteral> {
    @Override
    public Class<? extends StringLiteral> getEncodedClass() {
      return StringLiteral.class;
    }

    @Override
    public void serialize(SerializationContext context, StringLiteral lit, CodedOutputStream out)
        throws SerializationException, IOException {
      // Enable de-duplication of strings during encoding.
      // The encoder does not intern and de-duplicate Strings by default,
      // though it does for all other objects;
      // see skyframe.serialization.strings.StringCodec.getStrategy.
      // If that were to change, we could delete StringLiteralCodec.
      // (One wonders why Identifier.name strings are not similarly de-duped,
      // as they are as numerous and more repetitive than string literals.)
      context.serializeWithAdHocMemoizationStrategy(
          lit.getValue(), MemoizationStrategy.MEMOIZE_AFTER, out);
      out.writeInt32NoTag(lit.startOffset);
      out.writeInt32NoTag(lit.endOffset);
      context.serialize(lit.locs, out);
    }

    @Override
    public StringLiteral deserialize(DeserializationContext context, CodedInputStream in)
        throws SerializationException, IOException {
      String value =
          context.deserializeWithAdHocMemoizationStrategy(in, MemoizationStrategy.MEMOIZE_AFTER);
      int startOffset = in.readInt32();
      int endOffset = in.readInt32();
      FileLocations locs = context.deserialize(in);
      return new StringLiteral(locs, startOffset, value, endOffset);
    }
  }
}
