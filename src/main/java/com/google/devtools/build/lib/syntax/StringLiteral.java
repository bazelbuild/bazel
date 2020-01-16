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

  private final String value;

  StringLiteral(String value) {
    this.value = value;
  }

  /** Returns the value denoted by the string literal */
  public String getValue() {
    return value;
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
    public void serialize(
        SerializationContext context, StringLiteral stringLiteral, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      // The String instances referred to by StringLiterals are deduped by Parser, so therefore
      // memoization is guaranteed to be profitable.
      context.serializeWithAdHocMemoizationStrategy(
          stringLiteral.getValue(), MemoizationStrategy.MEMOIZE_AFTER, codedOut);
      context.serialize(stringLiteral.getStartLocation(), codedOut);
    }

    @Override
    public StringLiteral deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      String value =
          context.deserializeWithAdHocMemoizationStrategy(
              codedIn, MemoizationStrategy.MEMOIZE_AFTER);
      Lexer.LexerLocation location = context.deserialize(codedIn);
      StringLiteral stringLiteral = new StringLiteral(value);
      stringLiteral.setLocation(location);
      return stringLiteral;
    }
  }
}
