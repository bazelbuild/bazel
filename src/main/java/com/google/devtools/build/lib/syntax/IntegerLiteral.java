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

import java.io.IOException;

/**
 * Syntax node for an integer literal.
 */
public final class IntegerLiteral extends Expression {
  private final int value;

  public IntegerLiteral(Integer value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }

  @Override
  Object doEval(Environment env) {
    return value;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    buffer.append(String.valueOf(value));
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }
}
