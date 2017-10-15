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
 * Syntax node for comments.
 */
public final class Comment extends ASTNode {

  protected final String value;

  public Comment(String value) {
    this.value = value;
  }

  public String getValue() {
    return value;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    // We can't really print comments in the right place anyway, due to how their relative order
    // is lost in the representation of BuildFileAST. So don't bother word-wrapping and just print
    // it on a single line.
    printIndent(buffer, indentLevel);
    buffer.append("# ");
    buffer.append(value);
  }

  @Override
  public String toString() {
    return value;
  }
}
