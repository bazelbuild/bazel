// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.syntax;

import java.io.IOException;

/**
 * A class for flow statements (e.g. break and continue)
 */
public final class FlowStatement extends Statement {
  // TODO(laurentlb): This conflicts with Statement.Kind, maybe remove it?
  public enum Kind {
    BREAK("break"),
    CONTINUE("continue");

    private final String name;

    private Kind(String name) {
      this.name = name;
    }

    public String getName() {
      return name;
    }
  }

  private final Kind kind;

  /**
   *
   * @param kind The label of the statement (either break or continue)
   */
  public FlowStatement(Kind kind) {
    this.kind = kind;
  }

  public Kind getKind() {
    return kind;
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    buffer.append(kind.name);
    buffer.append('\n');
  }

  @Override
  public String toString() {
    return kind.name + "\n";
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Statement.Kind kind() {
    return Statement.Kind.FLOW;
  }
}
