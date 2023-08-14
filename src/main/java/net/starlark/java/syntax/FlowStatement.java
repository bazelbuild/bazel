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
package net.starlark.java.syntax;

/** A class for flow statements (break, continue, and pass) */
public final class FlowStatement extends Statement {

  private final TokenKind flowKind; // BREAK | CONTINUE | PASS
  private final int offset;

  /**
   * Constructs a new flow control statement.
   *
   * @param flowKind The specific kind of flow control statement (break, continue, or pass)
   */
  FlowStatement(FileLocations locs, TokenKind flowKind, int offset) {
    super(locs, Kind.FLOW);
    this.flowKind = flowKind;
    this.offset = offset;
  }

  public TokenKind getFlowKind() {
    return flowKind;
  }

  @Override
  public String toString() {
    return flowKind.toString() + "\n";
  }

  @Override
  public int getStartOffset() {
    return offset;
  }

  @Override
  public int getEndOffset() {
    return offset + flowKind.toString().length();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
