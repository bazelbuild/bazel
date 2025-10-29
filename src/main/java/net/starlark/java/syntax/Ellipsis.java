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

/** Syntax node for the singleton ellipsis expression. */
public final class Ellipsis extends Expression {

  private final int startOffset;

  Ellipsis(FileLocations locs, int startOffset) {
    super(locs, Kind.ELLIPSIS);
    this.startOffset = startOffset;
  }

  @Override
  public int getStartOffset() {
    return startOffset;
  }

  @Override
  public int getEndOffset() {
    return startOffset + 3;
  }

  @Override
  public String toString() {
    return "...";
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
