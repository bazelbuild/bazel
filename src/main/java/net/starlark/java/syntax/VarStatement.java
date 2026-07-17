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

import javax.annotation.Nullable;

/**
 * Syntax node for a variable type annotation appearing as its own statement ({@code foo : int}), as
 * opposed to in an assignment statement where there's an initializer on the right-hand side.
 *
 * <p>(The name of this class is meant to be reminiscent of the `var` keyword that some languages
 * use, although Python and Starlark have no special keyword for variable declarations.)
 */
public final class VarStatement extends Statement {

  private final Identifier identifier;

  private final Expression type;

  @Nullable private final DocComments docComments;

  /** Constructs a {@code VarStatement}. */
  VarStatement(
      FileLocations locs,
      Identifier identifier,
      Expression type,
      @Nullable DocComments docComments) {
    super(locs, Kind.VAR);
    this.identifier = identifier;
    this.type = type;
    this.docComments = docComments;
  }

  @Override
  public int getStartOffset() {
    return identifier.getStartOffset();
  }

  @Override
  public int getEndOffset() {
    return type.getEndOffset();
  }

  /** Returns the variable being declared and annotated. */
  public Identifier getIdentifier() {
    return identifier;
  }

  /** Returns the type expression associated with the variable. */
  public Expression getType() {
    return type;
  }

  /** Returns the Sphinx autodoc-style doc comments attached to this statement, if any. */
  @Nullable
  public DocComments getDocComments() {
    return docComments;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
