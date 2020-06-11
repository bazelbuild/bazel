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

import com.google.common.collect.ImmutableList;

/** Syntax node for an import statement. */
public final class LoadStatement extends Statement {

  /**
   * Binding represents a binding in a load statement. load("...", local = "orig")
   *
   * <p>If there's no alias, a single Identifier can be used for both local and orig.
   * TODO(adonovan): don't do that; be faithful to source.
   */
  public static final class Binding {
    private final Identifier local;
    private final Identifier orig;

    public Identifier getLocalName() {
      return local;
    }

    public Identifier getOriginalName() {
      return orig;
    }

    Binding(Identifier localName, Identifier originalName) {
      this.local = localName;
      this.orig = originalName;
    }
  }

  private final int loadOffset;
  private final StringLiteral module;
  private final ImmutableList<Binding> bindings;
  private final int rparenOffset;

  LoadStatement(
      FileLocations locs,
      int loadOffset,
      StringLiteral module,
      ImmutableList<Binding> bindings,
      int rparenOffset) {
    super(locs);
    this.loadOffset = loadOffset;
    this.module = module;
    this.bindings = bindings;
    this.rparenOffset = rparenOffset;
  }

  public ImmutableList<Binding> getBindings() {
    return bindings;
  }

  public StringLiteral getImport() {
    return module;
  }

  @Override
  public int getStartOffset() {
    return loadOffset;
  }

  @Override
  public int getEndOffset() {
    return rparenOffset + 1;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.LOAD;
  }
}
