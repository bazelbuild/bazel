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
import java.io.IOException;
import java.io.Serializable;
import java.util.List;

/** Syntax node for an import statement. */
public final class LoadStatement extends Statement {

  /**
   * Binding represents a binding in a load statement. load("...", local = "orig")
   *
   * <p>If there's no alias, a single Identifier can be used for both local and orig.
   */
  public static final class Binding implements Serializable {
    private final Identifier local;
    private final Identifier orig;

    public Identifier getLocalName() {
      return local;
    }

    public Identifier getOriginalName() {
      return orig;
    }

    public Binding(Identifier localName, Identifier originalName) {
      this.local = localName;
      this.orig = originalName;
    }
  }

  private final ImmutableList<Binding> bindings;
  private final StringLiteral imp;

  /**
   * Constructs an import statement.
   *
   * <p>{@code symbols} maps a symbol to the original name under which it was defined in the bzl
   * file that should be loaded. If aliasing is used, the value differs from its key's {@code
   * symbol.getName()}. Otherwise, both values are identical.
   */
  public LoadStatement(StringLiteral imp, List<Binding> bindings) {
    this.imp = imp;
    this.bindings = ImmutableList.copyOf(bindings);
  }

  public ImmutableList<Binding> getBindings() {
    return bindings;
  }

  public StringLiteral getImport() {
    return imp;
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    buffer.append("load(");
    imp.prettyPrint(buffer);
    for (Binding binding : bindings) {
      buffer.append(", ");
      Identifier local = binding.getLocalName();
      String origName = binding.getOriginalName().getName();
      if (origName.equals(local.getName())) {
        buffer.append('"');
        local.prettyPrint(buffer);
        buffer.append('"');
      } else {
        local.prettyPrint(buffer);
        buffer.append("=\"");
        buffer.append(origName);
        buffer.append('"');
      }
    }
    buffer.append(")\n");
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.LOAD;
  }
}
