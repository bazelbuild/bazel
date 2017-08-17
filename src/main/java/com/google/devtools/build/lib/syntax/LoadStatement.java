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
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.util.Map;

/**
 * Syntax node for an import statement.
 */
public final class LoadStatement extends Statement {

  private final ImmutableMap<Identifier, String> symbolMap;
  private final StringLiteral imp;

  /**
   * Constructs an import statement.
   *
   * <p>{@code symbols} maps a symbol to the original name under which it was defined in
   * the bzl file that should be loaded. If aliasing is used, the value differs from its key's
   * {@code symbol.getName()}. Otherwise, both values are identical.
   */
  public LoadStatement(StringLiteral imp, Map<Identifier, String> symbolMap) {
    this.imp = imp;
    this.symbolMap = ImmutableMap.copyOf(symbolMap);
  }

  public ImmutableMap<Identifier, String> getSymbolMap() {
    return symbolMap;
  }

  public ImmutableList<Identifier> getSymbols() {
    return ImmutableList.copyOf(symbolMap.keySet());
  }

  public StringLiteral getImport() {
    return imp;
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    buffer.append("load(");
    imp.prettyPrint(buffer);
    for (Identifier symbol : symbolMap.keySet()) {
      buffer.append(", ");
      String origName = symbolMap.get(symbol);
      if (origName.equals(symbol.getName())) {
        buffer.append('"');
        symbol.prettyPrint(buffer);
        buffer.append('"');
      } else {
        symbol.prettyPrint(buffer);
        buffer.append("=\"");
        buffer.append(origName);
        buffer.append('"');
      }
    }
    buffer.append(")\n");
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    if (env.getSemantics().incompatibleLoadArgumentIsLabel) {
      String s = imp.getValue();
      if (!s.startsWith("//") && !s.startsWith(":")) {
        throw new EvalException(
            getLocation(),
            "First argument of 'load' must be a label and start with either '//' or ':'. "
                + "Use --incompatible_load_argument_is_label=false to temporarily disable this "
                + "check.");
      }
    }

    for (Map.Entry<Identifier, String> entry : symbolMap.entrySet()) {
      try {
        Identifier name = entry.getKey();
        Identifier declared = new Identifier(entry.getValue());

        if (declared.isPrivate()) {
          throw new EvalException(getLocation(),
              "symbol '" + declared.getName() + "' is private and cannot be imported.");
        }
        // The key is the original name that was used to define the symbol
        // in the loaded bzl file.
        env.importSymbol(imp.getValue(), name, declared.getName());
      } catch (Environment.LoadFailedException e) {
        throw new EvalException(getLocation(), e.getMessage());
      }
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    for (Identifier symbol : symbolMap.keySet()) {
      env.declare(symbol.getName(), getLocation());
    }
  }
}
