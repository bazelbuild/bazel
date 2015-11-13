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

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.LoopLabels;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.vfs.PathFragment;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

import java.util.Map;

/**
 * Syntax node for an import statement.
 */
public final class LoadStatement extends Statement {

  private final ImmutableMap<Identifier, String> symbols;
  private final ImmutableList<Identifier> cachedSymbols; // to save time
  private final PathFragment importPath;
  private final StringLiteral pathString;

  /**
   * Constructs an import statement.
   *
   * <p>{@code symbols} maps a symbol to the original name under which it was defined in
   * the bzl file that should be loaded. If aliasing is used, the value differs from its key's
   * {@code symbol.getName()}. Otherwise, both values are identical.
   */
  LoadStatement(StringLiteral path, Map<Identifier, String> symbols) {
    this.symbols = ImmutableMap.copyOf(symbols);
    this.cachedSymbols = ImmutableList.copyOf(symbols.keySet());
    this.importPath = new PathFragment(path.getValue() + ".bzl");
    this.pathString = path;
  }

  public ImmutableList<Identifier> getSymbols() {
    return cachedSymbols;
  }

  public PathFragment getImportPath() {
    return importPath;
  }

  @Override
  public String toString() {
    return String.format("load(\"%s\", %s)", importPath, Joiner.on(", ").join(cachedSymbols));
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    for (Map.Entry<Identifier, String> entry : symbols.entrySet()) {
      try {
        Identifier current = entry.getKey();

        if (current.isPrivate()) {
          throw new EvalException(
              getLocation(), "symbol '" + current + "' is private and cannot be imported.");
        }
        // The key is the original name that was used to define the symbol
        // in the loaded bzl file.
        env.importSymbol(getImportPath(), current, entry.getValue());
      } catch (Environment.NoSuchVariableException | Environment.LoadFailedException e) {
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
    validatePath();
    for (Identifier symbol : cachedSymbols) {
      env.declare(symbol.getName(), getLocation());
    }
  }

  public StringLiteral getPath() {
    return pathString;
  }

  /**
   * Throws an exception if the path argument to load() starts with more than one forward
   * slash ('/')
   */
  public void validatePath() throws EvalException {
    String error = null;

    if (pathString.getValue().isEmpty()) {
      error = "Path argument to load() must not be empty.";
    } else if (pathString.getValue().startsWith("//")) {
      error =
          "First argument of load() is a path, not a label. "
          + "It should start with a single slash if it is an absolute path.";
    } else if (!importPath.isAbsolute() && importPath.segmentCount() > 1) {
      error = String.format(
          "Path '%s' is not valid. "
              + "It should either start with a slash or refer to a file in the current directory.",
          importPath);
    }

    if (error != null) {
      throw new EvalException(getLocation(), error);
    }
  }

  @Override
  ByteCodeAppender compile(
      VariableScope scope, Optional<LoopLabels> loopLabels, DebugInfo debugInfo) {
    throw new UnsupportedOperationException(
        "load statements should never appear in method bodies and"
            + " should never be compiled in general");
  }

  public boolean isAbsolute() {
    return getImportPath().isAbsolute();
  }
}
