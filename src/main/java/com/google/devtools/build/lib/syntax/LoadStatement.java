// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/**
 * Syntax node for an import statement.
 */
public final class LoadStatement extends Statement {

  public static final String PATH_ERROR_MSG = "Path '%s' is not valid. "
      + "It should either start with a slash or refer to a file in the current directory.";
  private final ImmutableList<Ident> symbols;
  private final PathFragment importPath;
  private final String pathString;

  /**
   * Constructs an import statement.
   */
  LoadStatement(String path, List<Ident> symbols) {
    this.symbols = ImmutableList.copyOf(symbols);
    this.importPath = new PathFragment(path + ".bzl");
    this.pathString = path;
  }

  public ImmutableList<Ident> getSymbols() {
    return symbols;
  }

  public PathFragment getImportPath() {
    return importPath;
  }

  @Override
  public String toString() {
    return String.format("load(\"%s\", %s)", importPath, Joiner.on(", ").join(symbols));
  }

  @Override
  void exec(Environment env) throws EvalException, InterruptedException {
    for (Ident i : symbols) {
      try {
        if (i.getName().startsWith("_")) {
          throw new EvalException(getLocation(), "symbol '" + i + "' is private and cannot "
              + "be imported");
        }
        env.importSymbol(getImportPath(), i.getName());
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
    validateLoadPath();
    
    if (!importPath.isAbsolute() && importPath.segmentCount() > 1) {
      throw new EvalException(getLocation(), String.format(PATH_ERROR_MSG, importPath));
    }
    for (Ident symbol : symbols) {
      env.declare(symbol.getName(), getLocation());
    }
  }

  /**
   * Throws an exception if the path argument to load() does starts with more than one forward
   * slash ('/')
   *
   * @throws EvalException if the path is empty or starts with two forward slashes
   */
  public void validateLoadPath() throws EvalException {
    String error = null;

    if (pathString.isEmpty()) {
      error = "Path argument to load() must not be empty";
    } else if (pathString.startsWith("//")) {
      error =
          "First argument of load() is a path, not a label. "
          + "It should start with a single slash if it is an absolute path.";
    }

    if (error != null) {
      throw new EvalException(getLocation(), error);
    }
  }
}
