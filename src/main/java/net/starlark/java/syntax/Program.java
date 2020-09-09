// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

/**
 * An opaque, executable representation of a valid Starlark program. Programs may
 * [eventually---TODO(adonovan)] be efficiently serialized and deserialized without parsing and
 * recompiling.
 */
public final class Program {

  private final Resolver.Function body;
  private final ImmutableList<String> loads;

  private Program(Resolver.Function body, ImmutableList<String> loads) {
    // TODO(adonovan): compile here.
    this.body = body;
    this.loads = loads;
  }

  // TODO(adonovan): eliminate once Eval no longer needs access to syntax.
  public Resolver.Function getResolvedFunction() {
    return body;
  }

  /** Returns the file name of this compiled program. */
  public String getFilename() {
    return body.getLocation().file();
  }

  /** Returns the list of load strings of this compiled program, in source order. */
  public ImmutableList<String> getLoads() {
    return loads;
  }

  /**
   * Resolves a file syntax tree in the specified environment and compiles it to a Program. This
   * operation mutates the syntax tree, both by resolving identifiers and recording local variables,
   * and in case of error, by appending to {@code file.errors()}.
   *
   * @throws SyntaxError.Exception in case of resolution error, or if the syntax tree already
   *     contained syntax scan/parse errors. Resolution errors are added to {@code file.errors()}.
   */
  public static Program compileFile(StarlarkFile file, Resolver.Module env)
      throws SyntaxError.Exception {
    Resolver.resolveFile(file, env);
    if (!file.ok()) {
      throw new SyntaxError.Exception(file.errors());
    }
    return compileResolvedFile(file);
  }

  /** Variant of {@link #compileFile} for an already-resolved file syntax tree. */
  // TODO(adonovan): eliminate. This is a stop-gap because Bazel's Skyframe functions
  // are currently split as parse/resolve + compile/run, not parse/resolve/compile + run.
  public static Program compileResolvedFile(StarlarkFile file) {
    Preconditions.checkState(file.ok());

    // Extract load statements.
    ImmutableList.Builder<String> loads = ImmutableList.builder();
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        LoadStatement load = (LoadStatement) stmt;
        String module = load.getImport().getValue();
        loads.add(module);
      }
    }

    return new Program(file.getResolvedFunction(), loads.build());
  }

  /**
   * Resolves an expression syntax tree in the specified environment and compiles it to a Program.
   * This operation mutates the syntax tree. The {@code options} must match those used when parsing
   * expression.
   *
   * @throws SyntaxError.Exception in case of resolution error.
   */
  public static Program compileExpr(Expression expr, Resolver.Module module, FileOptions options)
      throws SyntaxError.Exception {
    Resolver.Function body = Resolver.resolveExpr(expr, module, options);
    return new Program(body, /*loads=*/ ImmutableList.of());
  }
}
