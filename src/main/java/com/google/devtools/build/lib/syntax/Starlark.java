// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.Pair;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * The Starlark class defines the most important entry points, constants, and functions needed by
 * all clients of the Starlark interpreter.
 */
// TODO(adonovan): move these here:
// UNIVERSE, None, len, truth, str, iterate, equal, compare, getattr, index,
// slice, parse, exec, eval, and so on.
public final class Starlark {

  private Starlark() {} // uninstantiable

  // TODO(adonovan):
  //
  // The code below shows the API that is the destination toward which all of the recent
  // tiny steps are headed. It doesn't work yet, but it helps to remember our direction.
  //
  // The API assumes that the "universe" portion (None, len, str) of the "predeclared" lexical block
  // is always available, so clients needn't mention it in the API. Starlark.UNIVERSE will expose it
  // as a public constant.
  //
  // Q. is there any value to returning the Module as opposed to just its global bindings as a Map?
  // The Go implementation does the latter and it works well.
  // This would allow the the Module class to be private.
  // The Bazel "Label" function, and various Bazel caller whitelists, depend on
  // being able to dig the label metadata out of a function's module,
  // but this could be addressed with a StarlarkFunction.getModuleLabel accessor.
  // A. The Module has an associated mutability (that of the thread),
  // and it might benefit from a 'freeze' method.
  // (But longer term, we might be able to eliminate Thread.mutability,
  // and the concept of a shared Mutability entirely, as in go.starlark.net.)
  //
  // Any FlagRestrictedValues among 'predeclared' and 'env' maps are implicitly filtered by the
  // semantics or thread.semantics.
  //
  // For exec(file), 'predeclared' corresponds exactly to the predeclared environment (sans
  // UNIVERSE) as described in the language spec. For eval(expr), 'env' is the complete environment
  // in which the expression is evaluated, which might include a mixture of predeclared, global,
  // file-local, and function-local variables, as when (for example) the debugger evaluates an
  // expression as if at a particular point in the source. As far as 'eval' is concerned, there is
  // no difference in kind between these bindings.
  //
  // The API does not rely on StarlarkThread acting as an environment, or on thread.globals.
  //
  // These functions could be implemented today with minimal effort.
  // The challenge is to migrate all the callers from the old API,
  // and in particular to reduce their assumptions about thread.globals,
  // which is going away.

  // ---- One shot execution API: parse, compile, and execute ----

  /**
   * Parse the input as a file, validate it in the specified predeclared environment, compile it,
   * and execute it. On success, the module is returned; on failure, it throws an exception.
   */
  public static Module exec(
      StarlarkThread thread, ParserInput input, Map<String, Object> predeclared)
      throws SyntaxError, EvalException, InterruptedException {
    // Pseudocode:
    // file = StarlarkFile.parse(input)
    // validateFile(file, predeclared.keys, thread.semantics)
    // prog = compile(file.statements)
    // module = new module(predeclared)
    // toplevel = new StarlarkFunction(prog.toplevel, module)
    // call(thread, toplevel)
    // return module  # or module.globals?
    throw new UnsupportedOperationException();
  }

  /**
   * Parse the input as an expression, validate it in the specified environment, compile it, and
   * evaluate it. On success, the expression's value is returned; on failure, it throws an
   * exception.
   */
  public static Object eval(StarlarkThread thread, ParserInput input, Map<String, Object> env)
      throws SyntaxError, EvalException, InterruptedException {
    // Pseudocode:
    // StarlarkFunction fn = exprFunc(input, env, thread.semantics)
    // return call(thread, fn)
    throw new UnsupportedOperationException();
  }

  /**
   * Parse the input as a file, validate it in the specified environment, compile it, and execute
   * it. If the final statement is an expression, return its value.
   *
   * <p>This complicated function, which combines exec and eval, is intended for use in a REPL or
   * debugger. In case of parse of validation error, it throws SyntaxError. In case of execution
   * error, the function returns partial results: the incomplete module plus the exception.
   *
   * <p>Assignments in the input act as updates to a new module created by this function, which is
   * returned.
   *
   * <p>In a typical REPL, the module bindings may be provided as predeclared bindings to the next
   * call.
   *
   * <p>In a typical debugger, predeclared might contain the complete environment at a particular
   * point in a running program, including its predeclared, global, and local variables. Assignments
   * in the debugger affect only the ephemeral module created by this call, not the values of
   * bindings observable by the debugged Starlark program. Thus execAndEval("x = 1; x + x") will
   * return a value of 2, and a module containing x=1, but it will not affect the value of any
   * variable named x in the debugged program.
   *
   * <p>A REPL will typically set the legacy "load binds globally" semantics flag, otherwise the
   * names bound by a load statement will not be visible in the next REPL chunk.
   */
  public static ModuleAndValue execAndEval(
      StarlarkThread thread, ParserInput input, Map<String, Object> predeclared)
      throws SyntaxError {
    // Pseudocode:
    // file = StarlarkFile.parse(input)
    // validateFile(file, predeclared.keys, thread.semantics)
    // prog = compile(file.statements + [return lastexpr])
    // module = new module(predeclared)
    // toplevel = new StarlarkFunction(prog.toplevel, module)
    // value = call(thread, toplevel)
    // return (module, value, error)  # or module.globals?
    throw new UnsupportedOperationException();
  }

  /**
   * The triple returned by {@link #execAndEval}. At most one of {@code value} and {@code error} is
   * set.
   */
  public static class ModuleAndValue {
    /** The module, containing global values from top-level assignments. */
    public Module module;
    /** The value of the final expression, if any, on success. */
    @Nullable public Object value;
    /** An EvalException or InterruptedException, if execution failed. */
    @Nullable public Exception error;
  }

  // ---- Two-stage API: compilation and execution are separate ---

  /**
   * Parse the input as a file, validates it in the specified predeclared environment (a set of
   * names, optionally filtered by the semantics), and compiles it to a Program. It throws
   * SyntaxError in case of scan/parse/validation error.
   *
   * <p>In addition to the program, it returns the validated syntax tree. This permits clients such
   * as Bazel to inspect the syntax (for BUILD dialect checks, glob prefetching, etc.)
   */
  public static Pair<Program, StarlarkFile> compileFile(
      ParserInput input, //
      Set<String> predeclared,
      StarlarkSemantics semantics)
      throws SyntaxError {
    // Pseudocode:
    // file = StarlarkFile.parse(input)
    // validateFile(file, predeclared.keys, thread.semantics)
    // prog = compile(file.statements)
    // return (prog, file)
    throw new UnsupportedOperationException();
  }

  /**
   * An opaque executable representation of a StarlarkFile. Programs may be efficiently serialized
   * and deserialized without parsing and recompiling.
   */
  public static class Program {

    /**
     * Execute the toplevel function of a compiled program and returns the module populated by its
     * top-level assignments.
     *
     * <p>The keys of predeclared must match the set used when creating the Program.
     */
    public Module init(
        StarlarkThread thread, //
        Map<String, Object> predeclared,
        @Nullable Object label) // a regrettable Bazelism we needn't widely expose in the API
        throws EvalException, InterruptedException {
      // Pseudocode:
      // module = new module(predeclared, label=label)
      // toplevel = new StarlarkFunction(prog.toplevel, module)
      // call(thread, toplevel)
      // return module # or module.globals?
      throw new UnsupportedOperationException();
    }
  }

  /**
   * Parse the input as an expression, validates it in the specified environment, and returns a
   * callable Starlark no-argument function value that computes and returns the value of the
   * expression.
   */
  private static StarlarkFunction exprFunc(
      ParserInput input, //
      Map<String, Object> env,
      StarlarkSemantics semantics)
      throws SyntaxError {
    // Pseudocode:
    // expr = Expression.parse(input)
    // validateExpr(expr, env.keys, semantics)
    // prog = compile([return expr])
    // module = new module(env)
    // return new StarlarkFunction(prog.toplevel, module)
    throw new UnsupportedOperationException();
  }
}
