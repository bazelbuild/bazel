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
package net.starlark.java.eval;

import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.Resolver;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;

/**
 * Examples of typical API usage of the Starlark interpreter.<br>
 * This is not a test, but it is checked by the compiler.
 */
final class Examples {

  /**
   * This example reads, parses, compiles, and executes a Starlark file. It returns the module,
   * which holds the values of global variables.
   */
  Module execFile(String filename)
      throws IOException, SyntaxError.Exception, EvalException, InterruptedException {
    // Read input from the named file.
    ParserInput input = ParserInput.readFile(filename);

    // Create the module that will be populated by executing the file.
    // It holds the global variables, initially empty.
    // Its predeclared environment defines only the standard builtins:
    // None, True, len, and so on.
    Module module = Module.create();

    // Resolve, compile, and execute the file.
    //
    // The Mutability will be associated with all the values created by this thread.
    // The try-with-resources statement ensures that all values become frozen
    // after execution.
    try (Mutability mu = Mutability.create(input.getFile())) {
      StarlarkThread thread = new StarlarkThread(mu, StarlarkSemantics.DEFAULT);
      Starlark.execFile(input, FileOptions.DEFAULT, module, thread);
    }

    return module;
  }

  /**
   * This example evaluates a Starlark expression in the specified environment and returns its
   * value.
   */
  Object evalExpr(String expr, ImmutableMap<String, Object> env)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    // The apparent file name (for error messages) will be "<expr>".
    ParserInput input = ParserInput.fromString(expr, "<expr>");

    // Create the module in which the expression is evaluated.
    // It may define additional predeclared environment bindings.
    Module module = Module.withPredeclared(StarlarkSemantics.DEFAULT, env);

    // Resolve, compile, and execute the expression.
    try (Mutability mu = Mutability.create(input.getFile())) {
      StarlarkThread thread = new StarlarkThread(mu, StarlarkSemantics.DEFAULT);
      return Starlark.eval(input, FileOptions.DEFAULT, module, thread);
    }
  }

  /**
   * This advanced example reads, parses, and compiles a Starlark file to a Program, then later
   * executes it.
   */
  Module compileThenExecute()
      throws IOException, SyntaxError.Exception, EvalException, InterruptedException {
    // Read and parse the named file.
    ParserInput input = ParserInput.readFile("my/file.star");
    StarlarkFile file = StarlarkFile.parse(input);

    // Compile the program, with additional predeclared environment bindings.
    // TODO(adonovan): supply Starlark.UNIVERSE somehow.
    Program prog = Program.compileFile(file, Resolver.moduleWithPredeclared("zero", "square"));

    // . . .

    // TODO(adonovan): when supported, show how the compiled program can be
    // saved and reloaded, to avoid repeating the cost of parsing and
    // compilation.

    // Execute the compiled program to populate a module.
    // The module's predeclared environment must match the
    // names provided during compilation.
    Module module = Module.withPredeclared(StarlarkSemantics.DEFAULT, makeEnvironment());
    try (Mutability mu = Mutability.create(prog.getFilename())) {
      StarlarkThread thread = new StarlarkThread(mu, StarlarkSemantics.DEFAULT);
      Starlark.execFileProgram(prog, module, thread);
    }
    return module;
  }

  /** This function shows how to construct a callable Starlark value from a Java method. */
  ImmutableMap<String, Object> makeEnvironment() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env.put("zero", 0);
    Starlark.addMethods(env, new MyFunctions(), StarlarkSemantics.DEFAULT); // adds 'square'
    return env.build();
  }

  /**
   * The annotated methods of this class are added to the environment by {@link
   * Starlark#addMethods}.
   */
  static final class MyFunctions {
    @StarlarkMethod(
        name = "square",
        parameters = {@Param(name = "x")},
        doc = "Returns the square of its integer argument.")
    public StarlarkInt square(StarlarkInt x) {
      return StarlarkInt.multiply(x, x);
    }
  }
}
