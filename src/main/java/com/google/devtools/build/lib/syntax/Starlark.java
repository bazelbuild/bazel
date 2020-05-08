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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkInterfaceUtils;
import com.google.errorprone.annotations.CheckReturnValue;
import com.google.errorprone.annotations.FormatMethod;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * The Starlark class defines the most important entry points, constants, and functions needed by
 * all clients of the Starlark interpreter.
 */
// TODO(adonovan): move these here: equal, compare, getattr, index, parse, exec, eval, and so on.
public final class Starlark {

  private Starlark() {} // uninstantiable

  /** The Starlark None value. */
  public static final NoneType NONE = NoneType.NONE;

  /**
   * A sentinel value passed to optional parameters of SkylarkCallable-annotated methods to indicate
   * that no argument value was supplied.
   */
  public static final Object UNBOUND = new UnboundMarker();

  @Immutable
  private static final class UnboundMarker implements StarlarkValue {
    private UnboundMarker() {}

    @Override
    public String toString() {
      return "<unbound>";
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<unbound>");
    }
  }

  /**
   * The universal bindings predeclared in every Starlark file, such as None, True, len, and range.
   */
  public static final ImmutableMap<String, Object> UNIVERSE = makeUniverse();

  private static ImmutableMap<String, Object> makeUniverse() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env //
        .put("False", false)
        .put("True", true)
        .put("None", Starlark.NONE);
    addMethods(env, new MethodLibrary());
    return env.build();
  }

  /**
   * Reports whether the argument is a legal Starlark value: a string, boolean, integer, or
   * StarlarkValue.
   */
  public static boolean valid(Object x) {
    return x instanceof StarlarkValue
        || x instanceof String
        || x instanceof Boolean
        || x instanceof Integer;
  }

  /**
   * Returns {@code x} if it is a {@link #valid} Starlark value, otherwise throws
   * IllegalArgumentException.
   */
  public static <T> T checkValid(T x) {
    if (!valid(x)) {
      throw new IllegalArgumentException("invalid Starlark value: " + x.getClass());
    }
    return x;
  }

  /**
   * Converts a Java value {@code x} to a Starlark one, if x is not already a valid Starlark value.
   * A Java List or Map is converted to a Starlark list or dict, respectively, and null becomes
   * {@link #NONE}. Any other non-Starlark value causes the function to throw
   * IllegalArgumentException.
   *
   * <p>This function is applied to the results of @SkylarkCallable-annotated Java methods.
   */
  public static Object fromJava(Object x, @Nullable Mutability mutability) {
    if (x == null) {
      return NONE;
    } else if (Starlark.valid(x)) {
      return x;
    } else if (x instanceof List) {
      return StarlarkList.copyOf(mutability, (List<?>) x);
    } else if (x instanceof Map) {
      return Dict.copyOf(mutability, (Map<?, ?>) x);
    } else {
      throw new IllegalArgumentException(
          "cannot expose internal type to Starlark: " + x.getClass());
    }
  }

  /**
   * Returns the truth value of a valid Starlark value, as if by the Starlark expression {@code
   * bool(x)}.
   */
  public static boolean truth(Object x) {
    if (x instanceof Boolean) {
      return (Boolean) x;
    } else if (x instanceof StarlarkValue) {
      return ((StarlarkValue) x).truth();
    } else if (x instanceof String) {
      return !((String) x).isEmpty();
    } else if (x instanceof Integer) {
      return (Integer) x != 0;
    } else {
      throw new IllegalArgumentException("invalid Starlark value: " + x.getClass());
    }
  }

  /**
   * Checks whether the Freezable Starlark value is frozen or temporarily immutable due to active
   * iterators.
   *
   * @throws EvalException if the value is not mutable.
   */
  public static void checkMutable(Mutability.Freezable x) throws EvalException {
    if (x.mutability().isFrozen()) {
      throw Starlark.errorf("trying to mutate a frozen %s value", Starlark.type(x));
    }
    if (x.updateIteratorCount(0)) {
      throw Starlark.errorf(
          "%s value is temporarily immutable due to active for-loop iteration", Starlark.type(x));
    }
  }

  /**
   * Returns an iterable view of {@code x} if it is an iterable Starlark value; throws EvalException
   * otherwise.
   *
   * <p>Whereas the interpreter temporarily freezes the iterable value by bracketing {@code for}
   * loops and comprehensions in calls to {@link Freezable#updateIteratorCount}, iteration using
   * this method does not freeze the value. Callers should exercise care not to mutate the
   * underlying object during iteration.
   */
  public static Iterable<?> toIterable(Object x) throws EvalException {
    if (x instanceof StarlarkIterable) {
      return (Iterable<?>) x;
    }
    throw errorf("type '%s' is not iterable", type(x));
  }

  /**
   * Returns a new array containing the elements of Starlark iterable value {@code x}. A Starlark
   * value is iterable if it implements {@link StarlarkIterable}.
   */
  public static Object[] toArray(Object x) throws EvalException {
    // Specialize Sequence and Dict to avoid allocation and/or indirection.
    if (x instanceof Sequence) {
      return ((Sequence<?>) x).toArray();
    } else if (x instanceof Dict) {
      return ((Dict<?, ?>) x).keySet().toArray();
    } else {
      return Iterables.toArray(toIterable(x), Object.class);
    }
  }

  /**
   * Returns the length of a Starlark string, sequence (such as a list or tuple), dict, or other
   * iterable, as if by the Starlark expression {@code len(x)}, or -1 if the value is valid but has
   * no length.
   */
  public static int len(Object x) {
    if (x instanceof String) {
      return ((String) x).length();
    } else if (x instanceof Sequence) {
      return ((Sequence) x).size();
    } else if (x instanceof Dict) {
      return ((Dict) x).size();
    } else if (x instanceof StarlarkIterable) {
      // Iterables.size runs in constant time if x implements Collection.
      return Iterables.size((Iterable<?>) x);
    } else {
      checkValid(x);
      return -1; // valid but not a sequence
    }
  }

  /** Returns the name of the type of a value as if by the Starlark expression {@code type(x)}. */
  public static String type(Object x) {
    return classType(x.getClass());
  }

  /**
   * Returns the name of the type of instances of class c.
   *
   * <p>This function accepts any class, not just those of legal Starlark values, and may be used
   * for reporting error messages involving arbitrary Java classes, for example at the interface
   * between Starlark and Java.
   */
  // TODO(adonovan): reconsider allowing any classes other than String, Integer, Boolean, and
  // subclasses of StarlarkValue, with a special exception for Object.class meaning "any Starlark
  // value" (not: any Java object). Ditto for Depset.ElementType.
  public static String classType(Class<?> c) {
    // Check for "direct hits" first to avoid needing to scan for annotations.
    if (c.equals(String.class)) {
      return "string";
    } else if (c.equals(Integer.class)) {
      return "int";
    } else if (c.equals(Boolean.class)) {
      return "bool";
    }

    StarlarkBuiltin module = StarlarkInterfaceUtils.getStarlarkBuiltin(c);
    if (module != null) {
      String name = module.name();
      return module.namespace() ? name + " (a language module)" : name;

    } else if (StarlarkCallable.class.isAssignableFrom(c)) {
      // All callable values have historically been lumped together as "function".
      // TODO(adonovan): built-in types that don't use SkylarkModule should report
      // their own type string, but this is a breaking change as users often
      // use type(x)=="function" for Starlark and built-in functions.
      return "function";

    } else if (c.equals(Object.class)) {
      // "Unknown" is another unfortunate choice.
      // Object.class does mean "unknown" when talking about the type parameter
      // of a collection (List<Object>), but it also means "any" when used
      // as an argument to Sequence.cast, and more generally it means "value".
      return "unknown";

    } else if (List.class.isAssignableFrom(c)) {
      // Any class of java.util.List that isn't a Sequence.
      return "List";

    } else if (Map.class.isAssignableFrom(c)) {
      // Any class of java.util.Map that isn't a Dict.
      return "Map";

    } else {
      String simpleName = c.getSimpleName();
      return simpleName.isEmpty() ? c.getName() : simpleName;
    }
  }

  /** Returns the string form of a value as if by the Starlark expression {@code str(x)}. */
  public static String str(Object x) {
    return Printer.getPrinter().str(x).toString();
  }

  /** Returns the string form of a value as if by the Starlark expression {@code repr(x)}. */
  public static String repr(Object x) {
    return Printer.getPrinter().repr(x).toString();
  }

  /** Returns a string formatted as if by the Starlark expression {@code pattern % arguments}. */
  public static String format(String pattern, Object... arguments) {
    return Printer.getPrinter().format(pattern, arguments).toString();
  }

  /** Returns a string formatted as if by the Starlark expression {@code pattern % arguments}. */
  public static String formatWithList(String pattern, List<?> arguments) {
    return Printer.getPrinter().formatWithList(pattern, arguments).toString();
  }

  /** Returns a slice of a sequence as if by the Starlark operation {@code x[start:stop:step]}. */
  public static Object slice(
      Mutability mu, Object x, Object startObj, Object stopObj, Object stepObj)
      throws EvalException {
    int n;
    if (x instanceof String) {
      n = ((String) x).length();
    } else if (x instanceof Sequence) {
      n = ((Sequence) x).size();
    } else {
      throw errorf("invalid slice operand: %s", type(x));
    }

    int start;
    int stop;
    int step;

    // step
    if (stepObj == NONE) {
      step = 1;
    } else {
      step = toInt(stepObj, "slice step");
      if (step == 0) {
        throw errorf("slice step cannot be zero");
      }
    }

    // start, stop
    if (step > 0) {
      // positive stride: default indices are [0:n].
      if (startObj == NONE) {
        start = 0;
      } else {
        start = EvalUtils.toIndex(toInt(startObj, "start index"), n);
      }

      if (stopObj == NONE) {
        stop = n;
      } else {
        stop = EvalUtils.toIndex(toInt(stopObj, "stop index"), n);
      }

      if (stop < start) {
        stop = start; // => empty result
      }

    } else {
      // negative stride: default indices are effectively [n-1:-1],
      // though to get this effect using explicit indices requires
      // [n-1:-1-n:-1] because of the treatment of negative values.
      if (startObj == NONE) {
        start = n - 1;
      } else {
        start = toInt(startObj, "start index");
        if (start < 0) {
          start += n;
        }
        if (start >= n) {
          start = n - 1;
        }
      }

      if (stopObj == NONE) {
        stop = -1;
      } else {
        stop = toInt(stopObj, "stop index");
        if (stop < 0) {
          stop += n;
        }
        if (stop < -1) {
          stop = -1;
        }
      }

      if (start < stop) {
        start = stop; // => empty result
      }
    }

    // slice operation
    if (x instanceof String) {
      return StringModule.slice((String) x, start, stop, step);
    } else {
      return ((Sequence<?>) x).getSlice(mu, start, stop, step);
    }
  }

  static int toInt(Object x, String name) throws EvalException {
    if (x instanceof Integer) {
      return (Integer) x;
    }
    throw errorf("got %s for %s, want int", type(x), name);
  }

  /**
   * Calls the function-like value {@code fn} in the specified thread, passing it the given
   * positional and named arguments, as if by the Starlark expression {@code fn(*args, **kwargs)}.
   */
  public static Object call(
      StarlarkThread thread,
      Object fn,
      List<Object> args,
      Map<String, Object> kwargs)
      throws EvalException, InterruptedException {
    Object[] named = new Object[2 * kwargs.size()];
    int i = 0;
    for (Map.Entry<String, Object> e : kwargs.entrySet()) {
      named[i++] = e.getKey();
      named[i++] = e.getValue();
    }
    return fastcall(thread, fn, args.toArray(), named);
  }

  /**
   * Calls the function-like value {@code fn} in the specified thread, passing it the given
   * positional and named arguments in the "fastcall" array representation.
   *
   * <p>The caller must not subsequently modify or even inspect the two arrays.
   */
  public static Object fastcall(
      StarlarkThread thread, Object fn, Object[] positional, Object[] named)
      throws EvalException, InterruptedException {
    StarlarkCallable callable;
    if (fn instanceof StarlarkCallable) {
      callable = (StarlarkCallable) fn;
    } else {
      // @SkylarkCallable(selfCall)?
      MethodDescriptor desc =
          CallUtils.getSelfCallMethodDescriptor(thread.getSemantics(), fn.getClass());
      if (desc == null) {
        throw errorf("'%s' object is not callable", type(fn));
      }
      callable = new BuiltinCallable(fn, desc.getName(), desc);
    }

    thread.push(callable);
    try {
      return callable.fastcall(thread, positional, named);
    } finally {
      thread.pop();
    }
  }

  /**
   * Returns a new EvalException with no location and an error message produced by Java-style string
   * formatting ({@code String.format(format, args)}). Use {@code errorf("%s", msg)} to produce an
   * error message from a non-constant expression {@code msg}.
   */
  @FormatMethod
  @CheckReturnValue // don't forget to throw it
  public static EvalException errorf(String format, Object... args) {
    return new EvalException(null, String.format(format, args));
  }

  /** Equivalent to {@code addMethods(env, v, StarlarkSemantics.DEFAULT)}. */
  public static void addMethods(ImmutableMap.Builder<String, Object> env, Object v) {
    addMethods(env, v, StarlarkSemantics.DEFAULT);
  }

  /**
   * Adds to the environment {@code env} all {@code StarlarkCallable}-annotated fields and methods
   * of value {@code v}, filtered by the given semantics. The class of {@code v} must have or
   * inherit a {@link StarlarkBuiltin} or {@code SkylarkGlobalLibrary} annotation.
   */
  public static void addMethods(
      ImmutableMap.Builder<String, Object> env, Object v, StarlarkSemantics semantics) {
    Class<?> cls = v.getClass();
    if (!StarlarkInterfaceUtils.hasSkylarkGlobalLibrary(cls)
        && StarlarkInterfaceUtils.getStarlarkBuiltin(cls) == null) {
      throw new IllegalArgumentException(
          cls.getName() + " is annotated with neither @SkylarkGlobalLibrary nor @StarlarkBuiltin");
    }
    for (String name : CallUtils.getMethodNames(semantics, v.getClass())) {
      // We use the 2-arg (desc=null) BuiltinCallable constructor instead of passing
      // the descriptor that CallUtils.getMethod would return,
      // because most calls to addMethods pass StarlarkSemantics.DEFAULT,
      // which is probably incorrect for the call.
      // The effect is that the default semantics determine which methods appear in
      // env, but the thread's semantics determine which method calls succeed.
      env.put(name, new BuiltinCallable(v, name));
    }
  }

  /**
   * Adds to the environment {@code env} the value {@code v}, under its annotated name. The class of
   * {@code v} must have or inherit a {@link StarlarkBuiltin} annotation.
   */
  public static void addModule(ImmutableMap.Builder<String, Object> env, Object v) {
    Class<?> cls = v.getClass();
    StarlarkBuiltin annot = StarlarkInterfaceUtils.getStarlarkBuiltin(cls);
    if (annot == null) {
      throw new IllegalArgumentException(cls.getName() + " is not annotated with @StarlarkBuiltin");
    }
    env.put(annot.name(), v);
  }

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
      throws SyntaxError.Exception, EvalException, InterruptedException {
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
      throws SyntaxError.Exception, EvalException, InterruptedException {
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
   * debugger. In case of parse of validation error, it throws SyntaxError.Exception. In case of
   * execution error, the function returns partial results: the incomplete module plus the
   * exception.
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
      throws SyntaxError.Exception {
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
   * SyntaxError.Exception in case of scan/parse/validation error.
   *
   * <p>In addition to the program, it returns the validated syntax tree. This permits clients such
   * as Bazel to inspect the syntax (for BUILD dialect checks, glob prefetching, etc.)
   */
  public static Object /*Pair<Program, StarlarkFile>*/ compileFile(
      ParserInput input, //
      Set<String> predeclared,
      StarlarkSemantics semantics)
      throws SyntaxError.Exception {
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
      throws SyntaxError.Exception {
    // Pseudocode:
    // expr = Expression.parse(input)
    // validateExpr(expr, env.keys, semantics)
    // prog = compile([return expr])
    // module = new module(env)
    // return new StarlarkFunction(prog.toplevel, module)
    throw new UnsupportedOperationException();
  }

  /**
   * Starts the CPU profiler with the specified sampling period, writing a pprof profile to {@code
   * out}. All running Starlark threads are profiled. May be called concurrent with Starlark
   * execution.
   *
   * @throws IllegalStateException exception if the Starlark profiler is already running or if the
   *     operating system's profiling resources for this process are already in use.
   */
  public static void startCpuProfile(OutputStream out, Duration period) {
    CpuProfiler.start(out, period);
  }

  /**
   * Stops the profiler and waits for the log to be written. Throws an unchecked exception if the
   * profiler was not already started by a prior call to {@link #startCpuProfile}.
   */
  public static void stopCpuProfile() throws IOException {
    CpuProfiler.stop();
  }
}
