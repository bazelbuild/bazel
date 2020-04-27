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
import com.google.common.collect.ImmutableList;
import com.google.devtools.starlark.spelling.SpellChecker;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;

/** A StarlarkFunction is a function value created by a Starlark {@code def} statement. */
public final class StarlarkFunction implements StarlarkCallable {

  private final Resolver.Function rfn;
  private final Module module; // a function closes over its defining module
  private final Tuple<Object> defaultValues;

  StarlarkFunction(Resolver.Function rfn, Tuple<Object> defaultValues, Module module) {
    this.rfn = rfn;
    this.module = module;
    this.defaultValues = defaultValues;
  }

  boolean isToplevel() {
    return rfn.isToplevel;
  }

  /**
   * Returns the default value of the ith parameter ({@code 0 <= i < getParameterNames().size()}),
   * or null if the parameter is not optional. Residual parameters, if any, are always last, and
   * have no default value.
   */
  @Nullable
  public Object getDefaultValue(int i) {
    if (i >= 0) {
      // def f(a, b=1, *args, c, d=2, **kwargs) has defaults tuple (b=1, d=2).
      // TODO(adonovan): eliminate hole using a sentinel, to simplify this
      // and other run-time logic.
      int a = rfn.numMandatoryPositional;
      int b = rfn.numOptionalPositional;
      int c = rfn.numMandatoryNamedOnly; // the hole
      int d = rfn.numOptionalNamedOnly;
      if (i < a) {
        return null;
      } else if (i < a + b) {
        return defaultValues.get(i - a);
      } else if (i < a + b + c) {
        return null;
      } else if (i < a + b + c + d) {
        return defaultValues.get(i - a - c);
      } else if (i < rfn.parameterNames.size()) {
        return null; // *args or **kwargs   TODO(adonovan): make this an error.
      }
    }
    throw new IndexOutOfBoundsException();
  }

  /**
   * Returns the names of this function's parameters. The residual {@code *args} and {@code
   * **kwargs} parameters, if any, are always last.
   */
  public ImmutableList<String> getParameterNames() {
    return rfn.parameterNames;
  }

  /**
   * Reports whether this function has a residual positional arguments parameter, {@code def
   * f(*args)}.
   */
  public boolean hasVarargs() {
    return rfn.varargs != null;
  }

  /**
   * Reports whether this function has a residual keyword arguments parameter, {@code def
   * f(**kwargs)}.
   */
  public boolean hasKwargs() {
    return rfn.kwargs != null;
  }

  @Override
  public Location getLocation() {
    return rfn.location;
  }

  @Override
  public String getName() {
    return rfn.name;
  }

  /** Returns the value denoted by the function's doc string literal, or null if absent. */
  @Nullable
  public String getDocumentation() {
    if (rfn.body.isEmpty()) {
      return null;
    }
    Statement first = rfn.body.get(0);
    if (!(first instanceof ExpressionStatement)) {
      return null;
    }
    Expression expr = ((ExpressionStatement) first).getExpression();
    if (!(expr instanceof StringLiteral)) {
      return null;
    }
    return ((StringLiteral) expr).getValue();
  }

  public Module getModule() {
    return module;
  }

  @Override
  public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
      throws EvalException, InterruptedException {
    if (thread.mutability().isFrozen()) {
      throw Starlark.errorf("Trying to call in frozen environment");
    }
    if (thread.isRecursiveCall(this)) {
      throw Starlark.errorf("function '%s' called recursively", getName());
    }

    // Compute the effective parameter values
    // and update the corresponding variables.
    Object[] arguments = processArgs(thread.mutability(), positional, named);

    StarlarkThread.Frame fr = thread.frame(0);
    ImmutableList<String> names = rfn.parameterNames;
    for (int i = 0; i < names.size(); ++i) {
      fr.locals.put(names.get(i), arguments[i]);
    }

    return Eval.execFunctionBody(fr, rfn.body);
  }

  @Override
  public void repr(Printer printer) {
    Object label = module.getLabel();

    printer.append("<function " + getName());
    if (label != null) {
      printer.append(" from " + label);
    }
    printer.append(">");
  }

  // Checks the positional and named arguments to ensure they match the signature. It returns a new
  // array of effective parameter values corresponding to the parameters of the signature. Newly
  // allocated values (e.g. a **kwargs dict) use the Mutability mu.
  //
  // If the function has optional parameters, their default values are supplied by getDefaultValues.
  private Object[] processArgs(Mutability mu, Object[] positional, Object[] named)
      throws EvalException {
    ImmutableList<String> names = rfn.parameterNames;

    // TODO(adonovan): when we have flat frames, pass in the locals array here instead of
    // allocating.
    Object[] arguments = new Object[names.size()];

    // Note that this variable will be adjusted down if there are extra positionals,
    // after these extra positionals are dumped into starParam.
    int numPositionalArgs = positional.length;

    int numPositionalParams = rfn.numPositional();
    int numNamedParams = rfn.numPositional() + rfn.numNamedOnly();

    // positional arguments
    if (rfn.varargs != null) {
      Object varargs;
      if (numPositionalArgs > rfn.numPositional()) {
        varargs =
            Tuple.wrap(Arrays.copyOfRange(positional, rfn.numPositional(), numPositionalArgs));
        numPositionalArgs = numPositionalParams; // clip numPositionalArgs
      } else {
        varargs = Tuple.empty();
      }
      arguments[numNamedParams] = varargs;
    } else if (numPositionalArgs > numPositionalParams) {
      if (numPositionalParams > 0) {
        throw Starlark.errorf(
            "%s() accepts no more than %d positional argument%s but got %d",
            getName(), numPositionalParams, plural(numPositionalParams), numPositionalArgs);
      } else {
        throw Starlark.errorf(
            "%s() does not accept positional arguments, but got %d", getName(), numPositionalArgs);
      }
    }
    for (int i = 0; i < numPositionalArgs; i++) {
      arguments[i] = positional[i];
    }

    // **kwargs
    Dict<String, Object> kwargs = null;
    if (rfn.kwargs != null) {
      kwargs = Dict.of(mu);
      arguments[names.size() - 1] = kwargs;
    }

    List<String> missing = null;

    // named arguments
    for (int i = 0; i < named.length; i += 2) {
      String keyword = (String) named[i]; // safe
      Object value = named[i + 1];
      int pos = names.indexOf(keyword); // the list should be short, so linear scan is OK.
      if (0 <= pos && pos < numNamedParams) {
        // keyword is the name of a named parameter
        if (arguments[pos] != null) {
          throw Starlark.errorf("%s() got multiple values for parameter '%s'", getName(), keyword);
        }
        arguments[pos] = value;

      } else if (kwargs != null) {
        // residual keyword argument
        int sz = kwargs.size();
        kwargs.put(keyword, value, null);
        if (kwargs.size() == sz) {
          throw Starlark.errorf(
              "%s() got multiple values for keyword argument '%s'", getName(), keyword);
        }

      } else {
        // unexpected keyword argument
        if (missing == null) {
          missing = new ArrayList<>();
        }
        missing.add(keyword);
      }
    }
    if (missing != null) {
      // Give a spelling hint if there is exactly one.
      // More than that suggests the wrong function was called.
      throw Starlark.errorf(
          "%s() got unexpected keyword argument%s: %s%s",
          getName(),
          plural(missing.size()),
          Joiner.on(", ").join(missing),
          missing.size() == 1 ? SpellChecker.didYouMean(missing.get(0), names) : "");
    }

    // missing mandatory positionals?
    // numPositionalArgs > rfn.numMandatoryPositional is OK
    for (int i = numPositionalArgs; i < rfn.numMandatoryPositional; i++) {
      if (arguments[i] == null) {
        if (missing == null) {
          missing = new ArrayList<>();
        }
        missing.add(names.get(i));
      }
    }
    if (missing != null) {
      throw Starlark.errorf(
          "%s() missing %d required positional argument%s: %s",
          getName(), missing.size(), plural(missing.size()), Joiner.on(", ").join(missing));
    }

    // missing mandatory named-onlys?
    int endMandatoryNamedOnlyParams = rfn.numPositional() + rfn.numMandatoryNamedOnly;
    for (int i = numPositionalParams; i < endMandatoryNamedOnlyParams; i++) {
      if (arguments[i] == null) {
        if (missing == null) {
          missing = new ArrayList<>();
        }
        missing.add(names.get(i));
      }
    }
    if (missing != null) {
      throw Starlark.errorf(
          "%s() missing %d required keyword-only argument%s: %s",
          getName(), missing.size(), plural(missing.size()), Joiner.on(", ").join(missing));
    }

    // default values
    for (int i = Math.max(numPositionalArgs, rfn.numMandatoryPositional);
        i < numPositionalParams;
        i++) {
      if (arguments[i] == null) {
        arguments[i] = defaultValues.get(i - rfn.numMandatoryPositional);
      }
    }
    int numMandatoryParams = rfn.numMandatoryPositional + rfn.numMandatoryNamedOnly;
    for (int i = numMandatoryParams + rfn.numOptionalPositional; i < numNamedParams; i++) {
      if (arguments[i] == null) {
        arguments[i] = defaultValues.get(i - numMandatoryParams);
      }
    }

    return arguments;
  }

  private static String plural(int n) {
    return n == 1 ? "" : "s";
  }

  @Override
  public String toString() {
    StringBuilder out = new StringBuilder();
    out.append(getName());
    out.append('(');
    String sep = "";
    for (String param : getParameterNames()) {
      out.append(sep).append(param);
      sep = ", ";
    }
    out.append(')');
    return out.toString();
  }

  @Override
  public boolean isImmutable() {
    // Only correct because closures are not yet supported.
    return true;
  }
}
