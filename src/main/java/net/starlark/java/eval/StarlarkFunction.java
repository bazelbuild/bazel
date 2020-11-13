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
package net.starlark.java.eval;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.spelling.SpellChecker;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ExpressionStatement;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.Resolver;
import net.starlark.java.syntax.Statement;
import net.starlark.java.syntax.StringLiteral;

/** A StarlarkFunction is a function value created by a Starlark {@code def} statement. */
@StarlarkBuiltin(
    name = "function",
    category = "core",
    doc = "The type of functions declared in Starlark.")
public final class StarlarkFunction implements StarlarkCallable {

  private final Resolver.Function rfn;
  private final Module module; // a function closes over its defining module
  private final Tuple defaultValues;

  StarlarkFunction(Resolver.Function rfn, Tuple defaultValues, Module module) {
    this.rfn = rfn;
    this.module = module;
    this.defaultValues = defaultValues;
  }

  boolean isToplevel() {
    return rfn.isToplevel();
  }

  // TODO(adonovan): many functions would be simpler if
  // parameterNames excluded the *args and **kwargs parameters,
  // (whose names are immaterial to the callee anyway). Do that.
  // Also, reject getDefaultValue for varargs and kwargs.

  /**
   * Returns the default value of the ith parameter ({@code 0 <= i < getParameterNames().size()}),
   * or null if the parameter is required. Residual parameters, if any, are always last, and have no
   * default value.
   */
  @Nullable
  public Object getDefaultValue(int i) {
    if (i < 0 || i >= rfn.getParameters().size()) {
      throw new IndexOutOfBoundsException();
    }
    int nparams =
        rfn.getParameters().size() - (rfn.hasKwargs() ? 1 : 0) - (rfn.hasVarargs() ? 1 : 0);
    int prefix = nparams - defaultValues.size();
    if (i < prefix) {
      return null; // implicit prefix of mandatory parameters
    }
    if (i < nparams) {
      Object v = defaultValues.get(i - prefix);
      return v == MANDATORY ? null : v;
    }
    return null; // *args or *kwargs
  }

  /**
   * Returns the names of this function's parameters. The residual {@code *args} and {@code
   * **kwargs} parameters, if any, are always last.
   */
  public ImmutableList<String> getParameterNames() {
    return rfn.getParameterNames();
  }

  /**
   * Reports whether this function has a residual positional arguments parameter, {@code def
   * f(*args)}.
   */
  public boolean hasVarargs() {
    return rfn.hasVarargs();
  }

  /**
   * Reports whether this function has a residual keyword arguments parameter, {@code def
   * f(**kwargs)}.
   */
  public boolean hasKwargs() {
    return rfn.hasKwargs();
  }

  /** Returns the location of the function's defining identifier. */
  @Override
  public Location getLocation() {
    return rfn.getLocation();
  }

  /**
   * Returns the name of the function. Implicit functions (those not created by a def statement),
   * may have names such as "<toplevel>" or "<expr>".
   */
  @Override
  public String getName() {
    return rfn.getName();
  }

  /** Returns the value denoted by the function's doc string literal, or null if absent. */
  @Nullable
  public String getDocumentation() {
    if (rfn.getBody().isEmpty()) {
      return null;
    }
    Statement first = rfn.getBody().get(0);
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
    if (!thread.isRecursionAllowed() && thread.isRecursiveCall(this)) {
      throw Starlark.errorf("function '%s' called recursively", getName());
    }

    // Compute the effective parameter values
    // and update the corresponding variables.
    Object[] arguments = processArgs(thread.mutability(), positional, named);

    StarlarkThread.Frame fr = thread.frame(0);
    ImmutableList<String> names = rfn.getParameterNames();
    for (int i = 0; i < names.size(); ++i) {
      fr.locals.put(names.get(i), arguments[i]);
    }

    return Eval.execFunctionBody(fr, rfn.getBody());
  }

  @Override
  public void repr(Printer printer) {
    // TODO(adonovan): use the file name instead. But that's a breaking Bazel change.
    Object clientData = module.getClientData();

    printer.append("<function " + getName());
    if (clientData != null) {
      printer.append(" from " + clientData);
    }
    printer.append(">");
  }

  // Checks the positional and named arguments to ensure they match the signature. It returns a new
  // array of effective parameter values corresponding to the parameters of the signature. Newly
  // allocated values (e.g. a **kwargs dict) use the Mutability mu.
  //
  // If the function has optional parameters, their default values are supplied by getDefaultValue.
  private Object[] processArgs(Mutability mu, Object[] positional, Object[] named)
      throws EvalException {

    // This is the general schema of a function:
    //
    //   def f(p1, p2=dp2, p3=dp3, *args, k1, k2=dk2, k3, **kwargs)
    //
    // The p parameters are non-kwonly, and may be specified positionally.
    // The k parameters are kwonly, and must be specified by name.
    // The defaults tuple is (dp2, dp3, MANDATORY, dk2, MANDATORY).
    // The missing prefix (p1) is assumed to be all MANDATORY.
    //
    // Arguments are processed as follows:
    // - positional arguments are bound to a prefix of [p1, p2, p3].
    // - surplus positional arguments are bound to *args.
    // - keyword arguments are bound to any of {p1, p2, p3, k1, k2, k3};
    //   duplicate bindings are rejected.
    // - surplus keyword arguments are bound to **kwargs.
    // - defaults are bound to each parameter from p2 to k3 if no value was set.
    //   default values come from the tuple above.
    //   It is an error if the defaults tuple entry for an unset parameter is MANDATORY.

    ImmutableList<String> names = rfn.getParameterNames();

    // TODO(adonovan): when we have flat frames, pass in the locals array here instead of
    // allocating.
    Object[] arguments = new Object[names.size()];

    // nparams is the number of ordinary parameters.
    int nparams =
        rfn.getParameters().size() - (rfn.hasKwargs() ? 1 : 0) - (rfn.hasVarargs() ? 1 : 0);

    // numPositionalParams is the number of non-kwonly parameters.
    int numPositionalParams = nparams - rfn.numKeywordOnlyParams();

    // Too many positional args?
    int n = positional.length;
    if (n > numPositionalParams) {
      if (!rfn.hasVarargs()) {
        if (numPositionalParams > 0) {
          throw Starlark.errorf(
              "%s() accepts no more than %d positional argument%s but got %d",
              getName(), numPositionalParams, plural(numPositionalParams), n);
        } else {
          throw Starlark.errorf(
              "%s() does not accept positional arguments, but got %d", getName(), n);
        }
      }
      n = numPositionalParams;
    }
    // Inv: n is number of positional arguments that are not surplus.

    // Bind positional arguments to non-kwonly parameters.
    for (int i = 0; i < n; i++) {
      arguments[i] = positional[i];
    }

    // Bind surplus positional arguments to *args parameter.
    if (rfn.hasVarargs()) {
      arguments[nparams] = Tuple.wrap(Arrays.copyOfRange(positional, n, positional.length));
    }

    List<String> unexpected = null;

    // named arguments
    Dict<String, Object> kwargs = null;
    if (rfn.hasKwargs()) {
      kwargs = Dict.of(mu);
      arguments[rfn.getParameters().size() - 1] = kwargs;
    }
    for (int i = 0; i < named.length; i += 2) {
      String keyword = (String) named[i]; // safe
      Object value = named[i + 1];
      int pos = names.indexOf(keyword); // the list should be short, so linear scan is OK.
      if (0 <= pos && pos < nparams) {
        // keyword is the name of a named parameter
        if (arguments[pos] != null) {
          throw Starlark.errorf("%s() got multiple values for parameter '%s'", getName(), keyword);
        }
        arguments[pos] = value;

      } else if (kwargs != null) {
        // residual keyword argument
        int sz = kwargs.size();
        kwargs.putEntry(keyword, value);
        if (kwargs.size() == sz) {
          throw Starlark.errorf(
              "%s() got multiple values for keyword argument '%s'", getName(), keyword);
        }

      } else {
        // unexpected keyword argument
        if (unexpected == null) {
          unexpected = new ArrayList<>();
        }
        unexpected.add(keyword);
      }
    }
    if (unexpected != null) {
      // Give a spelling hint if there is exactly one.
      // More than that suggests the wrong function was called.
      throw Starlark.errorf(
          "%s() got unexpected keyword argument%s: %s%s",
          getName(),
          plural(unexpected.size()),
          Joiner.on(", ").join(unexpected),
          unexpected.size() == 1
              ? SpellChecker.didYouMean(unexpected.get(0), names.subList(0, nparams))
              : "");
    }

    // Apply defaults and report errors for missing required arguments.
    int m = nparams - defaultValues.size(); // first default
    List<String> missingPositional = null;
    List<String> missingKwonly = null;
    for (int i = n; i < nparams; i++) {
      // provided?
      if (arguments[i] != null) {
        continue;
      }

      // optional?
      if (i >= m) {
        Object dflt = defaultValues.get(i - m);
        if (dflt != MANDATORY) {
          arguments[i] = dflt;
          continue;
        }
      }

      // missing
      if (i < numPositionalParams) {
        if (missingPositional == null) {
          missingPositional = new ArrayList<>();
        }
        missingPositional.add(names.get(i));
      } else {
        if (missingKwonly == null) {
          missingKwonly = new ArrayList<>();
        }
        missingKwonly.add(names.get(i));
      }
    }
    if (missingPositional != null) {
      throw Starlark.errorf(
          "%s() missing %d required positional argument%s: %s",
          getName(),
          missingPositional.size(),
          plural(missingPositional.size()),
          Joiner.on(", ").join(missingPositional));
    }
    if (missingKwonly != null) {
      throw Starlark.errorf(
          "%s() missing %d required keyword-only argument%s: %s",
          getName(),
          missingKwonly.size(),
          plural(missingKwonly.size()),
          Joiner.on(", ").join(missingKwonly));
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
    // TODO(adonovan): include *, ** tokens.
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

  // The MANDATORY sentinel indicates a slot in the defaultValues
  // tuple corresponding to a required parameter. It is not visible
  // to Java or Starlark code.
  static final Object MANDATORY = new Mandatory();

  private static class Mandatory implements StarlarkValue {}
}
