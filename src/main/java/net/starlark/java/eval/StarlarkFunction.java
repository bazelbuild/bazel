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
import com.google.common.collect.Maps;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.spelling.SpellChecker;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.Resolver;

/** A StarlarkFunction is a function value created by a Starlark {@code def} statement. */
@StarlarkBuiltin(
    name = "function",
    category = "core",
    doc = "The type of functions declared in Starlark.")
public final class StarlarkFunction implements StarlarkCallable {

  final Resolver.Function rfn;
  private final Module module; // a function closes over its defining module

  // Index in Module.globals of ith Program global (Resolver.Binding(GLOBAL).index).
  // See explanation at Starlark.execFileProgram.
  final int[] globalIndex;

  // Default values of optional parameters.
  // Indices correspond to the subsequence of parameters after the initial
  // required parameters and before *args/**kwargs.
  // Contain MANDATORY for the required keyword-only parameters.
  private final Tuple defaultValues;

  // Cells (shared locals) of enclosing functions.
  // Indexed by Resolver.Binding(FREE).index values.
  private final Tuple freevars;

  // A stable identifier for this function instance.
  //
  // This may be mutated by export.
  private SymbolGenerator.Symbol<?> token;

  StarlarkFunction(
      Resolver.Function rfn,
      Module module,
      int[] globalIndex,
      Tuple defaultValues,
      Tuple freevars,
      SymbolGenerator.Symbol<?> token) {
    this.rfn = rfn;
    this.module = module;
    this.globalIndex = globalIndex;
    this.defaultValues = defaultValues;
    this.freevars = freevars;
    this.token = token;
  }

  // Sets a global variable, given its index in this function's compiled Program.
  void setGlobal(int progIndex, Object value) {
    module.setGlobalByIndex(globalIndex[progIndex], value);
  }

  // Gets the value of a global variable, given its index in this function's compiled Program.
  @Nullable
  Object getGlobal(int progIndex) {
    return module.getGlobalByIndex(globalIndex[progIndex]);
  }

  boolean isToplevel() {
    return rfn.isToplevel();
  }

  /** Whether this function is defined at the top level of a file. */
  public boolean isGlobal() {
    return module.getGlobal(getName()) == this;
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
   * Returns the names of this function's parameters.
   *
   * <p>The first {@code getNumOrdinaryParameters()} parameters in the returned list are ordinary
   * (non-residual, non-keyword-only); the following {@code getNumKeywordOnlyParameters()} are
   * keyword-only; and the residual {@code *args} and {@code **kwargs} parameters, if any, are
   * always last.
   */
  public ImmutableList<String> getParameterNames() {
    return rfn.getParameterNames();
  }

  /** Returns the number of ordinary (non-residual, non-keyword-only) parameters. */
  public int getNumOrdinaryParameters() {
    return rfn.getParameters().size()
        - (rfn.hasKwargs() ? 1 : 0)
        - (rfn.hasVarargs() ? 1 : 0)
        - rfn.numKeywordOnlyParams();
  }

  /** Returns the number of non-residual keyword-only parameters. */
  public int getNumKeywordOnlyParameters() {
    return rfn.numKeywordOnlyParams();
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
   * Returns the name of the function, or "lambda" if anonymous. Implicit functions (those not
   * created by a def statement), may have names such as "<toplevel>" or "<expr>".
   */
  @Override
  public String getName() {
    return rfn.getName();
  }

  /**
   * Returns the value denoted by the function's doc string literal (trimmed if necessary), or null
   * if absent.
   */
  @Nullable
  public String getDocumentation() {
    String documentation = rfn.getDocumentation();
    return documentation != null ? Starlark.trimDocString(documentation) : null;
  }

  public Module getModule() {
    return module;
  }

  @Override
  public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
      throws EvalException, InterruptedException {
    if (!thread.isRecursionAllowed() && thread.isRecursiveCall(this)) {
      throw Starlark.errorf("function '%s' called recursively", getName());
    }

    // Compute the effective parameter values
    // and update the corresponding variables.
    StarlarkThread.Frame fr = thread.frame(0);
    fr.locals = processArgs(thread.mutability(), positional, named);

    // Spill indicated locals to cells.
    for (int index : rfn.getCellIndices()) {
      fr.locals[index] = new Cell(fr.locals[index]);
    }

    return Eval.execFunctionBody(fr, rfn.getBody());
  }

  Cell getFreeVar(int index) {
    return (Cell) freevars.get(index);
  }

  void export(StarlarkThread thread, String name) {
    // Checks that thread is the one that defines the StarlarkFunction. It's possible for one
    // StarlarkFunction to be exported in different places.
    if (!token.getOwner().equals(thread.getOwner())) {
      return;
    }
    if (token.isGlobal()) {
      // Keeps only the first token if the same function is exported under multiple aliases.
      return;
    }
    token = token.exportAs(name);
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
  // array of effective parameter values corresponding to the parameters of the signature. The
  // returned array has size of locals and is directly pushed to the stack.
  // Newly allocated values (e.g. a **kwargs dict) use the Mutability mu.
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

    Object[] locals = new Object[rfn.getLocals().size()];

    // numOrdinaryParams is the number of ordinary (non-residual, non-kwonly) parameters.
    int numOrdinaryParams = getNumOrdinaryParameters();

    // nparams is the number of all non-residual parameters.
    int nparams = numOrdinaryParams + getNumKeywordOnlyParameters();

    // Too many positional args?
    int n = positional.length;
    if (n > numOrdinaryParams) {
      if (!rfn.hasVarargs()) {
        if (numOrdinaryParams > 0) {
          throw Starlark.errorf(
              "%s() accepts no more than %d positional argument%s but got %d",
              getName(), numOrdinaryParams, plural(numOrdinaryParams), n);
        } else {
          throw Starlark.errorf(
              "%s() does not accept positional arguments, but got %d", getName(), n);
        }
      }
      n = numOrdinaryParams;
    }
    // Inv: n is number of positional arguments that are not surplus.

    // Bind positional arguments to non-kwonly parameters.
    for (int i = 0; i < n; i++) {
      locals[i] = positional[i];
    }

    // Bind surplus positional arguments to *args parameter.
    if (rfn.hasVarargs()) {
      locals[nparams] = Tuple.wrap(Arrays.copyOfRange(positional, n, positional.length));
    }

    List<String> unexpected = null;

    // Named arguments.
    LinkedHashMap<String, Object> kwargs = null;
    if (rfn.hasKwargs()) {
      // To avoid Dict overhead, we populate a LinkedHashMap and then pass it to Dict.wrap()
      // afterwards. (The contract of Dict.wrap prohibits us from modifying the map once the Dict is
      // created.)
      kwargs = Maps.newLinkedHashMapWithExpectedSize(1);
    }
    for (int i = 0; i < named.length; i += 2) {
      String keyword = (String) named[i]; // safe
      Object value = named[i + 1];
      int pos = names.indexOf(keyword); // the list should be short, so linear scan is OK.
      if (0 <= pos && pos < nparams) {
        // keyword is the name of a named parameter
        if (locals[pos] != null) {
          throw Starlark.errorf("%s() got multiple values for parameter '%s'", getName(), keyword);
        }
        locals[pos] = value;

      } else if (kwargs != null) {
        // residual keyword argument
        if (kwargs.put(keyword, value) != null) {
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
    if (kwargs != null) {
      locals[rfn.getParameters().size() - 1] = Dict.wrap(mu, kwargs);
    }

    // Apply defaults and report errors for missing required arguments.
    int m = nparams - defaultValues.size(); // first default
    List<String> missingPositional = null;
    List<String> missingKwonly = null;
    for (int i = n; i < nparams; i++) {
      // provided?
      if (locals[i] != null) {
        continue;
      }

      // optional?
      if (i >= m) {
        Object dflt = defaultValues.get(i - m);
        if (dflt != MANDATORY) {
          locals[i] = dflt;
          continue;
        }
      }

      // missing
      if (i < numOrdinaryParams) {
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

    return locals;
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

  public SymbolGenerator.Symbol<?> getToken() {
    return token;
  }

  @Override
  public int hashCode() {
    return token.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof StarlarkFunction)) {
      return false;
    }
    return token.equals(((StarlarkFunction) obj).token);
  }

  @Override
  public boolean isImmutable() {
    // Only correct because closures are not yet supported.
    return true;
  }

  // The MANDATORY sentinel indicates a slot in the defaultValues
  // tuple corresponding to a required parameter.
  // It is not visible to Java or Starlark code.
  static final Object MANDATORY = new Mandatory();

  private static class Mandatory implements StarlarkValue {}

  // A Cell is a local variable shared between an inner and an outer function.
  // It is a StarlarkValue because it is a stack operand and a Tuple element,
  // but it is not visible to Java or Starlark code.
  static final class Cell implements StarlarkValue {
    Object x;

    Cell(Object x) {
      this.x = x;
    }
  }
}
