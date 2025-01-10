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
import net.starlark.java.eval.StarlarkThread.Frame;
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
    int nparams = getNumNonResidualParameters();
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
    return rfn.getNumOrdinaryParameters();
  }

  /** Returns the number of non-residual keyword-only parameters. */
  public int getNumKeywordOnlyParameters() {
    return rfn.numKeywordOnlyParams();
  }

  private int getNumNonResidualParameters() {
    return rfn.getNumNonResidualParameters();
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
    checkRecursive(thread);
    FastcallArgumentProcessor argumentProcessor = new FastcallArgumentProcessor(this);
    // Feed positional and named arguments into the argument processor.
    argumentProcessor.processPositionalAndNamed(positional, named, thread.mutability());
    return callWithArguments(thread, argumentProcessor);
  }

  @Override
  public Object positionalOnlyCall(StarlarkThread thread, Object... positional)
      throws EvalException, InterruptedException {
    checkRecursive(thread);
    FastcallArgumentProcessor argumentProcessor = new FastcallArgumentProcessor(this);
    // Feed only positional arguments into the argument processor.
    argumentProcessor.processPositionalOnly(positional, thread.mutability());
    return callWithArguments(thread, argumentProcessor);
  }

  @Override
  public StarlarkCallable.ArgumentProcessor requestArgumentProcessor(StarlarkThread thread) {
    return new ArgumentProcessor(this);
  }

  private Object callWithArguments(StarlarkThread thread, BaseArgumentProcessor argumentProcessor)
      throws EvalException, InterruptedException {
    checkRecursive(thread);
    Frame fr = thread.frame(0);
    fr.locals = argumentProcessor.retrieveFinishedLocals();

    spillIndicatedLocalsToCells(fr);

    return Eval.execFunctionBody(fr, rfn.getBody());
  }

  private void checkRecursive(StarlarkThread thread) throws EvalException {
    if (!thread.isRecursionAllowed() && thread.isRecursiveCall(this)) {
      throw Starlark.errorf("function '%s' called recursively", getName());
    }
  }

  private void spillIndicatedLocalsToCells(Frame fr) {
    for (int index : rfn.getCellIndices()) {
      fr.locals[index] = new Cell(fr.locals[index]);
    }
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

  // Checks the positional and named arguments to ensure they match the signature. It returns a new
  // array of effective parameter values corresponding to the parameters of the signature. The
  // returned array has size of locals and is directly pushed to the stack.
  // Newly allocated values (e.g. a **kwargs dict) use the Mutability mu.
  //
  // If the function has optional parameters, their default values are supplied by getDefaultValue.
  private abstract static class BaseArgumentProcessor
      implements StarlarkCallable.ArgumentProcessor {

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

    protected final StarlarkFunction owner;
    // Number of positional args that were set by the caller and bound to ordinary params (in other
    // words, not counting surplus positional args that were spilled to *args, and not counting
    // positional params that weren't set via args but were instead filled with defaults).
    protected int numNonSurplusPositionalArgs;
    // Local variable array for the function's call frame. It has the following layout:
    //
    // * The first owner.getNumOrdinaryParameters() entries are values of ordinary parameters
    //   * The first numNonSurplusPositionalArgs entries contain positional args, set by
    //     bindPositionalArgsToLocals()
    //   * The remaining entries contain keyword args (set by bindNamedArgsToLocals()) or default
    //     values (set by applyDefaultsReportMissingArgs())
    // * The next owner.getNumKeywordOnlyParameters() entries are values of keyword-only parameters,
    //   which may be either keyword args (set by bindNamedArgsToLocals()) or default values (set by
    //   applyDefaultsReportMissingArgs())
    // * An optional entry for *args - present if and only if the function takes varargs (set by
    //   bindSurplusPositionalArgsToVarArgs())
    // * An optional entry for **kwargs - present if and only if the function takes kwargs (set by
    //   bindNamedArgsToLocals())
    // * The remaining entries hold values of the function body's variables - these are left
    //   uninitialized by ArgumentProcessor, and will be set in the process of evaluating the
    //   function body.
    protected final Object[] locals;
    // unexpectedNamedArgs serves as accumulator for named arguments that can't be bound to any of
    // the function's parameters or to **kwargs. It is used to error-report all unexpected named
    // args, not just the first one that was encountered.
    @Nullable protected List<String> unexpectedNamedArgs;

    public BaseArgumentProcessor(StarlarkFunction owner) {
      this.owner = owner;
      this.locals = new Object[owner.rfn.getLocals().size()];
      this.unexpectedNamedArgs = null;
    }

    @Override
    public StarlarkCallable getCallable() {
      return owner;
    }

    protected int getKwargsIndex() {
      return owner.rfn.hasKwargs() ? owner.rfn.getParameters().size() - 1 : -1;
    }

    protected int getVarArgsIndex() {
      if (owner.rfn.hasVarargs()) {
        int index = owner.rfn.getParameters().size();
        return owner.rfn.hasKwargs() ? index - 2 : index - 1;
      }
      return -1;
    }

    private Object[] retrieveFinishedLocals() throws EvalException {
      applyDefaultsReportMissingArgs();
      return locals;
    }

    protected void addUnexpectedNamedArg(String keyword) {
      if (unexpectedNamedArgs == null) {
        unexpectedNamedArgs = new ArrayList<>();
      }
      unexpectedNamedArgs.add(keyword);
    }

    protected void checkUnexpectedNamedArgs() throws EvalException {
      if (unexpectedNamedArgs != null) {
        // Give a spelling hint if there is exactly one.
        // More than that suggests the wrong function was called.
        throw Starlark.errorf(
            "%s() got unexpected keyword argument%s: %s%s",
            owner.getName(),
            plural(unexpectedNamedArgs.size()),
            Joiner.on(", ").join(unexpectedNamedArgs),
            unexpectedNamedArgs.size() == 1
                ? SpellChecker.didYouMean(
                    unexpectedNamedArgs.get(0),
                    owner.getParameterNames().subList(0, owner.getNumNonResidualParameters()))
                : "");
      }
    }

    protected void applyDefaultsReportMissingArgs() throws EvalException {
      // Apply defaults and report errors for missing required arguments.
      // Inv: all params below positionalCount were bound (by bindPositionalArgsToLocals()).
      int numParams = owner.getNumNonResidualParameters();
      int firstDefault = numParams - owner.defaultValues.size(); // first default
      List<String> missingPositional = null;
      List<String> missingKwonly = null;
      for (int i = numNonSurplusPositionalArgs; i < numParams; i++) {
        // provided?
        if (locals[i] != null) {
          continue;
        }

        // optional?
        if (i >= firstDefault) {
          Object dflt = owner.defaultValues.get(i - firstDefault);
          if (dflt != MANDATORY) {
            locals[i] = dflt;
            continue;
          }
        }

        // missing
        if (i < owner.getNumOrdinaryParameters()) {
          if (missingPositional == null) {
            missingPositional = new ArrayList<>();
          }
          missingPositional.add(owner.getParameterNames().get(i));
        } else {
          if (missingKwonly == null) {
            missingKwonly = new ArrayList<>();
          }
          missingKwonly.add(owner.getParameterNames().get(i));
        }
      }
      if (missingPositional != null) {
        throw Starlark.errorf(
            "%s() missing %d required positional argument%s: %s",
            owner.getName(),
            missingPositional.size(),
            plural(missingPositional.size()),
            Joiner.on(", ").join(missingPositional));
      }
      if (missingKwonly != null) {
        throw Starlark.errorf(
            "%s() missing %d required keyword-only argument%s: %s",
            owner.getName(),
            missingKwonly.size(),
            plural(missingKwonly.size()),
            Joiner.on(", ").join(missingKwonly));
      }
    }
  }

  private static class FastcallArgumentProcessor extends BaseArgumentProcessor {

    public FastcallArgumentProcessor(StarlarkFunction owner) {
      super(owner);
    }

    void processPositionalAndNamed(Object[] positional, Object[] named, Mutability mu)
        throws EvalException {
      numNonSurplusPositionalArgs = getNumNonSurplusPositionalArgs(positional);
      bindPositionalArgsToLocals(positional);
      bindSurplusPositionalArgsToVarArgs(positional);
      bindNamedArgsToLocals(named, mu);
    }

    void processPositionalOnly(Object[] positional, Mutability mu) throws EvalException {
      numNonSurplusPositionalArgs = getNumNonSurplusPositionalArgs(positional);
      bindPositionalArgsToLocals(positional);
      bindSurplusPositionalArgsToVarArgs(positional);
      // Bind an empty dict to **kwargs if present. (The dict, unfortunately, needs to be mutable;
      // see https://github.com/bazelbuild/starlark/issues/295)
      if (owner.rfn.hasKwargs()) {
        locals[owner.rfn.getParameters().size() - 1] = Dict.of(mu);
      }
    }

    /**
     * Returns the number of positional arguments that should be bound to the function's positional
     * parameters (in other words, excluding surplus positionals that get bound to {@code *args}).
     * If the function doesn't take {@code *args}, verifies that the number of positional arguments
     * doesn't exceed the number of ordinary parameters.
     *
     * @param positional positional arguments passed by the caller
     */
    private int getNumNonSurplusPositionalArgs(Object[] positional) throws EvalException {
      int positionalCount = positional.length;
      int numOrdinaryParams = owner.getNumOrdinaryParameters();
      if (positionalCount > numOrdinaryParams) {
        if (!owner.rfn.hasVarargs()) {
          if (numOrdinaryParams > 0) {
            throw Starlark.errorf(
                "%s() accepts no more than %d positional argument%s but got %d",
                owner.getName(), numOrdinaryParams, plural(numOrdinaryParams), positionalCount);
          } else {
            throw Starlark.errorf(
                "%s() does not accept positional arguments, but got %d",
                owner.getName(), positionalCount);
          }
        }
        positionalCount = numOrdinaryParams;
      }
      return positionalCount;
    }

    private void bindPositionalArgsToLocals(Object[] positional) {
      // Inv: numNonSurplusPositionalArgs == getNumNonSurplusPositionalArgs(positional)
      for (int i = 0; i < numNonSurplusPositionalArgs; i++) {
        locals[i] = positional[i];
      }
    }

    private void bindSurplusPositionalArgsToVarArgs(Object[] positional) {
      // Inv: numNonSurplusPositionalArgs == getNumNonSurplusPositionalArgs(positional)
      if (owner.rfn.hasVarargs()) {
        locals[owner.getNumNonResidualParameters()] =
            Tuple.wrap(
                Arrays.copyOfRange(positional, numNonSurplusPositionalArgs, positional.length));
      }
    }

    private void bindNamedArgsToLocals(Object[] named, Mutability mu) throws EvalException {

      // Named arguments.
      LinkedHashMap<String, Object> kwargs = null;
      if (owner.rfn.hasKwargs()) {
        // To avoid Dict overhead, we populate a LinkedHashMap and then pass it to Dict.wrap()
        // afterwards. (The contract of Dict.wrap prohibits us from modifying the map once the Dict
        // is created.)
        kwargs = Maps.newLinkedHashMapWithExpectedSize(1);
      }
      for (int i = 0; i < named.length; i += 2) {
        String keyword = (String) named[i]; // safe
        Object value = named[i + 1];
        // The list should be short, so linear scan should still be OK for now.
        // TODO(b/380824219): Investigate caching between calls
        int pos = owner.getParameterNames().indexOf(keyword);
        if (0 <= pos && pos < owner.getNumNonResidualParameters()) {
          // keyword is the name of a named parameter
          if (locals[pos] != null) {
            throw Starlark.errorf(
                "%s() got multiple values for parameter '%s'", owner.getName(), keyword);
          }
          locals[pos] = value;

        } else if (kwargs != null) {
          // residual keyword argument
          if (kwargs.put(keyword, value) != null) {
            throw Starlark.errorf(
                "%s() got multiple values for parameter '%s'", owner.getName(), keyword);
          }

        } else {
          // unexpected keyword argument
          addUnexpectedNamedArg(keyword);
        }
      }
      checkUnexpectedNamedArgs();
      if (kwargs != null) {
        locals[owner.rfn.getParameters().size() - 1] = Dict.wrap(mu, kwargs);
      }
    }

    @Override
    public void addPositionalArg(Object value) throws EvalException {
      throw notForOutsideUse();
    }

    @Override
    public void addNamedArg(String name, Object value) throws EvalException {
      throw notForOutsideUse();
    }

    @Override
    public Object call(StarlarkThread thread) throws EvalException, InterruptedException {
      throw notForOutsideUse();
    }

    private IllegalStateException notForOutsideUse() {
      return new IllegalStateException(
          "FastcallArgumentProcessor is not intended to be used outside of StarlarkFunction.");
    }
  }

  private static class ArgumentProcessor extends BaseArgumentProcessor {
    // varArgs and kwargs are used to collect the respective arguments before transforming them into
    // Starlark values and binding them to the right slots in the locals array.
    @Nullable private ArrayList<Object> varArgs;
    @Nullable private LinkedHashMap<String, Object> kwargs;

    ArgumentProcessor(StarlarkFunction owner) {
      super(owner);
      this.varArgs = null;
      this.kwargs = null;
    }

    @Override
    public void addPositionalArg(Object value) throws EvalException {
      if (numNonSurplusPositionalArgs < owner.getNumOrdinaryParameters()) {
        locals[numNonSurplusPositionalArgs++] = value;
      } else if (owner.rfn.hasVarargs()) {
        if (varArgs == null) {
          varArgs = new ArrayList<>();
        }
        varArgs.add(value);
      } else {
        // This indicates an error condition which is then checked in call().
        numNonSurplusPositionalArgs++;
      }
    }

    private void setKwargToLocal(int index, Object value, String name) throws EvalException {
      if (locals[index] != null) {
        throwDoubleDefinedKeywordArg(name);
      }
      locals[index] = value;
    }

    protected void throwDoubleDefinedKeywordArg(String name) throws EvalException {
      throw Starlark.errorf("%s() got multiple values for parameter '%s'", owner.getName(), name);
    }

    @Override
    public void addNamedArg(String name, Object value) throws EvalException {
      int formalIndex = owner.getParameterNames().indexOf(name);
      if (0 <= formalIndex && formalIndex < owner.getNumNonResidualParameters()) {
        setKwargToLocal(formalIndex, value, name);
      } else {
        if (owner.rfn.hasKwargs()) {
          if (kwargs == null) {
            kwargs = Maps.newLinkedHashMapWithExpectedSize(1);
          }
          Object oldValue = kwargs.put(name, value);
          if (oldValue != null) {
            throwDoubleDefinedKeywordArg(name);
          }
        } else {
          addUnexpectedNamedArg(name);
        }
      }
    }

    @Override
    public Object call(StarlarkThread thread) throws EvalException, InterruptedException {
      int numOrdinaryParams = owner.getNumOrdinaryParameters();
      if (numNonSurplusPositionalArgs > numOrdinaryParams) {
        if (numOrdinaryParams > 0) {
          throw Starlark.errorf(
              "%s() accepts no more than %d positional argument%s but got %d",
              owner.getName(),
              numOrdinaryParams,
              plural(numOrdinaryParams),
              numNonSurplusPositionalArgs);
        } else {
          throw Starlark.errorf(
              "%s() does not accept positional arguments, but got %d",
              owner.getName(), numNonSurplusPositionalArgs);
        }
      }
      checkUnexpectedNamedArgs();
      if (owner.rfn.hasVarargs()) {
        locals[getVarArgsIndex()] =
            varArgs == null
                ? Tuple.empty()
                : varArgs.size() == 1
                    ? Tuple.of(varArgs.getFirst())
                    : Tuple.wrap(varArgs.toArray());
      }
      if (owner.rfn.hasKwargs()) {
        locals[getKwargsIndex()] =
            kwargs == null ? Dict.of(thread.mutability()) : Dict.wrap(thread.mutability(), kwargs);
      }
      return owner.callWithArguments(thread, this);
    }
  }
}
