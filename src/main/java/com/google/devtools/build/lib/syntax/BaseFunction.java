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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.util.Preconditions;

import net.bytebuddy.implementation.bytecode.StackManipulation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A base class for Skylark functions, whether builtin or user-defined.
 *
 * <p>Nomenclature:
 * We call "Parameters" the formal parameters of a function definition.
 * We call "Arguments" the actual values supplied at the call site.
 *
 * <p>The outer calling convention is like that of python3,
 * with named parameters that can be mandatory or optional, and also be positional or named-only,
 * and rest parameters for extra positional and keyword arguments.
 * Callers supply a {@code List<Object>} args for positional arguments
 * and a {@code Map<String, Object>} for keyword arguments,
 * where positional arguments will be resolved first, then keyword arguments,
 * with errors for a clash between the two, for missing mandatory parameter,
 * or for unexpected extra positional or keyword argument in absence of rest parameter.
 *
 * <p>The inner calling convention is to pass the underlying method
 * an {@code Object[]} of the type-checked argument values, one per expected parameter,
 * parameters being sorted as documented in {@link FunctionSignature}.
 *
 * <p>The function may provide default values for optional parameters not provided by the caller.
 * These default values can be null if there are no optional parameters or for builtin functions,
 * but not for user-defined functions that have optional parameters.
 */
// TODO(bazel-team):
// Provide optimized argument frobbing depending of FunctionSignature and CallerSignature
// (that FuncallExpression must supply), optimizing for the all-positional and all-keyword cases.
// Also, use better pure maps to minimize map O(n) re-creation events when processing keyword maps.
public abstract class BaseFunction implements SkylarkValue {

  // The name of the function
  private final String name;

  // A function signature, including defaults and types
  // never null after it is configured
  @Nullable protected FunctionSignature.WithValues<Object, SkylarkType> signature;

  // Location of the function definition, or null for builtin functions
  @Nullable protected Location location;

  // Some functions are also Namespaces or other Skylark entities.
  @Nullable protected Class<?> objectType;

  // Documentation for variables, if any
  @Nullable protected List<String> paramDoc;

  // The types actually enforced by the Skylark runtime, as opposed to those enforced by the JVM,
  // or those displayed to the user in the documentation.
  @Nullable protected List<SkylarkType> enforcedArgumentTypes;

  // Defaults to be used when configure(annotation) is called (after the function is constructed).
  @Nullable private Iterable<Object> unconfiguredDefaultValues;
  // The configure(annotation) function will include these defaults in the function signature.
  // We need to supply these defaultValues to the constructor, that will store them here, because
  // they can't be supplied via Java annotations, due to the limitations in the annotation facility.
  // (For extra brownies, we could supply them as Skylark expression strings, to be evaluated by our
  // evaluator without the help of any unconfigured functions, or to be processed at compile-time;
  // but we resolve annotations at runtime for now.)
  // Limitations in Java annotations mean we can't express them in the SkylarkSignature annotation.
  // (In the future, we could parse and evaluate simple Skylark expression strings, but then
  // we'd have to be very careful of circularities during initialization).
  // Note that though we want this list to be immutable, we don't use ImmutableList,
  // because that can't store nulls and nulls are essential for some BuiltinFunction-s.
  // We trust the user not to modify the list behind our back.


  /** Returns the name of this function. */
  public String getName() {
    return name;
  }

  /** Returns the signature of this function. */
  @Nullable public FunctionSignature.WithValues<Object, SkylarkType> getSignature() {
    return signature;
  }

  /** This function may also be viewed by Skylark as being of a special ObjectType */
  @Nullable public Class<?> getObjectType() {
    return objectType;
  }

  /** Returns true if the BaseFunction is configured. */
  public boolean isConfigured() {
    return signature != null;
  }

  /**
   * Creates an unconfigured BaseFunction with the given name.
   *
   * @param name the function name
   */
  public BaseFunction(String name) {
    this.name = name;
  }

  /**
   * Constructs a BaseFunction with a given name, signature and location.
   *
   * @param name the function name
   * @param signature the signature with default values and types
   * @param location the location of function definition
   */
  public BaseFunction(String name,
      @Nullable FunctionSignature.WithValues<Object, SkylarkType> signature,
      @Nullable Location location) {
    this(name);
    this.signature = signature;
    this.location = location;
  }

  /**
   * Constructs a BaseFunction with a given name, signature.
   *
   * @param name the function name
   * @param signature the signature, with default values and types
   */
  public BaseFunction(String name,
      @Nullable FunctionSignature.WithValues<Object, SkylarkType> signature) {
    this(name, signature, null);
  }

  /**
   * Constructs a BaseFunction with a given name and signature without default values or types.
   *
   * @param name the function name
   * @param signature the signature, without default values or types
   */
  public BaseFunction(String name, FunctionSignature signature) {
    this(name, FunctionSignature.WithValues.<Object, SkylarkType>create(signature), null);
  }

  /**
   * Constructs a BaseFunction with a given name and list of unconfigured defaults.
   *
   * @param name the function name
   * @param defaultValues a list of default values for the optional arguments to be configured.
   */
  public BaseFunction(String name, @Nullable Iterable<Object> defaultValues) {
    this(name);
    this.unconfiguredDefaultValues = defaultValues;
  }

  /** Get parameter documentation as a list corresponding to each parameter */
  public List<String> getParamDoc() {
    return paramDoc;
  }

  /**
   * The size of the array required by the callee.
   */
  protected int getArgArraySize() {
    return signature.getSignature().getShape().getArguments();
  }

  /**
   * The types that will be actually enforced by Skylark itself, so we may skip those already
   * enforced by the JVM during calls to BuiltinFunction, but also so we may lie to the user
   * in the automatically-generated documentation
   */
  public List<SkylarkType> getEnforcedArgumentTypes() {
    return enforcedArgumentTypes;
  }

  /**
   * Process the caller-provided arguments into an array suitable for the callee (this function).
   */
  public Object[] processArguments(List<Object> args,
      @Nullable Map<String, Object> kwargs,
      @Nullable Location loc)
      throws EvalException {

    Object[] arguments = new Object[getArgArraySize()];

    // extract function signature
    FunctionSignature sig = signature.getSignature();
    FunctionSignature.Shape shape = sig.getShape();
    ImmutableList<String> names = sig.getNames();
    List<Object> defaultValues = signature.getDefaultValues();

    // Note that this variable will be adjusted down if there are extra positionals,
    // after these extra positionals are dumped into starParam.
    int numPositionalArgs = args.size();

    int numMandatoryPositionalParams = shape.getMandatoryPositionals();
    int numOptionalPositionalParams = shape.getOptionalPositionals();
    int numMandatoryNamedOnlyParams = shape.getMandatoryNamedOnly();
    int numOptionalNamedOnlyParams = shape.getOptionalNamedOnly();
    boolean hasStarParam = shape.hasStarArg();
    boolean hasKwParam = shape.hasKwArg();
    int numPositionalParams = numMandatoryPositionalParams + numOptionalPositionalParams;
    int numNamedOnlyParams = numMandatoryNamedOnlyParams + numOptionalNamedOnlyParams;
    int numNamedParams = numPositionalParams + numNamedOnlyParams;
    int kwParamIndex = names.size() - 1; // only valid if hasKwParam

    // (1) handle positional arguments
    if (hasStarParam) {
      // Nota Bene: we collect extra positional arguments in a (tuple,) rather than a [list],
      // and this is actually the same as in Python.
      int starParamIndex = numNamedParams;
      if (numPositionalArgs > numPositionalParams) {
        arguments[starParamIndex] =
            Tuple.copyOf(args.subList(numPositionalParams, numPositionalArgs));
        numPositionalArgs = numPositionalParams; // clip numPositionalArgs
      } else {
        arguments[starParamIndex] = Tuple.EMPTY;
      }
    } else if (numPositionalArgs > numPositionalParams) {
      throw new EvalException(loc,
          numPositionalParams > 0
          ? "too many (" + numPositionalArgs + ") positional arguments in call to " + this
          : this + " does not accept positional arguments, but got " + numPositionalArgs);
    }

    for (int i = 0; i < numPositionalArgs; i++) {
      arguments[i] = args.get(i);
    }

    // (2) handle keyword arguments
    if (kwargs == null || kwargs.isEmpty()) {
      // Easy case (2a): there are no keyword arguments.
      // All arguments were positional, so check we had enough to fill all mandatory positionals.
      if (numPositionalArgs < numMandatoryPositionalParams) {
        throw new EvalException(loc, String.format(
            "insufficient arguments received by %s (got %s, expected at least %s)",
            this, numPositionalArgs, numMandatoryPositionalParams));
      }
      // We had no named argument, so fail if there were mandatory named-only parameters
      if (numMandatoryNamedOnlyParams > 0) {
        throw new EvalException(loc, String.format(
            "missing mandatory keyword arguments in call to %s", this));
      }
      // Fill in defaults for missing optional parameters, that were conveniently grouped together,
      // thanks to the absence of mandatory named-only parameters as checked above.
      if (defaultValues != null) {
        int j = numPositionalArgs - numMandatoryPositionalParams;
        int endOptionalParams = numPositionalParams + numOptionalNamedOnlyParams;
        for (int i = numPositionalArgs; i < endOptionalParams; i++) {
          arguments[i] = defaultValues.get(j++);
        }
      }
      // If there's a kwParam, it's empty.
      if (hasKwParam) {
        // TODO(bazel-team): create a fresh mutable dict, like Python does
        arguments[kwParamIndex] = ImmutableMap.<String, Object>of();
      }
    } else if (hasKwParam && numNamedParams == 0) {
      // Easy case (2b): there are no named parameters, but there is a **kwParam.
      // Therefore all keyword arguments go directly to the kwParam.
      // Note that *starParam and **kwParam themselves don't count as named.
      // Also note that no named parameters means no mandatory parameters that weren't passed,
      // and no missing optional parameters for which to use a default. Thus, no loops.
      // TODO(bazel-team): create a fresh mutable dict, like Python does
      arguments[kwParamIndex] = kwargs; // NB: not 2a means kwarg isn't null
    } else {
      // Hard general case (2c): some keyword arguments may correspond to named parameters
      HashMap<String, Object> kwArg = hasKwParam ? new HashMap<String, Object>() : null;

      // For nicer stabler error messages, start by checking against
      // an argument being provided both as positional argument and as keyword argument.
      ArrayList<String> bothPosKey = new ArrayList<>();
      for (int i = 0; i < numPositionalArgs; i++) {
        String name = names.get(i);
        if (kwargs.containsKey(name)) {
          bothPosKey.add(name);
        }
      }
      if (!bothPosKey.isEmpty()) {
        throw new EvalException(loc,
            String.format("argument%s '%s' passed both by position and by name in call to %s",
                (bothPosKey.size() > 1 ? "s" : ""), Joiner.on("', '").join(bothPosKey), this));
      }

      // Accept the arguments that were passed.
      for (Map.Entry<String, Object> entry : kwargs.entrySet()) {
        String keyword = entry.getKey();
        Object value = entry.getValue();
        int pos = names.indexOf(keyword); // the list should be short, so linear scan is OK.
        if (0 <= pos && pos < numNamedParams) {
          arguments[pos] = value;
        } else {
          if (!hasKwParam) {
            List<String> unexpected = Ordering.natural().sortedCopy(Sets.difference(
                kwargs.keySet(), ImmutableSet.copyOf(names.subList(0, numNamedParams))));
            throw new EvalException(loc, String.format("unexpected keyword%s '%s' in call to %s",
                    unexpected.size() > 1 ? "s" : "", Joiner.on("', '").join(unexpected), this));
          }
          if (kwArg.containsKey(keyword)) {
            throw new EvalException(loc, String.format(
                "%s got multiple values for keyword argument '%s'", this, keyword));
          }
          kwArg.put(keyword, value);
        }
      }
      if (hasKwParam) {
        // TODO(bazel-team): create a fresh mutable dict, like Python does
        arguments[kwParamIndex] = ImmutableMap.copyOf(kwArg);
      }

      // Check that all mandatory parameters were filled in general case 2c.
      // Note: it's possible that numPositionalArgs > numMandatoryPositionalParams but that's OK.
      for (int i = numPositionalArgs; i < numMandatoryPositionalParams; i++) {
        if (arguments[i] == null) {
          throw new EvalException(loc, String.format(
              "missing mandatory positional argument '%s' while calling %s",
              names.get(i), this));
        }
      }

      int endMandatoryNamedOnlyParams = numPositionalParams + numMandatoryNamedOnlyParams;
      for (int i = numPositionalParams; i < endMandatoryNamedOnlyParams; i++) {
        if (arguments[i] == null) {
          throw new EvalException(loc, String.format(
              "missing mandatory named-only argument '%s' while calling %s",
              names.get(i), this));
        }
      }

      // Get defaults for those parameters that weren't passed.
      if (defaultValues != null) {
        for (int i = Math.max(numPositionalArgs, numMandatoryPositionalParams);
             i < numPositionalParams; i++) {
          if (arguments[i] == null) {
            arguments[i] = defaultValues.get(i - numMandatoryPositionalParams);
          }
        }
        int numMandatoryParams = numMandatoryPositionalParams + numMandatoryNamedOnlyParams;
        for (int i = numMandatoryParams + numOptionalPositionalParams; i < numNamedParams; i++) {
          if (arguments[i] == null) {
            arguments[i] = defaultValues.get(i - numMandatoryParams);
          }
        }
      }
    } // End of general case 2c for argument passing.

    return arguments;
  }

  /** check types and convert as required */
  protected void canonicalizeArguments(Object[] arguments, Location loc) throws EvalException {
    // TODO(bazel-team): maybe link syntax.SkylarkType and package.Type,
    // so we can simultaneously typecheck and convert?
    // Note that a BuiltinFunction already does typechecking of simple types.

    List<SkylarkType> types = getEnforcedArgumentTypes();

    // Check types, if supplied
    if (types == null) {
      return;
    }
    List<String> names = signature.getSignature().getNames();
    int length = types.size();
    for (int i = 0; i < length; i++) {
      Object value = arguments[i];
      SkylarkType type = types.get(i);
      if (value != null && type != null && !type.contains(value)) {
        throw new EvalException(loc,
            String.format("expected %s for '%s' while calling %s but got %s instead: %s",
                type, names.get(i), getName(), EvalUtils.getDataTypeName(value, true), value));
      }
    }
  }

  /**
   * Returns the environment for the scope of this function.
   *
   * <p>Since this is a BaseFunction, we don't create a new environment.
   */
  @SuppressWarnings("unused") // For the exception
  protected Environment getOrCreateChildEnvironment(Environment parent) throws EvalException {
    return parent;
  }

  public static final StackManipulation call =
      ByteCodeUtils.invoke(
          BaseFunction.class,
          "call",
          List.class,
          Map.class,
          FuncallExpression.class,
          Environment.class);

  /**
   * The outer calling convention to a BaseFunction.
   *
   * @param args a list of all positional arguments (as in *starArg)
   * @param kwargs a map for key arguments (as in **kwArgs)
   * @param ast the expression for this function's definition
   * @param env the Environment in the function is called
   * @return the value resulting from evaluating the function with the given arguments
   * @throws EvalException-s containing source information.
   */
  public Object call(List<Object> args,
      @Nullable Map<String, Object> kwargs,
      @Nullable FuncallExpression ast,
      Environment env)
      throws EvalException, InterruptedException {
    Preconditions.checkState(isConfigured(), "Function %s was not configured", getName());

    // ast is null when called from Java (as there's no Skylark call site).
    Location loc = ast == null ? Location.BUILTIN : ast.getLocation();

    Object[] arguments = processArguments(args, kwargs, loc);
    canonicalizeArguments(arguments, loc);

    return call(arguments, ast, env);
  }

  /**
   * Inner call to a BaseFunction
   * subclasses need to @Override this method.
   *
   * @param args an array of argument values sorted as per the signature.
   * @param ast the source code for the function if user-defined
   * @param env the lexical environment of the function call
   * @throws InterruptedException may be thrown in the function implementations.
   */
  // Don't make it abstract, so that subclasses may be defined that @Override the outer call() only.
  protected Object call(Object[] args,
      @Nullable FuncallExpression ast, @Nullable Environment env)
      throws EvalException, ConversionException, InterruptedException {
    throw new EvalException(
        (ast == null) ? Location.BUILTIN : ast.getLocation(),
        String.format("function %s not implemented", getName()));
  }

  /**
   * Render this object in the form of an equivalent Python function signature.
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append(getName());
    if (signature != null) {
      sb.append('(');
      signature.toStringBuilder(sb);
      sb.append(')');
    } // if unconfigured, don't even output parentheses
    return sb.toString();
  }

  /** Configure a BaseFunction from a @SkylarkSignature annotation */
  public void configure(SkylarkSignature annotation) {
    Preconditions.checkState(!isConfigured()); // must not be configured yet

    this.paramDoc = new ArrayList<>();
    this.signature = SkylarkSignatureProcessor.getSignature(
        getName(), annotation, unconfiguredDefaultValues, paramDoc, getEnforcedArgumentTypes());
    this.objectType = annotation.objectType().equals(Object.class)
        ? null : annotation.objectType();
    configure();
  }

  /** Configure a function based on its signature */
  protected void configure() {
    // this function is called after the signature was initialized
    Preconditions.checkState(signature != null);
    enforcedArgumentTypes = signature.getTypes();
  }

  protected boolean hasSelfArgument() {
    Class<?> clazz = getObjectType();
    if (clazz == null) {
      return false;
    }
    List<SkylarkType> types = signature.getTypes();
    ImmutableList<String> names = signature.getSignature().getNames();

    return (!types.isEmpty() && types.get(0).canBeCastTo(clazz))
        || (!names.isEmpty() && names.get(0).equals("self"));
  }

  protected String getObjectTypeString() {
    Class<?> clazz = getObjectType();
    if (clazz == null) {
      return "";
    }
    return EvalUtils.getDataTypeNameFromClass(clazz, false) + ".";
  }

  /**
   * Returns [class.]function (depending on whether func belongs to a class).
   */
  public String getFullName() {
    return String.format("%s%s", getObjectTypeString(), getName());
  }

  /**
   * Returns the signature as "[className.]methodName(name1: paramType1, name2: paramType2, ...)"
   * or "[className.]methodName(paramType1, paramType2, ...)", depending on the value of showNames.
   */
  public String getShortSignature(boolean showNames) {
    StringBuilder builder = new StringBuilder();
    boolean hasSelf = hasSelfArgument();

    builder.append(getFullName()).append("(");
    signature.toStringBuilder(builder, showNames, false, false, hasSelf);
    builder.append(")");

    return builder.toString();
  }

  /**
   * Prints the types of the first {@code howManyArgsToPrint} given arguments as
   * "(type1, type2, ...)"
   */
  protected String printTypeString(Object[] args, int howManyArgsToPrint) {
    StringBuilder builder = new StringBuilder();
    builder.append("(");

    int start = hasSelfArgument() ? 1 : 0;
    for (int pos = start; pos < howManyArgsToPrint; ++pos) {
      builder.append(EvalUtils.getDataTypeName(args[pos]));

      if (pos < howManyArgsToPrint - 1) {
        builder.append(", ");
      }
    }
    builder.append(")");
    return builder.toString();
  }

  @Override
  public boolean equals(@Nullable Object other) {
    if (other instanceof BaseFunction) {
      BaseFunction that = (BaseFunction) other;
      // In theory, the location alone unambiguously identifies a given function. However, in
      // some test cases the location might not have a valid value, thus we also check the name.
      return Objects.equals(this.name, that.name) && Objects.equals(this.location, that.location);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, location);
  }

  @Nullable
  public Location getLocation() {
    return location;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.append(buffer, "<function " + getName() + ">");
  }
}
